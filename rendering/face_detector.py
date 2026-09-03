"""
Dynamic Shorts (v5) — «виртуальный оператор» для кропа 16:9 → 9:16.

Камера моделируется как физическая система (демпфированный осциллятор), а не
как мгновенное позиционирование по последнему детекту:

- каскад детекторов MediaPipe → YuNet → Haar (fail-soft, а не «слепота»);
- выбор главного лица по saliency с гистерезисом (без пинг-понга между героями);
- анти-джерк, low-pass фильтр, dead zone, ограничение скорости и ускорения;
- rule-of-thirds side bias, eye-level lift, защитные поля по краям;
- зум под целевую крупность лица (плавный, с ограничением скорости);
- жёсткая пересборка кадра на монтажной склейке (камера не «переезжает» через
  смену сцены);
- fallback Ken Burns вокруг последнего известного положения лица, а не прыжок
  в центр кадра.
"""

from __future__ import annotations

import bisect
import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from moviepy.editor import VideoClip, VideoFileClip

try:
    import mediapipe as mp
except ImportError:
    mp = None


# ============================================================
# Data structures
# ============================================================

@dataclass
class DynamicCropResult:
    clip: VideoClip
    analysis_points: int
    used_face_track: bool
    scaled_width: float
    crop_width: float
    scene_cuts: int = 0
    face_switches: int = 0
    zoom_range: Tuple[float, float] = (1.0, 1.0)
    camera_states: Tuple["CameraState", ...] = ()


@dataclass
class FaceDetection:
    center: np.ndarray
    size: np.ndarray
    score: float
    box: np.ndarray


@dataclass
class CameraState:
    time: float
    center: np.ndarray
    zoom: float
    has_face: bool


# ============================================================
# Профили поведения камеры
# ============================================================

# Базовые значения — «безопасный оператор». Профиль переопределяет часть из
# них, а явные ключи из config.yaml переопределяют профиль.
BASE_CAMERA_DEFAULTS: Dict[str, object] = {
    "target_width": 1080,
    "target_height": 1920,

    # анализ
    "analysis_fps": 8,
    "min_face_ratio": 0.06,
    "min_face_confidence": 0.75,
    "match_tolerance": 0.18,
    "max_miss_time": 2.5,
    "face_switch_margin": 0.15,
    "face_switch_hold_s": 0.5,

    # фильтрация входного сигнала
    "face_filter": 0.78,
    "max_face_step": 0.04,
    "stabilization_strength": 0.2,

    # движение камеры
    "smoothing": 0.92,
    "follow_stiffness": 9.0,
    "follow_damping": 2.2,
    "max_center_accel": 1.0,
    "max_center_speed": 0.45,
    "velocity_decay": 0.10,
    "velocity_soften": 0.8,
    "predictive_lead": 0.08,
    "human_lag": 0.03,
    "center_dead_zone": 0.035,

    # композиция
    "single_face_centering": True,
    "side_bias": 0.22,
    "side_bias_strength": 0.35,
    "eye_level_lift": 0.12,
    "face_margin": 0.08,

    # зум
    "face_coverage": 0.0,       # целевая высота лица в кадре (0 — зум выключен)
    "max_zoom": 1.0,
    "min_zoom": 1.0,
    "zoom_smoothing": 0.75,
    "max_zoom_speed": 0.25,

    # монтажные склейки
    "scene_cut_detection": True,
    "scene_cut_threshold": 0.06,       # абсолютный минимум различия кадров
    "scene_cut_relative": 4.0,         # ...и во сколько раз выше локального фона
    "scene_cut_window": 40,            # окно оценки фона (кадров анализа)
    "scene_cut_min_interval_s": 0.4,

    # fallback
    "fallback_recenter_s": 3.0,
    "ken_burns_period": 12.0,
    "ken_burns_pan_amplitude": 0.03,
    "ken_burns_tilt_amplitude": 0.02,
    "ken_burns_zoom_amplitude": 0.0,
}

CAMERA_PROFILES: Dict[str, Dict[str, float]] = {
    # почти не двигается: разговорные сцены, статичные планы
    "static": {
        "follow_stiffness": 2.0,
        "follow_damping": 5.0,
        "max_center_speed": 0.05,
        "max_center_accel": 0.35,
        "smoothing": 0.95,
        "predictive_lead": 0.0,
        "center_dead_zone": 0.06,
        "face_filter": 0.88,
    },
    # «руки оператора»: живо, но без рывков — режим по умолчанию
    "operator": {
        "follow_stiffness": 8.4,
        "follow_damping": 2.35,
        "max_center_speed": 0.40,
        "max_center_accel": 0.85,
        "smoothing": 0.93,
        "predictive_lead": 0.065,
        "center_dead_zone": 0.052,
        "face_filter": 0.80,
    },
    # экшен: быстрая реакция на перемещения
    "action": {
        "follow_stiffness": 15.0,
        "follow_damping": 1.2,
        "max_center_speed": 0.80,
        "max_center_accel": 1.4,
        "smoothing": 0.86,
        "predictive_lead": 0.10,
        "center_dead_zone": 0.03,
        "face_filter": 0.72,
    },
}


def resolve_camera_config(config: Optional[dict]) -> dict:
    """Собирает финальный конфиг камеры: базовые → профиль → явные ключи."""
    cfg = dict(BASE_CAMERA_DEFAULTS)
    user_cfg = dict(config or {})
    profile_name = str(user_cfg.get("profile", "operator") or "operator").lower()
    cfg.update(CAMERA_PROFILES.get(profile_name, CAMERA_PROFILES["operator"]))
    cfg["profile"] = profile_name if profile_name in CAMERA_PROFILES else "operator"
    for key, value in user_cfg.items():
        if key == "profile" or value is None:
            continue
        cfg[key] = value
    return cfg


# ============================================================
# Detection backend
# ============================================================

class _DetectorBackend:
    """Детектор лиц с fallback'ом: MediaPipe → YuNet → Haar."""

    def __init__(self, min_confidence: float):
        self._min_conf = float(np.clip(min_confidence, 0.0, 1.0))
        self._cascade = None
        self._mpdet = None
        self._yunet = None

        # Haar
        try:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                self._cascade = cascade
        except Exception:
            pass

        # MediaPipe
        if mp is not None:
            try:
                self._mpdet = mp.solutions.face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=max(0.1, self._min_conf),
                )
            except Exception:
                self._mpdet = None

        # YuNet (OpenCV ≥ 4.6)
        try:
            path = cv2.data.haarcascades + "face_detection_yunet_2023mar.onnx"
            self._yunet = cv2.FaceDetectorYN.create(
                model=path,
                config="",
                input_size=(320, 320),
                score_threshold=self._min_conf,
                nms_threshold=0.3,
                top_k=5000,
            )
        except Exception:
            self._yunet = None

    def close(self):
        if self._mpdet is not None:
            try:
                self._mpdet.close()
            except Exception:
                pass

    def detect(self, frame_rgb: np.ndarray, min_size: float) -> List[FaceDetection]:
        h, w = frame_rgb.shape[:2]
        out: List[FaceDetection] = []

        # MediaPipe
        if self._mpdet is not None:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            try:
                res = self._mpdet.process(frame_bgr)
            except Exception:
                res = None
            if res and res.detections:
                for d in res.detections:
                    box = d.location_data.relative_bounding_box
                    if box.width <= 0 or box.height <= 0:
                        continue
                    if min(box.width * w, box.height * h) < min_size:
                        continue
                    score = float(d.score[0]) if d.score else 0.0
                    x1, y1 = box.xmin, box.ymin
                    x2, y2 = x1 + box.width, y1 + box.height
                    center = np.array([x1 + box.width / 2, y1 + box.height / 2], np.float32)
                    size = np.array([box.width, box.height], np.float32)
                    out.append(FaceDetection(center, size, score, np.array([x1, y1, x2, y2], np.float32)))

        # YuNet fallback
        if self._yunet is not None and not out:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            self._yunet.setInputSize((w, h))
            try:
                _, faces = self._yunet.detect(frame_bgr)
            except Exception:
                faces = None
            if faces is not None:
                for f in faces:
                    x, y, fw, fh, conf = f[:5]
                    if conf < self._min_conf or min(fw, fh) < min_size:
                        continue
                    cx = (x + fw / 2) / w
                    cy = (y + fh / 2) / h
                    out.append(FaceDetection(np.array([cx, cy], np.float32),
                                             np.array([fw / w, fh / h], np.float32),
                                             float(conf),
                                             np.array([x / w, y / h, (x + fw) / w, (y + fh) / h], np.float32)))

        # Haar fallback
        if not out and self._cascade is not None:
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            faces = self._cascade.detectMultiScale(gray, 1.1, 5, minSize=(int(min_size), int(min_size)))
            for (x, y, fw, fh) in faces:
                cx = (x + fw / 2) / w
                cy = (y + fh / 2) / h
                out.append(FaceDetection(np.array([cx, cy], np.float32),
                                         np.array([fw / w, fh / h], np.float32),
                                         1.0,
                                         np.array([x / w, y / h, (x + fw) / w, (y + fh) / h], np.float32)))

        out.sort(key=lambda d: d.score, reverse=True)
        return out


# ============================================================
# Helpers
# ============================================================

_FRAME_CENTER = np.array([0.5, 0.5], np.float32)

# Коэффициенты сглаживания в конфиге заданы для анализа на 8 fps: пересчитываем
# их под фактический шаг, чтобы поведение камеры не менялось вместе с analysis_fps.
_REFERENCE_FPS = 8.0


def _fps_adjusted(alpha: float, dt: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 0.999))
    if alpha <= 0.0:
        return 0.0
    return float(alpha ** (dt * _REFERENCE_FPS))


def _face_saliency(det: FaceDetection) -> float:
    """Насколько лицо «главное» в кадре: уверенность + крупность + центральность."""
    area = float(max(0.0, det.size[0] * det.size[1]))
    area_norm = min(1.0, math.sqrt(area) / 0.35)
    dist = float(np.linalg.norm(det.center - _FRAME_CENTER)) / 0.7071
    return 0.45 * float(det.score) + 0.35 * area_norm + 0.20 * (1.0 - min(1.0, dist))


def _select_primary_face(
        dets: Sequence[FaceDetection],
        tracked: Optional[FaceDetection],
        *,
        match_tolerance: float,
        switch_margin: float,
        switch_pending: float,
        switch_hold_s: float,
        dt: float,
) -> Tuple[Optional[FaceDetection], float, bool]:
    """Выбирает главное лицо с гистерезисом.

    Возвращает (лицо, накопленное время «желания переключиться», был ли скачок
    фокуса). Переключение на другого героя происходит только если он заметно
    выигрывает по saliency и держит это преимущество ``switch_hold_s`` секунд —
    иначе камера пинг-понгует между лицами.
    """
    if not dets:
        return None, 0.0, False

    best = max(dets, key=_face_saliency)
    if tracked is None:
        return best, 0.0, True

    dists = [float(np.linalg.norm(d.center - tracked.center)) for d in dets]
    nearest_idx = int(np.argmin(dists))
    if dists[nearest_idx] > match_tolerance:
        # трек потерян — переприцеливаемся на самое «главное» лицо
        return best, 0.0, True

    continued = dets[nearest_idx]
    if best is not continued and _face_saliency(best) > _face_saliency(continued) + switch_margin:
        switch_pending += dt
        if switch_pending >= switch_hold_s:
            return best, 0.0, True
        return continued, switch_pending, False

    return continued, 0.0, False


def _downscale_gray(frame: np.ndarray, size: int = 32) -> np.ndarray:
    if frame.ndim == 3:
        gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
    else:
        gray = frame.astype(np.float32, copy=False)
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size), np.float32)
    ys = np.linspace(0, h - 1, size).astype(int)
    xs = np.linspace(0, w - 1, size).astype(int)
    return gray[np.ix_(ys, xs)].astype(np.float32, copy=False) / 255.0


def _ken_burns_offset(t: float, period: float, *, pan_amplitude: float, tilt_amplitude: float,
                      zoom_amplitude: float) -> Tuple[np.ndarray, float]:
    period = max(1e-3, float(period))
    phase = (2.0 * math.pi * t) / period
    offset = np.array(
        [pan_amplitude * math.sin(phase), tilt_amplitude * math.cos(phase * 0.7)],
        np.float32,
    )
    zoom = 1.0 + zoom_amplitude * math.sin(phase * 0.85)
    return offset, float(max(0.8, zoom))


def _interpolate_states(states: Sequence[CameraState]) -> Callable[[float], Tuple[np.ndarray, float]]:
    times = [s.time for s in states]
    centers = [s.center for s in states]
    zooms = [s.zoom for s in states]

    def _interp_vec(coll, t):
        if t <= times[0]:
            return coll[0]
        if t >= times[-1]:
            return coll[-1]
        i = bisect.bisect_right(times, t) - 1
        t0, t1 = times[i], times[i + 1]
        w = (t - t0) / (t1 - t0)
        return coll[i] * (1 - w) + coll[i + 1] * w

    def _interp_scalar(coll, t):
        if t <= times[0]:
            return float(coll[0])
        if t >= times[-1]:
            return float(coll[-1])
        i = bisect.bisect_right(times, t) - 1
        t0, t1 = times[i], times[i + 1]
        w = (t - t0) / (t1 - t0)
        return float(coll[i] * (1 - w) + coll[i + 1] * w)

    return lambda t: (_interp_vec(centers, t), _interp_scalar(zooms, t))


def _apply_virtual_camera(clip: VideoClip, path, target_width: int, target_height: int) -> VideoClip:
    if int(round(clip.h)) != target_height:
        clip = clip.resize(height=target_height)
    sw, sh = int(round(clip.w)), int(round(clip.h))

    def _fl(get_frame, t):
        frame = get_frame(t)
        c, z = path(t)
        cx, cy = c * np.array([sw, sh])
        z = max(1e-3, float(z))
        # целочисленный размер окна: при постоянном зуме он не «дышит» ±1px
        cw = int(min(sw, max(2, round(target_width / z))))
        ch = int(min(sh, max(2, round(target_height / z))))
        left = int(np.clip(round(cx - cw / 2.0), 0, sw - cw))
        top = int(np.clip(round(cy - ch / 2.0), 0, sh - ch))
        patch = frame[top:top + ch, left:left + cw]
        # при уменьшении INTER_AREA даёт меньше алиасинга, чем INTER_CUBIC
        interpolation = cv2.INTER_AREA if cw >= target_width else cv2.INTER_CUBIC
        return cv2.resize(patch, (target_width, target_height), interpolation=interpolation)

    return clip.fl(_fl, apply_to=["mask"])


# ============================================================
# Main logic
# ============================================================

def build_dynamic_short_clip(video_clip: VideoClip, config: Optional[dict] = None) -> DynamicCropResult:
    if video_clip is None:
        raise ValueError("video_clip must be provided")

    cfg = resolve_camera_config(config)

    target_width = int(cfg.get("target_width", 1080))
    target_height = int(cfg.get("target_height", 1920))

    analysis_fps = max(1.0, float(cfg.get("analysis_fps", 8)))
    dt = 1.0 / analysis_fps

    min_face_ratio = float(np.clip(cfg.get("min_face_ratio", 0.06), 0.01, 0.6))
    min_face_conf = float(np.clip(cfg.get("min_face_confidence", 0.75), 0.0, 1.0))

    smoothing = float(np.clip(cfg.get("smoothing", 0.92), 0.0, 0.999))
    stabilization_strength = float(np.clip(cfg.get("stabilization_strength", 0.2), 0.0, 1.0))
    velocity_decay = float(np.clip(cfg.get("velocity_decay", 0.10), 0.0, 1.0))
    velocity_soften = float(np.clip(cfg.get("velocity_soften", 0.8), 0.0, 1.0))
    follow_stiffness = float(max(cfg.get("follow_stiffness", 9.0), 0.0))
    follow_damping = float(max(cfg.get("follow_damping", 2.2), 0.0))
    max_center_accel = float(max(cfg.get("max_center_accel", 1.0), 0.0))
    predictive_lead = float(max(cfg.get("predictive_lead", 0.08), 0.0))
    face_filter = float(np.clip(cfg.get("face_filter", 0.78), 0.0, 0.999))
    max_face_step = float(max(cfg.get("max_face_step", 0.04), 0.001))
    face_margin = float(np.clip(cfg.get("face_margin", 0.08), 0.0, 0.49))
    side_bias_strength = float(np.clip(cfg.get("side_bias_strength", 0.35), 0.0, 1.0))
    side_bias = float(np.clip(cfg.get("side_bias", 0.22), 0.0, 0.49))
    single_face_centering = bool(cfg.get("single_face_centering", True))
    center_dead_zone = float(np.clip(cfg.get("center_dead_zone", 0.035), 0.0, 0.35))
    max_center_speed = float(max(cfg.get("max_center_speed", 0.45), 0.0))
    max_miss_time = float(max(cfg.get("max_miss_time", 2.5), 0.0))
    match_tolerance = float(np.clip(cfg.get("match_tolerance", 0.18), 0.01, 1.0))
    human_lag = float(np.clip(cfg.get("human_lag", 0.03), 0.0, 0.5))
    eye_level_lift = float(np.clip(cfg.get("eye_level_lift", 0.12), 0.0, 0.5))
    switch_margin = float(max(cfg.get("face_switch_margin", 0.15), 0.0))
    switch_hold_s = float(max(cfg.get("face_switch_hold_s", 0.5), 0.0))

    # --- зум ---
    target_face_ratio = float(max(cfg.get("face_coverage", 0.0) or 0.0, 0.0))
    max_zoom = float(max(cfg.get("max_zoom", 1.0) or 0.0, 0.0))
    min_zoom = float(np.clip(cfg.get("min_zoom", 1.0), 0.5, 2.5))
    zoom_enabled = target_face_ratio > 0.0 and max_zoom > 1.0
    max_zoom = max(1.0, max_zoom)
    zoom_smoothing = float(np.clip(cfg.get("zoom_smoothing", 0.75), 0.0, 0.999))
    max_zoom_speed = float(max(cfg.get("max_zoom_speed", 0.25), 0.0))

    # --- монтажные склейки ---
    cut_detection = bool(cfg.get("scene_cut_detection", True))
    cut_threshold = float(max(cfg.get("scene_cut_threshold", 0.06), 0.0))
    cut_relative = float(max(cfg.get("scene_cut_relative", 4.0), 0.0))
    cut_window = int(max(cfg.get("scene_cut_window", 40), 4))
    cut_min_interval = float(max(cfg.get("scene_cut_min_interval_s", 0.4), 0.0))

    # --- fallback ---
    fallback_recenter_s = float(max(cfg.get("fallback_recenter_s", 3.0), 0.0))
    ken_burns_period = float(cfg.get("ken_burns_period", 12.0))
    ken_burns_pan = float(cfg.get("ken_burns_pan_amplitude", 0.03))
    ken_burns_tilt = float(cfg.get("ken_burns_tilt_amplitude", 0.02))
    ken_burns_zoom = float(cfg.get("ken_burns_zoom_amplitude", 0.0))

    clip_scaled = video_clip.resize(height=target_height)
    sw, sh = float(clip_scaled.w), float(clip_scaled.h)

    backend = _DetectorBackend(min_face_conf)
    min_size = max(16.0, min(sw, sh) * min_face_ratio)

    camera_states: List[CameraState] = []
    velocity = np.zeros(2, np.float32)
    filtered_face_center: Optional[np.ndarray] = None
    camera_center: Optional[np.ndarray] = None
    smoothed_target: np.ndarray = _FRAME_CENTER.copy()
    camera_zoom = 1.0

    tracked_face: Optional[FaceDetection] = None
    last_seen_time = -1e9
    last_face_count = 0
    switch_pending = 0.0
    last_good_center = _FRAME_CENTER.copy()
    last_face_time = -1e9

    prev_small: Optional[np.ndarray] = None
    recent_diffs: deque = deque(maxlen=cut_window)
    last_cut_time = -1e9
    scene_cuts = 0
    face_switches = 0
    zoom_min_seen = 1.0
    zoom_max_seen = 1.0

    try:
        for i, frame in enumerate(clip_scaled.iter_frames(fps=analysis_fps, dtype="uint8")):
            t = i * dt

            # ----------------------------------------------
            # SCENE CUT: через склейку камера не «переезжает»
            # ----------------------------------------------
            is_cut = False
            if cut_detection:
                small = _downscale_gray(frame)
                if prev_small is not None:
                    diff = float(np.mean(np.abs(small - prev_small)))
                    # порог адаптивный: склейка должна выделяться и по абсолютной
                    # величине, и на фоне текущей динамики сцены — иначе экшен-
                    # сцены целиком превращаются в поток ложных склеек
                    background = float(np.median(recent_diffs)) if recent_diffs else 0.0
                    threshold = max(cut_threshold, cut_relative * background)
                    if diff >= threshold and (t - last_cut_time) >= cut_min_interval:
                        is_cut = True
                        last_cut_time = t
                        scene_cuts += 1
                    recent_diffs.append(diff)
                prev_small = small

            if is_cut:
                # идентичность лица через склейку не сохраняется
                tracked_face = None
                filtered_face_center = None
                switch_pending = 0.0
                velocity[:] = 0.0

            dets = backend.detect(frame, min_size)
            primary, switch_pending, switched = _select_primary_face(
                dets,
                tracked_face,
                match_tolerance=match_tolerance,
                switch_margin=switch_margin,
                switch_pending=switch_pending,
                switch_hold_s=switch_hold_s,
                dt=dt,
            )

            if primary is not None:
                if switched and tracked_face is not None:
                    face_switches += 1
                tracked_face = primary
                last_seen_time = t
                last_face_count = len(dets)
            elif tracked_face is not None and (t - last_seen_time) >= max_miss_time:
                # grace period истёк — забываем лицо
                tracked_face = None
                filtered_face_center = None

            has_face = tracked_face is not None and (t - last_seen_time) < max_miss_time

            single_face_active = single_face_centering and has_face and last_face_count == 1
            effective_face_filter = face_filter
            effective_smoothing = smoothing
            effective_dead_zone = center_dead_zone
            effective_max_speed = max_center_speed
            if single_face_active and stabilization_strength > 0.0:
                effective_face_filter = min(0.995, face_filter + stabilization_strength * 0.1)
                effective_smoothing = min(0.985, smoothing + stabilization_strength * 0.08)
                effective_dead_zone = min(0.35, center_dead_zone + stabilization_strength * 0.08)
                effective_max_speed = max(0.05, max_center_speed * (1.0 - 0.25 * stabilization_strength))

            effective_face_filter = _fps_adjusted(effective_face_filter, dt)
            effective_smoothing = _fps_adjusted(effective_smoothing, dt)
            effective_zoom_smoothing = _fps_adjusted(zoom_smoothing, dt)

            target_zoom = 1.0

            if has_face:
                raw_center = tracked_face.center.astype(np.float32)

                # ------------------------------------------
                # ANTI-JERK + LOW-PASS
                # ------------------------------------------
                if filtered_face_center is None or is_cut or switched:
                    # новая сцена/новый герой: не тянем фильтр из прошлого
                    filtered_face_center = raw_center.copy()
                else:
                    delta = raw_center - filtered_face_center
                    dist = float(np.linalg.norm(delta))
                    if dist > max_face_step:
                        raw_center = filtered_face_center + delta * (max_face_step / dist)
                    filtered_face_center = (
                            filtered_face_center * effective_face_filter
                            + raw_center * (1.0 - effective_face_filter)
                    )

                cx, cy = float(filtered_face_center[0]), float(filtered_face_center[1])

                # ------------------------------------------
                # EYE LEVEL: целимся выше центра лица
                # ------------------------------------------
                if eye_level_lift > 0.0:
                    cy = float(np.clip(cy - tracked_face.size[1] * eye_level_lift, 0.0, 1.0))
                cy = float(np.clip(cy, 0.05, 0.95))

                # ------------------------------------------
                # ADAPTIVE SIDE BIAS (rule of thirds)
                # ------------------------------------------
                if side_bias > 0.0 and not single_face_active:
                    biased_target = 0.5 - side_bias if cx < 0.5 else 0.5 + side_bias
                    edge_proximity = abs(cx - 0.5) * 2.0
                    adaptive_bias = side_bias_strength * (1.0 - edge_proximity)
                    cx = cx * (1.0 - adaptive_bias) + biased_target * adaptive_bias

                target_center = np.array(
                    [float(np.clip(cx, 0.0, 1.0)), float(np.clip(cy, 0.0, 1.0))], np.float32
                )
                last_good_center = target_center.copy()
                last_face_time = t

                # ------------------------------------------
                # ZOOM под целевую крупность лица
                # ------------------------------------------
                if zoom_enabled:
                    face_height = float(max(tracked_face.size[1], 1e-3))
                    target_zoom = float(np.clip(target_face_ratio / face_height, min_zoom, max_zoom))
            else:
                # ------------------------------------------
                # FALLBACK: Ken Burns вокруг последнего лица,
                # с медленным возвратом к центру кадра
                # ------------------------------------------
                anchor = last_good_center
                if fallback_recenter_s > 0 and last_face_time > -1e8:
                    blend = float(np.clip((t - last_face_time) / fallback_recenter_s, 0.0, 1.0))
                    anchor = last_good_center * (1.0 - blend) + _FRAME_CENTER * blend
                offset, kb_zoom = _ken_burns_offset(
                    t,
                    ken_burns_period,
                    pan_amplitude=ken_burns_pan,
                    tilt_amplitude=ken_burns_tilt,
                    zoom_amplitude=ken_burns_zoom,
                )
                target_center = np.clip(anchor + offset, 0.0, 1.0).astype(np.float32)
                target_zoom = kb_zoom if ken_burns_zoom else 1.0
                # в fallback'е дрожать нечему — мёртвую зону ослабляем
                effective_dead_zone *= 0.4

            # ----------------------------------------------
            # CAMERA FOLLOW PHYSICS
            # ----------------------------------------------
            if camera_center is None or is_cut:
                # первый кадр или монтажная склейка: ставим камеру сразу
                camera_center = target_center.copy()
                smoothed_target = target_center.copy()
                velocity[:] = 0.0
                camera_zoom = target_zoom
            else:
                prev_center = camera_center
                # smoothing — это лаг по ЦЕЛИ, а не гашение скорости:
                # если сглаживать саму позицию, камера физически не может
                # догнать героя (скорость умножается на 1 - smoothing).
                smoothed_target = (
                        smoothed_target * effective_smoothing
                        + target_center * (1.0 - effective_smoothing)
                ).astype(np.float32)
                desired = smoothed_target

                # ------------------------------------------
                # DEAD ZONE по ошибке слежения, а не по шагу:
                # мелкие смещения героя игнорируем совсем, а
                # при выходе из зоны тянемся к её границе —
                # иначе камера дрожит на самом краю зоны.
                # ------------------------------------------
                error_vec = desired - prev_center
                error_dist = float(np.linalg.norm(error_vec))
                if error_dist < effective_dead_zone:
                    velocity *= max(0.0, 1.0 - follow_damping * dt)
                    desired = prev_center.copy()
                elif error_dist > 1e-6:
                    desired = prev_center + error_vec * (
                            (error_dist - effective_dead_zone) / error_dist
                    )

                follow_boost, speed_boost = (1.5, 1.4) if error_dist > 0.10 else (1.0, 1.0)

                if predictive_lead > 0.0:
                    desired = np.clip(desired + velocity * predictive_lead, 0.0, 1.0)
                if human_lag > 0.0:
                    desired = desired * (1.0 - human_lag) + prev_center * human_lag

                error = desired - prev_center
                accel = error * follow_stiffness * follow_boost - velocity * follow_damping
                acc_norm = float(np.linalg.norm(accel))
                if max_center_accel > 0.0 and acc_norm > max_center_accel:
                    accel = accel * (max_center_accel / acc_norm)

                velocity = velocity + accel * dt
                velocity *= velocity_soften
                velocity *= max(0.0, 1.0 - velocity_decay)

                speed = float(np.linalg.norm(velocity))
                speed_limit = effective_max_speed * speed_boost
                if speed > speed_limit > 0:
                    velocity = velocity * (speed_limit / speed)

                next_center = prev_center + velocity * dt
                delta = next_center - prev_center
                dist = float(np.linalg.norm(delta))
                max_step = effective_max_speed * dt
                if dist > max_step > 0:
                    next_center = prev_center + delta * (max_step / dist)

                camera_center = next_center.astype(np.float32)

                # плавный зум с ограничением скорости
                smoothed_zoom = (
                        camera_zoom * effective_zoom_smoothing
                        + target_zoom * (1.0 - effective_zoom_smoothing)
                )
                if max_zoom_speed > 0:
                    max_dz = max_zoom_speed * dt
                    smoothed_zoom = float(
                        np.clip(smoothed_zoom, camera_zoom - max_dz, camera_zoom + max_dz)
                    )
                camera_zoom = float(np.clip(smoothed_zoom, min(1.0, min_zoom), max(1.0, max_zoom)))

            # ----------------------------------------------
            # FACE MARGIN GUARD: не режем уши/волосы краем кадра
            # ----------------------------------------------
            cx, cy = float(camera_center[0]), float(camera_center[1])
            if face_margin > 0.0:
                half_crop_w = min(0.5, (target_width / sw) / (2.0 * max(camera_zoom, 1e-3)))
                half_crop_h = min(0.5, (target_height / sh) / (2.0 * max(camera_zoom, 1e-3)))
                guard_x = max(face_margin, half_crop_w)
                guard_y = max(face_margin, half_crop_h)
                cx = float(np.clip(cx, guard_x, 1.0 - guard_x))
                cy = float(np.clip(cy, guard_y, 1.0 - guard_y))
                camera_center = np.array([cx, cy], np.float32)

            zoom_min_seen = min(zoom_min_seen, camera_zoom)
            zoom_max_seen = max(zoom_max_seen, camera_zoom)
            camera_states.append(
                CameraState(t, camera_center.copy(), float(camera_zoom), bool(has_face))
            )
    finally:
        backend.close()

    if not camera_states:
        duration = float(clip_scaled.duration or 1.0)
        center = _FRAME_CENTER.copy()
        camera_states = [
            CameraState(0.0, center, 1.0, False),
            CameraState(duration, center, 1.0, False),
        ]

    path = _interpolate_states(camera_states)
    cropped = _apply_virtual_camera(clip_scaled, path, target_width, target_height)
    used_face_track = any(state.has_face for state in camera_states)

    return DynamicCropResult(
        clip=cropped,
        analysis_points=len(camera_states),
        used_face_track=used_face_track,
        scaled_width=sw,
        crop_width=float(target_width),
        scene_cuts=scene_cuts,
        face_switches=face_switches,
        zoom_range=(round(zoom_min_seen, 3), round(zoom_max_seen, 3)),
        camera_states=tuple(camera_states),
    )


def analyze_face_activity(video_path, cfg) -> dict:
    dynamic_cfg = cfg.get("dynamic_shorts", {}) if isinstance(cfg, dict) else {}
    gpt_cfg = cfg.get("gpt", {}) if isinstance(cfg, dict) else {}

    analysis_interval_s = float(gpt_cfg.get("face_analysis_interval_s", 5.0) or 5.0)
    analysis_interval_s = max(0.1, analysis_interval_s)
    analysis_fps = 1.0 / analysis_interval_s
    min_face_ratio = float(np.clip(dynamic_cfg.get("min_face_ratio", 0.06), 0.01, 0.6))
    min_face_conf = float(np.clip(dynamic_cfg.get("min_face_confidence", 0.75), 0.0, 1.0))
    max_events = int(dynamic_cfg.get("max_face_events", 120))

    events: List[dict] = []
    backend = _DetectorBackend(min_face_conf)
    duration = 0.0
    face_time = 0.0

    try:
        with VideoFileClip(str(video_path)) as clip:
            duration = float(clip.duration or 0.0)
            if duration <= 0:
                return {
                    "events": [],
                    "summary": "no_video",
                }

            frame_duration = analysis_interval_s
            current = None

            sample_count = int(math.ceil(duration / analysis_interval_s))
            for frame_index in range(sample_count):
                t = min(duration, frame_index * analysis_interval_s)
                frame = clip.get_frame(t)
                h, w = frame.shape[:2]
                min_size = max(16.0, min(w, h) * min_face_ratio)
                detections = backend.detect(frame, min_size)
                face_count = len(detections)
                max_ratio = max((float(det.size[0] * det.size[1]) for det in detections), default=0.0)

                if face_count > 0:
                    if current is None:
                        current = {
                            "start": round(t, 3),
                            "end": round(min(t + frame_duration, duration), 3),
                            "faces": face_count,
                            "max_face_ratio": round(max_ratio, 4),
                        }
                    else:
                        current["end"] = round(min(t + frame_duration, duration), 3)
                        current["faces"] = max(current["faces"], face_count)
                        current["max_face_ratio"] = round(
                            max(current["max_face_ratio"], max_ratio), 4
                        )
                elif current is not None:
                    events.append(current)
                    current = None

            if current is not None:
                events.append(current)

    except Exception as exc:
        return {
            "events": [],
            "summary": f"error: {exc}",
        }
    finally:
        backend.close()

    face_time = sum(float(e["end"]) - float(e["start"]) for e in events)

    events.sort(key=lambda e: e["faces"], reverse=True)
    events = events[:max_events]
    events.sort(key=lambda e: e["start"])

    return {
        "events": events,
        "fps": round(analysis_fps, 4),
        "analysis_interval_s": round(analysis_interval_s, 3),
        "duration_s": round(duration, 3),
        "coverage_ratio": round(face_time / duration, 4) if duration > 0 else 0.0,
        "summary": "ok",
    }
