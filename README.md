# 🎬 AI Video Clipper

**English** · [Русский](README.ru.md)

**Long video → vertical Shorts.** Multi-signal moment scoring instead of prompting an LLM, plus a
virtual camera that reframes 9:16 like a human operator. One `config.yaml` drives the whole pipeline.

<p align="center">
  <a href="https://youtu.be/xu8sg_mXEh0">
    <img src="https://img.youtube.com/vi/xu8sg_mXEh0/maxresdefault.jpg" width="720"
         alt="Video walkthrough of the project — click to watch on YouTube">
  </a>
</p>
<p align="center">
  <b><a href="https://youtu.be/xu8sg_mXEh0">▶&nbsp; Watch the walkthrough — «Аниме ЗАВОД»</a></b><br>
  <sub>In Russian · on <a href="https://www.youtube.com/@AnimeFactorio">@AnimeFactorio</a>, the channel this was built for</sub>
</p>

A config-driven factory for short vertical videos. Feed it a long episode — get back ready-to-post
9:16 clips with a virtual camera, word-by-word subtitles, a watermark, cleaned metadata, and delivery
to Telegram.

The whole system is declarative: a channel's behaviour is described by a single `config.yaml`,
and the `pipeline` list inside it turns stages on and off.

```
episode.mp4  →  transcript  →  signals (audio / faces / cuts / tempo / hooks)  →  LLM moment selection
             →  segment assembly  →  virtual camera 9:16  →  layers (subtitles, watermark, music)
             →  metadata rebuild  →  Telegram
```

> **This is a showcase branch.** It is a trimmed-down MVP of a larger private project: it keeps the
> download → edit → deliver path and drops everything else (YouTube/TikTok upload automation,
> scheduling, analytics loop). All configuration values are reset to neutral defaults and the
> repository contains no API keys, no credentials, and no third-party media.

---

## 🧭 Why this exists

Cutting a Shorts clip by hand is not difficult. It is slow, and it is slow in a way that never gets
faster with practice: watch a 24-minute episode, find the two moments that still land without the
surrounding context, cut them, reframe 16:9 to a phone screen, time the subtitles word by word, export.
Roughly an hour per clip, almost all of it mechanical.

The obvious first attempt was to hand the transcript to a language model and ask for the best moments.
It produced consistently unusable output. The model picks lines that only work if you have seen the
episode, mistakes loud for dramatic, and has no idea what a scene looks like once it is cropped to a
vertical frame. Adding more prompt did not fix it, because the problem was not the wording — the model
was being asked a question the transcript could not answer.

What actually worked was to stop asking one model to be right about everything. The scene is described
along several independent axes first — speech density, audio peaks, face presence, scene cuts, tempo,
hooks — and candidate windows are ranked deterministically **before** the model sees any of them. The
model then does the one thing it is genuinely good at, judging whether a moment is interesting, and
nothing else. Whatever it returns is validated hard afterwards: durations are recomputed from the
segments, boundaries are snapped to speech pauses and cuts, and anything out of range is dropped.

Vertical framing turned out to be a separate problem of the same shape. A centred crop throws away half
the frame, and naively following the detected face gives a camera that twitches on every detection and
is genuinely unpleasant to watch. The fix was to stop treating it as a detection problem and model the
camera itself — a damped oscillator chasing a filtered target, with a dead zone, a speed limit, a
predictive lead, rule-of-thirds bias, and a Ken Burns drift for when there is no face to follow at all.

The last lesson was operational rather than algorithmic. A pipeline where a failure at stage nine costs
you stages one through eight is a pipeline nobody iterates on. So every stage writes an explicit
intermediate artifact, every optional dependency degrades instead of raising, stages can be switched on
and off individually, and a debug mode runs any single stage against a sandbox folder. That is why the
whole thing is driven by one `config.yaml` instead of command-line flags: the interesting work is
tuning, and tuning needs a file you can diff.

---

## 📚 Articles about the project

The engineering decisions behind this pipeline — the ones you cannot read off the code — are written up
on Habr.

**In English**

| Article | What it covers |
|---|---|
| [How I Built an "Anime Factory"](https://habr.com/ru/articles/1022250/) | System overview: why a monolithic end-to-end model loses to a set of independent signals, fail-soft over fail-fast, explicit intermediate artifacts |
| [I Taught a Virtual Camera to Behave Like a Human Operator](https://habr.com/ru/articles/1022298/) | Face tracking and 9:16 framing: detector cascade, tracking, stabilization, the camera as a damped oscillator, composition rules, Ken Burns as a fallback |

**In Russian**

| Article | What it covers |
|---|---|
| [Как я построил «аниме-завод»](https://habr.com/ru/articles/1014810/) | Обзор всей системы: три контура, почему монолитная end-to-end модель проигрывает набору независимых сигналов |
| [Как я выбираю моменты для Shorts](https://habr.com/ru/articles/1021552/) | Слой выбора момента: пять независимых источников сигнала, отбор через фильтрацию плохих вариантов, момент как монтажная сборка |
| [Я научил виртуальную камеру быть оператором](https://habr.com/ru/articles/1021278/) | Алгоритм face tracking и кадрирования 9:16 |

The thread running through all of them: **no single signal is sufficient.** A loud moment is not an
interesting one, a good line of dialogue is not one that works without context, and a perfect face
detection does not give you pleasant camera movement. The system is built out of weak signals combined,
and out of degrading correctly when one of them drops out.

---

## 🚀 Quick start

### Requirements

- **Python 3.12+**
- **`ffmpeg` and `ffprobe` on `PATH`** — invoked as commands by `ingestion/transcriber.py`,
  `analysis/audio_analyzer.py` and `publishing/spoof_metadata.py`. The binary bundled with
  `imageio-ffmpeg` is not enough: it has no `ffprobe`.
- **`OPENAI_API_KEY`** — validated at `config.py` import time; the app will not start without it.

### Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Installing ffmpeg:

```bash
winget install Gyan.FFmpeg   # Windows
brew install ffmpeg          # macOS
sudo apt install ffmpeg      # Debian/Ubuntu
```

Restart your terminal afterwards and verify: `ffmpeg -version`, `ffprobe -version`.

### Environment

```bash
cp .env.example .env
```

```ini
OPENAI_API_KEY=sk-...     # required
TELEGRAM_BOT_TOKEN=       # optional — only for the telegram_notify stage
TELEGRAM_CHAT_ID=
KODIK_TOKEN=              # optional — only for the kodik_download stage
```

`.env` is git-ignored. **No token is ever read from `config.yaml`** — secrets live in the environment only.

### Run

```bash
python app.py
```

`app.py` validates (and creates, where missing) the directory structure of every channel under
`channels/`, then runs each valid channel through its own `pipeline`. One channel failing does not
take down the others: the exception is logged and the channel lands in a summary list at the end.

> **Windows / PyCharm:** run with *this* project's interpreter (`.venv\Scripts\python.exe`), otherwise
> you get `ModuleNotFoundError: No module named 'yaml'`. Working directory must be the repo root.

---

## 🧩 Repository layout

```
app.py                          # entry point: channel validation + pipeline run
config.py                       # env, global paths, DEFAULT_CONFIG for new channels
core/channel_processor.py       # orchestrator for a single channel

infrastructure/
  check_structure.py            # creates missing folders and config.yaml

ingestion/
  autodownload.py               # episode download via Kodik
  parser.py                     # input video discovery
  transcriber.py                # ffmpeg audio extraction + Whisper

analysis/
  audio_analyzer.py             # RMS / ZCR / spectral centroid, loudness peaks
  moment_scorer.py              # deterministic scoring of candidate windows
  moment_validator.py           # repair and validation of what the LLM returned
  gpt_analyzer.py               # signal assembly and LLM moment selection
  subtitles_cleaner.py          # LLM post-processing of subtitle text

rendering/
  video_editor.py               # make_clips: the main render pipeline
  face_detector.py              # face detection + 9:16 virtual camera
  layers/                       # subtitle_renderer, title_renderer, watermark_renderer
  audio/enhancer.py             # soft-knee compressor for the dialogue track
  support/helper.py             # safe_filename, word chunking, transparent window

publishing/
  spoof_metadata.py             # container rebuild and metadata cleanup
  telegram_notifier.py          # media group delivery to Telegram

channels/DemoChannel/           # example channel: fully commented config.yaml
test_data/, output_test_data/   # sandbox for debug mode
docs/                           # ARCHITECTURE.md, CONFIG_REFERENCE.md, RUNBOOK.md
```

### Channel layout

```
channels/<ChannelName>/
├── config.yaml
├── input_videos/          # source episodes
├── output_clips/          # finished clips
├── assets/
│   ├── backgrounds/       # only needed when use_blurred_background: false
│   ├── musics/            # a track is picked at random
│   └── fonts/             # a font is picked at random per run
└── logs/
```

Missing folders and a baseline `config.yaml` (from `DEFAULT_CONFIG` in `config.py`) are created
automatically. If `channel_name` in the config does not match the folder name, you get a warning.

`channels/DemoChannel/assets/` ships with one open font and no media — bring your own music and
backgrounds; the repository deliberately carries no third-party licensed assets.

---

## ⚙️ Pipeline stages

Stages are listed in `channels/<Channel>/config.yaml → pipeline`.

| Stage | Module | What it does |
|---|---|---|
| `kodik_download` | `ingestion/autodownload.py` | Downloads episodes by title list via Kodik (legacy name `autodownload` still accepted) |
| `transcribe_video` | `ingestion/transcriber.py` | Extracts the first audio track via ffmpeg (16 kHz mono PCM) and transcribes it with Whisper |
| `analyze_moment` | `analysis/gpt_analyzer.py` | Builds the multi-signal payload and gets a list of moments with segments back from the LLM |
| `make_clips` | `rendering/video_editor.py` | Cuts and joins segments, assembles the final 1080×1920 canvas |
| `dynamic_shorts` | `rendering/face_detector.py` | Virtual camera: 9:16 framing that follows faces |
| `watermark` | `rendering/layers/watermark_renderer.py` | Text watermark with shadow and opacity |
| `enhance_audio` | `rendering/audio/enhancer.py` | Soft-knee compression of the original track, lifting quiet passages |
| `video_effects` | `rendering/video_editor.py` | Mirror, saturation, gamma, contrast, resize, FPS, speed |
| `music` | `rendering/video_editor.py` | Mixes in a random track from `assets/musics` |
| `subtitles` | `rendering/layers/subtitle_renderer.py` | Re-transcribes the clip, cleans the text with an LLM, renders word by word with `<hl>` highlighting and colour emoji |
| `title` | `rendering/layers/title_renderer.py` | Title card with automatic font sizing |
| `spoof_metadata` | `publishing/spoof_metadata.py` | Strips all metadata and rebuilds the container |
| `telegram_notify` | `publishing/telegram_notifier.py` | Sends new clips to a chat as a media group |

**Things worth knowing:**

- `pipeline` is a **set of enabled-stage flags, not an execution order** — the actual order is fixed in
  `channel_processor.py` and `make_clips()`.
- If at least one "feature" stage is enabled (`dynamic_shorts`, `video_effects`, `speed`, `watermark`,
  `enhance_audio`, `music`, `subtitles`, `title`), then **only** the stages listed in `pipeline` run —
  the rest are off even if their config section says `enabled: true`.
- If post-clip stages are enabled but `make_clips` is not in `pipeline`, it **runs anyway** — otherwise
  there would be nothing to apply the effects to.
- `spoof_metadata` deletes the original only after the container has been rebuilt successfully: an
  ffmpeg failure never costs you the clip.
- `telegram_notify` sends **new** files only — the `*.mp4` listing is snapshotted before processing.

---

## 🎯 How a moment gets picked

Full write-up: [«Как я выбираю моменты для Shorts»](https://habr.com/ru/articles/1021552/) (RU).

The naive approach — hand the transcript to a model — fails. The LLM picks lines that only work in the
context of the whole episode, mistakes loudness for drama, and cannot see how a scene looks in vertical.

So `analysis/gpt_analyzer.py` gives the model not raw text but a **pre-processed description of the
scene along several axes**:

| Signal | Source | What is extracted |
|---|---|---|
| **Speech** | Whisper | Timestamped segments, word density |
| **Audio** | `analyze_audio_peaks` | RMS, zero-crossing rate, spectral centroid, peak events with strength labels |
| **Faces** | `analyze_face_activity` | Face presence intervals, count, maximum frame coverage |
| **Visual** | `_detect_cuts` + `_build_visual_summary` | Scene changes from frame difference, motion intensity |
| **Tempo** | `_build_tempo_windows` | 5-second windows: speech density, pauses, cuts |
| **Emotion** | `_extract_emotion_peaks` | Intersection of textual markers with audio peaks |
| **Hooks** | `_extract_hooks` | Open questions, promises, unresolved conflicts |

These are ranked deterministically by `analysis/moment_scorer.py` before the LLM ever sees them, so the
prompt carries only the top candidates — cheaper, and far more stable than free-form selection.

Then comes **hard validation after the model answers** (`analysis/moment_validator.py`): each moment's
`duration` is recomputed as the sum of its segments, boundaries are snapped to speech pauses and scene
cuts, and moments outside `gpt.min_time`–`gpt.max_time` are dropped. If nothing survives, the request is
retried (up to 3 times) with a clarifying suffix. Network errors, rate limits and 502s are retried with
growing backoff; malformed JSON and an expired token fail immediately.

A moment is a **montage assembly**: `segments` may come from different parts of the episode, pauses are
cut out, the hook is always in the first segment, and the last one is a cliffhanger or a loop back to
the start.

The full payload sent to the model is saved next to the clips as `<video>_chatgpt_payload.txt` — the
"explicit intermediate artifact" without which debugging moment selection is impossible.

---

## 🎥 Virtual camera (`dynamic_shorts`)

Full write-up: [I Taught a Virtual Camera to Behave Like a Human Operator](https://habr.com/ru/articles/1022298/) (EN).

A centred 16:9 → 9:16 crop throws away half the frame, and naively following a face gives you a jittery
camera.

```
detection → tracking → stabilization → virtual camera → interpolation → crop
```

1. **Detector cascade** (`_DetectorBackend`): **MediaPipe** → **YuNet** (ONNX via OpenCV) →
   **Haar Cascade**. Any backend failing does not break the stage.
2. **Tracking** — nearest-neighbour within `match_tolerance`, with a `max_miss_time` grace period, so a
   brief loss of the face does not make the camera jump.
3. **Stabilization** — anti-jerk (a hard per-frame step limit) plus a `face_filter` low-pass.
4. **The camera as a damped oscillator**: `follow_stiffness` (tripod springiness), `follow_damping`,
   `max_center_speed` / `max_center_accel` (limits), `predictive_lead` (the operator looks slightly
   ahead), `human_lag` (a micro-delay that makes movement feel alive).
5. **Composition rules**: `side_bias` / `side_bias_strength` (rule of thirds), `eye_level_lift` (aim at
   the eyes, not the centre of the face), `center_dead_zone` (ignore micro-movement), `face_margin`
   (clearance from the edges).
6. **Ken Burns as a fallback**: with no faces in frame the camera does not freeze — it pans and zooms
   gently (`ken_burns_*`; zero the amplitudes to lock the frame instead).
7. **Interpolation**: analysis runs at `analysis_fps` (8 fps by default), camera states are interpolated
   up to the full frame rate, then the ROI is computed and the crop applied.

---

## 🛠 Configuration

Full reference: [`docs/CONFIG_REFERENCE.md`](docs/CONFIG_REFERENCE.md).
Live, fully commented example: [`channels/DemoChannel/config.yaml`](channels/DemoChannel/config.yaml).

| Section | Purpose |
|---|---|
| `video` | Output canvas size (1080×1920 by default) |
| `background` | Blurred background from the source (`use_blurred_background`, `blur_strength`), a static image, or a transparent PNG window; `fit_mode: contain` or `cover` |
| `dynamic_shorts` | Everything about detection, tracking, camera physics, composition and Ken Burns |
| `watermark` | Text, colour, size, opacity, shadow, position, padding |
| `video_effects` | `mirror`, `color_saturation`, `gamma`, `contrast`, `resize`, `frame_rate` (number or `random`), `speed` |
| `audio_enhancer` | Compressor: `threshold`, `ratio`, `soft_knee`, `makeup_gain`, `noise_floor`, `output_ceiling` |
| `music` | Volume of the mixed-in track relative to the original |
| `subtitles` | Font, colour, stroke, shadow, alignment, `words_per_chunk`, `<hl>` highlight style, Whisper and LLM models, prompts (`normal` / `meta_ad`) |
| `moment_scoring` | Signal weights and candidate selection before the LLM |
| `moment_validation` | Boundary snapping and rejection rules after the LLM |
| `gpt` | Model, `min_time`/`max_time` and `min_count`/`max_count` ranges, audience, platform, tone, and the main selection prompt |
| `kodik_download` | Title list in the form `["Title", "1-3", "Dub studio"]` |

On subtitles specifically: with `improve_transcript_quality: true` the clip is **re-transcribed** with a
heavier model (`enhanced_whisper_model`), then the text is cleaned by an LLM (`enhancer_model`) — spelling
is fixed and exactly one key word is wrapped in `<hl>…</hl>`, which the renderer paints in a separate
style. Colour emoji are rendered via Pilmoji using the font at `emoji_font_path`.

---

## 🧪 Debug mode

Set `debug: true` in a channel config:

- input comes from `test_data/`, output goes to `output_test_data/`;
- Kodik download is disabled;
- intermediate artifacts are written: `<video>_transcript.txt`, `<video>_moments.json`,
  `<video>_chatgpt_payload.txt`;
- if moments were not computed but post-clip stages are enabled, the whole video is used as one segment,
  so subtitles, watermark and effects can each be debugged in isolation.

This follows directly from the principle in the first article: every stage must be debuggable on its own,
without recomputing all the others.

---

## ✅ Tests and CI

```bash
pip install -r requirements-ci.txt

pytest                    # everything (~60 s)
pytest -m "not render"    # fast layer, no real rendering (~30 s)
pytest -m render          # only stages that actually render a clip (~28 s)
```

The tests are hermetic: OpenAI, Telegram and Kodik are mocked, and the `whisper` module is replaced by a
stub in `tests/conftest.py` (otherwise CI would pull torch and download models). `mediapipe` and
`anime_parsers_ru` are deliberately not installed in CI — the code has to degrade to the fallback
detectors, and that is part of what is being checked.

| File | What it covers |
|---|---|
| `test_check_structure.py` | Channel structure creation and the default `config.yaml` |
| `test_channel_processor.py` | Orchestration: which stages run, automatic `make_clips`, debug mode, clip replacement after `spoof_metadata` |
| `test_transcriber.py` | ffmpeg track extraction, Whisper model choice, text and `<hl>` normalization |
| `test_audio_analyzer.py` | RMS / ZCR / spectral centroid, peak-to-event merging |
| `test_moment_scorer.py` | Signal time grid, window scoring, NMS candidate selection |
| `test_moment_validator.py` | Segment repair, rejection of empty moments, heuristic top-up |
| `test_gpt_analyzer.py` | Payload assembly, retries, fallback on a garbage model response |
| `test_face_detector.py` | Detector cascade, hero-selection hysteresis, Ken Burns, camera interpolation |
| `test_rendering_layers.py` | Subtitles, title card, watermark, dialogue compressor |
| `test_publishing.py` | Metadata rebuild and Telegram media group delivery |
| `test_channel_configs.py` | Channel configs in the repo: known stages, prompt rendering, font presence |
| `test_render_stages.py` | **Every `make_clips` stage really renders a valid mp4** |
| `test_pipeline_e2e.py` | End-to-end channel run: video → transcript → clip → metadata → Telegram |

CI (`.github/workflows/ci.yml`) runs module compilation, the fast layer and the render layer as separate
steps on every push and PR — so you can see immediately whether logic or rendering broke.

---

## 📤 Delivery

- **Telegram** works out of the box once `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set: new clips
  are sent as a media group with moment titles and a normalized title name.
- **`spoof_metadata`** rebuilds the container without re-encoding (`-c copy`), wipes the original
  metadata (`-map_metadata -1`) and writes plausible editor metadata in its place.
- **Upload automation for YouTube and TikTok is not part of this branch.** It lives in the private
  project; the showcase stops at "a finished clip on disk, delivered to Telegram".

---

## 🐳 Docker

```bash
docker build -t ai-video-clipper .
docker run --rm --env-file .env -v "$PWD/channels:/app/channels" ai-video-clipper
```

The image installs ffmpeg and the OpenCV system libraries. Mount `channels/` from the host so that
source videos and rendered clips stay outside the container.

---

## ⚠️ Known limitations

- `OPENAI_API_KEY` is required even for runs without an LLM — the check sits at `config.py` import time.
- The **legacy** `openai==0.28.0` API (`openai.ChatCompletion`) is used, not the current SDK.
- Kodik is blocked by many ISPs: the resolver for `kodik-api.com` is checked before the request, and if
  it is unreachable the stage is skipped softly and the pipeline continues with already-downloaded videos.
- `pilmoji 2.0.4` requires `emoji 1.x` — the version is pinned in `requirements.txt`; do not bump it blindly.
- Whisper models are downloaded on first run; `large-v3-turbo` is significantly heavier than `base`.

---

## 📖 Documentation

The docs below are in Russian.

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — modules and data flow
- [`docs/CONFIG_REFERENCE.md`](docs/CONFIG_REFERENCE.md) — `config.yaml` reference
- [`docs/RUNBOOK.md`](docs/RUNBOOK.md) — running the pipeline and diagnosing common problems

## 🔁 Suggested workflow

1. Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and the articles above.
2. Create a channel under `channels/`, fill in `assets/` (fonts, music, backgrounds if needed).
3. Do a debug run against `test_data/` with `debug: true` and inspect the intermediate artifacts.
4. Switch to `debug: false`, drop episodes into `input_videos/`, run `python app.py`.
5. Check the result in `channels/<Channel>/output_clips/`.

---

## 📄 License

Distributed under the **I_Alakey License** — see [`LICENSE`](LICENSE).

### Bundled fonts

Both fonts shipped in `assets/fonts/` are under the **SIL Open Font License 1.1**, which permits
redistribution. Full license texts live in [`assets/fonts/LICENSES/`](assets/fonts/LICENSES).

| Font | Used for | Source |
|---|---|---|
| Montserrat ExtraBold | subtitles, titles, watermark | [google/fonts](https://github.com/google/fonts/tree/main/ofl/montserrat) — static instance at `wght=800` |
| Noto Color Emoji | colour emoji inside subtitles | [google/fonts](https://github.com/google/fonts/tree/main/ofl/notocoloremoji) |

`make_clips` picks a font **at random** from the channel's `assets/fonts/`, so dropping more faces in
there gives you per-run variety. Keep whatever you add redistributable — "free for personal and
commercial use" usually covers *using* a font, not shipping the file in a public repository.
