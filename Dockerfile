# --- БАЗОВЫЙ ОБРАЗ ---
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

# --- СИСТЕМНЫЕ ПАКЕТЫ ---
# ffmpeg — обязателен (moviepy, whisper, spoof_metadata);
# libGL/libglib — нужны opencv; git — для установки whisper из репозитория.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- ЗАВИСИМОСТИ (отдельным слоем, чтобы кэшировались) ---
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --- ПРОЕКТ ---
COPY . .

# --- ПРОВЕРКА СБОРКИ ---
RUN python -c "import moviepy, whisper; print('MoviePy and Whisper installed')"

# Каналы монтируются снаружи: docker run -v "$PWD/channels:/app/channels" ...
VOLUME ["/app/channels"]

CMD ["python", "app.py"]
