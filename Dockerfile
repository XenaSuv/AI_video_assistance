FROM python:3.12-slim

# System deps: ffmpeg for moviepy, imagemagick for TextClip
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    imagemagick \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

# Allow ImageMagick to read text/@ files (needed by moviepy.TextClip)
RUN sed -i 's|<policy domain="path" rights="none" pattern="@\*"/>||' \
    /etc/ImageMagick-6/policy.xml || true

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/scheduler.py"]
