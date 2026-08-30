FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY speaker_recognition ./speaker_recognition

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN uv pip install --system --no-cache -e ".[server]"

RUN mkdir -p /data/embeddings

EXPOSE 8099

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import json, urllib.request; data=json.load(urllib.request.urlopen('http://localhost:8099/health')); assert data.get('status') == 'healthy'"

ENV HOST=0.0.0.0 \
    PORT=8099 \
    LOG_LEVEL=INFO \
    ACCESS_LOG=true \
    EMBEDDINGS_DIR=/data/embeddings \
    API_TOKEN= \
    ALLOW_INSECURE_REMOTE=false

CMD ["python", "-m", "speaker_recognition"]
