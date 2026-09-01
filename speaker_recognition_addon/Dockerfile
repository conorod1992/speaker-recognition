FROM ghcr.io/home-assistant/base-debian:bookworm AS builder

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        python3 \
        python3-dev \
        python3-venv \
    && python3 -m venv /opt/speaker-recognition \
    && /opt/speaker-recognition/bin/pip install --no-cache-dir --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN /opt/speaker-recognition/bin/pip install --no-cache-dir -r requirements.txt
# Keep the RNNoise runtime packaged for future experimentation, but production
# startup/API code does not import it or allocate RNNoise state.
RUN /opt/speaker-recognition/bin/pip install --no-cache-dir --no-deps pyrnnoise==0.4.3


FROM ghcr.io/home-assistant/base-debian:bookworm

ARG BUILD_ARCH="unknown"
ARG BUILD_VERSION="3.0.0"

LABEL \
    io.hass.type="app" \
    io.hass.version="${BUILD_VERSION}" \
    io.hass.arch="${BUILD_ARCH}" \
    org.opencontainers.image.title="Home Assistant App: Speaker Recognition" \
    org.opencontainers.image.description="Speaker recognition service using voice embeddings for identification." \
    org.opencontainers.image.source="https://github.com/conorod1992/speaker-recognition" \
    org.opencontainers.image.licenses="MIT"

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libsndfile1 \
        python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/speaker-recognition /opt/speaker-recognition
COPY speaker_recognition ./speaker_recognition

ENV PATH="/opt/speaker-recognition/bin:${PATH}"

COPY rootfs /

RUN chmod a+x /etc/s6-overlay/s6-rc.d/speaker-recognition/run \
    && SPEAKER_RECOGNITION_SKIP_WARMUP=1 /opt/speaker-recognition/bin/python -c \
        "from speaker_recognition.api import app; from speaker_recognition.neural_denoise import rnnoise_frame_size; assert app.title and rnnoise_frame_size() == 480"
