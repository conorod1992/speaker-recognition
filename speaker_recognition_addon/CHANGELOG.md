# Changelog

## 1.1.0

- Make the app directory a complete Supervisor Docker build context.
- Replace deprecated `build.yaml` configuration with a multi-architecture Dockerfile.
- Use a Debian base and CPU-only PyTorch packages compatible with `amd64` and `aarch64`.
- Register and smoke-test the s6 service and preserve embeddings under `/share`.
