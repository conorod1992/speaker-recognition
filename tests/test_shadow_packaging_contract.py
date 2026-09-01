"""Smoke contracts for the optional model runtime packaging."""

from pathlib import Path


def test_supervisor_runtime_packages_ecapa_dependencies() -> None:
    requirements = Path("speaker_recognition_addon/requirements.txt").read_text(
        encoding="utf-8"
    )
    assert "torchaudio==2.8.0+cpu" in requirements
    assert "speechbrain==1.1.0" in requirements
    assert "huggingface-hub==1.5.0" in requirements


def test_model_cache_is_persistent_in_supported_deployments() -> None:
    addon_config = Path("speaker_recognition_addon/config.yaml").read_text(encoding="utf-8")
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert "/share/speaker_recognition/models" in addon_config
    assert "./models:/data/models" in compose
    assert 'VOLUME ["/data/embeddings"]' in dockerfile
    assert 'VOLUME ["/data/models"]' in dockerfile
