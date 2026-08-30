"""Contracts for release, packaging, and distribution correctness."""

from pathlib import Path

from speaker_recognition.client import SyncSpeakerRecognitionClient

ROOT = Path(__file__).parents[1]


def test_manual_release_uses_builtin_token_and_tracked_version() -> None:
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "secrets.PAT" not in workflow
    assert "contents: write" in workflow
    assert "GH_TOKEN: ${{ github.token }}" in workflow
    assert 'gh release create "$VERSION"' in workflow
    assert "semantic-release" not in workflow
    assert "pyproject.toml" in workflow


def test_publish_jobs_checkout_immutable_release_tag() -> None:
    workflow = (ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8")
    assert workflow.count("ref: ${{ needs.version.outputs.version }}") == 3
    assert "Set release versions" not in workflow
    assert "uv sync --locked --all-groups" in workflow
    assert "gh release upload" in workflow


def test_addon_consumes_the_images_publish_builds() -> None:
    config = (ROOT / "speaker_recognition_addon/config.yaml").read_text(
        encoding="utf-8"
    )
    publish = (ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8")
    assert "image: ghcr.io/conorod1992/{arch}-speaker-recognition-addon" in config
    assert "ghcr.io/${{ github.repository_owner }}/${{ matrix.arch }}-speaker-recognition-addon" in publish


def test_standalone_container_has_valid_package_metadata_and_persistent_volume() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY pyproject.toml README.md LICENSE.md ./" in dockerfile
    assert 'VOLUME ["/data/embeddings"]' in dockerfile
    assert "EMBEDDINGS_DIR=/data/embeddings" in dockerfile


def test_environment_example_matches_runtime_names_and_secure_defaults() -> None:
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "EMBEDDINGS_DIR=" in env_example
    assert "EMBEDDINGS_PATH=" not in env_example
    assert "API_TOKEN=" in env_example
    assert "ALLOW_INSECURE_REMOTE=false" in env_example


def test_packaged_client_can_authenticate_remote_backend() -> None:
    client = SyncSpeakerRecognitionClient(
        "http://speaker-recognition.example:8099",
        api_token="secret-token",
    )
    http_client = client._ensure_client()
    try:
        assert http_client.headers["Authorization"] == "Bearer secret-token"
    finally:
        client.close()
