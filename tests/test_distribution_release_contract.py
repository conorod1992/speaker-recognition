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


def test_manual_release_is_hard_pinned_to_master() -> None:
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "if: github.ref == 'refs/heads/master'" in workflow
    assert "ref: master" in workflow
    assert 'sha=$(git rev-parse HEAD)' in workflow
    assert '--target "$SOURCE_SHA"' in workflow
    assert '--target "$GITHUB_SHA"' not in workflow


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
    image_name = (
        "ghcr.io/${{ github.repository_owner }}/"
        "${{ matrix.arch }}-speaker-recognition-addon"
    )
    assert image_name in publish


def test_compose_uses_fork_image_and_requires_authentication() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "ghcr.io/conorod1992/speaker-recognition:latest" in compose
    assert "ghcr.io/eulemitkeule/speaker-recognition" not in compose
    assert "API_TOKEN=${API_TOKEN:?" in compose
    assert "ALLOW_INSECURE_REMOTE=false" in compose
    assert "./embeddings:/data/embeddings" in compose


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


def test_readme_examples_match_current_api_and_storage_contract() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "./embeddings:/data/embeddings" in readme
    assert "./embeddings:/app/embeddings" not in readme
    assert '"audio_input"' not in readme
    assert "Authorization: Bearer YOUR_TOKEN" in readme
    assert 'api_token="replace-with-the-backend-token"' in readme
    assert "[MIT License](LICENSE.md)" in readme
    assert "python-3.9-blue" in readme
    assert "supports Python 3.9." in readme


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
