"""Contracts for release, packaging, and distribution correctness."""

from pathlib import Path

from speaker_recognition.client import SyncSpeakerRecognitionClient

ROOT = Path(__file__).parents[1]


def test_manual_release_uses_builtin_token_and_explicit_target_version() -> None:
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "secrets.PAT" not in workflow
    assert "contents: write" in workflow
    assert "GH_TOKEN: ${{ github.token }}" in workflow
    assert 'description: "Release version (X.Y.Z)"' in workflow
    assert 'gh release create "$VERSION"' in workflow
    assert "semantic-release" not in workflow
    assert "scripts/set_release_version.py" in workflow


def test_manual_release_is_hard_pinned_to_master_and_source_sha() -> None:
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "if: github.ref == 'refs/heads/master'" in workflow
    assert workflow.count("ref: master") >= 2
    assert 'sha=$(git rev-parse HEAD)' in workflow
    assert "Refuse to publish from a stale source commit" in workflow
    assert 'git push origin HEAD:master' in workflow
    assert '--target "$RELEASE_SHA"' in workflow
    assert '--target "$GITHUB_SHA"' not in workflow


def test_release_builds_and_verifies_images_before_exposing_addon_version() -> None:
    """Supervisor cannot see a new config version until its image tags exist."""
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert "Build amd64 add-on" not in workflow  # matrix expands this at runtime
    assert "Build ${{ matrix.arch }} add-on" in workflow
    assert "amd64-speaker-recognition-addon:$VERSION" in workflow
    assert "aarch64-speaker-recognition-addon:$VERSION" in workflow
    assert "Verify published image manifests" in workflow
    assert "Commit version only after installable images exist" in workflow
    assert workflow.index("Verify published image manifests") < workflow.index(
        "git push origin HEAD:master"
    )
    assert "No version bump or GitHub release has been published" in workflow


def test_normal_release_does_not_depend_on_post_publish_trigger() -> None:
    """The recovery publisher must not start a second release-time build race."""
    publish = (ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8")
    assert "name: Republish release" in publish
    assert "types:\n      - published" not in publish
    assert "Recovery-only workflow" in publish


def test_republish_jobs_checkout_immutable_release_tag() -> None:
    workflow = (ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8")
    assert workflow.count("ref: ${{ needs.version.outputs.version }}") == 3
    assert "uv sync --locked --all-groups" in workflow
    assert "gh release upload" in workflow


def test_addon_consumes_the_images_release_workflow_builds() -> None:
    config = (ROOT / "speaker_recognition_addon/config.yaml").read_text(
        encoding="utf-8"
    )
    release = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "image: ghcr.io/conorod1992/{arch}-speaker-recognition-addon" in config
    image_name = (
        "ghcr.io/${{ github.repository_owner }}/"
        "${{ matrix.arch }}-speaker-recognition-addon"
    )
    assert image_name in release


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
