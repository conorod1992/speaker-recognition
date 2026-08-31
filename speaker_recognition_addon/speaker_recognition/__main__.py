"""Main entry point for speaker recognition service."""

import logging
import warnings

import typer
import uvicorn
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning, module="webrtcvad")

from speaker_recognition.api import app  # noqa: E402
from speaker_recognition.const import (  # noqa: E402
    DEFAULT_ACCESS_LOG,
    DEFAULT_ALLOW_INSECURE_REMOTE,
    DEFAULT_API_TOKEN,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_HOST,
    DEFAULT_LOG_LEVEL,
    DEFAULT_PORT,
    ENV_ACCESS_LOG,
    ENV_ALLOW_INSECURE_REMOTE,
    ENV_API_TOKEN,
    ENV_EMBEDDINGS_DIR,
    ENV_HOST,
    ENV_LOG_LEVEL,
    ENV_PORT,
)
from speaker_recognition.logging_config import configure_logging  # noqa: E402
from speaker_recognition.models import config  # noqa: E402
from speaker_recognition.recognizer import recognizer  # noqa: E402

load_dotenv()
cli = typer.Typer(name="speaker-recognition", help="Speaker Recognition Service")

_LOGGER = logging.getLogger(__name__)


@cli.command()
def serve(
    host: str = typer.Option(
        DEFAULT_HOST,
        "--host",
        "-h",
        help="Host to bind the server to",
        envvar=ENV_HOST,
    ),
    port: int = typer.Option(
        DEFAULT_PORT,
        "--port",
        "-p",
        help="Port to bind the server to",
        envvar=ENV_PORT,
    ),
    log_level: str = typer.Option(
        DEFAULT_LOG_LEVEL,
        "--log-level",
        "-l",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        envvar=ENV_LOG_LEVEL,
    ),
    access_log: bool = typer.Option(
        DEFAULT_ACCESS_LOG,
        "--access-log/--no-access-log",
        help="Enable or disable access logging",
        envvar=ENV_ACCESS_LOG,
    ),
    embeddings_dir: str = typer.Option(
        DEFAULT_EMBEDDINGS_DIR,
        "--embeddings-dir",
        "-e",
        help="Directory to store voice embeddings",
        envvar=ENV_EMBEDDINGS_DIR,
    ),
    api_token: str = typer.Option(
        DEFAULT_API_TOKEN,
        "--api-token",
        help="Bearer token required for non-loopback API clients",
        envvar=ENV_API_TOKEN,
        show_default=False,
    ),
    allow_insecure_remote: bool = typer.Option(
        DEFAULT_ALLOW_INSECURE_REMOTE,
        "--allow-insecure-remote/--no-allow-insecure-remote",
        help="Explicitly permit unauthenticated non-loopback API clients",
        envvar=ENV_ALLOW_INSECURE_REMOTE,
    ),
) -> None:
    """Start the Speaker Recognition Service."""

    config.host = host
    config.port = port
    config.log_level = log_level.upper()
    config.access_log = access_log
    config.embeddings_directory = embeddings_dir
    config.api_token = api_token
    config.allow_insecure_remote = allow_insecure_remote

    recognizer.embeddings_directory = config.embeddings_directory

    configure_logging(config.log_level)

    _LOGGER.info("Starting Speaker Recognition Service...")
    _LOGGER.info("Host: %s", config.host)
    _LOGGER.info("Port: %s", config.port)
    _LOGGER.info("Log Level: %s", config.log_level)
    _LOGGER.info("Embeddings Directory: %s", config.embeddings_directory)
    _LOGGER.info(
        "Remote API authentication: %s",
        "disabled by explicit override"
        if config.allow_insecure_remote
        else (
            "bearer token configured"
            if config.api_token
            else "trusted local hosts only"
        ),
    )

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        access_log=config.access_log,
        log_config=None,
    )


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
