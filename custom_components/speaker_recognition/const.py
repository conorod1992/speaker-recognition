"""Constants for speaker recognition integration."""

from typing import Any, Mapping

DOMAIN = "speaker_recognition"

# Configuration keys
CONF_BACKEND_URL = "backend_url"
CONF_VOICE_SAMPLES = "voice_samples"
CONF_USER = "user"
CONF_SAMPLES = "samples"
CONF_SAMPLE = "sample"
CONF_FINISH_ENROLLMENT = "finish_enrollment"
CONF_ENROLLMENT_ACTION = "enrollment_action"
CONF_PENDING_ENROLLMENT = "pending_enrollment"

# Sub-entry types
CONF_ENTRY_TYPE = "entry_type"
ENTRY_TYPE_MAIN = "main"
ENTRY_TYPE_STT = "stt"
ENTRY_TYPE_CONVERSATION = "conversation"

# STT configuration
CONF_STT_ENTITY = "stt_entity"

# Conversation configuration
CONF_CONVERSATION_ENTITY = "conversation_entity"
CONF_MIN_CONFIDENCE = "min_confidence"

# Enrollment
ENROLLMENT_PHRASES = (
    "The morning light is warm across the kitchen table.",
    "Please turn on the hallway lamp before it gets dark.",
    "My favorite music sounds best on a quiet afternoon.",
    "A small bird landed beside the open garden gate.",
    "Tomorrow I will remember to water all the plants.",
    "Home should feel comfortable, calm, and welcoming.",
)
MIN_ENROLLMENT_SAMPLES = 5
MAX_ENROLLMENT_SAMPLES = len(ENROLLMENT_PHRASES)
ENROLLMENT_SAMPLE_RATE = 16000
MIN_ENROLLMENT_SECONDS = 0.5
MAX_ENROLLMENT_SECONDS = 10.0

# Defaults
DEFAULT_BACKEND_URL = "http://localhost:8099"
DEFAULT_MIN_CONFIDENCE = 0.0


def effective_backend_url(data: Mapping[str, Any], options: Mapping[str, Any]) -> str:
    """Return the current backend URL using runtime precedence."""
    value = options.get(CONF_BACKEND_URL, data.get(CONF_BACKEND_URL))
    return value if isinstance(value, str) else DEFAULT_BACKEND_URL
