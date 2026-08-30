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
CONF_USE_BASIC_DSP = "use_basic_dsp"

# Conversation configuration
CONF_CONVERSATION_ENTITY = "conversation_entity"
CONF_MIN_CONFIDENCE = "min_confidence"

# Defaults
DEFAULT_BACKEND_URL = "http://localhost:8099"
DEFAULT_MIN_CONFIDENCE = 0.0
DEFAULT_USE_BASIC_DSP = False


def effective_backend_url(data: Mapping[str, Any], options: Mapping[str, Any]) -> str:
    """Return the current backend URL using runtime precedence."""
    value = options.get(CONF_BACKEND_URL, data.get(CONF_BACKEND_URL))
    return value if isinstance(value, str) else DEFAULT_BACKEND_URL


def effective_use_basic_dsp(data: Mapping[str, Any], options: Mapping[str, Any]) -> bool:
    """Return whether streaming basic DSP is enabled for an STT proxy."""
    return bool(options.get(CONF_USE_BASIC_DSP, data.get(CONF_USE_BASIC_DSP, False)))
