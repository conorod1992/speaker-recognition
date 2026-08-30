"""Config flow for Speaker Recognition integration."""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.components import media_source
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import selector

from .const import (
    CONF_BACKEND_TOKEN,
    CONF_BACKEND_URL,
    CONF_CONVERSATION_ENTITY,
    CONF_ENTRY_TYPE,
    CONF_ENROLLMENT_ACTION,
    CONF_FINISH_ENROLLMENT,
    CONF_MIN_CONFIDENCE,
    CONF_PENDING_ENROLLMENT,
    CONF_SAMPLE,
    CONF_SAMPLES,
    CONF_STT_ENTITY,
    CONF_USE_BASIC_DSP,
    CONF_USER,
    CONF_VOICE_SAMPLES,
    DEFAULT_BACKEND_URL,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_USE_BASIC_DSP,
    DOMAIN,
    ENTRY_TYPE_CONVERSATION,
    ENTRY_TYPE_MAIN,
    ENTRY_TYPE_STT,
    effective_backend_token,
    effective_backend_url,
    effective_use_basic_dsp,
)
from .proxy import proxy_unique_id, validate_proxy_source

from .audio import decode_wav, read_bounded_wav

ENROLLMENT_PHRASES = (
    "The morning light is warm across the kitchen table.",
    "Please turn on the hallway lamp before it gets dark.",
    "My favorite music sounds best on a quiet afternoon.",
    "A small bird landed beside the open garden gate.",
    "Tomorrow I will remember to water all the plants.",
    "Home should feel comfortable, calm, and welcoming.",
)
MIN_ENROLLMENT_SAMPLES = 5


async def _build_user_selector(hass: HomeAssistant) -> selector.SelectSelector:
    """Build a Home Assistant user selector for guided enrollment."""
    users = await hass.auth.async_get_users()
    return selector.SelectSelector(
        selector.SelectSelectorConfig(
            options=[
                selector.SelectOptionDict(value=user.id, label=user.name or user.id)
                for user in users
                if not user.system_generated
            ],
            mode=selector.SelectSelectorMode.DROPDOWN,
        )
    )


async def _async_validate_enrollment_sample(
    hass: HomeAssistant, media_item: object
) -> None:
    """Reject unsupported, empty, or implausibly short enrollment media."""
    if not isinstance(media_item, dict):
        raise ValueError("Invalid media selection")
    media_id = media_item.get("media_content_id")
    if not isinstance(media_id, str) or not media_id.startswith("media-source://"):
        raise ValueError("Invalid media selection")
    resolved_media = await media_source.async_resolve_media(hass, media_id, None)
    if resolved_media.path is None:
        raise ValueError("Selected media must be a local Home Assistant media file")
    audio_data = await hass.async_add_executor_job(read_bounded_wav, resolved_media.path)
    pcm_data, sample_rate = await hass.async_add_executor_job(decode_wav, audio_data)
    if sample_rate <= 0 or len(pcm_data) < sample_rate:
        raise ValueError("Enrollment sample must contain at least 0.5 seconds of audio")


def _replace_enrolled_user(
    voice_samples: list[dict[str, Any]],
    user_id: str,
    media_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replace one user's enrollment without changing unrelated users."""
    retained = [sample for sample in voice_samples if sample.get(CONF_USER) != user_id]
    retained.append(
        {
            CONF_USER: user_id,
            CONF_SAMPLES: media_items,
            "sample_metadata": [
                {"phrase": phrase} for phrase in ENROLLMENT_PHRASES[: len(media_items)]
            ],
        }
    )
    return retained


class EnrollmentFlowMixin:
    """Shared phrase-by-phrase enrollment steps for config and options flows."""

    hass: HomeAssistant
    _enrollment_user_id: str
    _enrollment_samples: list[dict[str, Any]]
    _retry_sample_index: int | None

    async def async_step_enrollment_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Select the Home Assistant user to enroll or retrain."""
        if user_input is not None:
            self._enrollment_user_id = user_input[CONF_USER]
            self._enrollment_samples = []
            self._retry_sample_index = None
            return await self.async_step_enrollment_sample()

        return self.async_show_form(
            step_id="enrollment_user",
            data_schema=vol.Schema(
                {vol.Required(CONF_USER): await _build_user_selector(self.hass)}
            ),
        )

    async def async_step_enrollment_sample(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Collect and validate one prompted WAV sample."""
        errors: dict[str, str] = {}
        sample_index = (
            self._retry_sample_index
            if self._retry_sample_index is not None
            else len(self._enrollment_samples)
        )

        if user_input is not None:
            try:
                await _async_validate_enrollment_sample(
                    self.hass, user_input[CONF_SAMPLE]
                )
            except Exception:
                errors[CONF_SAMPLE] = "invalid_enrollment_sample"
            else:
                media_item = user_input[CONF_SAMPLE]
                if self._retry_sample_index is None:
                    self._enrollment_samples.append(media_item)
                else:
                    self._enrollment_samples[self._retry_sample_index] = media_item
                    self._retry_sample_index = None
                    return await self.async_step_enrollment_review()

                accepted = len(self._enrollment_samples)
                if accepted == len(ENROLLMENT_PHRASES) or (
                    accepted >= MIN_ENROLLMENT_SAMPLES
                    and user_input.get(CONF_FINISH_ENROLLMENT, False)
                ):
                    return await self.async_step_enrollment_review()
                sample_index = accepted

        schema: dict[vol.Marker, object] = {
            vol.Required(CONF_SAMPLE): selector.MediaSelector(
                selector.MediaSelectorConfig(
                    accept=["audio/wav", "audio/x-wav", "audio/wave"]
                )
            )
        }
        if sample_index + 1 >= MIN_ENROLLMENT_SAMPLES:
            schema[vol.Optional(CONF_FINISH_ENROLLMENT, default=False)] = bool

        return self.async_show_form(
            step_id="enrollment_sample",
            data_schema=vol.Schema(schema),
            errors=errors,
            description_placeholders={
                "phrase": ENROLLMENT_PHRASES[sample_index],
                "sample_number": str(sample_index + 1),
                "sample_total": str(len(ENROLLMENT_PHRASES)),
                "accepted": str(len(self._enrollment_samples)),
            },
        )

    async def async_step_enrollment_review(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Finish enrollment or retry any accepted sample."""
        if user_input is not None:
            action = user_input[CONF_ENROLLMENT_ACTION]
            if action == "finish":
                return await self.async_step_enrollment_complete()
            self._retry_sample_index = int(action[len("retry_") :])
            return await self.async_step_enrollment_sample()

        options = [selector.SelectOptionDict(value="finish", label="Finish enrollment")]
        options.extend(
            selector.SelectOptionDict(
                value=f"retry_{index}", label=f"Retry sample {index + 1}"
            )
            for index in range(len(self._enrollment_samples))
        )
        return self.async_show_form(
            step_id="enrollment_review",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_ENROLLMENT_ACTION, default="finish"
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=options,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    )
                }
            ),
            description_placeholders={"accepted": str(len(self._enrollment_samples))},
        )

    async def async_step_enrollment_complete(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Show accepted count before rebuilding the user's reference."""
        if user_input is not None:
            return await self._async_save_enrollment()
        return self.async_show_form(
            step_id="enrollment_complete",
            data_schema=vol.Schema({}),
            description_placeholders={"accepted": str(len(self._enrollment_samples))},
        )

    async def _async_save_enrollment(self) -> ConfigFlowResult:
        """Persist collected enrollment configuration in the concrete flow."""
        raise NotImplementedError


def _get_main_config_entry(hass: HomeAssistant) -> ConfigEntry | None:
    """Get the main config entry if it exists."""
    entries = hass.config_entries.async_entries(DOMAIN)
    for entry in entries:
        if entry.data.get(CONF_ENTRY_TYPE) == ENTRY_TYPE_MAIN:
            return entry
    return None


class SpeakerRecognitionConfigFlow(EnrollmentFlowMixin, ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Speaker Recognition."""

    VERSION = 2
    MINOR_VERSION = 0

    _pending_backend_url: str
    _pending_backend_token: str

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        main_entry = _get_main_config_entry(self.hass)

        if main_entry is None:
            return await self.async_step_main(user_input)

        return await self.async_step_menu()

    async def async_step_menu(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Show menu to add STT or Conversation proxy."""
        return self.async_show_menu(
            step_id="menu",
            menu_options=["add_stt", "add_conversation"],
        )

    async def async_step_main(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle main configuration."""
        errors: dict[str, str] = {}

        if user_input is not None:
            await self.async_set_unique_id(ENTRY_TYPE_MAIN)
            self._abort_if_unique_id_configured()
            self._pending_backend_url = user_input[CONF_BACKEND_URL]
            self._pending_backend_token = user_input.get(CONF_BACKEND_TOKEN, "")
            return await self.async_step_enrollment_menu()

        return self.async_show_form(
            step_id="main",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_BACKEND_URL, default=DEFAULT_BACKEND_URL
                    ): selector.TextSelector(),
                    vol.Optional(CONF_BACKEND_TOKEN, default=""): selector.TextSelector(),
                }
            ),
            errors=errors,
        )

    async def async_step_enrollment_menu(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Offer guided enrollment during initial setup."""
        return self.async_show_menu(
            step_id="enrollment_menu",
            menu_options=["enrollment_user", "finish_setup"],
        )

    async def async_step_finish_setup(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Finish initial setup without enrolling a speaker."""
        return self._async_create_main_entry([])

    async def _async_save_enrollment(self) -> ConfigFlowResult:
        return self._async_create_main_entry(
            _replace_enrolled_user(
                [], self._enrollment_user_id, self._enrollment_samples
            )
        )

    def _async_create_main_entry(
        self, voice_samples: list[dict[str, Any]]
    ) -> ConfigFlowResult:
        options: dict[str, Any] = {CONF_VOICE_SAMPLES: voice_samples}
        if voice_samples:
            options[CONF_PENDING_ENROLLMENT] = voice_samples[0][CONF_USER]
        return self.async_create_entry(
            title="Speaker Recognition",
            data={
                CONF_ENTRY_TYPE: ENTRY_TYPE_MAIN,
                CONF_BACKEND_URL: self._pending_backend_url,
                CONF_BACKEND_TOKEN: self._pending_backend_token,
            },
            options=options,
        )

    async def async_step_add_stt(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Add STT proxy entity."""
        errors: dict[str, str] = {}

        if user_input is not None:
            stt_entity = user_input[CONF_STT_ENTITY]
            source_error = validate_proxy_source(self.hass, ENTRY_TYPE_STT, stt_entity)
            if source_error is not None:
                errors["base"] = source_error
            else:
                await self.async_set_unique_id(proxy_unique_id(ENTRY_TYPE_STT, stt_entity))
                self._abort_if_unique_id_configured()

                return self.async_create_entry(
                    title=f"STT: {stt_entity.split('.', 1)[-1]}",
                    data={
                        CONF_ENTRY_TYPE: ENTRY_TYPE_STT,
                        CONF_STT_ENTITY: stt_entity,
                        CONF_USE_BASIC_DSP: user_input.get(
                            CONF_USE_BASIC_DSP, DEFAULT_USE_BASIC_DSP
                        ),
                    },
                )

        return self.async_show_form(
            step_id="add_stt",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_STT_ENTITY): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain=Platform.STT,
                        ),
                    ),
                    vol.Optional(
                        CONF_USE_BASIC_DSP, default=DEFAULT_USE_BASIC_DSP
                    ): selector.BooleanSelector(),
                }
            ),
            errors=errors,
        )

    async def async_step_add_conversation(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Add Conversation proxy entity."""
        errors: dict[str, str] = {}

        if user_input is not None:
            conversation_entity = user_input[CONF_CONVERSATION_ENTITY]
            source_error = validate_proxy_source(
                self.hass, ENTRY_TYPE_CONVERSATION, conversation_entity
            )
            if source_error is not None:
                errors["base"] = source_error
            else:
                await self.async_set_unique_id(
                    proxy_unique_id(ENTRY_TYPE_CONVERSATION, conversation_entity)
                )
                self._abort_if_unique_id_configured()

                return self.async_create_entry(
                    title=f"Conversation: {conversation_entity.split('.', 1)[-1]}",
                    data={
                        CONF_ENTRY_TYPE: ENTRY_TYPE_CONVERSATION,
                        CONF_CONVERSATION_ENTITY: conversation_entity,
                        CONF_MIN_CONFIDENCE: user_input[CONF_MIN_CONFIDENCE],
                    },
                )

        return self.async_show_form(
            step_id="add_conversation",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_CONVERSATION_ENTITY): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="conversation",
                        ),
                    ),
                    vol.Required(
                        CONF_MIN_CONFIDENCE, default=DEFAULT_MIN_CONFIDENCE
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0.0,
                            max=1.0,
                            step=0.05,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                }
            ),
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> SpeakerRecognitionOptionsFlow:
        """Get the options flow for this handler."""
        return SpeakerRecognitionOptionsFlow()


class SpeakerRecognitionOptionsFlow(EnrollmentFlowMixin, OptionsFlow):
    """Handle options flow for Speaker Recognition."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        entry_type = self.config_entry.data.get(CONF_ENTRY_TYPE, ENTRY_TYPE_MAIN)

        if entry_type == ENTRY_TYPE_MAIN:
            return self.async_show_menu(
                step_id="init", menu_options=["main_options", "enrollment_user"]
            )
        if entry_type == ENTRY_TYPE_STT:
            return await self.async_step_stt_options(user_input)
        return await self.async_step_conversation_options(user_input)

    async def async_step_main_options(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage main config options."""
        if user_input is not None:
            return self.async_create_entry(
                title="",
                data={
                    CONF_BACKEND_URL: user_input[CONF_BACKEND_URL],
                    CONF_BACKEND_TOKEN: user_input.get(CONF_BACKEND_TOKEN, ""),
                    CONF_VOICE_SAMPLES: self.config_entry.options.get(
                        CONF_VOICE_SAMPLES, []
                    ),
                },
            )

        current_url = effective_backend_url(
            self.config_entry.data, self.config_entry.options
        )
        current_token = effective_backend_token(
            self.config_entry.data, self.config_entry.options
        )
        return self.async_show_form(
            step_id="main_options",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_BACKEND_URL, default=current_url
                    ): selector.TextSelector(),
                    vol.Optional(
                        CONF_BACKEND_TOKEN, default=current_token
                    ): selector.TextSelector(),
                }
            ),
        )

    async def _async_save_enrollment(self) -> ConfigFlowResult:
        current_voice_samples = self.config_entry.options.get(CONF_VOICE_SAMPLES, [])
        current_url = effective_backend_url(
            self.config_entry.data, self.config_entry.options
        )
        current_token = effective_backend_token(
            self.config_entry.data, self.config_entry.options
        )
        return self.async_create_entry(
            title="",
            data={
                CONF_BACKEND_URL: current_url,
                CONF_BACKEND_TOKEN: current_token,
                CONF_VOICE_SAMPLES: _replace_enrolled_user(
                    current_voice_samples,
                    self._enrollment_user_id,
                    self._enrollment_samples,
                ),
            },
        )

    async def async_step_stt_options(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage STT proxy options."""
        errors: dict[str, str] = {}

        if user_input is not None:
            stt_entity = user_input[CONF_STT_ENTITY]
            source_error = validate_proxy_source(
                self.hass,
                ENTRY_TYPE_STT,
                stt_entity,
                exclude_entry_id=self.config_entry.entry_id,
            )
            if source_error is not None:
                errors["base"] = source_error
            else:
                return self.async_create_entry(
                    title="",
                    data={
                        CONF_STT_ENTITY: stt_entity,
                        CONF_USE_BASIC_DSP: user_input.get(
                            CONF_USE_BASIC_DSP, DEFAULT_USE_BASIC_DSP
                        ),
                    },
                )

        current_stt_entity = self.config_entry.options.get(
            CONF_STT_ENTITY, self.config_entry.data.get(CONF_STT_ENTITY)
        )
        current_use_basic_dsp = effective_use_basic_dsp(
            self.config_entry.data, self.config_entry.options
        )

        return self.async_show_form(
            step_id="stt_options",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_STT_ENTITY, default=current_stt_entity
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain=Platform.STT,
                        ),
                    ),
                    vol.Optional(
                        CONF_USE_BASIC_DSP, default=current_use_basic_dsp
                    ): selector.BooleanSelector(),
                }
            ),
            errors=errors,
        )

    async def async_step_conversation_options(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage Conversation proxy options."""
        errors: dict[str, str] = {}

        if user_input is not None:
            conversation_entity = user_input[CONF_CONVERSATION_ENTITY]
            source_error = validate_proxy_source(
                self.hass,
                ENTRY_TYPE_CONVERSATION,
                conversation_entity,
                exclude_entry_id=self.config_entry.entry_id,
            )
            if source_error is not None:
                errors["base"] = source_error
            else:
                return self.async_create_entry(
                    title="",
                    data={
                        CONF_CONVERSATION_ENTITY: conversation_entity,
                        CONF_MIN_CONFIDENCE: user_input[CONF_MIN_CONFIDENCE],
                    },
                )

        current_conversation_entity = self.config_entry.options.get(
            CONF_CONVERSATION_ENTITY,
            self.config_entry.data.get(CONF_CONVERSATION_ENTITY),
        )
        current_min_confidence = self.config_entry.options.get(
            CONF_MIN_CONFIDENCE,
            self.config_entry.data.get(CONF_MIN_CONFIDENCE, DEFAULT_MIN_CONFIDENCE),
        )

        return self.async_show_form(
            step_id="conversation_options",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_CONVERSATION_ENTITY, default=current_conversation_entity
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="conversation",
                        ),
                    ),
                    vol.Required(
                        CONF_MIN_CONFIDENCE, default=current_min_confidence
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0.0,
                            max=1.0,
                            step=0.05,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                }
            ),
            errors=errors,
        )
