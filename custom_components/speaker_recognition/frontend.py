"""Frontend registration for Speaker Recognition enrollment."""

from __future__ import annotations

from pathlib import Path

from homeassistant.components import frontend, panel_custom
from homeassistant.components.http import StaticPathConfig
from homeassistant.core import HomeAssistant

from .const import DOMAIN
from .enhancement import async_register_enhancement_websocket

PANEL_URL_PATH = "speaker-recognition"
PANEL_ELEMENT = "speaker-recognition-enhancement-panel"
STATIC_URL = "/speaker-recognition-static"


async def async_register_frontend(hass: HomeAssistant) -> None:
    """Register static assets and the enrollment/diagnostics panel."""
    www_path = Path(__file__).parent / "www"
    await hass.http.async_register_static_paths(
        [StaticPathConfig(STATIC_URL, str(www_path), False)]
    )
    async_register_enhancement_websocket(hass)
    await panel_custom.async_register_panel(
        hass,
        webcomponent_name=PANEL_ELEMENT,
        frontend_url_path=PANEL_URL_PATH,
        module_url=f"{STATIC_URL}/speaker-recognition-enhancement-panel.js",
        sidebar_title="Speaker Recognition",
        sidebar_icon="mdi:account-voice",
        require_admin=True,
        config={},
        config_panel_domain=DOMAIN,
    )


def async_unregister_frontend(hass: HomeAssistant) -> None:
    """Remove the panel registration."""
    frontend.async_remove_panel(hass, PANEL_URL_PATH)
