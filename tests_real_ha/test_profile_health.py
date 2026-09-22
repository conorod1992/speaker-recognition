"""Real HA profile diagnostics preserve admin and user eligibility boundaries."""
from unittest.mock import AsyncMock, patch

import pytest

from tests_real_ha.test_promoted_samples import promotion_environment  # noqa: F401


@pytest.mark.asyncio
async def test_profile_health_names_eligibility_and_unavailable(hass, promotion_environment):
    alice, bob, entry, backend, history, decision, send = promotion_environment
    rows = [{"user_id": alice.id, "nearest_user_id": bob.id,
             "separation": .03, "low_separation": True,
             "sample_warnings": [{"sample_index": 1, "competing_user_id": bob.id}]},
            {"user_id": "removed", "nearest_user_id": alice.id}]
    with patch.object(entry.runtime_data, "_async_get", AsyncMock(return_value={"profiles": rows})) as get:
        result = (await send("profile_health"))["result"]
        get.assert_awaited_once_with("/profiles/diagnostics")
        assert len(result["profiles"]) == 1
        row = result["profiles"][0]
        assert row["user_name"] == "Alice" and row["nearest_user_name"] == "Bob"
        assert row["sample_warnings"][0]["competing_user_name"] == "Bob"
        await hass.auth.async_update_user(bob, is_active=False)
        row = (await send("profile_health"))["result"]["profiles"][0]
        assert row["nearest_user_id"] is None and not row["sample_warnings"]
    with patch.object(entry.runtime_data, "_async_get", AsyncMock(side_effect=RuntimeError("offline"))):
        assert (await send("profile_health"))["result"] == {"available": False, "profiles": []}

@pytest.mark.asyncio
async def test_profile_health_requires_admin(hass, promotion_environment, hass_ws_client, hass_read_only_access_token):
    client = await hass_ws_client(hass, access_token=hass_read_only_access_token)
    await client.send_json_auto_id({"type": "speaker_recognition/profile_health"})
    assert (await client.receive_json())["error"]["code"] == "unauthorized"
