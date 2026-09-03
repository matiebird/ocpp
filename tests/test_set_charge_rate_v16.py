"""Tests for the simplified OCPP 1.6 set_charge_rate implementation.

These tests use the production ChargePoint class (v1.6) and monkeypatch only
the collaborators set_charge_rate depends on:
- get_configuration(...)
- call(...)
- notify_ha(...)

They avoid any parallel/dummy implementation of ChargePoint.
"""

from types import SimpleNamespace

import pytest

from custom_components.ocpp.ocppv16 import ChargePoint as ChargePointv16
from custom_components.ocpp.enums import (
    Profiles as prof,
    ConfigurationKey as ckey,
    HAChargerStatuses as cstat,
)
from ocpp.v16.enums import (
    ChargePointStatus,
    ClearChargingProfileStatus,
    ChargingProfileStatus,
    ChargingProfilePurposeType,
    ChargingProfileKindType,
    ChargingRateUnitType,
)


def _mark_ocular(cp):
    """Give a minimal charge point the commissioned Ocular identity/state."""
    cp.id = "ocular"
    cp.settings = SimpleNamespace(cpid="ocular")
    cp._metrics = {}
    cp._active_tx = {}
    return cp


@pytest.fixture
def cp_v16():
    """Provide a minimally-initialized v1.6 ChargePoint instance.

    We bypass __init__ and set only the attributes used by set_charge_rate.
    """
    cp = object.__new__(ChargePointv16)  # type: ignore[misc]
    # What set_charge_rate reads:
    cp._attr_supported_features = prof.SMART  # can be overridden in tests
    cp._ocpp_version = "1.6"
    cp.active_transaction_id = 0
    cp._active_tx = {}
    # set_charge_rate calls these (we’ll monkeypatch per-test):
    # - cp.get_configuration(key)
    # - cp.call(req)
    # - cp.notify_ha(msg)
    return cp


@pytest.mark.asyncio
async def test_ocular_rejects_watt_limit_without_sending_ocpp(cp_v16, monkeypatch):
    """A watt request must not silently become the default 32 amp limit."""
    _mark_ocular(cp_v16)
    calls = []

    async def fake_call(req):
        calls.append(req)
        return SimpleNamespace(status=ChargingProfileStatus.accepted)

    monkeypatch.setattr(cp_v16, "call", fake_call)

    assert await cp_v16.set_charge_rate(limit_watts=1000, conn_id=1) is False
    assert calls == []


@pytest.mark.asyncio
async def test_ocular_rejects_custom_profile_without_sending_ocpp(cp_v16, monkeypatch):
    """The advanced custom-profile escape hatch cannot bypass Ocular safety."""
    _mark_ocular(cp_v16)
    calls = []

    async def fake_call(req):
        calls.append(req)
        return SimpleNamespace(status=ChargingProfileStatus.accepted)

    monkeypatch.setattr(cp_v16, "call", fake_call)

    unsafe = {
        "chargingProfileId": 99,
        "stackLevel": 9,
        "chargingProfilePurpose": "TxProfile",
        "chargingProfileKind": "Relative",
        "chargingSchedule": {
            "chargingRateUnit": "A",
            "chargingSchedulePeriod": [{"startPeriod": 0, "limit": 0}],
        },
    }
    assert await cp_v16.set_charge_rate(profile=unsafe, conn_id=1) is False
    assert calls == []


@pytest.mark.asyncio
async def test_ocular_allows_only_exact_active_transaction_zero_amp_pause(
    cp_v16, monkeypatch
):
    """The supervised pause path sends one exact transaction-bound 0 A profile."""
    _mark_ocular(cp_v16)
    cp_v16._active_tx[1] = 1234
    calls = []

    async def fake_call(req):
        calls.append(req)
        return SimpleNamespace(status=ChargingProfileStatus.accepted)

    monkeypatch.setattr(cp_v16, "call", fake_call)
    pause = {
        "chargingProfileId": 1,
        "stackLevel": 0,
        "chargingProfileKind": "Relative",
        "chargingProfilePurpose": "TxProfile",
        "chargingSchedule": {
            "chargingRateUnit": "A",
            "chargingSchedulePeriod": [{"startPeriod": 0, "limit": 0}],
        },
        "transactionId": 1234,
    }

    assert await cp_v16.set_charge_rate(profile=pause, conn_id=1) is True
    assert len(calls) == 1
    assert calls[0].cs_charging_profiles == pause


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_tx_id", [True, 1.5, "1", "bad", -7])
async def test_ocular_active_state_without_transaction_id_fails_closed(
    cp_v16, monkeypatch, raw_tx_id
):
    """Never install a default profile over an active but unidentified session."""
    _mark_ocular(cp_v16)
    cp_v16._metrics[(1, cstat.status_connector.value)] = SimpleNamespace(
        value=ChargePointStatus.charging.value
    )
    cp_v16._active_tx[1] = raw_tx_id
    calls = []

    async def fake_call(req):
        calls.append(req)
        return SimpleNamespace(status=ChargingProfileStatus.accepted)

    monkeypatch.setattr(cp_v16, "call", fake_call)

    assert await cp_v16.set_charge_rate(limit_amps=16, conn_id=1) is False
    assert calls == []


@pytest.mark.asyncio
async def test_ocular_legacy_migration_is_tracked_per_profile_id(cp_v16, monkeypatch):
    """A migrated connector must not suppress cleanup for another connector."""
    calls = []

    async def fake_call(req):
        calls.append(req)
        return SimpleNamespace(status=ClearChargingProfileStatus.unknown)

    monkeypatch.setattr(cp_v16, "call", fake_call)

    assert await cp_v16._clear_ocular_legacy_profiles(1) is True
    assert await cp_v16._clear_ocular_legacy_profiles(2) is True
    assert [req.id for req in calls] == [1000, 2001, 3001, 2002, 3002]


@pytest.mark.asyncio
async def test_custom_profile_path_exception_triggers_notify_and_returns_false(
    cp_v16, monkeypatch
):
    """1) When a custom profile is provided and the CP call raises, return False and notify HA."""
    # notify capture
    notices = []

    async def fake_notify(msg, title="Ocpp integration"):
        notices.append(msg)
        return True

    async def fake_call(_req):
        raise RuntimeError("boom")

    # get_configuration shouldn't be touched in this path
    async def fake_get_conf(_key):
        pytest.fail("get_configuration should not be called for custom profile")

    monkeypatch.setattr(cp_v16, "notify_ha", fake_notify)
    monkeypatch.setattr(cp_v16, "call", fake_call)
    monkeypatch.setattr(cp_v16, "get_configuration", fake_get_conf)

    profile = {
        "chargingProfileId": 123,
        "stackLevel": 1,
        "chargingProfileKind": ChargingProfileKindType.relative.value,
        "chargingProfilePurpose": ChargingProfilePurposeType.charge_point_max_profile.value,
        "chargingSchedule": {
            "chargingRateUnit": ChargingRateUnitType.amps.value,
            "chargingSchedulePeriod": [{"startPeriod": 0, "limit": 16}],
        },
    }

    ok = await cp_v16.set_charge_rate(profile=profile, conn_id=2)
    assert ok is False
    assert len(notices) == 1
    assert "Set charging profile failed" in notices[0]


@pytest.mark.asyncio
async def test_smart_charging_not_supported_returns_false_no_notify(
    cp_v16, monkeypatch
):
    """2) If the charger doesn't advertise SMART profile, return False without notifications."""
    cp_v16._attr_supported_features = prof.NONE

    notices = []

    async def fake_notify(msg, title="Ocpp integration"):
        notices.append(msg)
        return True

    # get_configuration and call should not be called
    async def fake_get_conf(_key):
        pytest.fail("get_configuration should not be called when SMART not supported")

    async def fake_call(_req):
        pytest.fail("call should not be called when SMART not supported")

    monkeypatch.setattr(cp_v16, "notify_ha", fake_notify)
    monkeypatch.setattr(cp_v16, "get_configuration", fake_get_conf)
    monkeypatch.setattr(cp_v16, "call", fake_call)

    ok = await cp_v16.set_charge_rate(limit_amps=16, conn_id=2)
    assert ok is False
    assert notices == []


@pytest.mark.asyncio
async def test_cpmax_exception_falls_back_to_txdefault_accepted_returns_true(
    cp_v16, monkeypatch
):
    """3) CPMax path raises -> fallback to TxDefault which is accepted -> return True."""

    # Allow both A and stack level
    async def fake_get_conf(key: str):
        if key == ckey.charging_schedule_allowed_charging_rate_unit.value:
            return "Current"  # supports Amps
        if key == ckey.charge_profile_max_stack_level.value:
            return "2"
        pytest.fail(f"Unexpected get_configuration key: {key}")

    # First SetChargingProfile (CPMax connectorId=0) raises, second (TxDefault connectorId=2) accepted
    async def fake_call(req):
        purpose = req.cs_charging_profiles["chargingProfilePurpose"]
        if purpose == ChargingProfilePurposeType.charge_point_max_profile.value:
            raise RuntimeError("transport error")
        if purpose == ChargingProfilePurposeType.tx_default_profile.value:
            return SimpleNamespace(status=ChargingProfileStatus.accepted)
        return SimpleNamespace(status=ChargingProfileStatus.rejected)

    notices = []

    async def fake_notify(msg, title="Ocpp integration"):
        notices.append(msg)
        return True

    monkeypatch.setattr(cp_v16, "get_configuration", fake_get_conf)
    monkeypatch.setattr(cp_v16, "call", fake_call)
    monkeypatch.setattr(cp_v16, "notify_ha", fake_notify)

    ok = await cp_v16.set_charge_rate(limit_amps=16, conn_id=2)
    assert ok is True
    # No user-facing warning necessary when fallback succeeds
    assert notices == []


@pytest.mark.asyncio
async def test_cpmax_rejected_txdefault_accepted_returns_true(cp_v16, monkeypatch):
    """4) CPMax rejected -> TxDefault accepted -> return True."""

    async def fake_get_conf(key: str):
        if key == ckey.charging_schedule_allowed_charging_rate_unit.value:
            return "Current"
        if key == ckey.charge_profile_max_stack_level.value:
            return "3"
        pytest.fail(f"Unexpected get_configuration key: {key}")

    async def fake_call(req):
        purpose = req.cs_charging_profiles["chargingProfilePurpose"]
        if purpose == ChargingProfilePurposeType.charge_point_max_profile.value:
            return SimpleNamespace(status=ChargingProfileStatus.rejected)
        if purpose == ChargingProfilePurposeType.tx_default_profile.value:
            return SimpleNamespace(status=ChargingProfileStatus.accepted)
        return SimpleNamespace(status=ChargingProfileStatus.rejected)

    notices = []

    async def fake_notify(msg, title="Ocpp integration"):
        notices.append(msg)
        return True

    monkeypatch.setattr(cp_v16, "get_configuration", fake_get_conf)
    monkeypatch.setattr(cp_v16, "call", fake_call)
    monkeypatch.setattr(cp_v16, "notify_ha", fake_notify)

    ok = await cp_v16.set_charge_rate(limit_amps=10, conn_id=2)
    assert ok is True
    assert notices == []
