# -*- coding: utf-8 -*-
"""
Base policy for the restaurant HVAC problem.

The environment calls select_action(state) once per hour and expects a dict:
    {
        "HeatPowerRoom1": float,
        "HeatPowerRoom2": float,
        "VentilationON": int
    }

This is a fast, rule-based policy based on the Part A overrule logic:
- low-temperature overrule -> heater at max until OK threshold,
- high-temperature overrule -> heater off,
- humidity overrule -> ventilation on,
- ventilation inertia -> keep ventilation on during forced-on period.

For non-overruled hours, it uses a simple proportional comfort rule so it can
serve as a base policy for rollout / policy-function approximation.
"""

from __future__ import annotations

from typing import Any, Dict
import math


# -----------------------------------------------------------------------------
# Parameter loading
# -----------------------------------------------------------------------------

def _load_params() -> Dict[str, Any]:
    """Load Data.v2_SystemCharacteristics if available; otherwise use safe fallbacks.

    The exact key names may differ between course releases, so this function
    accepts several common aliases.
    """
    try:
        import Data.v2_SystemCharacteristics  # type: ignore

        if hasattr(Data.v2_SystemCharacteristics, "get_fixed_data"):
            return dict(Data.v2_SystemCharacteristics.get_fixed_data())
        if hasattr(Data.v2_SystemCharacteristics, "fetch_data"):
            return dict(Data.v2_SystemCharacteristics.fetch_data())
    except Exception:
        pass
    return {}


PARAMS = _load_params()


def _p(*names: str, default: Any) -> Any:
    """Return the first matching parameter name from Data.v2_SystemCharacteristics."""
    for name in names:
        if name in PARAMS:
            return PARAMS[name]
    return default


# Conservative defaults; overwritten automatically if Data.v2_SystemCharacteristics exists.
P_MAX = float(_p("heating_max_power", "P_r", "Pr", "PowerMax", "max_heating_power", default=3.0))
P_MAX_R1 = float(_p("heating_max_power_room1", "P1", "P_room1", default=P_MAX))
P_MAX_R2 = float(_p("heating_max_power_room2", "P2", "P_room2", default=P_MAX))

T_LOW = float(_p("T_low", "Tlow", "temperature_low", "TempLow", default=18.0))
T_OK = float(_p("T_OK", "TOK", "temperature_ok", "TempOK", default=20.0))
T_HIGH = float(_p("T_high", "THigh", "temperature_high", "TempHigh", default=24.0))
H_HIGH = float(_p("H_high", "Hhigh", "humidity_high", "HumHigh", default=60.0))

# Optional dynamics parameters used only for a one-step heat estimate.
ZETA_EXCH = float(_p("heat_exchange_coeff", "zeta_exch", "zeta_exchange", default=0.6))
ZETA_LOSS = float(_p("thermal_loss_coeff", "zeta_loss", default=0.1))
ZETA_CONV = float(_p("heating_efficiency_coeff", "zeta_conv", default=1.0))
ZETA_COOL = float(_p("heat_vent_coeff", "zeta_cool", default=0.7))
ZETA_OCC = float(_p("heat_occupancy_coeff", "zeta_occ", default=0.02))

OUTDOOR_TEMP = _p("outdoor_temperature", "T_out", "Tout", default=None)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _num(x: Any, default: float = 0.0) -> float:
    """Safely coerce a state value to a finite float."""
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def _bin(x: Any) -> int:
    """Safely coerce a state value to 0/1."""
    return int(_num(x, 0.0) >= 0.5)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _outdoor_temperature(t: int) -> float:
    """Outdoor temperature for hour t, if available."""
    if isinstance(OUTDOOR_TEMP, (list, tuple)) and len(OUTDOOR_TEMP) > 0:
        return _num(OUTDOOR_TEMP[min(max(t, 0), len(OUTDOOR_TEMP) - 1)], default=5.0)
    return _num(OUTDOOR_TEMP, default=5.0)


def _is_forced_ventilation(state: Dict[str, Any]) -> bool:
    """Apply humidity overrule and ventilation inertia."""
    humidity = _num(state.get("H"), default=0.0)
    vent_counter = int(round(_num(state.get("vent_counter"), default=0.0)))

    # In the Part A/MDP formulation, ventilation must stay on during its forced-on
    # inertia period. Some environments encode this as remaining hours, others as
    # consecutive hours already on. Treat positive values below 3 as forced-on.
    return humidity > H_HIGH or vent_counter in (1, 2)


def _price_is_cheap(state: Dict[str, Any]) -> bool:
    """Classify whether the current price is attractive relative to recent price."""
    price = _num(state.get("price_t"), default=4.0)
    previous = _num(state.get("price_previous"), default=price)
    return price <= previous or price <= 4.0


def _heater_power_for_room(
    temperature: float,
    other_temperature: float,
    occupancy: float,
    low_override_active: int,
    ventilation_on: int,
    p_max: float,
    state: Dict[str, Any],
) -> float:
    """Rule-based heater command for one room."""
    # Overrule controller from Part A: high temperature has priority.
    if temperature > T_HIGH:
        return 0.0

    if low_override_active == 1 or temperature < T_LOW:
        return p_max

    # If there is no overrule, choose a temperature target. When electricity is
    # cheap, preheat toward T_OK; otherwise only protect a small buffer above T_LOW.
    target = T_OK if _price_is_cheap(state) else max(T_LOW + 0.5, min(T_OK, 0.5 * (T_LOW + T_OK)))

    # Approximate next temperature without heating, using the same linear terms as
    # the Part A formulation. This keeps the policy aligned with the solution sheet.
    t = int(round(_num(state.get("current_time"), default=0.0)))
    t_out = _outdoor_temperature(t)
    predicted_without_heat = (
        temperature
        + ZETA_EXCH * (other_temperature - temperature)
        + ZETA_LOSS * (t_out - temperature)
        - ZETA_COOL * ventilation_on
        + ZETA_OCC * occupancy
    )

    # Heat enough to reach the target in one step, if possible.
    needed = (target - predicted_without_heat) / max(ZETA_CONV, 1e-9)

    # Add a small safety margin when close to T_LOW to avoid triggering overrule.
    if temperature <= T_LOW + 0.75:
        needed = max(needed, 0.5 * p_max)

    return _clip(needed, 0.0, p_max)


# -----------------------------------------------------------------------------
# Required policy function
# -----------------------------------------------------------------------------

def select_action(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return here-and-now HVAC actions for the current state."""
    t1 = _num(state.get("T1"), default=T_OK)
    t2 = _num(state.get("T2"), default=T_OK)
    occ1 = _num(state.get("Occ1"), default=0.0)
    occ2 = _num(state.get("Occ2"), default=0.0)
    low1 = _bin(state.get("low_override_r1", 0))
    low2 = _bin(state.get("low_override_r2", 0))

    ventilation_on = 1 if _is_forced_ventilation(state) else 0

    # Optional proactive ventilation: if humidity is very close to the threshold,
    # vent during relatively cheap hours to avoid a later forced start.
    humidity = _num(state.get("H"), default=0.0)
    if ventilation_on == 0 and humidity >= H_HIGH - 2.0 and _price_is_cheap(state):
        ventilation_on = 1

    p1 = _heater_power_for_room(t1, t2, occ1, low1, ventilation_on, P_MAX_R1, state)
    p2 = _heater_power_for_room(t2, t1, occ2, low2, ventilation_on, P_MAX_R2, state)

    HereAndNowActions = {
        "HeatPowerRoom1": float(p1),
        "HeatPowerRoom2": float(p2),
        "VentilationON": int(ventilation_on),
    }

    return HereAndNowActions


# Small smoke test. Remove or comment out before handing in if your checker requires
# absolutely no top-level execution beyond definitions.
if __name__ == "__main__":
    test_state = {
        "T1": 18.4,
        "T2": 20.1,
        "H": 58.5,
        "Occ1": 35,
        "Occ2": 20,
        "price_t": 3.5,
        "price_previous": 4.0,
        "vent_counter": 0,
        "low_override_r1": 0,
        "low_override_r2": 0,
        "current_time": 2,
    }
    print(select_action(test_state))
