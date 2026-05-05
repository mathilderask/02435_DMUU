# -*- coding: utf-8 -*-
"""
Dummy policy for the restaurant HVAC problem.

Task 6 point 2 asks for a dummy policy that never turns on the
ventilation nor any heater, leaving everything to the environment's
overrule controllers.

The environment calls select_action(state) once per hour and expects:
    {
        "HeatPowerRoom1": float,
        "HeatPowerRoom2": float,
        "VentilationON": int
    }
"""

from __future__ import annotations

from typing import Any, Dict


def select_action(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return the dummy here-and-now action.

    This policy intentionally does not use the state. If temperature or
    humidity constraints are violated, the environment/checks should apply
    the built-in overrule logic.
    """
    HereAndNowActions = {
        "HeatPowerRoom1": 0.0,
        "HeatPowerRoom2": 0.0,
        "VentilationON": 0,
    }

    return HereAndNowActions


if __name__ == "__main__":
    # Small smoke test. Safe to remove before hand-in.
    print(select_action({}))
