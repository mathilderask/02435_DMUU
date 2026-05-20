"""
Task 5 Hybrid policy: ADP rollout policy with continuous ADP base policy.

The trained ADP policy is used as the base policy. Rollout is used only to
improve the current here-and-now action. For each candidate current action,
future trajectories are simulated and all future decisions are chosen by the
fixed ADP base policy.

No stochastic programming policy and no empirical evaluation data are used here.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np

import ADP_policy_27 as ADP


# ============================================================
# Rollout configuration
# ============================================================
N_ROLLOUT_SCENARIOS = 3
PRINT_RUNTIME = True

HEAT_GRID_FRACTIONS = [0.0, 0.25, 0.50, 0.75, 1.0]
VENT_GRID = [0, 1]


# ============================================================
# Helpers
# ============================================================
def safe_float(x: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.

    Parameters:
        x: value to convert
        default: default value if conversion fails

    Returns:
        float: converted value or default
    """
    try:
        return float(x)
    except Exception:
        return float(default)


def immediate_cost(state: Dict[str, Any], action: Dict[str, Any], params: Dict[str, Any]) -> float:
    """
    Immediate electricity cost after applying controller overrules.

    Parameters:
        state: current system state
        action: current control action
        params: fixed problem data

    Returns:
        float: one-period electricity cost
    """
    h1, h2, v = ADP.apply_overrules(
        state,
        safe_float(action.get("HeatPowerRoom1", 0.0)),
        safe_float(action.get("HeatPowerRoom2", 0.0)),
        int(round(safe_float(action.get("VentilationON", 0), 0))),
        params,
    )

    price = safe_float(state.get("price_t", 4.0), 4.0)
    return float(price * (h1 + h2 + params["Pvent"] * v))


def simulate_transition(
    state: Dict[str, Any],
    action: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Apply overrules and simulate the next state.

    Parameters:
        state: current system state
        action: current control action
        params: fixed problem data

    Returns:
        dict: next system state
    """
    h1, h2, v = ADP.apply_overrules(
        state,
        safe_float(action.get("HeatPowerRoom1", 0.0)),
        safe_float(action.get("HeatPowerRoom2", 0.0)),
        int(round(safe_float(action.get("VentilationON", 0), 0))),
        params,
    )

    return ADP.simulate_next_state(state, h1, h2, v, params, mode="sample")


def candidate_current_actions(
    base_action: Dict[str, Any],
    state: Dict[str, Any],
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Generate candidate here-and-now actions for the rollout step.

    Parameters:
        base_action: action suggested by the ADP base policy
        state: current system state
        params: fixed problem data

    Returns:
        list: de-duplicated candidate actions after overrules
    """
    Pmax = params["Pmax"]

    h1_base = float(np.clip(safe_float(base_action.get("HeatPowerRoom1", 0.0)), 0.0, Pmax))
    h2_base = float(np.clip(safe_float(base_action.get("HeatPowerRoom2", 0.0)), 0.0, Pmax))
    v_base = int(round(safe_float(base_action.get("VentilationON", 0), 0)))

    raw_candidates: List[Tuple[float, float, int]] = [(h1_base, h2_base, v_base)]

    for f1 in HEAT_GRID_FRACTIONS:
        for f2 in HEAT_GRID_FRACTIONS:
            for v in VENT_GRID:
                raw_candidates.append((f1 * Pmax, f2 * Pmax, int(v)))

    candidates = []
    seen = set()

    for h1, h2, v in raw_candidates:
        h1 = float(np.clip(h1, 0.0, Pmax))
        h2 = float(np.clip(h2, 0.0, Pmax))
        v = int(np.clip(v, 0, 1))

        eff_h1, eff_h2, eff_v = ADP.apply_overrules(state, h1, h2, v, params)
        key = (round(eff_h1, 6), round(eff_h2, 6), int(eff_v))

        if key in seen:
            continue

        seen.add(key)
        candidates.append({
            "HeatPowerRoom1": float(eff_h1),
            "HeatPowerRoom2": float(eff_h2),
            "VentilationON": int(eff_v),
        })

    return candidates


def rollout_cost_after_current_action(
    state: Dict[str, Any],
    current_action: Dict[str, Any],
    params: Dict[str, Any],
) -> float:
    """
    Estimate the cost of a current action followed by the ADP base policy.

    Parameters:
        state: current system state
        current_action: candidate here-and-now action
        params: fixed problem data

    Returns:
        float: average simulated rollout cost
    """
    H_day = params["num_timeslots"]
    scenario_costs = []

    for _ in range(N_ROLLOUT_SCENARIOS):
        s = dict(state)
        total = immediate_cost(s, current_action, params)
        s = simulate_transition(s, current_action, params)

        while int(round(safe_float(s.get("current_time", 0), 0))) < H_day:
            action = ADP.select_action(s)
            total += immediate_cost(s, action, params)
            s = simulate_transition(s, action, params)

        scenario_costs.append(float(total))

    return float(np.mean(scenario_costs))


# ============================================================
# Submitted policy function
# ============================================================
def select_action(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Select the current action using ADP rollout.

    Parameters:
        state: current system state

    Returns:
        dict: here-and-now action
    """
    start_time = time.perf_counter()
    params = ADP.get_fixed_params()

    # The ADP policy is the base policy and is also included as candidate.
    base_action = ADP.select_action(state)
    best_action = base_action
    best_cost = float("inf")

    candidates = candidate_current_actions(base_action, state, params)
    evaluated_candidates = 0

    for action in candidates:
        rollout_cost = rollout_cost_after_current_action(state, action, params)
        evaluated_candidates += 1

        if rollout_cost < best_cost:
            best_cost = rollout_cost
            best_action = action

    result = {
        "HeatPowerRoom1": float(best_action["HeatPowerRoom1"]),
        "HeatPowerRoom2": float(best_action["HeatPowerRoom2"]),
        "VentilationON": int(best_action["VentilationON"]),
    }

    if PRINT_RUNTIME:
        elapsed = time.perf_counter() - start_time
        t = int(round(safe_float(state.get("current_time", -1), -1)))
        print(
            f"Hybrid rollout policy call | t={t:02d} | "
            f"runtime={elapsed:.2f}s | "
            f"scenarios={N_ROLLOUT_SCENARIOS} | "
            f"candidates={evaluated_candidates}/{len(candidates)} | "
            f"action=({result['HeatPowerRoom1']:.2f}, "
            f"{result['HeatPowerRoom2']:.2f}, {result['VentilationON']})",
            flush=True,
        )

    return result
