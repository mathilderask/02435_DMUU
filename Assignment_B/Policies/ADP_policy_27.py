"""
ADP policy for Task 4.

The submitted policy uses the trained value-function coefficients below.
Running this file with --train-and-update retrains the coefficients using
process-generated forward trajectories and updates THETA_BY_TIME in place.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pyomo.environ as pyo

import Data.v2_SystemCharacteristics as SystemCharacteristics
import Data.PriceProcessRestaurant as PriceProcessRestaurant
import Data.OccupancyProcessRestaurant as OccupancyProcessRestaurant
import Policies.Dummy_policy_27 as DummyPolicy


# ============================================================
# Trained value-function coefficients
# ============================================================
# --- THETA_BY_TIME_START ---
N_FEATURES = 11
THETA_BY_TIME = {
    0: np.array([3.7326917745, 0.0000000000, 0.0000000000, 0.0000000000, 1.1575770348, -0.0661124316, -0.0377782793, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000], dtype=float),
    1: np.array([11.8997436256, 0.4336990038, -0.3327693229, -0.2159556778, 1.0153922887, -0.0928623227, -0.0118439272, -0.1172258771, 0.0000000000, 0.0000000000, 0.0000000000], dtype=float),
    2: np.array([7357.6135762490, -414.3698638347, 3.9539246278, 0.7383220991, 12.5096073520, 0.1816075682, 0.0104654625, 1.4124710814, 2.6932835590, -0.4438259381, 0.0000000000], dtype=float),
    3: np.array([174.4949789182, -3.1404303449, -9.5733596843, 1.3249941971, 15.3975442796, 0.0235571281, -0.0199038825, 11.0633425650, -5.3119219402, -8.1795234777, 0.0000000000], dtype=float),
    4: np.array([227.0706999655, -7.5966417766, -4.5730885387, 0.4256105061, 14.7646197568, -0.0034607931, -0.1246266048, -0.3852455467, -4.5956871338, -0.8793860156, 0.0000000000], dtype=float),
    5: np.array([254.4684326231, -5.9454849967, -7.5327753392, 0.4895155555, 12.1293485074, -0.0631996342, -0.0596764280, -0.9475076078, -0.7972544933, 0.8801349595, 0.0000000000], dtype=float),
    6: np.array([21.8758422913, -2.2545895828, -0.6668002076, 0.4588371459, 8.0268668115, 0.0765127209, 0.0517763696, -1.4786861166, 6.4829118027, 8.1458184003, 0.0000000000], dtype=float),
    7: np.array([10.3522413650, -1.1394257910, -1.5929904780, 0.6368149270, 4.7720574808, 0.1070472659, 0.0487010517, -0.1499403207, 0.0653611368, 1.7424830789, 0.0000000000], dtype=float),
    8: np.array([10.7971325691, -2.3315890220, -0.7797953991, 0.6498974360, 5.1533543810, -0.0015580764, 0.0449040035, 1.4913261208, 9.5856823954, 14.4013672016, 0.0000000000], dtype=float),
    9: np.array([19.0306601472, -1.6396803049, -0.3457511180, 0.1640150451, 3.6785315769, 0.0412915697, -0.0421961018, 0.5771992382, 8.2368689942, 11.4705736688, 0.0000000000], dtype=float),
}
# --- THETA_BY_TIME_END ---


# ============================================================
# Training configuration
# ============================================================
RIDGE = 5.0
ETA_UPDATE_ALPHA = 0.25
TARGET_CLIP_MIN = 0.0


# ============================================================
# Helpers
# ============================================================
def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def get_fixed_params() -> Dict[str, Any]:
    fixed = SystemCharacteristics.get_fixed_data()

    def get_first(*keys: str, default: Any = None) -> Any:
        for k in keys:
            if k in fixed:
                return fixed[k]
        return default

    return {
        "Pmax": float(fixed["heating_max_power"]),
        "Pvent": float(fixed["ventilation_power"]),
        "Tlow": float(fixed["temp_min_comfort_threshold"]),
        "TOK": float(fixed["temp_OK_threshold"]),
        "THigh": float(fixed["temp_max_comfort_threshold"]),
        "HHigh": float(fixed["humidity_threshold"]),
        "z_exch": float(fixed["heat_exchange_coeff"]),
        "z_loss": float(fixed["thermal_loss_coeff"]),
        "z_conv": float(fixed["heating_efficiency_coeff"]),
        "z_cool": float(fixed["heat_vent_coeff"]),
        "z_occ": float(fixed["heat_occupancy_coeff"]),
        "eta_occ": float(fixed["humidity_occupancy_coeff"]),
        "eta_vent": float(fixed["humidity_vent_coeff"]),
        "Uvent": int(fixed["vent_min_up_time"]),
        "Tout": list(fixed["outdoor_temperature"]),
        "num_timeslots": int(fixed["num_timeslots"]),
        "T_init": float(get_first("T1", "initial_temperature", default=21.0)),
        "H_init": float(get_first("H", "initial_humidity", default=40.0)),
    }


def feature_vector(state: Dict[str, Any], num_timeslots: int) -> np.ndarray:
    t = int(round(safe_float(state.get("current_time", 0), 0)))
    remaining_time = max(0, num_timeslots - t)

    return np.array([
        1.0,
        safe_float(state.get("T1", 21.0), 21.0),
        safe_float(state.get("T2", 21.0), 21.0),
        safe_float(state.get("H", 40.0), 40.0),
        safe_float(state.get("price_t", 4.0), 4.0),
        safe_float(state.get("Occ1", 30.0), 30.0),
        safe_float(state.get("Occ2", 20.0), 20.0),
        safe_float(state.get("vent_counter", 0), 0),
        safe_float(state.get("low_override_r1", 0), 0),
        safe_float(state.get("low_override_r2", 0), 0),
        float(remaining_time),
    ], dtype=float)


def approximate_value_with_theta_dict(
    state: Dict[str, Any],
    theta_by_time: Dict[int, np.ndarray],
    num_timeslots: int,
) -> float:
    t = int(round(safe_float(state.get("current_time", 0), 0)))
    if t >= num_timeslots:
        return 0.0
    t = max(0, min(t, num_timeslots - 1))

    theta = theta_by_time.get(t)
    if theta is None:
        return 0.0

    return float(np.dot(theta, feature_vector(state, num_timeslots)))


def approximate_value(state: Dict[str, Any], num_timeslots: int) -> float:
    return approximate_value_with_theta_dict(state, THETA_BY_TIME, num_timeslots)


# ============================================================
# Exogenous process sampling
# ============================================================
def sample_next_price(current_price: float, previous_price: float) -> float:
    return float(PriceProcessRestaurant.price_model(float(current_price), float(previous_price)))


def sample_next_occupancy(occ1: float, occ2: float) -> Tuple[float, float]:
    next_occ1, next_occ2 = OccupancyProcessRestaurant.next_occupancy_levels(float(occ1), float(occ2))
    return float(next_occ1), float(next_occ2)


def expected_next_exogenous(state: Dict[str, Any], samples: int = 7) -> Tuple[float, float, float]:
    price_t = safe_float(state.get("price_t", 4.0), 4.0)
    price_prev = safe_float(state.get("price_previous", 4.0), 4.0)
    occ1 = safe_float(state.get("Occ1", 30.0), 30.0)
    occ2 = safe_float(state.get("Occ2", 20.0), 20.0)

    price_vals = []
    occ1_vals = []
    occ2_vals = []
    for _ in range(samples):
        price_vals.append(sample_next_price(price_t, price_prev))
        o1, o2 = sample_next_occupancy(occ1, occ2)
        occ1_vals.append(o1)
        occ2_vals.append(o2)

    return float(np.mean(price_vals)), float(np.mean(occ1_vals)), float(np.mean(occ2_vals))


def sampled_next_exogenous_means(state: Dict[str, Any], samples: int) -> Tuple[float, float, float]:
    """Sample K next exogenous states and return their means.

    This keeps the Bellman target continuous and avoids discrete action enumeration.
    The value-function terms in price and occupancy are linear, so the sample mean
    is equivalent to averaging those parts of the value approximation.
    """
    return expected_next_exogenous(state, samples=max(1, int(samples)))


# ============================================================
# System dynamics and overrules
# ============================================================
def apply_overrules(
    state: Dict[str, Any],
    heat1: float,
    heat2: float,
    ventilation: int,
    params: Dict[str, Any],
) -> Tuple[float, float, int]:
    Pmax = params["Pmax"]

    heat1 = float(heat1)
    heat2 = float(heat2)
    ventilation = int(ventilation)

    T1 = safe_float(state.get("T1", 21.0), 21.0)
    T2 = safe_float(state.get("T2", 21.0), 21.0)
    H = safe_float(state.get("H", 40.0), 40.0)

    vent_counter = int(round(safe_float(state.get("vent_counter", 0), 0)))
    low1 = int(safe_float(state.get("low_override_r1", 0), 0) > 0.5)
    low2 = int(safe_float(state.get("low_override_r2", 0), 0) > 0.5)

    if T1 >= params["THigh"]:
        heat1 = 0.0
    elif T1 <= params["Tlow"] or low1 == 1:
        heat1 = Pmax

    if T2 >= params["THigh"]:
        heat2 = 0.0
    elif T2 <= params["Tlow"] or low2 == 1:
        heat2 = Pmax

    if H >= params["HHigh"]:
        ventilation = 1

    if 0 < vent_counter < params["Uvent"]:
        ventilation = 1

    return (
        float(np.clip(heat1, 0.0, Pmax)),
        float(np.clip(heat2, 0.0, Pmax)),
        int(np.clip(ventilation, 0, 1)),
    )


def simulate_next_state(
    state: Dict[str, Any],
    heat1: float,
    heat2: float,
    ventilation: int,
    params: Dict[str, Any],
    mode: str = "sample",
) -> Dict[str, Any]:
    t = int(round(safe_float(state.get("current_time", 0), 0)))
    H_day = params["num_timeslots"]

    T1 = safe_float(state.get("T1", 21.0), 21.0)
    T2 = safe_float(state.get("T2", 21.0), 21.0)
    Hum = safe_float(state.get("H", 40.0), 40.0)

    Occ1 = safe_float(state.get("Occ1", 30.0), 30.0)
    Occ2 = safe_float(state.get("Occ2", 20.0), 20.0)

    price_t = safe_float(state.get("price_t", 4.0), 4.0)
    price_prev = safe_float(state.get("price_previous", 4.0), 4.0)

    Tout = params["Tout"][t % H_day]

    T1_next = (
        T1
        + params["z_exch"] * (T2 - T1)
        + params["z_loss"] * (Tout - T1)
        + params["z_conv"] * heat1
        - params["z_cool"] * ventilation
        + params["z_occ"] * Occ1
    )

    T2_next = (
        T2
        + params["z_exch"] * (T1 - T2)
        + params["z_loss"] * (Tout - T2)
        + params["z_conv"] * heat2
        - params["z_cool"] * ventilation
        + params["z_occ"] * Occ2
    )

    H_next = Hum + params["eta_occ"] * (Occ1 + Occ2) - params["eta_vent"] * ventilation

    if mode == "expected":
        price_next, Occ1_next, Occ2_next = expected_next_exogenous(state, samples=7)
    else:
        price_next = sample_next_price(price_t, price_prev)
        Occ1_next, Occ2_next = sample_next_occupancy(Occ1, Occ2)

    low1 = int(safe_float(state.get("low_override_r1", 0), 0) > 0.5)
    low2 = int(safe_float(state.get("low_override_r2", 0), 0) > 0.5)

    low1_next = int(T1_next <= params["Tlow"] or (low1 == 1 and T1_next < params["TOK"]))
    low2_next = int(T2_next <= params["Tlow"] or (low2 == 1 and T2_next < params["TOK"]))

    if ventilation == 1:
        vent_counter_next = int(round(safe_float(state.get("vent_counter", 0), 0))) + 1
    else:
        vent_counter_next = 0

    return {
        "T1": float(T1_next),
        "T2": float(T2_next),
        "H": float(H_next),
        "Occ1": float(Occ1_next),
        "Occ2": float(Occ2_next),
        "price_t": float(price_next),
        "price_previous": float(price_t),
        "vent_counter": int(vent_counter_next),
        "low_override_r1": int(low1_next),
        "low_override_r2": int(low2_next),
        "current_time": int(t + 1),
    }


# ============================================================
# Continuous one-step Bellman problem
# ============================================================
def solve_continuous_bellman(
    state: Dict[str, Any],
    theta_next: np.ndarray,
    params: Dict[str, Any],
    price_next: float,
    occ1_next: float,
    occ2_next: float,
    solver_time_limit: float = 3.0,
    mip_gap: float = 0.01,
) -> Tuple[float, Dict[str, Any]]:
    """Solve the continuous one-step ADP Bellman problem.

    Returns
    -------
    objective_value, action
    """
    Pmax = params["Pmax"]
    Pvent = params["Pvent"]
    Tlow = params["Tlow"]
    TOK = params["TOK"]
    THigh = params["THigh"]
    HHigh = params["HHigh"]
    H_day = params["num_timeslots"]
    BIG_M = 100.0

    t = int(round(safe_float(state.get("current_time", 0), 0)))
    remaining_next = max(0, H_day - (t + 1))

    T1 = safe_float(state.get("T1", 21.0), 21.0)
    T2 = safe_float(state.get("T2", 21.0), 21.0)
    Hum = safe_float(state.get("H", 40.0), 40.0)
    Occ1 = safe_float(state.get("Occ1", 30.0), 30.0)
    Occ2 = safe_float(state.get("Occ2", 20.0), 20.0)
    price_t = safe_float(state.get("price_t", 4.0), 4.0)
    vent_counter = int(round(safe_float(state.get("vent_counter", 0), 0)))
    low1 = int(safe_float(state.get("low_override_r1", 0), 0) > 0.5)
    low2 = int(safe_float(state.get("low_override_r2", 0), 0) > 0.5)

    if T1 <= Tlow:
        low1 = 1
    if T2 <= Tlow:
        low2 = 1

    Tout = params["Tout"][t % H_day]
    theta_next = np.array(theta_next, dtype=float)

    m = pyo.ConcreteModel()
    m.R = pyo.Set(initialize=[1, 2])

    m.pc = pyo.Var(m.R, bounds=(0.0, Pmax))
    m.vb = pyo.Var(domain=pyo.Binary)
    m.pf = pyo.Var(m.R, bounds=(0.0, Pmax))
    m.ve = pyo.Var(domain=pyo.Binary)

    m.T1_next = pyo.Var()
    m.T2_next = pyo.Var()
    m.H_next = pyo.Var()

    m.low1_next = pyo.Var(domain=pyo.Binary)
    m.low2_next = pyo.Var(domain=pyo.Binary)
    m.z1_below_low = pyo.Var(domain=pyo.Binary)
    m.z2_below_low = pyo.Var(domain=pyo.Binary)
    m.y1_ok_next = pyo.Var(domain=pyo.Binary)
    m.y2_ok_next = pyo.Var(domain=pyo.Binary)

    m.V_next = pyo.Var(bounds=(0.0, None))
    m.cons = pyo.ConstraintList()

    # Effective heating overrules
    if T1 >= THigh:
        m.cons.add(m.pf[1] == 0.0)
        m.cons.add(m.pc[1] == 0.0)
    elif low1 == 1:
        m.cons.add(m.pf[1] == Pmax)
        m.cons.add(m.pc[1] == Pmax)
    else:
        m.cons.add(m.pf[1] == m.pc[1])

    if T2 >= THigh:
        m.cons.add(m.pf[2] == 0.0)
        m.cons.add(m.pc[2] == 0.0)
    elif low2 == 1:
        m.cons.add(m.pf[2] == Pmax)
        m.cons.add(m.pc[2] == Pmax)
    else:
        m.cons.add(m.pf[2] == m.pc[2])

    # Effective ventilation overrules
    if Hum >= HHigh or (0 < vent_counter < params["Uvent"]):
        m.cons.add(m.ve == 1)
        m.cons.add(m.vb == 1)
    else:
        m.cons.add(m.ve == m.vb)

    # Dynamics
    m.cons.add(
        m.T1_next ==
        T1
        + params["z_exch"] * (T2 - T1)
        + params["z_loss"] * (Tout - T1)
        + params["z_conv"] * m.pf[1]
        - params["z_cool"] * m.ve
        + params["z_occ"] * Occ1
    )
    m.cons.add(
        m.T2_next ==
        T2
        + params["z_exch"] * (T1 - T2)
        + params["z_loss"] * (Tout - T2)
        + params["z_conv"] * m.pf[2]
        - params["z_cool"] * m.ve
        + params["z_occ"] * Occ2
    )
    m.cons.add(m.H_next == Hum + params["eta_occ"] * (Occ1 + Occ2) - params["eta_vent"] * m.ve)

    # Low-temperature latch next state
    m.cons.add(m.T1_next <= Tlow + BIG_M * (1 - m.z1_below_low))
    m.cons.add(m.T1_next >= Tlow - BIG_M * m.z1_below_low)
    m.cons.add(m.T2_next <= Tlow + BIG_M * (1 - m.z2_below_low))
    m.cons.add(m.T2_next >= Tlow - BIG_M * m.z2_below_low)

    m.cons.add(m.T1_next >= TOK - BIG_M * (1 - m.y1_ok_next))
    m.cons.add(m.T1_next <= TOK + BIG_M * m.y1_ok_next)
    m.cons.add(m.T2_next >= TOK - BIG_M * (1 - m.y2_ok_next))
    m.cons.add(m.T2_next <= TOK + BIG_M * m.y2_ok_next)

    m.cons.add(m.low1_next >= m.z1_below_low)
    m.cons.add(m.low1_next >= low1 - m.y1_ok_next)
    m.cons.add(m.low1_next <= m.z1_below_low + low1)
    m.cons.add(m.low1_next <= m.z1_below_low + (1 - m.y1_ok_next))

    m.cons.add(m.low2_next >= m.z2_below_low)
    m.cons.add(m.low2_next >= low2 - m.y2_ok_next)
    m.cons.add(m.low2_next <= m.z2_below_low + low2)
    m.cons.add(m.low2_next <= m.z2_below_low + (1 - m.y2_ok_next))

    vent_counter_next_expr = (vent_counter + 1) * m.ve

    value_expr = (
        theta_next[0]
        + theta_next[1] * m.T1_next
        + theta_next[2] * m.T2_next
        + theta_next[3] * m.H_next
        + theta_next[4] * price_next
        + theta_next[5] * occ1_next
        + theta_next[6] * occ2_next
        + theta_next[7] * vent_counter_next_expr
        + theta_next[8] * m.low1_next
        + theta_next[9] * m.low2_next
        + theta_next[10] * remaining_next
    )
    m.cons.add(m.V_next >= value_expr)
    m.cons.add(m.V_next >= TARGET_CLIP_MIN)

    immediate_cost = price_t * (m.pf[1] + m.pf[2] + Pvent * m.ve)
    m.obj = pyo.Objective(expr=immediate_cost + m.V_next, sense=pyo.minimize)

    solver = pyo.SolverFactory("gurobi")
    solver.options["OutputFlag"] = 0
    solver.options["TimeLimit"] = solver_time_limit
    solver.options["MIPGap"] = mip_gap
    results = solver.solve(m, tee=False)

    ok = (
        results.solver.status == pyo.SolverStatus.ok
        and results.solver.termination_condition in [
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
            pyo.TerminationCondition.maxTimeLimit,
        ]
    )
    if not ok:
        raise RuntimeError(f"Continuous ADP solve failed: {results.solver.status}, {results.solver.termination_condition}")

    h1 = pyo.value(m.pf[1])
    h2 = pyo.value(m.pf[2])
    v = pyo.value(m.ve)
    obj = pyo.value(m.obj)

    if h1 is None or h2 is None or v is None or obj is None:
        raise RuntimeError("Continuous ADP solve returned uninitialized values.")

    action = {
        "HeatPowerRoom1": float(max(0.0, min(Pmax, h1))),
        "HeatPowerRoom2": float(max(0.0, min(Pmax, h2))),
        "VentilationON": int(float(v) > 0.5),
    }
    return float(obj), action


def select_action_with_theta_continuous(state: Dict[str, Any], theta_by_time: Dict[int, np.ndarray]) -> Dict[str, Any]:
    params = get_fixed_params()
    H_day = params["num_timeslots"]
    t = int(round(safe_float(state.get("current_time", 0), 0)))

    theta_next = theta_by_time.get(t + 1)
    if theta_next is None or t + 1 >= H_day:
        theta_next = np.zeros(len(feature_vector(state, H_day)))

    price_next, occ1_next, occ2_next = expected_next_exogenous(state, samples=7)

    try:
        _, action = solve_continuous_bellman(
            state=state,
            theta_next=theta_next,
            params=params,
            price_next=price_next,
            occ1_next=occ1_next,
            occ2_next=occ2_next,
            solver_time_limit=3.0,
            mip_gap=0.01,
        )
        return action
    except Exception:
        return DummyPolicy.select_action(state)


def select_action(state: Dict[str, Any]) -> Dict[str, Any]:
    return select_action_with_theta_continuous(state, THETA_BY_TIME)


# ============================================================
# Offline training
# ============================================================
def sample_training_state(t: int) -> Dict[str, Any]:
    return {
        "T1": float(np.random.uniform(17.0, 24.0)),
        "T2": float(np.random.uniform(17.0, 24.0)),
        "H": float(np.random.uniform(35.0, 70.0)),
        "Occ1": float(np.random.uniform(20.0, 50.0)),
        "Occ2": float(np.random.uniform(10.0, 30.0)),
        "price_t": float(np.random.uniform(0.0, 12.0)),
        "price_previous": float(np.random.uniform(0.0, 12.0)),
        "vent_counter": int(np.random.choice([0, 1, 2])),
        "low_override_r1": int(np.random.choice([0, 1])),
        "low_override_r2": int(np.random.choice([0, 1])),
        "current_time": int(t),
    }


def bellman_target_continuous(
    state: Dict[str, Any],
    theta_next: np.ndarray,
    params: Dict[str, Any],
    K_next: int,
) -> float:
    price_next, occ1_next, occ2_next = sampled_next_exogenous_means(state, samples=K_next)

    try:
        obj, _ = solve_continuous_bellman(
            state=state,
            theta_next=theta_next,
            params=params,
            price_next=price_next,
            occ1_next=occ1_next,
            occ2_next=occ2_next,
            solver_time_limit=3.0,
            mip_gap=0.01,
        )
        return float(max(TARGET_CLIP_MIN, obj))
    except Exception:
        # Conservative fallback: evaluate dummy action one step + value.
        h1, h2, v = apply_overrules(state, 0.0, 0.0, 0, params)
        immediate = safe_float(state.get("price_t", 4.0), 4.0) * (h1 + h2 + params["Pvent"] * v)
        next_state = simulate_next_state(state, h1, h2, v, params, mode="sample")
        t_next = int(round(safe_float(state.get("current_time", 0), 0))) + 1
        future = approximate_value_with_theta_dict(next_state, {t_next: theta_next}, params["num_timeslots"])
        return float(max(TARGET_CLIP_MIN, immediate + future))


def fit_theta_ridge(states: list[Dict[str, Any]], targets: list[float], num_timeslots: int, ridge: float = RIDGE) -> np.ndarray:
    X = np.array([feature_vector(s, num_timeslots) for s in states], dtype=float)
    y = np.array(targets, dtype=float)

    if len(states) == 0:
        return np.zeros(X.shape[1] if X.ndim == 2 else N_FEATURES)

    Xs = X.copy()
    means = Xs[:, 1:].mean(axis=0)
    stds = Xs[:, 1:].std(axis=0)
    stds[stds < 1e-8] = 1.0
    Xs[:, 1:] = (Xs[:, 1:] - means) / stds

    if ridge > 0:
        penalty = ridge * np.eye(Xs.shape[1])
        penalty[0, 0] = 0.0
        try:
            beta_scaled = np.linalg.solve(Xs.T @ Xs + penalty, Xs.T @ y)
        except np.linalg.LinAlgError:
            beta_scaled, *_ = np.linalg.lstsq(Xs.T @ Xs + penalty, Xs.T @ y, rcond=None)
    else:
        beta_scaled, *_ = np.linalg.lstsq(Xs, y, rcond=None)

    beta = np.zeros_like(beta_scaled)
    beta[1:] = beta_scaled[1:] / stds
    beta[0] = beta_scaled[0] - np.sum(beta_scaled[1:] * means / stds)
    return beta


# ============================================================
# Forward-backward training
# ============================================================
def choose_forward_action(state: Dict[str, Any], theta_by_time: Dict[int, np.ndarray], use_dummy_policy: bool) -> Dict[str, Any]:
    if use_dummy_policy:
        return DummyPolicy.select_action(state)
    return select_action_with_theta_continuous(state, theta_by_time)


def simulate_policy_transition(
    state: Dict[str, Any],
    action: Dict[str, Any],
    params: Dict[str, Any],
    mode: str = "sample",
) -> Dict[str, Any]:
    h1_raw = float(action["HeatPowerRoom1"])
    h2_raw = float(action["HeatPowerRoom2"])
    v_raw = int(action["VentilationON"])
    h1, h2, v = apply_overrules(state, h1_raw, h2_raw, v_raw, params)
    return simulate_next_state(state, h1, h2, v, params, mode=mode)


def generate_forward_pass_states(
    theta_by_time: Dict[int, np.ndarray],
    n_trajectories: int = 200,
    seed: int = 1234,
    mode: str = "sample",
    use_dummy_policy: bool = False,
) -> Dict[int, list[Dict[str, Any]]]:
    """Forward pass: simulate trajectories from the provided stochastic process models.

    No empirical CSV files are loaded here. This keeps training separate from the
    empirical evaluation days. Initial exogenous values are sampled from broad
    ranges matching the assignment data scale; all subsequent price and occupancy
    transitions are generated by PriceProcessRestaurant and
    OccupancyProcessRestaurant through simulate_next_state(..., mode="sample").
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    params = get_fixed_params()
    H_day = params["num_timeslots"]
    states_by_t = {t: [] for t in range(H_day)}

    for _ in range(n_trajectories):
        price_0 = float(rng.uniform(0.0, 12.0))
        state = {
            "T1": params["T_init"],
            "T2": params["T_init"],
            "H": params["H_init"],
            "Occ1": float(rng.uniform(20.0, 50.0)),
            "Occ2": float(rng.uniform(10.0, 30.0)),
            "price_t": price_0,
            "price_previous": price_0,
            "vent_counter": 0,
            "low_override_r1": 0,
            "low_override_r2": 0,
            "current_time": 0,
        }

        for t in range(H_day):
            state["current_time"] = t
            states_by_t[t].append(dict(state))

            action = choose_forward_action(
                state,
                theta_by_time,
                use_dummy_policy=use_dummy_policy,
            )
            state = simulate_policy_transition(state, action, params, mode=mode)

    return states_by_t


def choose_states_for_backward(
    t: int,
    forward_states_by_t: Dict[int, list[Dict[str, Any]]],
    N_states: int,
) -> list[Dict[str, Any]]:
    candidates = forward_states_by_t.get(t, [])
    if len(candidates) == 0:
        return [sample_training_state(t) for _ in range(N_states)]

    idx = np.random.choice(len(candidates), size=N_states, replace=True)
    return [dict(candidates[i]) for i in idx]


def train_theta_forward_backward(
    n_iterations: int = 10,
    N_states: int = 300,
    K_next: int = 10,
    n_forward_trajectories: int = 300,
    seed: int = 321,
    ridge: float = RIDGE,
    alpha: float = ETA_UPDATE_ALPHA,
) -> Dict[int, np.ndarray]:
    global THETA_BY_TIME
    np.random.seed(seed)

    params = get_fixed_params()
    H_day = params["num_timeslots"]
    n_features = len(feature_vector(sample_training_state(0), H_day))

    theta = {t: np.zeros(n_features) for t in range(H_day)}

    for it in range(1, n_iterations + 1):
        use_dummy_policy = (it == 1)

        print(f"\nForward-backward iteration {it}/{n_iterations} | forward policy: {'Dummy_policy_27' if use_dummy_policy else 'continuous ADP'}")

        forward_states = generate_forward_pass_states(
            theta,
            n_trajectories=n_forward_trajectories,
            seed=seed + 1000 * it,
            mode="sample",
            use_dummy_policy=use_dummy_policy,
        )

        theta_fit = {}
        theta_next = np.zeros(n_features)
        total_delta = 0.0
        total_norm = 0.0

        for t in reversed(range(H_day)):
            states_t = choose_states_for_backward(t, forward_states, N_states)
            targets_t = [bellman_target_continuous(s, theta_next, params, K_next) for s in states_t]
            fitted = fit_theta_ridge(states_t, targets_t, H_day, ridge=ridge)

            old = theta.get(t, np.zeros(n_features))
            updated = (1.0 - alpha) * old + alpha * fitted
            theta_fit[t] = updated
            theta_next = updated

            delta = float(np.linalg.norm(updated - old))
            norm = float(np.linalg.norm(old))
            total_delta += delta
            total_norm += norm

            print(
                f"  t={t}: target_mean={np.mean(targets_t):7.2f}, "
                f"std={np.std(targets_t):7.2f}, "
                f"eta_delta={delta:9.3f}"
            )

        theta = theta_fit
        rel_delta = total_delta / (1e-6 + total_norm)
        print(f"  relative eta change: {rel_delta:.4f}")

    THETA_BY_TIME = theta
    return theta


# ============================================================
# Safe theta updater
# ============================================================
def format_theta_block(theta_by_time: Dict[int, np.ndarray]) -> str:
    start_marker = "# --- " + "THETA_BY_TIME_START ---"
    end_marker = "# --- " + "THETA_BY_TIME_END ---"

    lines = [
        start_marker,
        f"N_FEATURES = {len(next(iter(theta_by_time.values())))}",
        "THETA_BY_TIME = {",
    ]
    for t in sorted(theta_by_time.keys()):
        values = ", ".join(f"{float(v):.10f}" for v in theta_by_time[t])
        lines.append(f"    {t}: np.array([{values}], dtype=float),")
    lines.extend(["}", end_marker])
    return "\n".join(lines)


def update_theta_in_file(theta_by_time: Dict[int, np.ndarray], file_path: Optional[Path] = None) -> None:
    if file_path is None:
        file_path = Path(__file__)
    else:
        file_path = Path(file_path)

    text = file_path.read_text(encoding="utf-8")
    start_marker = "# --- " + "THETA_BY_TIME_START ---"
    end_marker = "# --- " + "THETA_BY_TIME_END ---"

    start_idx = text.find(start_marker)
    if start_idx == -1:
        raise RuntimeError("Could not find THETA_BY_TIME_START marker.")

    end_idx = text.find(end_marker, start_idx)
    if end_idx == -1:
        raise RuntimeError("Could not find THETA_BY_TIME_END marker after start marker.")
    end_idx += len(end_marker)

    updated = text[:start_idx] + format_theta_block(theta_by_time) + text[end_idx:]
    file_path.write_text(updated, encoding="utf-8")
    print(f"Updated THETA_BY_TIME in {file_path}")


# ============================================================
# Command line interface
# ============================================================
if __name__ == "__main__":
    if "--train-and-update" in sys.argv:
        print("Continuous forward-backward ADP training")
        print("Iteration 1 uses DummyPolicy.py.")
        print("Iterations 2...K use the current continuous ADP policy.")
        print("No discrete action enumeration is used.\n")

        theta = train_theta_forward_backward(
            n_iterations=10,
            N_states=300,
            K_next=10,
            n_forward_trajectories=300,
            seed=321,
            ridge=RIDGE,
            alpha=ETA_UPDATE_ALPHA,
        )
        update_theta_in_file(theta)

    else:
        print("Use --train-and-update to train and update THETA_BY_TIME.")
