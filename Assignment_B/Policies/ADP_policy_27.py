"""
Task 4 ADP policy: continuous forward-backward ADP with dummy initial policy.

Straightforward training logic:
1) Start with zero value-function coefficients.
2) Forward-backward iteration 1: generate visited states using DummyPolicy.select_action,
   i.e. zero heating and zero ventilation commands.
3) Forward-backward iterations 2...K: generate visited states using the current
   continuous ADP policy.
4) After each forward pass, refit the value function backwards using ridge
   regression and gradual eta updates.

For submission, only select_action(state) is needed. Running this file with
--train-and-update updates THETA_BY_TIME in-place.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyomo.environ as pyo

import Data.v2_SystemCharacteristics as SystemCharacteristics
import Data.PriceProcessRestaurant as PriceProcessRestaurant
import Data.OccupancyProcessRestaurant as OccupancyProcessRestaurant
import DummyPolicy


# ============================================================
# Trained value-function coefficients
# Replace this dictionary after training.
# ============================================================
# --- THETA_BY_TIME_START ---
N_FEATURES = 11
THETA_BY_TIME = {
    0: np.array([-13.9733426161, 0.0000000000, 0.0000000000, 0.0000000000, 9.5709986829, -0.1763070768, -0.1018769506, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000], dtype=float),
    1: np.array([37.3473611755, -1.5787883563, 0.1792552768, -0.7634177772, 17.6414092912, -0.4499183037, -0.1085335466, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000], dtype=float),
    2: np.array([22.9192648820, -6.2763503670, 2.8628937027, 0.4475803516, 16.5261505188, 0.1313908713, 0.0119699410, 1.7425591402, -3.7958495549, 0.0000000000, 0.0000000000], dtype=float),
    3: np.array([142.4278017360, -6.9422082970, -5.8411226209, 1.6472903003, 13.7927915152, -0.0512152682, -0.1986714591, 12.5789752126, -1.3231263129, 0.0000000000, 0.0000000000], dtype=float),
    4: np.array([330.8498869273, -6.1551699715, -9.3158884526, 0.1415508616, 11.5582200912, -0.3345009111, -0.2769839911, -2.2361021780, 2.6169002400, -0.4444039031, 0.0000000000], dtype=float),
    5: np.array([85.9143143295, -4.1482166045, -0.4667509770, 0.1438176250, 6.6624265572, 0.2181195586, -0.0870859761, -4.1915946939, 11.4246104886, 7.8466465774, 0.0000000000], dtype=float),
    6: np.array([54.4548401406, -1.5273083664, -1.8019691848, 0.2539239445, 4.2220029099, 0.0601025526, 0.2808357357, -3.3828758151, 3.7269622837, 5.7181506348, 0.0000000000], dtype=float),
    7: np.array([32.5317452573, -2.3688704429, -1.3738156696, 0.4583020896, 5.0729307658, 0.0735469091, 0.3710105464, -0.3688022362, 19.0737835147, 5.2664744929, 0.0000000000], dtype=float),
    8: np.array([-13.5941868365, -1.2313829177, -0.2784195020, 0.4366945679, 7.4023561004, -0.0560423370, 0.1891649968, 0.7526885405, 12.5045500662, 12.2495367647, 0.0000000000], dtype=float),
    9: np.array([-24.8985393762, -0.2308081111, 0.2166787426, 0.1720850672, 6.4226297655, -0.0520300909, 0.0610423709, 1.6281291481, 6.6780024340, 7.5461295624, 0.0000000000], dtype=float),
}
# --- THETA_BY_TIME_END ---


# ============================================================
# Training configuration
# ============================================================
RIDGE = 1.0                 # stabilizes least-squares fits
ETA_UPDATE_ALPHA = 0.35      # gradual eta update in forward-backward iterations
TARGET_CLIP_MIN = 0.0        # costs should not become negative
# Forward-backward training uses only visited states from the forward pass.
# First forward pass: dummy policy. Later forward passes: continuous ADP policy.


# ============================================================
# Basic helpers
# ============================================================
def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def get_fixed_params():
    fixed = SystemCharacteristics.get_fixed_data()

    def get_first(*keys, default=None):
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


def feature_vector(state, num_timeslots):
    """
    Keep the colleague's simple feature representation.
    This is intentionally less expressive but more robust than the earlier
    high-dimensional value function.
    """
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


def approximate_value_with_theta_dict(state, theta_by_time, num_timeslots):
    t = int(round(safe_float(state.get("current_time", 0), 0)))
    if t >= num_timeslots:
        return 0.0
    t = max(0, min(t, num_timeslots - 1))

    theta = theta_by_time.get(t)
    if theta is None:
        return 0.0

    return float(np.dot(theta, feature_vector(state, num_timeslots)))


def approximate_value(state, num_timeslots):
    return approximate_value_with_theta_dict(state, THETA_BY_TIME, num_timeslots)


# ============================================================
# Price and occupancy processes
# ============================================================
def sample_next_price(current_price, previous_price):
    """Sample next price using the provided assignment price process."""
    return float(PriceProcessRestaurant.price_model(float(current_price), float(previous_price)))


def expected_next_price(current_price, previous_price, samples=7):
    """Monte Carlo estimate of expected next price using the provided process."""
    vals = [sample_next_price(current_price, previous_price) for _ in range(samples)]
    return float(np.mean(vals))


def sample_next_occupancy(occ1, occ2):
    """Sample next occupancies using the provided assignment occupancy process."""
    next_occ1, next_occ2 = OccupancyProcessRestaurant.next_occupancy_levels(float(occ1), float(occ2))
    return float(next_occ1), float(next_occ2)


def expected_next_occupancy(occ1, occ2, samples=7):
    """Monte Carlo estimate of expected next occupancies using the provided process."""
    vals = [sample_next_occupancy(occ1, occ2) for _ in range(samples)]
    return float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals]))


# ============================================================
# Dynamics and overrules
# ============================================================
def apply_overrules(state, heat1, heat2, ventilation, params):
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

    return float(np.clip(heat1, 0.0, Pmax)), float(np.clip(heat2, 0.0, Pmax)), int(np.clip(ventilation, 0, 1))


def simulate_next_state(state, heat1, heat2, ventilation, params, mode="expected"):
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

    if mode == "sample":
        price_next = sample_next_price(price_t, price_prev)
        Occ1_next, Occ2_next = sample_next_occupancy(Occ1, Occ2)
    else:
        price_next = expected_next_price(price_t, price_prev)
        Occ1_next, Occ2_next = expected_next_occupancy(Occ1, Occ2)

    low1 = int(safe_float(state.get("low_override_r1", 0), 0) > 0.5)
    low2 = int(safe_float(state.get("low_override_r2", 0), 0) > 0.5)

    low1_next = int(T1_next <= params["Tlow"] or (low1 == 1 and T1_next < params["TOK"]))
    low2_next = int(T2_next <= params["Tlow"] or (low2 == 1 and T2_next < params["TOK"]))

    # Keep colleague's convention because it performed reasonably well.
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
# ADP decision rule: robust discrete Bellman action enumeration
# ============================================================
def get_candidate_actions(Pmax):
    heat_candidates = [0.0, 0.25 * Pmax, 0.50 * Pmax, 0.75 * Pmax, Pmax]
    vent_candidates = [0, 1]
    return [(h1, h2, v) for h1 in heat_candidates for h2 in heat_candidates for v in vent_candidates]


def evaluate_action_with_theta(state, action, params, theta_by_time, next_mode="expected"):
    h1_raw, h2_raw, v_raw = action
    h1, h2, v = apply_overrules(state, h1_raw, h2_raw, v_raw, params)

    immediate_cost = safe_float(state.get("price_t", 4.0), 4.0) * (h1 + h2 + params["Pvent"] * v)
    next_state = simulate_next_state(state, h1, h2, v, params, mode=next_mode)
    future_cost = approximate_value_with_theta_dict(next_state, theta_by_time, params["num_timeslots"])

    # Prevent artificial negative value from dominating action choice.
    future_cost = max(TARGET_CLIP_MIN, future_cost)
    return immediate_cost + future_cost, h1, h2, v


def evaluate_action(state, action, params):
    return evaluate_action_with_theta(state, action, params, THETA_BY_TIME, next_mode="expected")



# ============================================================
# Alternative ADP decision rule: continuous Pyomo Bellman optimization
# ============================================================
def _expected_exogenous_for_next_state(state):
    """Expected next exogenous state used inside the continuous one-step Bellman problem."""
    price_t = safe_float(state.get("price_t", 4.0), 4.0)
    price_prev = safe_float(state.get("price_previous", 4.0), 4.0)
    occ1 = safe_float(state.get("Occ1", 30.0), 30.0)
    occ2 = safe_float(state.get("Occ2", 20.0), 20.0)
    return (
        expected_next_price(price_t, price_prev),
        *expected_next_occupancy(occ1, occ2),
    )


def select_action_with_theta_continuous(state, theta_by_time, solver_time_limit=3, mip_gap=0.01):
    """
    Continuous-action ADP policy.

    It solves the same one-step ADP Bellman decision as the grid-search policy,
    but heat inputs are continuous variables in [0, Pmax]. Ventilation remains binary.

    This is useful as a diagnostic/comparison model. It is more faithful to the
    continuous action space, but also more likely to exploit inaccurate value-function
    slopes than the robust discrete action enumeration.
    """
    params = get_fixed_params()
    Pmax = params["Pmax"]
    Pvent = params["Pvent"]
    Tlow = params["Tlow"]
    TOK = params["TOK"]
    THigh = params["THigh"]
    HHigh = params["HHigh"]
    H_day = params["num_timeslots"]
    BIG_M = 100.0

    t = int(round(safe_float(state.get("current_time", 0), 0)))
    theta_next = theta_by_time.get(t + 1)
    if theta_next is None or t + 1 >= H_day:
        theta_next = np.zeros(len(feature_vector(state, H_day)))
    theta_next = np.array(theta_next, dtype=float)

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

    price_next, occ1_next, occ2_next = _expected_exogenous_for_next_state(state)
    remaining_next = max(0, H_day - (t + 1))
    Tout = params["Tout"][t % H_day]

    try:
        m = pyo.ConcreteModel()
        m.R = pyo.Set(initialize=[1, 2])

        # Commanded and effective actions
        m.pc = pyo.Var(m.R, bounds=(0.0, Pmax))
        m.vb = pyo.Var(domain=pyo.Binary)
        m.pf = pyo.Var(m.R, bounds=(0.0, Pmax))
        m.ve = pyo.Var(domain=pyo.Binary)

        # Next states
        m.T1_next = pyo.Var()
        m.T2_next = pyo.Var()
        m.H_next = pyo.Var()

        # Binary next-state memory indicators
        m.low1_next = pyo.Var(domain=pyo.Binary)
        m.low2_next = pyo.Var(domain=pyo.Binary)
        m.z1_below_low = pyo.Var(domain=pyo.Binary)
        m.z2_below_low = pyo.Var(domain=pyo.Binary)
        m.y1_ok_next = pyo.Var(domain=pyo.Binary)
        m.y2_ok_next = pyo.Var(domain=pyo.Binary)

        # Nonnegative wrapper for approximate value to avoid artificial rewards
        m.V_next = pyo.Var(bounds=(0.0, None))

        m.cons = pyo.ConstraintList()

        # Effective heating overrules at current state
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

        # Effective ventilation overrules at current state
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

        # Colleague-style ventilation counter convention:
        # if ventilation is ON, next counter increments; otherwise zero.
        vent_counter_next_expr = (vent_counter + 1) * m.ve

        # Feature order:
        # [1, T1, T2, H, price_t, Occ1, Occ2, vent_counter, low1, low2, remaining_time]
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

        # Return effective decisions, consistent with the colleague base script.
        h1 = pyo.value(m.pf[1])
        h2 = pyo.value(m.pf[2])
        v = pyo.value(m.ve)
        if h1 is None or h2 is None or v is None:
            raise RuntimeError("Continuous ADP solve returned uninitialized action variables.")

        return {
            "HeatPowerRoom1": float(max(0.0, min(Pmax, h1))),
            "HeatPowerRoom2": float(max(0.0, min(Pmax, h2))),
            "VentilationON": int(float(v) > 0.5),
        }

    except Exception:
        # Safe fallback if the continuous solve fails. The evaluator applies overrules.
        return DummyPolicy.select_action(state)

def select_action_with_theta(state, theta_by_time):
    params = get_fixed_params()
    best_score = float("inf")
    best_h1, best_h2, best_v = 0.0, 0.0, 0

    for action in get_candidate_actions(params["Pmax"]):
        score, h1, h2, v = evaluate_action_with_theta(state, action, params, theta_by_time, next_mode="expected")
        if score < best_score:
            best_score = score
            best_h1, best_h2, best_v = h1, h2, v

    return {
        "HeatPowerRoom1": float(max(0.0, min(params["Pmax"], best_h1))),
        "HeatPowerRoom2": float(max(0.0, min(params["Pmax"], best_h2))),
        "VentilationON": int(best_v),
    }


def select_action(state):
    return select_action_with_theta_continuous(state, THETA_BY_TIME)


# ============================================================
# Offline training
# ============================================================
def sample_training_state(t):
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


def value_with_theta(state, theta, num_timeslots):
    return float(np.dot(theta, feature_vector(state, num_timeslots)))


def bellman_target(state, theta_next, params, K_next):
    best_value = float("inf")

    # Build a small theta dictionary containing only the next stage. This lets us reuse
    # the same value helper while computing backward targets.
    t_next = int(round(safe_float(state.get("current_time", 0), 0))) + 1
    theta_next_dict = {t_next: theta_next}

    for action in get_candidate_actions(params["Pmax"]):
        h1_raw, h2_raw, v_raw = action
        h1, h2, v = apply_overrules(state, h1_raw, h2_raw, v_raw, params)

        immediate_cost = safe_float(state.get("price_t", 4.0), 4.0) * (h1 + h2 + params["Pvent"] * v)
        future_values = []

        for _ in range(K_next):
            next_state = simulate_next_state(state, h1, h2, v, params, mode="sample")
            fv = approximate_value_with_theta_dict(next_state, theta_next_dict, params["num_timeslots"])
            future_values.append(max(TARGET_CLIP_MIN, fv))

        target = immediate_cost + float(np.mean(future_values))
        target = max(TARGET_CLIP_MIN, target)
        if target < best_value:
            best_value = target

    return best_value


def fit_theta_ridge(states, targets, num_timeslots, ridge=RIDGE):
    X = np.array([feature_vector(s, num_timeslots) for s in states], dtype=float)
    y = np.array(targets, dtype=float)

    if len(states) == 0:
        return np.zeros(X.shape[1] if X.ndim == 2 else 11)

    # Standardize non-intercept columns to improve ridge conditioning.
    Xs = X.copy()
    means = Xs[:, 1:].mean(axis=0)
    stds = Xs[:, 1:].std(axis=0)
    stds[stds < 1e-8] = 1.0
    Xs[:, 1:] = (Xs[:, 1:] - means) / stds

    penalty = ridge * np.eye(Xs.shape[1])
    penalty[0, 0] = 0.0  # do not regularize intercept

    beta_scaled = np.linalg.solve(Xs.T @ Xs + penalty, Xs.T @ y)

    # Convert coefficients back to original feature scale.
    beta = np.zeros_like(beta_scaled)
    beta[1:] = beta_scaled[1:] / stds
    beta[0] = beta_scaled[0] - np.sum(beta_scaled[1:] * means / stds)
    return beta


def train_theta_backward_induction(N_states=300, K_next=10, seed=123, ridge=RIDGE):
    """Basic sampling-based backward induction, but with ridge and target clipping."""
    global THETA_BY_TIME
    np.random.seed(seed)

    params = get_fixed_params()
    H_day = params["num_timeslots"]
    n_features = len(feature_vector(sample_training_state(0), H_day))

    theta_by_time = {}
    theta_next = np.zeros(n_features)

    for t in reversed(range(H_day)):
        states_t = [sample_training_state(t) for _ in range(N_states)]
        targets_t = [bellman_target(s, theta_next, params, K_next) for s in states_t]
        theta_t = fit_theta_ridge(states_t, targets_t, H_day, ridge=ridge)

        theta_by_time[t] = theta_t
        theta_next = theta_t

        print(
            f"BI theta[{t}] | mean={np.mean(targets_t):.2f}, "
            f"std={np.std(targets_t):.2f}, min={np.min(targets_t):.2f}, max={np.max(targets_t):.2f}"
        )

    THETA_BY_TIME = theta_by_time
    return theta_by_time


# ============================================================
# Forward-backward mechanism
# ============================================================
def load_empirical_data():
    """Load data from common relative paths. Falls back to None if not found."""
    candidate_dirs = [Path("Assignment_B/Data"), Path("Data"), Path("../../../../../Downloads/")]
    for base in candidate_dirs:
        price_path = base / "v2_PriceData.csv"
        occ1_path = base / "OccupancyRoom1.csv"
        occ2_path = base / "OccupancyRoom2.csv"
        if price_path.exists() and occ1_path.exists() and occ2_path.exists():
            return {
                "price": pd.read_csv(price_path).to_numpy(),
                "occ1": pd.read_csv(occ1_path).to_numpy(),
                "occ2": pd.read_csv(occ2_path).to_numpy(),
            }
    return None


def dummy_policy_action(state):
    """Use the actual imported dummy policy for the first forward pass."""
    return DummyPolicy.select_action(state)


def choose_forward_action(state, theta_by_time, use_dummy_policy):
    """First iteration: imported dummy policy. Later iterations: continuous ADP."""
    if use_dummy_policy:
        return dummy_policy_action(state)
    return select_action_with_theta_continuous(state, theta_by_time)


def simulate_policy_transition(state, action, params, mode="sample"):
    """Apply environment/controller overrules and simulate one transition.

    The submitted policies return commanded actions. For internal forward-pass
    training we mimic the evaluator by applying the same overrule logic before
    updating the physical state.
    """
    h1_raw = float(action["HeatPowerRoom1"])
    h2_raw = float(action["HeatPowerRoom2"])
    v_raw = int(action["VentilationON"])
    h1, h2, v = apply_overrules(state, h1_raw, h2_raw, v_raw, params)
    return simulate_next_state(state, h1, h2, v, params, mode=mode)


def generate_forward_pass_states(theta_by_time, n_trajectories=200, seed=1234, mode="sample", use_dummy_policy=False):
    """Forward pass: simulate trajectories and collect visited states."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    params = get_fixed_params()
    H_day = params["num_timeslots"]
    states_by_t = {t: [] for t in range(H_day)}
    data = load_empirical_data()

    if data is None:
        # If empirical data cannot be loaded, simulate from random initial exogenous states.
        for _ in range(n_trajectories):
            state = {
                "T1": params["T_init"],
                "T2": params["T_init"],
                "H": params["H_init"],
                "Occ1": float(rng.uniform(20, 50)),
                "Occ2": float(rng.uniform(10, 30)),
                "price_t": float(rng.uniform(0, 12)),
                "price_previous": float(rng.uniform(0, 12)),
                "vent_counter": 0,
                "low_override_r1": 0,
                "low_override_r2": 0,
                "current_time": 0,
            }
            for t in range(H_day):
                state["current_time"] = t
                states_by_t[t].append(dict(state))
                action = choose_forward_action(state, theta_by_time, use_dummy_policy=use_dummy_policy)
                state = simulate_policy_transition(state, action, params, mode=mode)
        return states_by_t

    price = data["price"]
    occ1 = data["occ1"]
    occ2 = data["occ2"]
    num_days = min(price.shape[0], occ1.shape[0], occ2.shape[0])
    H_eval = min(H_day, price.shape[1], occ1.shape[1], occ2.shape[1])

    for _ in range(n_trajectories):
        d = int(rng.integers(0, num_days))
        state = {
            "T1": params["T_init"],
            "T2": params["T_init"],
            "H": params["H_init"],
            "Occ1": float(occ1[d, 0]),
            "Occ2": float(occ2[d, 0]),
            "price_t": float(price[d, 0]),
            "price_previous": float(price[d, 0]),
            "vent_counter": 0,
            "low_override_r1": 0,
            "low_override_r2": 0,
            "current_time": 0,
        }

        for t in range(H_eval):
            state["current_time"] = t
            state["price_t"] = float(price[d, t])
            state["Occ1"] = float(occ1[d, t])
            state["Occ2"] = float(occ2[d, t])
            if t > 0:
                state["price_previous"] = float(price[d, t - 1])

            states_by_t[t].append(dict(state))
            action = choose_forward_action(state, theta_by_time, use_dummy_policy=use_dummy_policy)
            state = simulate_policy_transition(state, action, params, mode=mode)

    return states_by_t


def choose_states_for_backward(t, forward_states_by_t, N_states):
    """Sample training states from the states visited in the forward pass."""
    candidates = forward_states_by_t.get(t, [])

    if len(candidates) == 0:
        # Should rarely happen, but keeps training robust if a data file has fewer columns.
        return [sample_training_state(t) for _ in range(N_states)]

    idx = np.random.choice(len(candidates), size=N_states, replace=True)
    return [dict(candidates[i]) for i in idx]


def train_theta_forward_backward(
    n_iterations=5,
    N_states=300,
    K_next=10,
    n_forward_trajectories=300,
    seed=321,
    ridge=RIDGE,
    alpha=ETA_UPDATE_ALPHA,
):
    """
    Straightforward forward-backward ADP training.

    Theta is initialized at zero. The first forward pass uses the dummy policy
    from DummyPolicy.py. All later forward passes use the current continuous ADP
    policy. No dummy share, no mixed schedule, no alternative forward policy.
    """
    global THETA_BY_TIME
    np.random.seed(seed)

    params = get_fixed_params()
    H_day = params["num_timeslots"]
    n_features = len(feature_vector(sample_training_state(0), H_day))

    theta = {t: np.zeros(n_features) for t in range(H_day)}

    for it in range(1, n_iterations + 1):
        use_dummy_policy = (it == 1)

        print(f"\nForward-backward iteration {it}/{n_iterations} | forward policy: {'DummyPolicy' if use_dummy_policy else 'continuous ADP'}")

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
            targets_t = [bellman_target(s, theta_next, params, K_next) for s in states_t]
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



def format_theta_block(theta_by_time):
    """Compact THETA_BY_TIME block: one row per time step."""
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


def update_theta_in_file(theta_by_time, file_path=None):
    """
    Safely replace only the top THETA_BY_TIME block.

    Important: do not use split(marker)[1], because the marker text may also
    appear later in helper functions and can truncate the file.
    """
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

    new_block = format_theta_block(theta_by_time)
    updated = text[:start_idx] + new_block + text[end_idx:]

    file_path.write_text(updated, encoding="utf-8")
    print(f"Updated THETA_BY_TIME in {file_path}")


# ============================================================
# Command line interface
# ============================================================
if __name__ == "__main__":
    if "--train-and-update" in sys.argv:
        print("Forward-backward ADP training")
        print("Iteration 1 uses DummyPolicy.py behaviour: zero heating and zero ventilation commands.")
        print("Iterations 2...K use the current continuous ADP policy.\n")

        theta = train_theta_forward_backward(
            n_iterations=5,
            N_states=300,
            K_next=10,
            n_forward_trajectories=300,
            seed=321,
            ridge=RIDGE,
            alpha=ETA_UPDATE_ALPHA,
        )
        update_theta_in_file(theta)

    else:
        # Keep the submitted file quiet by default. The evaluator imports and calls select_action(state).
        print("Use --train-and-update to train and update THETA_BY_TIME.")
