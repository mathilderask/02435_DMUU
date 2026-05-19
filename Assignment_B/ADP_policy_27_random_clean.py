"""
Task 4: Approximate Dynamic Programming policy with random state sampling.

This file contains both:

1. train_adp()
   Offline training by sampling-based approximate backward induction with randomly
   sampled training states.

2. select_action(state)
   Online ADP policy called by the evaluator. It uses the trained ADP_THETA
   dictionary and returns the here-and-now action.

Method:
    V_hat_t(s_t) = eta_t^T phi(s_t)

Training:
    - terminal value V_hat_T = 0
    - work backwards in time
    - sample states randomly
    - compute Bellman targets using a one-step optimization problem
    - fit eta_t by ordinary linear regression

Usage:
    Train and automatically update ADP_THETA:
        python ADP_policy_27_random_clean.py --train-and-update

    Print trained coefficients without updating the file:
        python ADP_policy_27_random_clean.py --train-print

    Smoke-test select_action():
        python ADP_policy_27_random_clean.py --smoke-test

Important:
    The evaluator only calls select_action(state). It does not call train_adp().
"""

import sys
import re
from pathlib import Path

import numpy as np
import pyomo.environ as pyo

import Data.v2_SystemCharacteristics as SystemCharacteristics
import Data.PriceProcessRestaurant as PriceProcessRestaurant
import Data.OccupancyProcessRestaurant as OccupancyProcessRestaurant


# =========================================================
# Trained value-function coefficients
# =========================================================
# Replace this dictionary by running:
#     python ADP_policy_27_random_clean.py --train-and-update
#
# Each vector has 16 entries and corresponds to FEATURE_NAMES.
# --- ADP_THETA_START ---
ADP_THETA = {
    0: [404.2823992337, 24.9925524948, -8.8441113839, -0.3645506439, -0.3514060962, 1.0959177458, 1.8522562422, -9.4106463980, -9.4814606595, -2.8010397450, 0.3386480207, -0.0000000000, 0.0000000000, 14.0934618557, 19.4207267626, 9.8581956011],
    1: [465.7501224144, 26.0642181648, -11.0483680525, -1.0672629763, -0.7899421178, 1.6484823533, -2.3269874058, -10.6209455534, -11.3429821071, -1.1615801706, -5.9919495003, -0.0000000000, 0.0000000000, 12.1035436884, 18.6323001619, 13.7008781884],
    2: [522.1619370597, 27.4111687142, -10.6020948219, -0.9837753723, -0.7407612523, 0.7532227732, 1.8284877992, -10.2981166947, -13.4969600768, -5.3556252483, -6.4537902781, -0.0000000000, 0.0000000000, 5.2862845684, 26.9030301979, 17.1284115837],
    3: [359.5103512130, 25.7625416688, -9.2416813758, -0.1569668757, -0.5001240051, 1.1415119371, 0.1157338286, -10.3825745159, -9.3177454683, -5.4002809233, -2.4130381231, 0.0000000000, -0.0000000000, 4.7033897320, 28.2742550333, 19.1194986836],
    4: [282.7605165986, 23.9983517697, -7.9554228475, 0.0505955124, -0.1179854934, 1.3935976514, 0.6067737349, -8.9148650492, -8.9471674385, -1.7201843279, -1.8431556903, -0.0000000000, -0.0000000000, -2.3048657724, 11.3730177964, 20.8950518718],
    5: [219.9981607722, 18.8864496173, -5.9259196375, 0.2435391662, -0.2233604837, 1.4209225158, -1.1384535009, -6.4521944303, -9.0247680627, 1.0492091931, -1.5206282165, 0.0000000000, 0.0000000000, 9.2190421106, 14.9932445049, 17.1207644317],
    6: [252.9722811270, 16.1232329850, -5.7090482796, 0.5181742978, -0.2774306943, 0.7158984802, 1.5207102133, -10.9031608606, -4.4853822858, -5.0810875904, 1.0694207087, 0.0000000000, 0.0000000000, 7.8799761748, 12.4451965811, 28.6904058765],
    7: [209.8424468487, 11.9755814418, -2.3415343152, 0.2279595358, -0.1346264770, 0.2098921826, 1.9861297661, -7.8945201454, -4.1602239024, -4.0925718239, -0.5199758955, 0.0000000000, 0.0000000000, 9.8503675214, 20.5488640831, 29.7695578866],
    8: [90.5537845606, 7.8966671770, -1.2488346980, 0.0320336233, 0.0682313034, -0.0081794540, 1.8682816406, -2.1261279013, -3.5702241018, -0.1423681250, -1.8691088057, -0.0000000000, -0.0000000000, 6.8952668135, 21.2582772501, 29.9046642095],
    9: [-58.8983269435, 3.1148309616, 0.3235277667, -0.0233615252, 0.0403285662, 0.0417934916, 0.8153487871, 1.9759814974, -0.3705212405, 3.3486645568, -1.0486863115, -0.0000000000, 0.0000000000, 4.4297402884, 15.3890807396, 21.1578056491],
    10: [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
}
# --- ADP_THETA_END ---


FEATURE_NAMES = [
    "constant",
    "price_t",
    "price_previous",
    "occ1_t",
    "occ2_t",
    "H_t",
    "humidity_excess",
    "T1_t",
    "T2_t",
    "T1_deficit_to_TOK",
    "T2_deficit_to_TOK",
    "T1_excess_above_Thigh",
    "T2_excess_above_Thigh",
    "vent_counter",
    "low_override_r1",
    "low_override_r2",
]

N_FEATURES = len(FEATURE_NAMES)


# =========================================================
# Problem data
# =========================================================
def get_problem_data():
    """Load fixed system parameters from the provided data file."""
    params = SystemCharacteristics.get_fixed_data()

    return {
        "Pmax": float(params["heating_max_power"]),
        "Pvent": float(params["ventilation_power"]),
        "Tlow": float(params["temp_min_comfort_threshold"]),
        "TOK": float(params["temp_OK_threshold"]),
        "THigh": float(params["temp_max_comfort_threshold"]),
        "HHigh": float(params["humidity_threshold"]),
        "z_exch": float(params["heat_exchange_coeff"]),
        "z_loss": float(params["thermal_loss_coeff"]),
        "z_conv": float(params["heating_efficiency_coeff"]),
        "z_cool": float(params["heat_vent_coeff"]),
        "z_occ": float(params["heat_occupancy_coeff"]),
        "eta_occ": float(params["humidity_occupancy_coeff"]),
        "eta_vent": float(params["humidity_vent_coeff"]),
        "Tout": list(params["outdoor_temperature"]),
        "T_HORIZON": int(params["num_timeslots"]),
    }


DATA = get_problem_data()


def safe_float(x, default=0.0):
    """Safely convert a value to float."""
    try:
        return float(x)
    except Exception:
        return float(default)


# =========================================================
# Value-function features
# =========================================================
def feature_vector(state):
    """
    Feature vector phi(s_t).

    The value function is linear in the fitted coefficients:

        V_hat_t(s_t) = eta_t^T phi(s_t)

    Positive-part basis functions are included as features, but the value
    function remains linear in eta.
    """
    T1 = float(state["T1"])
    T2 = float(state["T2"])
    H = float(state["H"])
    price = float(state["price_t"])
    price_prev = float(state["price_previous"])
    occ1 = float(state["Occ1"])
    occ2 = float(state["Occ2"])
    vent_counter = float(state["vent_counter"])
    low1 = float(state["low_override_r1"])
    low2 = float(state["low_override_r2"])

    return np.array([
        1.0,
        price,
        price_prev,
        occ1,
        occ2,
        H,
        max(0.0, H - DATA["HHigh"]),
        T1,
        T2,
        max(0.0, DATA["TOK"] - T1),
        max(0.0, DATA["TOK"] - T2),
        max(0.0, T1 - DATA["THigh"]),
        max(0.0, T2 - DATA["THigh"]),
        vent_counter,
        low1,
        low2,
    ], dtype=float)


def sample_next_exogenous(state):
    """Sample next price and occupancies from the provided stochastic processes."""
    next_price = PriceProcessRestaurant.price_model(
        float(state["price_t"]),
        float(state["price_previous"]),
    )
    next_occ1, next_occ2 = OccupancyProcessRestaurant.next_occupancy_levels(
        float(state["Occ1"]),
        float(state["Occ2"]),
    )
    return float(next_price), float(next_occ1), float(next_occ2)


# =========================================================
# Shared one-step ADP optimization
# =========================================================
def solve_one_step_adp_problem(state, theta_next, n_next_samples, solver_time_limit):
    """
    Solve the one-step ADP optimization problem:

        min_a c(s_t, a_t) + E[ V_hat_{t+1}(s_{t+1}) ]

    The same model is used:
    - during training, to compute Bellman targets;
    - online, to compute the here-and-now decision.
    """
    Pmax = DATA["Pmax"]
    Pvent = DATA["Pvent"]
    Tlow = DATA["Tlow"]
    TOK = DATA["TOK"]
    THigh = DATA["THigh"]
    HHigh = DATA["HHigh"]
    z_exch = DATA["z_exch"]
    z_loss = DATA["z_loss"]
    z_conv = DATA["z_conv"]
    z_cool = DATA["z_cool"]
    z_occ = DATA["z_occ"]
    eta_occ = DATA["eta_occ"]
    eta_vent = DATA["eta_vent"]
    Tout = DATA["Tout"]
    T_HORIZON = DATA["T_HORIZON"]

    BIG_M = 100.0

    T1 = float(state["T1"])
    T2 = float(state["T2"])
    H = float(state["H"])
    occ1 = float(state["Occ1"])
    occ2 = float(state["Occ2"])
    price = float(state["price_t"])
    price_prev = float(state["price_previous"])
    vent_counter = int(state["vent_counter"])
    low1 = int(state["low_override_r1"])
    low2 = int(state["low_override_r2"])
    current_time = int(state["current_time"])

    if T1 < Tlow:
        low1 = 1
    if T2 < Tlow:
        low2 = 1

    # One-step exogenous samples.
    samples = []
    for _ in range(n_next_samples):
        next_price, next_occ1, next_occ2 = sample_next_exogenous({
            "price_t": price,
            "price_previous": price_prev,
            "Occ1": occ1,
            "Occ2": occ2,
        })
        samples.append((next_price, next_occ1, next_occ2))

    prob = 1.0 / n_next_samples

    m = pyo.ConcreteModel()

    m.K = pyo.RangeSet(0, n_next_samples - 1)
    m.R = pyo.Set(initialize=[1, 2])

    # Commanded actions returned to the environment.
    m.pc = pyo.Var(m.R, bounds=(0.0, Pmax))
    m.vb = pyo.Var(domain=pyo.Binary)

    # Effective actions after controller logic.
    m.pf = pyo.Var(m.R, bounds=(0.0, Pmax))
    m.ve = pyo.Var(domain=pyo.Binary)

    # Next physical states.
    m.T1_next = pyo.Var(m.K)
    m.T2_next = pyo.Var(m.K)
    m.H_next = pyo.Var(m.K)

    # Positive-part features for next states.
    # These are exact max(0, x) terms. The equality is modeled with binaries so
    # the Bellman problem remains bounded even if a fitted coefficient is negative.
    m.h_excess = pyo.Var(m.K, bounds=(0.0, BIG_M))
    m.t1_deficit = pyo.Var(m.K, bounds=(0.0, BIG_M))
    m.t2_deficit = pyo.Var(m.K, bounds=(0.0, BIG_M))
    m.t1_high_excess = pyo.Var(m.K, bounds=(0.0, BIG_M))
    m.t2_high_excess = pyo.Var(m.K, bounds=(0.0, BIG_M))

    m.h_excess_pos = pyo.Var(m.K, domain=pyo.Binary)
    m.t1_deficit_pos = pyo.Var(m.K, domain=pyo.Binary)
    m.t2_deficit_pos = pyo.Var(m.K, domain=pyo.Binary)
    m.t1_high_excess_pos = pyo.Var(m.K, domain=pyo.Binary)
    m.t2_high_excess_pos = pyo.Var(m.K, domain=pyo.Binary)

    # Next low-temperature latch features.
    m.low1_next = pyo.Var(m.K, domain=pyo.Binary)
    m.low2_next = pyo.Var(m.K, domain=pyo.Binary)
    m.z1_below_low = pyo.Var(m.K, domain=pyo.Binary)
    m.z2_below_low = pyo.Var(m.K, domain=pyo.Binary)
    m.y1_ok_next = pyo.Var(m.K, domain=pyo.Binary)
    m.y2_ok_next = pyo.Var(m.K, domain=pyo.Binary)

    m.cons = pyo.ConstraintList()

    # Effective heating logic at current state.
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

    # Effective ventilation logic at current state.
    if H > HHigh or vent_counter > 0:
        m.cons.add(m.ve == 1)
        m.cons.add(m.vb == 1)
    else:
        m.cons.add(m.ve >= m.vb)

    # Approximate next ventilation counter feature.
    if vent_counter == 0:
        vent_counter_next_expr = m.ve
    elif vent_counter == 1:
        vent_counter_next_expr = 2.0 * m.ve
    else:
        vent_counter_next_expr = 0.0

    tout = Tout[current_time % T_HORIZON]

    for k, (price_next, occ1_next, occ2_next) in enumerate(samples):
        m.cons.add(
            m.T1_next[k] ==
            T1
            + z_exch * (T2 - T1)
            + z_loss * (tout - T1)
            + z_conv * m.pf[1]
            - z_cool * m.ve
            + z_occ * occ1
        )

        m.cons.add(
            m.T2_next[k] ==
            T2
            + z_exch * (T1 - T2)
            + z_loss * (tout - T2)
            + z_conv * m.pf[2]
            - z_cool * m.ve
            + z_occ * occ2
        )

        m.cons.add(
            m.H_next[k] ==
            H + eta_occ * (occ1 + occ2) - eta_vent * m.ve
        )

        # y = max(0, x), modeled exactly using Big-M and a binary selector.
        h_expr = m.H_next[k] - HHigh
        m.cons.add(m.h_excess[k] >= h_expr)
        m.cons.add(m.h_excess[k] <= h_expr + BIG_M * (1 - m.h_excess_pos[k]))
        m.cons.add(m.h_excess[k] <= BIG_M * m.h_excess_pos[k])

        t1_def_expr = TOK - m.T1_next[k]
        m.cons.add(m.t1_deficit[k] >= t1_def_expr)
        m.cons.add(m.t1_deficit[k] <= t1_def_expr + BIG_M * (1 - m.t1_deficit_pos[k]))
        m.cons.add(m.t1_deficit[k] <= BIG_M * m.t1_deficit_pos[k])

        t2_def_expr = TOK - m.T2_next[k]
        m.cons.add(m.t2_deficit[k] >= t2_def_expr)
        m.cons.add(m.t2_deficit[k] <= t2_def_expr + BIG_M * (1 - m.t2_deficit_pos[k]))
        m.cons.add(m.t2_deficit[k] <= BIG_M * m.t2_deficit_pos[k])

        t1_high_expr = m.T1_next[k] - THigh
        m.cons.add(m.t1_high_excess[k] >= t1_high_expr)
        m.cons.add(m.t1_high_excess[k] <= t1_high_expr + BIG_M * (1 - m.t1_high_excess_pos[k]))
        m.cons.add(m.t1_high_excess[k] <= BIG_M * m.t1_high_excess_pos[k])

        t2_high_expr = m.T2_next[k] - THigh
        m.cons.add(m.t2_high_excess[k] >= t2_high_expr)
        m.cons.add(m.t2_high_excess[k] <= t2_high_expr + BIG_M * (1 - m.t2_high_excess_pos[k]))
        m.cons.add(m.t2_high_excess[k] <= BIG_M * m.t2_high_excess_pos[k])

        # Next low-temperature latch state.
        m.cons.add(m.T1_next[k] <= Tlow + BIG_M * (1 - m.z1_below_low[k]))
        m.cons.add(m.T1_next[k] >= Tlow - BIG_M * m.z1_below_low[k])
        m.cons.add(m.T2_next[k] <= Tlow + BIG_M * (1 - m.z2_below_low[k]))
        m.cons.add(m.T2_next[k] >= Tlow - BIG_M * m.z2_below_low[k])

        m.cons.add(m.T1_next[k] >= TOK - BIG_M * (1 - m.y1_ok_next[k]))
        m.cons.add(m.T1_next[k] <= TOK + BIG_M * m.y1_ok_next[k])
        m.cons.add(m.T2_next[k] >= TOK - BIG_M * (1 - m.y2_ok_next[k]))
        m.cons.add(m.T2_next[k] <= TOK + BIG_M * m.y2_ok_next[k])

        m.cons.add(m.low1_next[k] >= m.z1_below_low[k])
        m.cons.add(m.low1_next[k] >= low1 - m.y1_ok_next[k])
        m.cons.add(m.low1_next[k] <= m.z1_below_low[k] + low1)
        m.cons.add(m.low1_next[k] <= m.z1_below_low[k] + (1 - m.y1_ok_next[k]))

        m.cons.add(m.low2_next[k] >= m.z2_below_low[k])
        m.cons.add(m.low2_next[k] >= low2 - m.y2_ok_next[k])
        m.cons.add(m.low2_next[k] <= m.z2_below_low[k] + low2)
        m.cons.add(m.low2_next[k] <= m.z2_below_low[k] + (1 - m.y2_ok_next[k]))

    immediate_cost = price * (m.pf[1] + m.pf[2] + Pvent * m.ve)

    future_value = 0.0
    for k, (price_next, occ1_next, occ2_next) in enumerate(samples):
        future_value += prob * (
            theta_next[0]
            + theta_next[1] * price_next
            + theta_next[2] * price
            + theta_next[3] * occ1_next
            + theta_next[4] * occ2_next
            + theta_next[5] * m.H_next[k]
            + theta_next[6] * m.h_excess[k]
            + theta_next[7] * m.T1_next[k]
            + theta_next[8] * m.T2_next[k]
            + theta_next[9] * m.t1_deficit[k]
            + theta_next[10] * m.t2_deficit[k]
            + theta_next[11] * m.t1_high_excess[k]
            + theta_next[12] * m.t2_high_excess[k]
            + theta_next[13] * vent_counter_next_expr
            + theta_next[14] * m.low1_next[k]
            + theta_next[15] * m.low2_next[k]
        )

    m.obj = pyo.Objective(expr=immediate_cost + future_value, sense=pyo.minimize)

    solver = pyo.SolverFactory("gurobi")
    solver.options["OutputFlag"] = 0
    solver.options["TimeLimit"] = solver_time_limit
    results = solver.solve(m, tee=False)

    status_ok = (
        results.solver.status == pyo.SolverStatus.ok
        and results.solver.termination_condition in [
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
            pyo.TerminationCondition.maxTimeLimit,
        ]
    )
    if not status_ok:
        raise RuntimeError(
            f"Solver failed in one-step ADP problem: "
            f"{results.solver.status}, {results.solver.termination_condition}"
        )

    objective_value = pyo.value(m.obj)
    if objective_value is None or not np.isfinite(objective_value):
        raise RuntimeError("No valid ADP objective value extracted.")

    action = {
        "HeatPowerRoom1": float(max(0.0, min(Pmax, pyo.value(m.pc[1])))),
        "HeatPowerRoom2": float(max(0.0, min(Pmax, pyo.value(m.pc[2])))),
        "VentilationON": 1 if float(pyo.value(m.vb)) > 0.5 else 0,
    }

    return float(objective_value), action


# =========================================================
# Online policy called by the evaluator
# =========================================================
def select_action(state):
    """Return here-and-now actions for the current observed state."""
    Pmax = DATA["Pmax"]
    Tlow = DATA["Tlow"]
    THigh = DATA["THigh"]
    HHigh = DATA["HHigh"]

    current_state = {
        "current_time": int(round(safe_float(state.get("current_time", 0), 0))),
        "T1": safe_float(state.get("T1", 21.0), 21.0),
        "T2": safe_float(state.get("T2", 21.0), 21.0),
        "H": safe_float(state.get("H", 40.0), 40.0),
        "price_t": safe_float(state.get("price_t", 4.0), 4.0),
        "price_previous": safe_float(state.get("price_previous", 4.0), 4.0),
        "Occ1": safe_float(state.get("Occ1", 30.0), 30.0),
        "Occ2": safe_float(state.get("Occ2", 20.0), 20.0),
        "vent_counter": int(round(safe_float(state.get("vent_counter", 0), 0))),
        "low_override_r1": 1 if safe_float(state.get("low_override_r1", 0), 0) > 0.5 else 0,
        "low_override_r2": 1 if safe_float(state.get("low_override_r2", 0), 0) > 0.5 else 0,
    }

    if current_state["T1"] < Tlow:
        current_state["low_override_r1"] = 1
    if current_state["T2"] < Tlow:
        current_state["low_override_r2"] = 1

    N_NEXT_SAMPLES = 10
    SOLVER_TIME_LIMIT = 4

    theta_next = ADP_THETA.get(
        current_state["current_time"] + 1,
        [0.0] * N_FEATURES,
    )

    try:
        _, action = solve_one_step_adp_problem(
            current_state,
            theta_next,
            n_next_samples=N_NEXT_SAMPLES,
            solver_time_limit=SOLVER_TIME_LIMIT,
        )
        return action

    except Exception:
        # Safe fallback using controller overrules.
        low1 = int(current_state["low_override_r1"])
        low2 = int(current_state["low_override_r2"])
        T1 = float(current_state["T1"])
        T2 = float(current_state["T2"])
        H = float(current_state["H"])
        vent_counter = int(current_state["vent_counter"])

        p1_fb = Pmax if low1 == 1 and T1 < THigh else 0.0
        p2_fb = Pmax if low2 == 1 and T2 < THigh else 0.0
        v_fb = 1 if H > HHigh or vent_counter > 0 else 0

        return {
            "HeatPowerRoom1": float(p1_fb),
            "HeatPowerRoom2": float(p2_fb),
            "VentilationON": int(v_fb),
        }


# =========================================================
# Offline training
# =========================================================
TRAINING_RNG_SEED = 27
N_STATE_SAMPLES = 120
N_TRAIN_NEXT_SAMPLES = 8
TRAINING_SOLVER_TIME_LIMIT = 2


def sample_training_state(t, rng):
    """
    Randomly sample a physically plausible state for stage t.

    This is the basic sampling-based approximate backward induction version.
    """
    T1 = rng.uniform(16.0, 25.5)
    T2 = rng.uniform(16.0, 25.5)
    H = rng.uniform(35.0, 82.0)

    price_prev = rng.uniform(0.0, 12.0)
    price = rng.uniform(0.0, 12.0)

    occ1 = rng.uniform(20.0, 50.0)
    occ2 = rng.uniform(10.0, 30.0)

    vent_counter = int(rng.choice([0, 1, 2], p=[0.65, 0.20, 0.15]))

    low1 = int(T1 < DATA["Tlow"] or (T1 < DATA["TOK"] and rng.random() < 0.5))
    low2 = int(T2 < DATA["Tlow"] or (T2 < DATA["TOK"] and rng.random() < 0.5))

    return {
        "current_time": int(t),
        "T1": float(T1),
        "T2": float(T2),
        "H": float(H),
        "price_t": float(price),
        "price_previous": float(price_prev),
        "Occ1": float(occ1),
        "Occ2": float(occ2),
        "vent_counter": int(vent_counter),
        "low_override_r1": int(low1),
        "low_override_r2": int(low2),
    }


def fit_theta(states, targets):
    """
    Ordinary linear regression fit:

        eta_t = argmin_eta ||Phi eta - y||^2
    """
    Phi = np.vstack([feature_vector(s) for s in states])
    y = np.array(targets, dtype=float)

    eta, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    return eta


def train_adp():
    """
    Sampling-based approximate backward induction.

    Terminal condition:
        V_hat_T(s_T) = 0

    Then for t = T-1, ..., 0:
        sample random states
        compute Bellman targets using theta_{t+1}
        fit theta_t by linear regression
    """
    rng = np.random.default_rng(TRAINING_RNG_SEED)

    theta = {DATA["T_HORIZON"]: np.zeros(N_FEATURES)}
    summary = {}

    for t in reversed(range(DATA["T_HORIZON"])):
        states_t = [sample_training_state(t, rng) for _ in range(N_STATE_SAMPLES)]
        theta_next = theta[t + 1]

        targets_t = []
        failed_targets = 0

        for s in states_t:
            try:
                target, _ = solve_one_step_adp_problem(
                    s,
                    theta_next,
                    n_next_samples=N_TRAIN_NEXT_SAMPLES,
                    solver_time_limit=TRAINING_SOLVER_TIME_LIMIT,
                )
                target = float(target)

            except Exception:
                # Finite fallback for rare numerical failures.
                target = 1000.0
                failed_targets += 1

            targets_t.append(target)

        theta[t] = fit_theta(states_t, targets_t)

        summary[t] = {
            "mean_target": float(np.mean(targets_t)),
            "std_target": float(np.std(targets_t)),
            "min_target": float(np.min(targets_t)),
            "max_target": float(np.max(targets_t)),
            "failed_targets": int(failed_targets),
        }

        print(
            f"Fitted theta[{t}] | "
            f"mean={summary[t]['mean_target']:.2f}, "
            f"std={summary[t]['std_target']:.2f}, "
            f"min={summary[t]['min_target']:.2f}, "
            f"max={summary[t]['max_target']:.2f}, "
            f"failed={summary[t]['failed_targets']}"
        )

    return theta, summary


# =========================================================
# Utilities
# =========================================================
def format_theta_dict(theta):
    """Return ADP_THETA as Python code."""
    lines = ["ADP_THETA = {"]
    for t in sorted(theta.keys()):
        values = ", ".join(f"{x:.10f}" for x in theta[t])
        lines.append(f"    {t}: [{values}],")
    lines.append("}")
    return "\n".join(lines)


def print_theta_for_copy(theta):
    """Print trained coefficients in a format that can be pasted into ADP_THETA."""
    print("\nFEATURE_NAMES = [")
    for name in FEATURE_NAMES:
        print(f'    "{name}",')
    print("]\n")
    print(format_theta_dict(theta))


def update_adp_theta_in_file(theta, file_path=None):
    """Replace the ADP_THETA block in this file."""
    if file_path is None:
        file_path = Path(__file__)
    else:
        file_path = Path(file_path)

    text = file_path.read_text(encoding="utf-8")
    new_theta_code = format_theta_dict(theta)

    pattern = (
        r"# --- ADP_THETA_START ---\n"
        r"ADP_THETA = \{.*?\}\n"
        r"# --- ADP_THETA_END ---"
    )

    replacement = (
        "# --- ADP_THETA_START ---\n"
        + new_theta_code
        + "\n# --- ADP_THETA_END ---"
    )

    updated, n_replacements = re.subn(pattern, replacement, text, flags=re.DOTALL)

    if n_replacements != 1:
        raise RuntimeError(
            "Could not uniquely locate ADP_THETA block. "
            "Check that the ADP_THETA_START and ADP_THETA_END markers are present."
        )

    file_path.write_text(updated, encoding="utf-8")
    print(f"\nUpdated ADP_THETA in: {file_path}")


def debug_value_predictions(n_samples=50, seed=123):
    """Print value-function predictions on sampled states."""
    rng = np.random.default_rng(seed)

    print("\nValue-function prediction check:")
    for t in range(DATA["T_HORIZON"] + 1):
        values = []

        for _ in range(n_samples):
            if t == DATA["T_HORIZON"]:
                values.append(0.0)
                continue

            s = sample_training_state(t, rng)
            theta_t = np.array(ADP_THETA.get(t, [0.0] * N_FEATURES), dtype=float)
            value = float(theta_t @ feature_vector(s))
            values.append(value)

        print(
            f"t={t}: "
            f"mean={np.mean(values):8.2f}, "
            f"std={np.std(values):8.2f}, "
            f"min={np.min(values):8.2f}, "
            f"max={np.max(values):8.2f}"
        )


def smoke_test_select_action(n_tests=10, seed=321):
    """Run select_action on a few sampled states."""
    rng = np.random.default_rng(seed)
    Pmax = DATA["Pmax"]

    print("\nselect_action smoke test:")
    failures = 0

    for i in range(n_tests):
        t = int(rng.integers(0, DATA["T_HORIZON"]))
        s = sample_training_state(t, rng)

        try:
            action = select_action(s)
            ok = (
                0.0 <= float(action["HeatPowerRoom1"]) <= Pmax
                and 0.0 <= float(action["HeatPowerRoom2"]) <= Pmax
                and int(action["VentilationON"]) in [0, 1]
            )
            if not ok:
                failures += 1

            print(f"test={i:02d}, t={t}, action={action}, ok={ok}")

        except Exception as exc:
            failures += 1
            print(f"test={i:02d}, t={t}, FAILED: {exc}")

    print(f"Smoke test failures: {failures}/{n_tests}")


if __name__ == "__main__":
    if "--check-values" in sys.argv:
        debug_value_predictions()

    elif "--smoke-test" in sys.argv:
        smoke_test_select_action()

    elif "--train-and-update" in sys.argv:
        theta, _ = train_adp()
        update_adp_theta_in_file(theta)

    elif "--train-print" in sys.argv:
        theta, _ = train_adp()
        print_theta_for_copy(theta)

    else:
        print(
            "No action selected. Use one of:\n"
            "  --train-and-update\n"
            "  --train-print\n"
            "  --check-values\n"
            "  --smoke-test"
        )
