"""
Task 4: Approximate Dynamic Programming policy with advanced lecture-aligned training.

This file contains both:

1. train_adp()
   Offline training by sampling-based approximate backward induction.
   Running this file directly prints a trained ADP_THETA dictionary.

2. select_action(state)
   Submitted online ADP policy.
   The policy uses the ADP_THETA dictionary and returns the here-and-now action.

Workflow:
    1. Run this file locally with:
           python ADP_policy_27_advanced.py --train-and-update

       This trains the value function and automatically replaces the
       ADP_THETA dictionary in this file.

    2. Submit the updated same file as:
           ADP policy [group number].py

Important:
    The teacher's evaluator will call select_action(state). It will not call train_adp().
    Training is included in this file only to keep the implementation reproducible.
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
# Replace this dictionary with the output printed by train_adp().
# Each vector has 19 entries and corresponds to FEATURE_NAMES.
# --- ADP_THETA_START ---
ADP_THETA = {
    0: [-31.4435170938, 29.9791079691, -16.7502111663, 1.5224146828, -0.2827850548, 0.7317240084, 0.1029033137, 2.8236831872, -4.3622062069, 0.0000000000, 0.0000000000, 0.2618602871, 2.9284843864, 1.4463952975, -0.3572262260, 0.0857657092, -0.7513067085, 7.7589892023, 11.1167507588],
    1: [-44.7229494760, 25.4910283512, -13.4398713438, 0.8608952405, 0.2061909845, 1.0223687477, -3.7249757198, 2.2027545340, -3.5889710853, 0.0000000000, 0.0000000000, 0.6524051513, 1.4175262111, 1.2407903814, -0.0972400619, 0.1025742783, 0.8007472560, 8.6723466896, 3.2428445050],
    2: [-40.0902064920, 21.7408945256, -9.9114770241, -0.2227825292, 0.6795271830, 1.0009625985, -1.2210769926, -1.7442453064, -0.4011806668, 0.0000000000, 0.0000000000, 0.3728395136, 1.5172274215, 0.7324540141, 0.0188848202, 0.1351781032, 1.2884042098, 7.2625559501, 2.3913804963],
    3: [-56.5864601147, 19.3728261161, -8.2288455123, -0.7043980331, 1.5097431104, 1.0905798476, 0.6033134977, 1.5715083590, 1.2589932858, 0.0000000000, 0.0000000000, -0.2136527289, 0.7212071658, 1.2414319457, 0.0122476844, -0.0436961153, 1.1706812924, 6.0892374830, 6.7030540144],
    4: [-39.8937617563, 17.9782300630, -8.7847891197, -0.2571273079, 0.7753871628, 1.1117028632, -1.9754403539, -1.4599682886, 0.5965644809, 0.0000000000, 0.0000000000, 0.3085009583, 1.1828048136, 0.6653317895, 0.0483904177, 0.0659145350, 3.1579420664, 4.8341721130, 4.6890648937],
    5: [3.3743165816, 13.9097781498, -6.4947721532, 0.0647206166, 0.1596162097, 0.8793825600, -2.1417591308, 2.7085882211, -0.6664444621, 0.0000000000, 0.0000000000, 0.4155361792, 0.3556184360, 1.4564746042, 0.0531704009, -0.0383133746, 3.0375919140, 4.9882516862, 9.3366794504],
    6: [89.3220028964, 8.9724816408, -4.6331002983, 0.1165606296, 0.2194375645, 0.5538783964, -0.5043389011, -0.0852499415, 1.1852554996, 0.0000000000, 0.0000000000, 0.2390821845, 1.7259955207, 0.2494556584, -0.0778857376, 0.1197568185, 1.1319717209, 7.3629980604, 9.5874515117],
    7: [246.8759711885, 7.4150768466, -5.5838832183, -0.2392913906, 0.2096578862, 0.2527323363, -1.2332533696, -2.7914084977, 1.8425879951, 0.0000000000, 0.0000000000, 0.7189904104, 0.8367306530, 0.8376523367, 0.1015580386, -0.0851591528, 5.0678708386, 7.7788152574, 10.6697757311],
    8: [467.1001756103, 4.6411881775, -1.7244064854, -0.1560839892, -0.2744647620, 0.1439540443, -0.4604852237, 0.1651129824, -0.4661361969, 0.0000000000, 0.0000000000, 0.2160354150, 0.4724093663, 0.4230064237, -0.0078993976, 0.0357038211, 5.1307013282, 8.3468555381, 10.5113174249],
    9: [764.6311809444, 0.5187639952, 0.1123896488, -0.1007096221, -0.1249375087, -0.0166149827, -0.0916111470, 0.6242828903, -0.9177591667, 0.0000000000, 0.0000000000, 0.0856407422, -0.1236007911, 0.1850578863, 0.0068232541, 0.0060989965, 0.8253041038, 1.6174878040, 3.3533136863],
    10: [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
}
# --- ADP_THETA_END ---


FEATURE_NAMES = [
    "constant",
    "price_t",
    "price_previous",
    "occ1_t",
    "occ2_t",

    "humidity_excess",
    "humidity_excess_sq",

    "T1_deficit_to_TOK",
    "T2_deficit_to_TOK",
    "T1_deficit_sq",
    "T2_deficit_sq",

    "T1_excess_above_Thigh",
    "T2_excess_above_Thigh",
    "T1_excess_sq",
    "T2_excess_sq",

    "vent_counter",
    "low_override_r1",
    "low_override_r2",
]

N_FEATURES = len(FEATURE_NAMES)


# =========================================================
# Shared fixed problem data
# =========================================================
def get_problem_data():
    """Load fixed system parameters from the provided data file."""
    params = SystemCharacteristics.get_fixed_data()

    return {
        "Pmax": float(params["heating_max_power"]),
        "Pvent": float(params["ventilation_power"]),
        "Uvent": int(params.get("vent_min_up_time", 3)),
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


def feature_vector(state):
    """
    Full linear value-function feature vector.

    The value function is linear in eta:

        V_hat_t(s) = eta_t @ phi(s)

    The features include positive-part basis functions. This is still a
    linear value function approximation because the approximation is linear
    in the fitted coefficients eta.
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

    h_excess = max(0.0, H - DATA["HHigh"])
    t1_deficit = max(0.0, DATA["TOK"] - T1)
    t2_deficit = max(0.0, DATA["TOK"] - T2)
    t1_high = max(0.0, T1 - DATA["THigh"])
    t2_high = max(0.0, T2 - DATA["THigh"])

    return np.array([
        1.0,
        price,
        price_prev,
        occ1,
        occ2,
        H,
        h_excess,
        t1_deficit,
        t2_deficit,
        t1_high,
        t2_high,
        price * h_excess,
        price * t1_deficit,
        price * t2_deficit,
        occ1 * t1_deficit,
        occ2 * t2_deficit,
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
# Online ADP policy
# =========================================================
def select_action(state):
    """
    ADP policy using a stage-dependent linear value-function approximation.

    Online decision:
        min_a immediate_cost(s_t, a_t)
              + E[ V_hat_{t+1}(s_{t+1}) ]

    where
        V_hat_t(s_t) = eta_t^T phi(s_t).

    The feature vector contains positive-part features for humidity excess,
    temperature deficits, and high-temperature excess.
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

    # Online tuning parameters
    N_NEXT_SAMPLES = 10
    SOLVER_TIME_LIMIT = 4
    BIG_M = 100.0

    # ---------------------------------------------------------
    # Read current state
    # ---------------------------------------------------------
    T1 = safe_float(state.get("T1", 21.0), 21.0)
    T2 = safe_float(state.get("T2", 21.0), 21.0)
    H = safe_float(state.get("H", 40.0), 40.0)

    occ1 = safe_float(state.get("Occ1", 30.0), 30.0)
    occ2 = safe_float(state.get("Occ2", 20.0), 20.0)

    price = safe_float(state.get("price_t", 4.0), 4.0)
    price_prev = safe_float(state.get("price_previous", 4.0), 4.0)

    vent_counter = int(round(safe_float(state.get("vent_counter", 0), 0)))
    low1 = 1 if safe_float(state.get("low_override_r1", 0), 0) > 0.5 else 0
    low2 = 1 if safe_float(state.get("low_override_r2", 0), 0) > 0.5 else 0

    current_time = int(round(safe_float(state.get("current_time", 0), 0)))

    # Safety correction for inconsistent low-temperature latch values.
    if T1 < Tlow:
        low1 = 1
    if T2 < Tlow:
        low2 = 1

    theta_next = ADP_THETA.get(current_time + 1, [0.0] * N_FEATURES)

    # ---------------------------------------------------------
    # Generate one-step exogenous samples
    # ---------------------------------------------------------
    samples = []
    for _ in range(N_NEXT_SAMPLES):
        next_price, next_occ1, next_occ2 = sample_next_exogenous({
            "price_t": price,
            "price_previous": price_prev,
            "Occ1": occ1,
            "Occ2": occ2,
        })
        samples.append((next_price, next_occ1, next_occ2))

    prob = 1.0 / N_NEXT_SAMPLES

    try:
        m = pyo.ConcreteModel()

        m.K = pyo.RangeSet(0, N_NEXT_SAMPLES - 1)
        m.R = pyo.Set(initialize=[1, 2])

        # Commanded actions returned to the environment
        m.pc = pyo.Var(m.R, bounds=(0.0, Pmax))
        m.vb = pyo.Var(domain=pyo.Binary)

        # Effective actions after overrule/inertia logic
        m.pf = pyo.Var(m.R, bounds=(0.0, Pmax))
        m.ve = pyo.Var(domain=pyo.Binary)

        # Next physical states for each uncertainty sample
        m.T1_next = pyo.Var(m.K)
        m.T2_next = pyo.Var(m.K)
        m.H_next = pyo.Var(m.K)

        # Positive-part features for the next state.
        # These are modeled as exact max(0, x) terms using binary selectors.
        # This is necessary because fitted value-function coefficients may be negative;
        # with only lower bounds, the optimizer could otherwise make these variables
        # arbitrarily large and the Bellman problem becomes unbounded.
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

        # Next low-temperature latch features
        m.low1_next = pyo.Var(m.K, domain=pyo.Binary)
        m.low2_next = pyo.Var(m.K, domain=pyo.Binary)
        m.z1_below_low = pyo.Var(m.K, domain=pyo.Binary)
        m.z2_below_low = pyo.Var(m.K, domain=pyo.Binary)
        m.y1_ok_next = pyo.Var(m.K, domain=pyo.Binary)
        m.y2_ok_next = pyo.Var(m.K, domain=pyo.Binary)

        m.cons = pyo.ConstraintList()

        # -----------------------------------------------------
        # Effective heating logic at the current state
        # -----------------------------------------------------
        if T1 >= THigh:
            m.cons.add(m.pf[1] == 0.0)
            # Command is irrelevant under high-temperature shutoff, but we fix it
            # to avoid extracting an uninitialized Pyomo variable.
            m.cons.add(m.pc[1] == 0.0)
        elif low1 == 1:
            m.cons.add(m.pf[1] == Pmax)
            # Command is irrelevant under low-temperature overrule, but returning
            # max power is consistent with the enforced effective action.
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

        # -----------------------------------------------------
        # Effective ventilation logic at the current state
        # -----------------------------------------------------
        if H > HHigh or vent_counter > 0:
            m.cons.add(m.ve == 1)
            # Command is irrelevant when ventilation is forced by humidity or inertia,
            # but we fix it to avoid extracting an uninitialized Pyomo variable.
            m.cons.add(m.vb == 1)
        else:
            m.cons.add(m.ve >= m.vb)

        # Approximate next ventilation inertia feature.
        # This follows the same convention as the SP policy:
        # counter 0: ventilation is not in the forced-on period;
        # counter 1: ventilation must stay ON now and next stage;
        # counter 2: ventilation must stay ON now only.
        if vent_counter == 0:
            vent_counter_next_expr = m.ve
        elif vent_counter == 1:
            vent_counter_next_expr = 2.0 * m.ve
        else:
            vent_counter_next_expr = 0.0

        # -----------------------------------------------------
        # One-step dynamics and next-state features
        # -----------------------------------------------------
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

            # Positive-part features, modeled exactly as y = max(0, x).
            # For y = max(0, x):
            #   y >= x, y >= 0, y <= x + M(1-z), y <= Mz
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

            # Next threshold indicators for low-temperature latch update.
            # z_below_low = 1 if T_next <= Tlow
            m.cons.add(m.T1_next[k] <= Tlow + BIG_M * (1 - m.z1_below_low[k]))
            m.cons.add(m.T1_next[k] >= Tlow - BIG_M * m.z1_below_low[k])
            m.cons.add(m.T2_next[k] <= Tlow + BIG_M * (1 - m.z2_below_low[k]))
            m.cons.add(m.T2_next[k] >= Tlow - BIG_M * m.z2_below_low[k])

            # y_ok_next = 1 if T_next >= TOK
            m.cons.add(m.T1_next[k] >= TOK - BIG_M * (1 - m.y1_ok_next[k]))
            m.cons.add(m.T1_next[k] <= TOK + BIG_M * m.y1_ok_next[k])
            m.cons.add(m.T2_next[k] >= TOK - BIG_M * (1 - m.y2_ok_next[k]))
            m.cons.add(m.T2_next[k] <= TOK + BIG_M * m.y2_ok_next[k])

            # Low-temperature latch update:
            # low_next = 1 if below low, or if current low latch is active and OK has not been reached.
            m.cons.add(m.low1_next[k] >= m.z1_below_low[k])
            m.cons.add(m.low1_next[k] >= low1 - m.y1_ok_next[k])
            m.cons.add(m.low1_next[k] <= m.z1_below_low[k] + low1)
            m.cons.add(m.low1_next[k] <= m.z1_below_low[k] + (1 - m.y1_ok_next[k]))

            m.cons.add(m.low2_next[k] >= m.z2_below_low[k])
            m.cons.add(m.low2_next[k] >= low2 - m.y2_ok_next[k])
            m.cons.add(m.low2_next[k] <= m.z2_below_low[k] + low2)
            m.cons.add(m.low2_next[k] <= m.z2_below_low[k] + (1 - m.y2_ok_next[k]))

        immediate_cost = price * (m.pf[1] + m.pf[2] + Pvent * m.ve)

        # -----------------------------------------------------
        # Full approximate future value with nonnegative protection.
        #
        # The value function is a cost-to-go approximation. Allowing negative
        # approximate values gives the optimizer an artificial reward and was one
        # reason for aggressive heating/ventilation. We therefore model
        # V_next[k] = max(0, theta_next @ phi(s_next[k])) using linear constraints.
        # -----------------------------------------------------
        m.V_next = pyo.Var(m.K, bounds=(0.0, None))

        for k, (price_next, occ1_next, occ2_next) in enumerate(samples):
            theta_phi = (
                theta_next[0]
                + theta_next[1] * price_next
                + theta_next[2] * price
                + theta_next[3] * occ1_next
                + theta_next[4] * occ2_next
                + theta_next[5] * m.H_next[k]
                + theta_next[6] * m.h_excess[k]
                + theta_next[7] * m.t1_deficit[k]
                + theta_next[8] * m.t2_deficit[k]
                + theta_next[9] * m.t1_high_excess[k]
                + theta_next[10] * m.t2_high_excess[k]
                + theta_next[11] * price_next * m.h_excess[k]
                + theta_next[12] * price_next * m.t1_deficit[k]
                + theta_next[13] * price_next * m.t2_deficit[k]
                + theta_next[14] * occ1_next * m.t1_deficit[k]
                + theta_next[15] * occ2_next * m.t2_deficit[k]
                + theta_next[16] * vent_counter_next_expr
                + theta_next[17] * m.low1_next[k]
                + theta_next[18] * m.low2_next[k]
            )
            m.cons.add(m.V_next[k] >= theta_phi)

        future_value = sum(prob * m.V_next[k] for k in m.K)

        m.obj = pyo.Objective(expr=immediate_cost + future_value, sense=pyo.minimize)

        solver = pyo.SolverFactory("gurobi")
        solver.options["OutputFlag"] = 0
        solver.options["TimeLimit"] = SOLVER_TIME_LIMIT
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
                f"Solver failed in online ADP policy: "
                f"{results.solver.status}, {results.solver.termination_condition}"
            )

        p1 = pyo.value(m.pc[1])
        p2 = pyo.value(m.pc[2])
        v = pyo.value(m.vb)

        if p1 is None or p2 is None or v is None:
            raise RuntimeError("No valid ADP decision extracted.")

        return {
            "HeatPowerRoom1": float(max(0.0, min(Pmax, p1))),
            "HeatPowerRoom2": float(max(0.0, min(Pmax, p2))),
            "VentilationON": 1 if float(v) > 0.5 else 0,
        }

    except Exception:
        # Safe fallback
        p1_fb = Pmax if low1 == 1 and T1 < THigh else 0.0
        p2_fb = Pmax if low2 == 1 and T2 < THigh else 0.0
        v_fb = 1 if H > HHigh or vent_counter > 0 else 0

        return {
            "HeatPowerRoom1": float(p1_fb),
            "HeatPowerRoom2": float(p2_fb),
            "VentilationON": int(v_fb),
        }


def debug_bellman_action_ranking(state):
    """
    Diagnostic:
    Compare a small set of manually chosen actions by decomposing the ADP Bellman score into:

        immediate cost
        approximate future value
        total score

    This helps detect whether the value function rewards bad actions such as full heating
    or unnecessary ventilation.
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

    # Current state
    T1 = safe_float(state.get("T1", 21.0), 21.0)
    T2 = safe_float(state.get("T2", 21.0), 21.0)
    H = safe_float(state.get("H", 40.0), 40.0)

    occ1 = safe_float(state.get("Occ1", 30.0), 30.0)
    occ2 = safe_float(state.get("Occ2", 20.0), 20.0)

    price = safe_float(state.get("price_t", 4.0), 4.0)
    price_prev = safe_float(state.get("price_previous", 4.0), 4.0)

    vent_counter = int(round(safe_float(state.get("vent_counter", 0), 0)))
    low1 = 1 if safe_float(state.get("low_override_r1", 0), 0) > 0.5 else 0
    low2 = 1 if safe_float(state.get("low_override_r2", 0), 0) > 0.5 else 0

    current_time = int(round(safe_float(state.get("current_time", 0), 0)))

    if T1 < Tlow:
        low1 = 1
    if T2 < Tlow:
        low2 = 1

    theta_next = np.array(
        ADP_THETA.get(current_time + 1, [0.0] * N_FEATURES),
        dtype=float,
    )

    tout = Tout[current_time % T_HORIZON]

    def effective_action(p1_cmd, p2_cmd, v_cmd):
        """Apply the same immediate override logic as select_action."""
        if T1 >= THigh:
            p1_eff = 0.0
        elif low1 == 1:
            p1_eff = Pmax
        else:
            p1_eff = float(p1_cmd)

        if T2 >= THigh:
            p2_eff = 0.0
        elif low2 == 1:
            p2_eff = Pmax
        else:
            p2_eff = float(p2_cmd)

        if H > HHigh or vent_counter > 0:
            v_eff = 1
        else:
            v_eff = int(v_cmd)

        return p1_eff, p2_eff, v_eff

    def update_vent_counter_local(current_counter, effective_ventilation):
        if current_counter == 0:
            return 1 if effective_ventilation == 1 else 0
        if current_counter == 1:
            return 2 if effective_ventilation == 1 else 0
        return 0

    def simulate_next_state_for_action(p1_cmd, p2_cmd, v_cmd):
        p1_eff, p2_eff, v_eff = effective_action(p1_cmd, p2_cmd, v_cmd)

        T1_next = (
            T1
            + z_exch * (T2 - T1)
            + z_loss * (tout - T1)
            + z_conv * p1_eff
            - z_cool * v_eff
            + z_occ * occ1
        )

        T2_next = (
            T2
            + z_exch * (T1 - T2)
            + z_loss * (tout - T2)
            + z_conv * p2_eff
            - z_cool * v_eff
            + z_occ * occ2
        )

        H_next = H + eta_occ * (occ1 + occ2) - eta_vent * v_eff

        low1_next = int(T1_next < Tlow or (low1 == 1 and T1_next < TOK))
        low2_next = int(T2_next < Tlow or (low2 == 1 and T2_next < TOK))
        vent_counter_next = update_vent_counter_local(vent_counter, v_eff)

        next_state = {
            "current_time": current_time + 1,
            "T1": float(T1_next),
            "T2": float(T2_next),
            "H": float(H_next),
            "price_t": price,              # keep exogenous fixed for this diagnostic
            "price_previous": price_prev,
            "Occ1": occ1,
            "Occ2": occ2,
            "vent_counter": int(vent_counter_next),
            "low_override_r1": int(low1_next),
            "low_override_r2": int(low2_next),
        }

        immediate_cost = price * (p1_eff + p2_eff + Pvent * v_eff)
        future_value_raw = float(theta_next @ feature_vector(next_state))
        future_value_clipped = max(0.0, future_value_raw)
        total_score = immediate_cost + future_value_clipped

        return {
            "p1_cmd": p1_cmd,
            "p2_cmd": p2_cmd,
            "v_cmd": v_cmd,
            "p1_eff": p1_eff,
            "p2_eff": p2_eff,
            "v_eff": v_eff,
            "T1_next": T1_next,
            "T2_next": T2_next,
            "H_next": H_next,
            "low1_next": low1_next,
            "low2_next": low2_next,
            "vent_counter_next": vent_counter_next,
            "immediate_cost": immediate_cost,
            "future_value_raw": future_value_raw,
            "future_value_clipped": future_value_clipped,
            "total_score": total_score,
        }

    candidate_actions = [
        (0.0, 0.0, 0),
        (Pmax, 0.0, 0),
        (0.0, Pmax, 0),
        (Pmax, Pmax, 0),
        (0.0, 0.0, 1),
        (Pmax, 0.0, 1),
        (0.0, Pmax, 1),
        (Pmax, Pmax, 1),
        (0.5 * Pmax, 0.5 * Pmax, 0),
        (0.5 * Pmax, 0.5 * Pmax, 1),
    ]

    rows = [simulate_next_state_for_action(*a) for a in candidate_actions]
    rows = sorted(rows, key=lambda x: x["total_score"])

    print("\n================ ADP Bellman action ranking ================")
    print("Current state:")
    print(
        f"t={current_time}, "
        f"T1={T1:.2f}, T2={T2:.2f}, H={H:.2f}, "
        f"price={price:.2f}, occ1={occ1:.2f}, occ2={occ2:.2f}, "
        f"vent_counter={vent_counter}, low1={low1}, low2={low2}"
    )

    print("\nCandidate actions ranked by ADP Bellman score:")
    header = (
        "rank | cmd(p1,p2,v) | eff(p1,p2,v) | "
        "next(T1,T2,H) | next flags | immediate | V_raw | V_clip | total"
    )
    print(header)
    print("-" * len(header))

    for i, r in enumerate(rows, start=1):
        print(
            f"{i:>4} | "
            f"({r['p1_cmd']:.1f},{r['p2_cmd']:.1f},{r['v_cmd']}) | "
            f"({r['p1_eff']:.1f},{r['p2_eff']:.1f},{r['v_eff']}) | "
            f"({r['T1_next']:.2f},{r['T2_next']:.2f},{r['H_next']:.2f}) | "
            f"vc={r['vent_counter_next']}, low=({r['low1_next']},{r['low2_next']}) | "
            f"{r['immediate_cost']:>9.2f} | "
            f"{r['future_value_raw']:>7.2f} | "
            f"{r['future_value_clipped']:>7.2f} | "
            f"{r['total_score']:>7.2f}"
        )

    print("============================================================\n")


# =========================================================
# Offline training configuration
# =========================================================
TRAINING_RNG_SEED = 27
N_STATE_SAMPLES = 180       # sampled states per stage for Bellman fitting
N_TRAIN_NEXT_SAMPLES = 12    # uncertainty samples per Bellman target
RIDGE = 1.0  # ridge regularization to stabilize correlated ADP features
TRAINING_BIG_M = 100.0
TRAINING_SOLVER_TIME_LIMIT = 2

# Empirical state sampling:
# Use the provided 100-day price and occupancy data to sample exogenous
# components of training states instead of broad artificial uniform ranges.
USE_EMPIRICAL_DATA_SAMPLING = True

# Forward-pass state sampling:
# If enabled, part of the training states are generated by simulating a simple
# greedy policy forward through empirical daily trajectories. This follows the
# lecture idea that training states should be sampled in regions that a plausible
# policy actually visits.
USE_FORWARD_PASS_SAMPLING = True
FORWARD_PASS_SHARE = 0.7
N_FORWARD_TRAJECTORIES = 120

# Lecture-aligned training extensions.
USE_OIH_WARM_START = True
OIH_STATE_SAMPLES = 25
OIH_TRAJECTORIES_PER_STATE = 3
OIH_SOLVER_TIME_LIMIT = 2

N_ADP_ITERATIONS = 4
ADP_FORWARD_TRAJECTORIES = 80
ETA_UPDATE_ALPHA = 0.30
CLIP_NEGATIVE_TARGETS = True


# No target clipping is used in the base version. Bellman targets are fitted
# directly by linear regression, consistent with the sampling-based approximate
# backward induction procedure.


def _candidate_data_paths(filename):
    """Return candidate locations for the CSV data files."""
    base = Path(__file__).resolve().parent
    return [
        base / filename,
        base / "Data" / filename,
        base.parent / filename,
        base.parent / "Data" / filename,
    ]


def _read_csv_numeric(path):
    """
    Read a CSV file into a numeric numpy array without requiring pandas.

    The assignment data files use header rows. We therefore skip the first row
    and parse all remaining entries as floats.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if not parts or all(p == "" for p in parts):
                continue
            rows.append([float(p) for p in parts])
    return np.array(rows, dtype=float)


def load_empirical_data():
    """
    Load the provided price and occupancy data if available.

    Price data:
        columns: previous price for first timeslot, then prices for timeslots 1,...,10
    Occupancy data:
        columns: timeslots 0,...,9

    If the files are not found, None is returned and the code falls back to
    broad random sampling.
    """
    filenames = {
        "price": "v2_PriceData.csv",
        "occ1": "OccupancyRoom1.csv",
        "occ2": "OccupancyRoom2.csv",
    }

    paths = {}
    for key, filename in filenames.items():
        found = None
        for candidate in _candidate_data_paths(filename):
            if candidate.exists():
                found = candidate
                break
        if found is None:
            return None
        paths[key] = found

    price = _read_csv_numeric(paths["price"])
    occ1 = _read_csv_numeric(paths["occ1"])
    occ2 = _read_csv_numeric(paths["occ2"])

    # Basic shape check. We expect 100 days, but only require enough columns.
    if price.shape[1] < DATA["T_HORIZON"] + 1:
        return None
    if occ1.shape[1] < DATA["T_HORIZON"] or occ2.shape[1] < DATA["T_HORIZON"]:
        return None

    return {
        "price": price,
        "occ1": occ1,
        "occ2": occ2,
    }


EMPIRICAL_DATA = load_empirical_data()


def empirical_exogenous_state(t, rng, day_idx=None):
    """
    Sample price and occupancy values from the empirical data.

    For t=0, the previous price is taken from the first column of the price file.
    For t>0, the previous price is the empirical price of the previous timeslot.
    """
    if EMPIRICAL_DATA is None:
        return None

    price_data = EMPIRICAL_DATA["price"]
    occ1_data = EMPIRICAL_DATA["occ1"]
    occ2_data = EMPIRICAL_DATA["occ2"]

    n_days = min(price_data.shape[0], occ1_data.shape[0], occ2_data.shape[0])

    if day_idx is None:
        day_idx = int(rng.integers(0, n_days))
    else:
        day_idx = int(day_idx) % n_days

    t = int(t)
    price_t = float(price_data[day_idx, t + 1])
    if t == 0:
        price_prev = float(price_data[day_idx, 0])
    else:
        price_prev = float(price_data[day_idx, t])

    occ1 = float(occ1_data[day_idx, t])
    occ2 = float(occ2_data[day_idx, t])

    return price_t, price_prev, occ1, occ2, day_idx


def update_vent_counter(current_counter, effective_ventilation):
    """
    Update the ventilation inertia counter using the convention used in the policy:
        0: not inside forced-on period
        1: ventilation must be ON now and next stage
        2: ventilation must be ON now only
    """
    effective_ventilation = int(effective_ventilation)

    if current_counter == 0:
        return 1 if effective_ventilation == 1 else 0
    if current_counter == 1:
        return 2 if effective_ventilation == 1 else 0
    return 0


def greedy_rollout_action(state):
    """
    Simple one-step greedy rollout policy used only for forward-pass sampling.

    It minimizes immediate cost and relies on the controller overrules:
    - heating is only applied if the low-temperature overrule is active,
    - ventilation is only applied if forced by humidity or existing inertia.
    """
    Pmax = DATA["Pmax"]
    Tlow = DATA["Tlow"]
    THigh = DATA["THigh"]
    HHigh = DATA["HHigh"]

    T1 = float(state["T1"])
    T2 = float(state["T2"])
    H = float(state["H"])
    vent_counter = int(state["vent_counter"])

    low1 = int(state["low_override_r1"])
    low2 = int(state["low_override_r2"])

    if T1 < Tlow:
        low1 = 1
    if T2 < Tlow:
        low2 = 1

    p1 = Pmax if low1 == 1 and T1 < THigh else 0.0
    p2 = Pmax if low2 == 1 and T2 < THigh else 0.0
    v = 1 if H > HHigh or vent_counter > 0 else 0

    return float(p1), float(p2), int(v)


def simulate_one_step_state(state, action, t):
    """
    Simulate one physical transition for forward-pass state sampling.

    This is only used offline to create realistic training states. The online
    ADP policy still optimizes the one-step Bellman problem.
    """
    Pmax = DATA["Pmax"]
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

    T1 = float(state["T1"])
    T2 = float(state["T2"])
    H = float(state["H"])
    occ1 = float(state["Occ1"])
    occ2 = float(state["Occ2"])
    vent_counter = int(state["vent_counter"])

    p1_cmd, p2_cmd, v_cmd = action

    low1 = int(state["low_override_r1"])
    low2 = int(state["low_override_r2"])
    if T1 < Tlow:
        low1 = 1
    if T2 < Tlow:
        low2 = 1

    if T1 >= THigh:
        p1_eff = 0.0
    elif low1 == 1:
        p1_eff = Pmax
    else:
        p1_eff = float(p1_cmd)

    if T2 >= THigh:
        p2_eff = 0.0
    elif low2 == 1:
        p2_eff = Pmax
    else:
        p2_eff = float(p2_cmd)

    if H > HHigh or vent_counter > 0:
        v_eff = 1
    else:
        v_eff = int(v_cmd)

    tout = Tout[int(t) % T_HORIZON]

    T1_next = (
        T1
        + z_exch * (T2 - T1)
        + z_loss * (tout - T1)
        + z_conv * p1_eff
        - z_cool * v_eff
        + z_occ * occ1
    )

    T2_next = (
        T2
        + z_exch * (T1 - T2)
        + z_loss * (tout - T2)
        + z_conv * p2_eff
        - z_cool * v_eff
        + z_occ * occ2
    )

    H_next = H + eta_occ * (occ1 + occ2) - eta_vent * v_eff

    low1_next = int(T1_next < Tlow or (low1 == 1 and T1_next < TOK))
    low2_next = int(T2_next < Tlow or (low2 == 1 and T2_next < TOK))
    vent_counter_next = update_vent_counter(vent_counter, v_eff)

    return float(T1_next), float(T2_next), float(H_next), int(vent_counter_next), int(low1_next), int(low2_next)


def generate_forward_pass_states(n_trajectories, rng):
    """
    Generate visited training states by simulating a simple greedy policy forward.

    The exogenous price and occupancy values are taken from empirical daily
    trajectories. This produces states that are dynamically consistent across
    stages, instead of sampling each stage independently.
    """
    states_by_t = {t: [] for t in range(DATA["T_HORIZON"])}

    if EMPIRICAL_DATA is None:
        return states_by_t

    price_data = EMPIRICAL_DATA["price"]
    n_days = price_data.shape[0]

    for _ in range(n_trajectories):
        day_idx = int(rng.integers(0, n_days))

        # Initial endogenous state. These defaults match the typical starting
        # conditions used elsewhere in the assignment code.
        T1 = 21.0
        T2 = 21.0
        H = 40.0
        vent_counter = 0
        low1 = 0
        low2 = 0

        for t in range(DATA["T_HORIZON"]):
            exo = empirical_exogenous_state(t, rng, day_idx=day_idx)
            if exo is None:
                break

            price_t, price_prev, occ1, occ2, _ = exo

            state = {
                "current_time": int(t),
                "T1": float(T1),
                "T2": float(T2),
                "H": float(H),
                "price_t": float(price_t),
                "price_previous": float(price_prev),
                "Occ1": float(occ1),
                "Occ2": float(occ2),
                "vent_counter": int(vent_counter),
                "low_override_r1": int(low1),
                "low_override_r2": int(low2),
            }

            states_by_t[t].append(state)

            action = greedy_rollout_action(state)
            T1, T2, H, vent_counter, low1, low2 = simulate_one_step_state(
                state, action, t
            )

    return states_by_t


def choose_training_states(t, rng, rollout_states_by_t=None):
    """
    Choose training states for a stage.

    If forward-pass sampling is enabled, a share of the states is drawn from
    greedy policy rollouts and the remaining states are independently sampled
    from empirical/broad distributions.
    """
    states = []

    if (
        USE_FORWARD_PASS_SAMPLING
        and rollout_states_by_t is not None
        and len(rollout_states_by_t.get(t, [])) > 0
    ):
        n_forward = int(round(FORWARD_PASS_SHARE * N_STATE_SAMPLES))
        candidates = rollout_states_by_t[t]

        for _ in range(n_forward):
            idx = int(rng.integers(0, len(candidates)))
            states.append(dict(candidates[idx]))

    while len(states) < N_STATE_SAMPLES:
        states.append(sample_training_state(t, rng))

    return states[:N_STATE_SAMPLES]



def sample_training_state(t, rng):
    """
    Sample a physically plausible state for stage t.

    Exogenous components are sampled from the empirical price/occupancy data
    when available. The endogenous components are still sampled from broad
    physically plausible ranges, unless forward-pass sampling is used.
    """
    t = int(t)

    T1 = rng.uniform(16.0, 25.5)
    T2 = rng.uniform(16.0, 25.5)
    H = rng.uniform(35.0, 82.0)

    exo = empirical_exogenous_state(t, rng) if USE_EMPIRICAL_DATA_SAMPLING else None
    if exo is not None:
        price, price_prev, occ1, occ2, _ = exo
    else:
        # Fallback if the CSV files are unavailable.
        price_prev = rng.uniform(0.0, 12.0)
        price = rng.uniform(0.0, 12.0)
        occ1 = rng.uniform(20.0, 50.0)
        occ2 = rng.uniform(10.0, 30.0)

    vent_counter = int(rng.choice([0, 1, 2], p=[0.65, 0.20, 0.15]))

    # Low-temperature latch cannot be inferred from temperature in the hysteresis band.
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


def solve_one_step_training_target(state, theta_next):
    """
    Compute one Bellman target for a sampled state:

        y_t = min_a c(s_t, a_t)
              + E[ V_hat_{t+1}(s_{t+1}) ]

    subject to one-step temperature, humidity and controller dynamics.
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

    t = int(state["current_time"])
    samples = [sample_next_exogenous(state) for _ in range(N_TRAIN_NEXT_SAMPLES)]
    prob = 1.0 / N_TRAIN_NEXT_SAMPLES

    T1 = float(state["T1"])
    T2 = float(state["T2"])
    H = float(state["H"])
    occ1 = float(state["Occ1"])
    occ2 = float(state["Occ2"])
    price = float(state["price_t"])
    vent_counter = int(state["vent_counter"])
    low1 = int(state["low_override_r1"])
    low2 = int(state["low_override_r2"])

    if T1 < Tlow:
        low1 = 1
    if T2 < Tlow:
        low2 = 1

    m = pyo.ConcreteModel()
    m.K = pyo.RangeSet(0, N_TRAIN_NEXT_SAMPLES - 1)
    m.R = pyo.Set(initialize=[1, 2])

    # Commanded actions
    m.pc = pyo.Var(m.R, bounds=(0.0, Pmax))
    m.vb = pyo.Var(domain=pyo.Binary)

    # Effective actions
    m.pf = pyo.Var(m.R, bounds=(0.0, Pmax))
    m.ve = pyo.Var(domain=pyo.Binary)

    # Next states and next-state features
    m.T1_next = pyo.Var(m.K)
    m.T2_next = pyo.Var(m.K)
    m.H_next = pyo.Var(m.K)

    # Positive-part features are modeled exactly as max(0, x) using binaries.
    # Without upper/equality logic, negative fitted coefficients can make the
    # Bellman target problem unbounded.
    m.h_excess = pyo.Var(m.K, bounds=(0.0, TRAINING_BIG_M))
    m.t1_deficit = pyo.Var(m.K, bounds=(0.0, TRAINING_BIG_M))
    m.t2_deficit = pyo.Var(m.K, bounds=(0.0, TRAINING_BIG_M))
    m.t1_high_excess = pyo.Var(m.K, bounds=(0.0, TRAINING_BIG_M))
    m.t2_high_excess = pyo.Var(m.K, bounds=(0.0, TRAINING_BIG_M))

    m.h_excess_pos = pyo.Var(m.K, domain=pyo.Binary)
    m.t1_deficit_pos = pyo.Var(m.K, domain=pyo.Binary)
    m.t2_deficit_pos = pyo.Var(m.K, domain=pyo.Binary)
    m.t1_high_excess_pos = pyo.Var(m.K, domain=pyo.Binary)
    m.t2_high_excess_pos = pyo.Var(m.K, domain=pyo.Binary)

    m.low1_next = pyo.Var(m.K, domain=pyo.Binary)
    m.low2_next = pyo.Var(m.K, domain=pyo.Binary)
    m.z1_below_low = pyo.Var(m.K, domain=pyo.Binary)
    m.z2_below_low = pyo.Var(m.K, domain=pyo.Binary)
    m.y1_ok_next = pyo.Var(m.K, domain=pyo.Binary)
    m.y2_ok_next = pyo.Var(m.K, domain=pyo.Binary)

    m.cons = pyo.ConstraintList()

    # Effective heating at current state
    if T1 >= THigh:
        m.cons.add(m.pf[1] == 0.0)
    elif low1 == 1:
        m.cons.add(m.pf[1] == Pmax)
    else:
        m.cons.add(m.pf[1] == m.pc[1])

    if T2 >= THigh:
        m.cons.add(m.pf[2] == 0.0)
    elif low2 == 1:
        m.cons.add(m.pf[2] == Pmax)
    else:
        m.cons.add(m.pf[2] == m.pc[2])

    # Effective ventilation at current state
    if H > HHigh or vent_counter > 0:
        m.cons.add(m.ve == 1)
    else:
        m.cons.add(m.ve >= m.vb)

    # Approximate next ventilation inertia feature
    if vent_counter == 0:
        vent_counter_next_expr = m.ve
    elif vent_counter == 1:
        vent_counter_next_expr = 2.0 * m.ve
    else:
        vent_counter_next_expr = 0.0

    tout = Tout[t % T_HORIZON]

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

        # Positive-part features, modeled exactly as y = max(0, x).
        h_expr = m.H_next[k] - HHigh
        m.cons.add(m.h_excess[k] >= h_expr)
        m.cons.add(m.h_excess[k] <= h_expr + TRAINING_BIG_M * (1 - m.h_excess_pos[k]))
        m.cons.add(m.h_excess[k] <= TRAINING_BIG_M * m.h_excess_pos[k])

        t1_def_expr = TOK - m.T1_next[k]
        m.cons.add(m.t1_deficit[k] >= t1_def_expr)
        m.cons.add(m.t1_deficit[k] <= t1_def_expr + TRAINING_BIG_M * (1 - m.t1_deficit_pos[k]))
        m.cons.add(m.t1_deficit[k] <= TRAINING_BIG_M * m.t1_deficit_pos[k])

        t2_def_expr = TOK - m.T2_next[k]
        m.cons.add(m.t2_deficit[k] >= t2_def_expr)
        m.cons.add(m.t2_deficit[k] <= t2_def_expr + TRAINING_BIG_M * (1 - m.t2_deficit_pos[k]))
        m.cons.add(m.t2_deficit[k] <= TRAINING_BIG_M * m.t2_deficit_pos[k])

        t1_high_expr = m.T1_next[k] - THigh
        m.cons.add(m.t1_high_excess[k] >= t1_high_expr)
        m.cons.add(m.t1_high_excess[k] <= t1_high_expr + TRAINING_BIG_M * (1 - m.t1_high_excess_pos[k]))
        m.cons.add(m.t1_high_excess[k] <= TRAINING_BIG_M * m.t1_high_excess_pos[k])

        t2_high_expr = m.T2_next[k] - THigh
        m.cons.add(m.t2_high_excess[k] >= t2_high_expr)
        m.cons.add(m.t2_high_excess[k] <= t2_high_expr + TRAINING_BIG_M * (1 - m.t2_high_excess_pos[k]))
        m.cons.add(m.t2_high_excess[k] <= TRAINING_BIG_M * m.t2_high_excess_pos[k])

        # Next low-overrule state
        m.cons.add(m.T1_next[k] <= Tlow + TRAINING_BIG_M * (1 - m.z1_below_low[k]))
        m.cons.add(m.T1_next[k] >= Tlow - TRAINING_BIG_M * m.z1_below_low[k])
        m.cons.add(m.T2_next[k] <= Tlow + TRAINING_BIG_M * (1 - m.z2_below_low[k]))
        m.cons.add(m.T2_next[k] >= Tlow - TRAINING_BIG_M * m.z2_below_low[k])

        m.cons.add(m.T1_next[k] >= TOK - TRAINING_BIG_M * (1 - m.y1_ok_next[k]))
        m.cons.add(m.T1_next[k] <= TOK + TRAINING_BIG_M * m.y1_ok_next[k])
        m.cons.add(m.T2_next[k] >= TOK - TRAINING_BIG_M * (1 - m.y2_ok_next[k]))
        m.cons.add(m.T2_next[k] <= TOK + TRAINING_BIG_M * m.y2_ok_next[k])

        m.cons.add(m.low1_next[k] >= m.z1_below_low[k])
        m.cons.add(m.low1_next[k] >= low1 - m.y1_ok_next[k])
        m.cons.add(m.low1_next[k] <= m.z1_below_low[k] + low1)
        m.cons.add(m.low1_next[k] <= m.z1_below_low[k] + (1 - m.y1_ok_next[k]))

        m.cons.add(m.low2_next[k] >= m.z2_below_low[k])
        m.cons.add(m.low2_next[k] >= low2 - m.y2_ok_next[k])
        m.cons.add(m.low2_next[k] <= m.z2_below_low[k] + low2)
        m.cons.add(m.low2_next[k] <= m.z2_below_low[k] + (1 - m.y2_ok_next[k]))

    immediate_cost = price * (m.pf[1] + m.pf[2] + Pvent * m.ve)

    # Correct 19-feature future value, matching feature_vector() and select_action().
    # We also protect against negative cost-to-go values by modeling
    # V_next[k] = max(0, theta_next @ phi(s_next[k])).
    m.V_next = pyo.Var(m.K, bounds=(0.0, None))

    for k, (price_next, occ1_next, occ2_next) in enumerate(samples):
        theta_phi = (
            theta_next[0]
            + theta_next[1] * price_next
            + theta_next[2] * price
            + theta_next[3] * occ1_next
            + theta_next[4] * occ2_next
            + theta_next[5] * m.H_next[k]
            + theta_next[6] * m.h_excess[k]
            + theta_next[7] * m.t1_deficit[k]
            + theta_next[8] * m.t2_deficit[k]
            + theta_next[9] * m.t1_high_excess[k]
            + theta_next[10] * m.t2_high_excess[k]
            + theta_next[11] * price_next * m.h_excess[k]
            + theta_next[12] * price_next * m.t1_deficit[k]
            + theta_next[13] * price_next * m.t2_deficit[k]
            + theta_next[14] * occ1_next * m.t1_deficit[k]
            + theta_next[15] * occ2_next * m.t2_deficit[k]
            + theta_next[16] * vent_counter_next_expr
            + theta_next[17] * m.low1_next[k]
            + theta_next[18] * m.low2_next[k]
        )
        m.cons.add(m.V_next[k] >= theta_phi)

    future_value = sum(prob * m.V_next[k] for k in m.K)

    m.obj = pyo.Objective(expr=immediate_cost + future_value, sense=pyo.minimize)

    solver = pyo.SolverFactory("gurobi")
    solver.options["OutputFlag"] = 0
    solver.options["TimeLimit"] = TRAINING_SOLVER_TIME_LIMIT
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
            f"Solver failed while computing Bellman target: "
            f"{results.solver.status}, {results.solver.termination_condition}"
        )

    value = pyo.value(m.obj)
    if value is None or not np.isfinite(value):
        raise RuntimeError("Could not compute Bellman target.")

    return float(value)



# =========================================================
# OiH warm-start and forward-backward ADP training helpers
# =========================================================
def sample_future_trajectory_from_state(state, rng):
    """Sample one complete exogenous trajectory from the current state to end of day."""
    t0 = int(state["current_time"])
    horizon = DATA["T_HORIZON"] - t0

    price = float(state["price_t"])
    price_prev = float(state["price_previous"])
    occ1 = float(state["Occ1"])
    occ2 = float(state["Occ2"])

    trajectory = []
    for h in range(horizon):
        t_abs = t0 + h
        trajectory.append({
            "price": float(price),
            "occ1": float(occ1),
            "occ2": float(occ2),
            "tout": float(DATA["Tout"][t_abs % DATA["T_HORIZON"]]),
        })
        if h < horizon - 1:
            next_price = PriceProcessRestaurant.price_model(price, price_prev)
            next_occ1, next_occ2 = OccupancyProcessRestaurant.next_occupancy_levels(occ1, occ2)
            price_prev = price
            price = float(next_price)
            occ1 = float(next_occ1)
            occ2 = float(next_occ2)

    return trajectory


def solve_oih_trajectory_cost(state, trajectory):
    """
    Deterministic optimal-in-hindsight cost from a sampled state and future trajectory.

    This is used only offline to create warm-start targets. It is not used by
    select_action() and therefore does not turn the submitted policy into SP.
    """
    Pmax = DATA["Pmax"]
    Pvent = DATA["Pvent"]
    Uvent = DATA["Uvent"]
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

    h = len(trajectory)
    if h <= 0:
        return 0.0

    BIG_M = TRAINING_BIG_M

    T1_0 = float(state["T1"])
    T2_0 = float(state["T2"])
    H_0 = float(state["H"])
    low1_0 = int(state["low_override_r1"])
    low2_0 = int(state["low_override_r2"])
    vent_counter_0 = int(state["vent_counter"])

    if T1_0 < Tlow:
        low1_0 = 1
    if T2_0 < Tlow:
        low2_0 = 1

    m = pyo.ConcreteModel()
    m.S = pyo.RangeSet(0, h)       # state nodes
    m.D = pyo.RangeSet(0, h - 1)   # decision stages
    m.R = pyo.Set(initialize=[1, 2])

    m.Temp = pyo.Var(m.S, m.R)
    m.Hum = pyo.Var(m.S)

    m.pc = pyo.Var(m.D, m.R, bounds=(0.0, Pmax))
    m.vb = pyo.Var(m.D, domain=pyo.Binary)
    m.pf = pyo.Var(m.D, m.R, bounds=(0.0, Pmax))
    m.ve = pyo.Var(m.D, domain=pyo.Binary)

    m.y_low = pyo.Var(m.S, m.R, domain=pyo.Binary)
    m.y_ok = pyo.Var(m.S, m.R, domain=pyo.Binary)
    m.y_high = pyo.Var(m.S, m.R, domain=pyo.Binary)
    m.z_below_low = pyo.Var(m.S, m.R, domain=pyo.Binary)
    m.start_vent = pyo.Var(m.D, domain=pyo.Binary)

    m.cons = pyo.ConstraintList()

    m.cons.add(m.Temp[0, 1] == T1_0)
    m.cons.add(m.Temp[0, 2] == T2_0)
    m.cons.add(m.Hum[0] == H_0)
    m.cons.add(m.y_low[0, 1] == low1_0)
    m.cons.add(m.y_low[0, 2] == low2_0)

    # Threshold detection for all state nodes.
    for s in range(h + 1):
        for r in [1, 2]:
            m.cons.add(m.Temp[s, r] >= THigh - BIG_M * (1 - m.y_high[s, r]))
            m.cons.add(m.Temp[s, r] <= THigh + BIG_M * m.y_high[s, r])

            m.cons.add(m.Temp[s, r] >= TOK - BIG_M * (1 - m.y_ok[s, r]))
            m.cons.add(m.Temp[s, r] <= TOK + BIG_M * m.y_ok[s, r])

            m.cons.add(m.Temp[s, r] <= Tlow + BIG_M * (1 - m.z_below_low[s, r]))
            m.cons.add(m.Temp[s, r] >= Tlow - BIG_M * m.z_below_low[s, r])

    # Low-temperature hysteresis update.
    for s in range(1, h + 1):
        for r in [1, 2]:
            m.cons.add(m.y_low[s, r] >= m.z_below_low[s, r])
            m.cons.add(m.y_low[s, r] >= m.y_low[s - 1, r] - m.y_ok[s, r])
            m.cons.add(m.y_low[s, r] <= m.z_below_low[s, r] + m.y_low[s - 1, r])
            m.cons.add(m.y_low[s, r] <= m.z_below_low[s, r] + (1 - m.y_ok[s, r]))

    # Effective heating and ventilation logic.
    for d in range(h):
        for r in [1, 2]:
            m.cons.add(m.pf[d, r] <= Pmax * (1 - m.y_high[d, r]))
            m.cons.add(m.pf[d, r] >= Pmax * (m.y_low[d, r] - m.y_high[d, r]))
            m.cons.add(m.pf[d, r] <= m.pc[d, r] + Pmax * (m.y_low[d, r] + m.y_high[d, r]))
            m.cons.add(m.pf[d, r] >= m.pc[d, r] - Pmax * (m.y_low[d, r] + m.y_high[d, r]))

        # Humidity overrule and commanded ventilation.
        m.cons.add(m.Hum[d] <= HHigh + BIG_M * m.ve[d])
        m.cons.add(m.ve[d] >= m.vb[d])

        # Ventilation startup detection.
        if d == 0:
            prev_on = 1 if vent_counter_0 > 0 else 0
            m.cons.add(m.start_vent[d] >= m.ve[d] - prev_on)
            m.cons.add(m.start_vent[d] <= m.ve[d])
            m.cons.add(m.start_vent[d] <= 1 - prev_on)
        else:
            m.cons.add(m.start_vent[d] >= m.ve[d] - m.ve[d - 1])
            m.cons.add(m.start_vent[d] <= m.ve[d])
            m.cons.add(m.start_vent[d] <= 1 - m.ve[d - 1])

    # Existing ventilation inertia from the observed state.
    if vent_counter_0 == 1:
        m.cons.add(m.ve[0] == 1)
        if h >= 2:
            m.cons.add(m.ve[1] == 1)
    elif vent_counter_0 == 2:
        m.cons.add(m.ve[0] == 1)

    # Minimum up-time if ventilation starts inside the trajectory.
    for d in range(h):
        for dd in range(d + 1, min(h, d + Uvent)):
            m.cons.add(m.ve[dd] >= m.start_vent[d])

    # Physical dynamics.
    for d in range(h):
        occ1 = float(trajectory[d]["occ1"])
        occ2 = float(trajectory[d]["occ2"])
        tout = float(trajectory[d]["tout"])

        m.cons.add(
            m.Temp[d + 1, 1]
            == m.Temp[d, 1]
            + z_exch * (m.Temp[d, 2] - m.Temp[d, 1])
            + z_loss * (tout - m.Temp[d, 1])
            + z_conv * m.pf[d, 1]
            - z_cool * m.ve[d]
            + z_occ * occ1
        )
        m.cons.add(
            m.Temp[d + 1, 2]
            == m.Temp[d, 2]
            + z_exch * (m.Temp[d, 1] - m.Temp[d, 2])
            + z_loss * (tout - m.Temp[d, 2])
            + z_conv * m.pf[d, 2]
            - z_cool * m.ve[d]
            + z_occ * occ2
        )
        m.cons.add(m.Hum[d + 1] == m.Hum[d] + eta_occ * (occ1 + occ2) - eta_vent * m.ve[d])

    m.obj = pyo.Objective(
        expr=sum(
            float(trajectory[d]["price"]) * (m.pf[d, 1] + m.pf[d, 2] + Pvent * m.ve[d])
            for d in range(h)
        ),
        sense=pyo.minimize,
    )

    solver = pyo.SolverFactory("gurobi")
    solver.options["OutputFlag"] = 0
    solver.options["TimeLimit"] = OIH_SOLVER_TIME_LIMIT
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
        raise RuntimeError("OiH deterministic solve failed")

    value = pyo.value(m.obj)
    if value is None or not np.isfinite(value):
        raise RuntimeError("OiH deterministic solve returned invalid value")
    return float(max(0.0, value))


def oih_warm_start_theta(rng):
    """Fit initial eta_t values from optimal-in-hindsight trajectory costs."""
    theta = {DATA["T_HORIZON"]: np.zeros(N_FEATURES)}
    summary = {}

    print("\nOiH warm-start:")
    for t in reversed(range(DATA["T_HORIZON"])):
        states_t = [sample_training_state(t, rng) for _ in range(OIH_STATE_SAMPLES)]
        targets_t = []
        failed = 0

        for s in states_t:
            values = []
            for _ in range(OIH_TRAJECTORIES_PER_STATE):
                try:
                    traj = sample_future_trajectory_from_state(s, rng)
                    values.append(solve_oih_trajectory_cost(s, traj))
                except Exception:
                    failed += 1

            if values:
                target = float(np.mean(values))
            else:
                target = 1000.0
            if CLIP_NEGATIVE_TARGETS:
                target = max(0.0, target)
            targets_t.append(target)

        theta[t] = fit_theta(states_t, targets_t)
        summary[("oih", t)] = {
            "mean_target": float(np.mean(targets_t)),
            "std_target": float(np.std(targets_t)),
            "min_target": float(np.min(targets_t)),
            "max_target": float(np.max(targets_t)),
            "failed_targets": int(failed),
        }
        print(
            f"OiH theta[{t}] | "
            f"mean={summary[('oih', t)]['mean_target']:.2f}, "
            f"std={summary[('oih', t)]['std_target']:.2f}, "
            f"min={summary[('oih', t)]['min_target']:.2f}, "
            f"max={summary[('oih', t)]['max_target']:.2f}, "
            f"failed={summary[('oih', t)]['failed_targets']}"
        )

    return theta, summary


def generate_policy_forward_pass_states(theta, n_trajectories, rng):
    """Forward pass using the current ADP policy implied by theta."""
    states_by_t = {t: [] for t in range(DATA["T_HORIZON"])}

    if EMPIRICAL_DATA is None:
        return generate_forward_pass_states(n_trajectories, rng)

    global ADP_THETA
    old_theta = ADP_THETA
    ADP_THETA = {int(k): [float(x) for x in v] for k, v in theta.items()}

    try:
        price_data = EMPIRICAL_DATA["price"]
        n_days = price_data.shape[0]

        for _ in range(n_trajectories):
            day_idx = int(rng.integers(0, n_days))
            T1 = 21.0
            T2 = 21.0
            H = 40.0
            vent_counter = 0
            low1 = 0
            low2 = 0

            for t in range(DATA["T_HORIZON"]):
                exo = empirical_exogenous_state(t, rng, day_idx=day_idx)
                if exo is None:
                    break
                price_t, price_prev, occ1, occ2, _ = exo
                state = {
                    "current_time": int(t),
                    "T1": float(T1),
                    "T2": float(T2),
                    "H": float(H),
                    "price_t": float(price_t),
                    "price_previous": float(price_prev),
                    "Occ1": float(occ1),
                    "Occ2": float(occ2),
                    "vent_counter": int(vent_counter),
                    "low_override_r1": int(low1),
                    "low_override_r2": int(low2),
                }
                states_by_t[t].append(dict(state))

                try:
                    a = select_action(state)
                    action = (
                        float(a["HeatPowerRoom1"]),
                        float(a["HeatPowerRoom2"]),
                        int(a["VentilationON"]),
                    )
                except Exception:
                    action = greedy_rollout_action(state)

                T1, T2, H, vent_counter, low1, low2 = simulate_one_step_state(state, action, t)

    finally:
        ADP_THETA = old_theta

    return states_by_t


def fit_theta(states, targets):
    """
    Linear regression fit:

        eta_t = argmin_eta ||Phi eta - y||^2

    The base version uses ordinary least squares. If RIDGE is set to a positive
    value for tuning, the same function becomes a ridge regression fit.
    """
    Phi = np.vstack([feature_vector(s) for s in states])
    y = np.array(targets, dtype=float)

    if RIDGE > 0.0:
        reg = RIDGE * np.eye(Phi.shape[1])
        reg[0, 0] = 0.0  # do not regularize the intercept
        A = Phi.T @ Phi + reg
        b = Phi.T @ y
        return np.linalg.solve(A, b)

    eta, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    return eta


def train_adp():
    """
    Advanced pure ADP training recipe:

    1. Optional OiH warm-start for physically meaningful initial targets.
    2. Repeated forward-backward fitted value improvement.
    3. Ridge regularization, target clipping, and gradual eta updates.

    The submitted online policy remains a one-step ADP Bellman policy.
    """
    rng = np.random.default_rng(TRAINING_RNG_SEED)
    summary = {}

    if USE_OIH_WARM_START:
        theta, oih_summary = oih_warm_start_theta(rng)
        summary.update(oih_summary)
    else:
        theta = {t: np.zeros(N_FEATURES) for t in range(DATA["T_HORIZON"] + 1)}

    theta[DATA["T_HORIZON"]] = np.zeros(N_FEATURES)

    for iteration in range(1, N_ADP_ITERATIONS + 1):
        print(f"\nForward-backward ADP iteration {iteration}/{N_ADP_ITERATIONS}:")

        if USE_FORWARD_PASS_SAMPLING:
            rollout_states_by_t = generate_policy_forward_pass_states(
                theta, ADP_FORWARD_TRAJECTORIES, rng
            )
        else:
            rollout_states_by_t = None

        theta_fit = {DATA["T_HORIZON"]: np.zeros(N_FEATURES)}

        for t in reversed(range(DATA["T_HORIZON"])):
            states_t = choose_training_states(t, rng, rollout_states_by_t)
            targets_t = []
            failed_targets = 0

            # Fitted-value / approximate-policy-iteration style target:
            # evaluate one-step decisions against the previous theta estimate.
            theta_next = theta[t + 1]

            for s in states_t:
                try:
                    target = float(solve_one_step_training_target(s, theta_next))
                    if CLIP_NEGATIVE_TARGETS:
                        target = max(0.0, target)
                except Exception:
                    target = 1000.0
                    failed_targets += 1

                targets_t.append(target)

            theta_fit[t] = fit_theta(states_t, targets_t)

            # Gradual update of eta, as recommended in the ADP lecture variants.
            theta[t] = (1.0 - ETA_UPDATE_ALPHA) * theta[t] + ETA_UPDATE_ALPHA * theta_fit[t]

            key = (iteration, t)
            summary[key] = {
                "mean_target": float(np.mean(targets_t)),
                "std_target": float(np.std(targets_t)),
                "min_target": float(np.min(targets_t)),
                "max_target": float(np.max(targets_t)),
                "failed_targets": int(failed_targets),
            }

            print(
                f"Iter {iteration} theta[{t}] | "
                f"mean={summary[key]['mean_target']:.2f}, "
                f"std={summary[key]['std_target']:.2f}, "
                f"min={summary[key]['min_target']:.2f}, "
                f"max={summary[key]['max_target']:.2f}, "
                f"failed={summary[key]['failed_targets']}"
            )

    return theta, summary


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
    """
    Replace the ADP_THETA block in this file.

    This is only used when running:
        python ADP_policy_27_advanced.py --train-and-update

    The submitted policy only needs select_action(state), so this update logic
    does not affect the teacher's evaluator.
    """
    if file_path is None:
        file_path = Path(__file__)
    else:
        file_path = Path(file_path)

    text = file_path.read_text(encoding="utf-8")
    new_theta_code = format_theta_dict(theta)

    pattern = (
        r"# --- ADP_THETA_START ---\n"
        r".*?"
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
            f"Could not uniquely locate ADP_THETA marker block. Found {n_replacements} matches."
        )

    file_path.write_text(updated, encoding="utf-8")
    print(f"\nUpdated ADP_THETA in: {file_path}")



def debug_value_predictions(n_samples=50, seed=123):
    """
    Print value-function predictions on sampled states.

    This checks whether the fitted value functions have a sensible scale.
    Expected pattern:
        earlier stages should generally have larger values than later stages,
        and predictions should not explode to extremely large magnitudes.
    """
    rng = np.random.default_rng(seed)

    print("\nValue-function prediction check:")
    for t in range(DATA["T_HORIZON"] + 1):
        values = []

        for _ in range(n_samples):
            if t == DATA["T_HORIZON"]:
                # Terminal value is defined as zero.
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
    """
    Run select_action on a few sampled states.

    This checks that the submitted policy function returns feasible-looking
    actions and does not crash after ADP_THETA has been updated.
    """
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


def run_post_training_checks():
    """
    Convenience wrapper after --train-and-update.

    Run this using:
        python ADP_policy_27_advanced.py --check-values
        python ADP_policy_27_advanced.py --smoke-test
    """
    debug_value_predictions()
    smoke_test_select_action()


if __name__ == "__main__":
    if "--check-values" in sys.argv:
        debug_value_predictions()

    elif "--smoke-test" in sys.argv:
        smoke_test_select_action()

    elif "--check-all" in sys.argv:
        run_post_training_checks()

    elif "--debug-ranking" in sys.argv:
        state = {
            "current_time": 0,
            "T1": 21.0,
            "T2": 21.0,
            "H": 40.0,
            "price_t": 4.0,
            "price_previous": 4.0,
            "Occ1": 30.0,
            "Occ2": 20.0,
            "vent_counter": 0,
            "low_override_r1": 0,
            "low_override_r2": 0,
        }

        debug_bellman_action_ranking(state)
        print("ADP select_action gives:")
        print(select_action(state))

    else:
        theta, summary = train_adp()

        if "--train-and-update" in sys.argv:
            update_adp_theta_in_file(theta)
            print(
                "\nNext recommended checks:\n"
                "    python ADP_policy_27_advanced.py --check-values\n"
                "    python ADP_policy_27_advanced.py --smoke-test\n"
            )
        else:
            print_theta_for_copy(theta)
            print(
                "\nTo update this file automatically, run:\n"
                "    python ADP_policy_27_advanced.py --train-and-update\n"
            )
