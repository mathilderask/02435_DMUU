"""
Task 4: Approximate Dynamic Programming policy with empirical forward-pass training.

This file contains both:

1. train_adp()
   Offline training by sampling-based approximate backward induction.
   Running this file directly prints a trained ADP_THETA dictionary.

2. select_action(state)
   Submitted online ADP policy.
   The policy uses the ADP_THETA dictionary and returns the here-and-now action.

Workflow:
    1. Run this file locally with:
           python ADP_policy_27_empirical_forward_clean.py --train-and-update

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
# Each vector has 14 entries and corresponds to FEATURE_NAMES.
# --- ADP_THETA_START ---
ADP_THETA = {
    0: [-128.4080699542, 48.4636137843, -27.4724035876, 0.8972895423, 0.7368029632, 0.5770753779, 3.9925908877, 15.5107385628, 6.7046851928, 0.0000000000, 0.0000000000, 7.2936934463, 6.8913657138, 17.2959421607],
    1: [-106.3505311259, 44.7128081729, -23.1403415815, 0.8168586983, -0.5653723681, 0.5249174901, 1.7564126922, 7.2742556248, 9.1561616799, 0.0000000000, 0.0000000000, 10.1953033268, 27.9835609489, 15.9544735164],
    2: [-144.3376381590, 42.9432792599, -21.8817844556, 0.4842681982, 0.3156935353, 0.9302515539, 1.8078588991, 9.7467061042, 6.8033916626, 0.0000000000, 0.0000000000, 1.5115645727, 17.6731366615, 22.9439472794],
    3: [-143.6846620424, 41.2317855064, -21.0399032146, 0.1875015765, 0.4618391544, 1.1437503861, -0.2588224536, 6.6576121166, 2.4728005216, 0.0000000000, 0.0000000000, 2.8207172765, 28.3272598382, 21.9147082171],
    4: [-87.0109054627, 37.9244974482, -19.2811521935, -0.3829739511, -0.1360584571, 0.6781875103, 0.5940601398, 4.8069554528, 5.5585202711, 0.0000000000, 0.0000000000, 6.0550073129, 27.9832831717, 14.5382456255],
    5: [-80.8292005112, 29.5078842226, -14.7350420638, 0.0628266041, 0.6105671702, 0.3418142272, 2.4221890518, 4.4022004557, 5.9322928943, 0.0000000000, 0.0000000000, 2.9721920810, 11.7787269384, 12.3424868869],
    6: [-70.3523005026, 22.4826209823, -11.7474163817, 0.5554760193, -0.3561159824, 0.5624090850, 0.9853659213, 3.4038244654, 2.9678042140, 0.0000000000, 0.0000000000, 1.9999956499, 15.9766414569, 24.9744311014],
    7: [-75.2816994777, 19.0281248330, -8.7878705307, 0.5907577011, 0.0066915339, 0.3663053545, 1.4936349631, 3.0958420720, 3.0439620684, 0.0000000000, 0.0000000000, 3.1413174868, 20.9242963181, 20.3192053782],
    8: [-54.8813193168, 13.9907485138, -4.1117343582, 0.0693865970, 0.0812597615, 0.2066905038, 0.6946278632, 3.2047119373, 0.9207931763, 0.0000000000, 0.0000000000, 4.4227277444, 20.8181404144, 13.9821195729],
    9: [-28.0414022293, 6.2580386353, 0.4566994973, 0.0000920494, -0.0111777335, 0.0523733625, 0.1644808435, -0.0429451739, 0.6757141610, 0.0000000000, 0.0000000000, 1.0663730996, 13.3207609259, 12.1860714911],
    10: [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
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
# Shared fixed problem data
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

    return np.array([
        1.0,
        price,
        price_prev,
        occ1,
        occ2,
        H,
        max(0.0, H - DATA["HHigh"]),
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
        # Full approximate future value
        # -----------------------------------------------------
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
                + theta_next[7] * m.t1_deficit[k]
                + theta_next[8] * m.t2_deficit[k]
                + theta_next[9] * m.t1_high_excess[k]
                + theta_next[10] * m.t2_high_excess[k]
                + theta_next[11] * vent_counter_next_expr
                + theta_next[12] * m.low1_next[k]
                + theta_next[13] * m.low2_next[k]
            )

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


# =========================================================
# Offline training configuration
# =========================================================
TRAINING_RNG_SEED = 27
N_STATE_SAMPLES = 120       # sampled states per stage
N_TRAIN_NEXT_SAMPLES = 8    # uncertainty samples per Bellman target
RIDGE = 1e-2  # ridge regularization to stabilize correlated ADP features
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
N_FORWARD_TRAJECTORIES = 200

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
            + theta_next[7] * m.t1_deficit[k]
            + theta_next[8] * m.t2_deficit[k]
            + theta_next[9] * m.t1_high_excess[k]
            + theta_next[10] * m.t2_high_excess[k]
            + theta_next[11] * vent_counter_next_expr
            + theta_next[12] * m.low1_next[k]
            + theta_next[13] * m.low2_next[k]
        )

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
        A = Phi.T @ Phi + RIDGE * np.eye(Phi.shape[1])
        b = Phi.T @ y
        return np.linalg.solve(A, b)

    eta, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    return eta


def train_adp():
    """
    Sampling-based approximate backward induction.

    Terminal condition:
        V_hat_T(s_T) = 0

    Then for t = T-1, ..., 0:
        sample states
        compute Bellman targets using theta_{t+1}
        fit theta_t by linear regression
    """
    rng = np.random.default_rng(TRAINING_RNG_SEED)

    theta = {DATA["T_HORIZON"]: np.zeros(N_FEATURES)}
    summary = {}

    rollout_states_by_t = None
    if USE_FORWARD_PASS_SAMPLING:
        rollout_states_by_t = generate_forward_pass_states(N_FORWARD_TRAJECTORIES, rng)

    for t in reversed(range(DATA["T_HORIZON"])):
        states_t = choose_training_states(t, rng, rollout_states_by_t)
        targets_t = []

        theta_next = theta[t + 1]

        failed_targets = 0

        for s in states_t:
            try:
                # Bellman target used directly, without clipping.
                target = float(solve_one_step_training_target(s, theta_next))

            except Exception:
                # Conservative finite fallback target for rare numerical failures.
                # In normal training this should remain zero, as indicated by
                # failed_targets in the printed summary.
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
    """Replace the complete ADP_THETA marker block in this file."""
    if file_path is None:
        file_path = Path(__file__)
    else:
        file_path = Path(file_path)

    text = file_path.read_text(encoding="utf-8")
    new_theta_code = format_theta_dict(theta)

    # Replace everything between the markers, including comments.
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
            f"Could not uniquely locate ADP_THETA marker block. "
            f"Found {n_replacements} matching blocks."
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
        python ADP_policy_27_empirical_forward_clean.py --check-values
        python ADP_policy_27_empirical_forward_clean.py --smoke-test
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

    else:
        theta, summary = train_adp()

        if "--train-and-update" in sys.argv:
            update_adp_theta_in_file(theta)
            print(
                "\nNext recommended checks:\n"
                "    python ADP_policy_27_empirical_forward_clean.py --check-values\n"
                "    python ADP_policy_27_empirical_forward_clean.py --smoke-test\n"
            )
        else:
            print_theta_for_copy(theta)
            print(
                "\nTo update this file automatically, run:\n"
                "    python ADP_policy_27_empirical_forward_clean.py --train-and-update\n"
            )
