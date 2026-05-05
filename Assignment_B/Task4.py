 # -*- coding: utf-8 -*-
"""
Task 4: Approximate Dynamic Programming Policy

Save this file as:
ADP policy [your group number].py

The policy uses a linear value function approximation:

    V_hat(s) = theta^T phi(s)

At each hour, the policy:
    1. reads the current state from the environment,
    2. evaluates a finite set of candidate actions,
    3. applies the overrule controller logic,
    4. simulates one step forward,
    5. chooses the action minimizing:

        immediate electricity cost + approximate future value

The bottom test block only runs if this file is executed directly.
It is not used when the teacher imports the policy.
"""

import numpy as np

import Data.v2_SystemCharacteristics as SystemCharacteristics
import Data.OccupancyProcessRestaurant as OccupancyProcessRestaurant


# ============================================================
# Linear value function coefficients
#
# Feature order:
# [1, T1, T2, H, price, Occ1, Occ2, vent_counter,
#  low_override_r1, low_override_r2, remaining_time]
# ============================================================

THETA = np.array([
    0.0,      # constant
    -2.0,     # T1: warmer room means less future heating
    -2.0,     # T2: warmer room means less future heating
    1.5,      # H: high humidity may cause ventilation cost
    1.0,      # price: high price increases future expected cost
    0.15,     # Occ1: occupancy affects heat and humidity
    0.15,     # Occ2: occupancy affects heat and humidity
    2.0,      # vent_counter: ventilation inertia may create future cost
    5.0,      # low override room 1: future forced heating risk
    5.0,      # low override room 2: future forced heating risk
    -0.5      # remaining time
])


# ============================================================
# Helper functions
# ============================================================

def safe_float(x, default=0.0):
    """
    Safely convert a value to float.
    """
    try:
        return float(x)
    except Exception:
        return float(default)


def expected_next_price(current_price, previous_price):
    """
    Expected transition of the provided price process.

    We use the deterministic expectation instead of importing
    PriceProcessRestaurant because the provided file may execute plotting code
    when imported.
    """
    mean_price = 4.0
    reversion_strength = 0.12
    price_cap = 12.0
    price_floor = 0.0

    mean_reversion = reversion_strength * (mean_price - current_price)

    next_price = (
        current_price
        + 0.6 * (current_price - previous_price)
        + mean_reversion
    )

    return float(np.clip(next_price, price_floor, price_cap))


def expected_next_occupancy(occ1, occ2, samples=5):
    """
    Expected next occupancy using the teacher's occupancy process.

    The process is stochastic, so we average a few samples.
    This is still very fast.
    """
    values = [
        OccupancyProcessRestaurant.next_occupancy_levels(occ1, occ2)
        for _ in range(samples)
    ]

    occ1_next = np.mean([v[0] for v in values])
    occ2_next = np.mean([v[1] for v in values])

    return float(occ1_next), float(occ2_next)


def feature_vector(state, num_timeslots):
    """
    Construct the ADP feature vector phi(s).
    """
    current_time = int(round(safe_float(state.get("current_time", 0), 0)))
    remaining_time = max(0, num_timeslots - current_time)

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
        float(remaining_time)
    ])


def approximate_value(state, num_timeslots):
    """
    Linear value function approximation.
    """
    return float(np.dot(THETA, feature_vector(state, num_timeslots)))


def apply_overrules(
    state,
    heat1,
    heat2,
    ventilation,
    Pmax,
    Tlow,
    TOK,
    THigh,
    HHigh,
    Uvent
):
    """
    Apply the HVAC overrule controller logic.

    This converts the chosen candidate action into the effective action.
    """
    heat1 = float(heat1)
    heat2 = float(heat2)
    ventilation = int(ventilation)

    T1 = safe_float(state.get("T1", 21.0), 21.0)
    T2 = safe_float(state.get("T2", 21.0), 21.0)
    H = safe_float(state.get("H", 40.0), 40.0)

    vent_counter = int(round(safe_float(state.get("vent_counter", 0), 0)))
    low_override_r1 = 1 if safe_float(state.get("low_override_r1", 0), 0) > 0.5 else 0
    low_override_r2 = 1 if safe_float(state.get("low_override_r2", 0), 0) > 0.5 else 0

    # High-temperature overrule has priority.
    if T1 >= THigh:
        heat1 = 0.0
    elif T1 <= Tlow or low_override_r1 == 1:
        heat1 = Pmax

    if T2 >= THigh:
        heat2 = 0.0
    elif T2 <= Tlow or low_override_r2 == 1:
        heat2 = Pmax

    # Humidity overrule.
    if H >= HHigh:
        ventilation = 1

    # Ventilation inertia.
    if 0 < vent_counter < Uvent:
        ventilation = 1

    heat1 = float(np.clip(heat1, 0.0, Pmax))
    heat2 = float(np.clip(heat2, 0.0, Pmax))
    ventilation = int(np.clip(ventilation, 0, 1))

    return heat1, heat2, ventilation


def simulate_next_state(
    state,
    heat1,
    heat2,
    ventilation,
    params
):
    """
    Simulate one hour forward using the system dynamics.
    """
    Pmax = params["Pmax"]
    Tlow = params["Tlow"]
    TOK = params["TOK"]
    THigh = params["THigh"]
    HHigh = params["HHigh"]

    z_exch = params["z_exch"]
    z_loss = params["z_loss"]
    z_conv = params["z_conv"]
    z_cool = params["z_cool"]
    z_occ = params["z_occ"]

    eta_occ = params["eta_occ"]
    eta_vent = params["eta_vent"]

    Tout = params["Tout"]
    num_timeslots = params["num_timeslots"]

    current_time = int(round(safe_float(state.get("current_time", 0), 0)))
    outdoor_temp = Tout[current_time % num_timeslots]

    T1 = safe_float(state.get("T1", 21.0), 21.0)
    T2 = safe_float(state.get("T2", 21.0), 21.0)
    H = safe_float(state.get("H", 40.0), 40.0)

    Occ1 = safe_float(state.get("Occ1", 30.0), 30.0)
    Occ2 = safe_float(state.get("Occ2", 20.0), 20.0)

    price_t = safe_float(state.get("price_t", 4.0), 4.0)
    price_previous = safe_float(state.get("price_previous", 4.0), 4.0)

    low_override_r1 = 1 if safe_float(state.get("low_override_r1", 0), 0) > 0.5 else 0
    low_override_r2 = 1 if safe_float(state.get("low_override_r2", 0), 0) > 0.5 else 0

    # Temperature dynamics.
    T1_next = (
        T1
        + z_exch * (T2 - T1)
        + z_loss * (outdoor_temp - T1)
        + z_conv * heat1
        - z_cool * ventilation
        + z_occ * Occ1
    )

    T2_next = (
        T2
        + z_exch * (T1 - T2)
        + z_loss * (outdoor_temp - T2)
        + z_conv * heat2
        - z_cool * ventilation
        + z_occ * Occ2
    )

    # Humidity dynamics.
    H_next = (
        H
        + eta_occ * (Occ1 + Occ2)
        - eta_vent * ventilation
    )

    # Exogenous expected transitions.
    price_next = expected_next_price(price_t, price_previous)
    Occ1_next, Occ2_next = expected_next_occupancy(Occ1, Occ2)

    # Hysteresis update.
    low_override_r1_next = int(
        T1_next <= Tlow
        or (
            low_override_r1 == 1
            and T1_next < TOK
        )
    )

    low_override_r2_next = int(
        T2_next <= Tlow
        or (
            low_override_r2 == 1
            and T2_next < TOK
        )
    )

    # Ventilation counter update.
    if ventilation == 1:
        vent_counter_next = int(round(safe_float(state.get("vent_counter", 0), 0))) + 1
    else:
        vent_counter_next = 0

    next_state = {
        "T1": float(T1_next),
        "T2": float(T2_next),
        "H": float(H_next),
        "Occ1": float(Occ1_next),
        "Occ2": float(Occ2_next),
        "price_t": float(price_next),
        "price_previous": float(price_t),
        "vent_counter": int(vent_counter_next),
        "low_override_r1": int(low_override_r1_next),
        "low_override_r2": int(low_override_r2_next),
        "current_time": int(current_time + 1)
    }

    return next_state


# ============================================================
# Required policy function
# ============================================================

def select_action(state):

    # =========================================================
    # Fixed problem data
    # =========================================================
    fixed = SystemCharacteristics.get_fixed_data()

    Pmax = float(fixed["heating_max_power"])
    Pvent = float(fixed["ventilation_power"])

    Tlow = float(fixed["temp_min_comfort_threshold"])
    TOK = float(fixed["temp_OK_threshold"])
    THigh = float(fixed["temp_max_comfort_threshold"])
    HHigh = float(fixed["humidity_threshold"])

    z_exch = float(fixed["heat_exchange_coeff"])
    z_loss = float(fixed["thermal_loss_coeff"])
    z_conv = float(fixed["heating_efficiency_coeff"])
    z_cool = float(fixed["heat_vent_coeff"])
    z_occ = float(fixed["heat_occupancy_coeff"])

    eta_occ = float(fixed["humidity_occupancy_coeff"])
    eta_vent = float(fixed["humidity_vent_coeff"])

    Uvent = int(fixed["vent_min_up_time"])
    Tout = list(fixed["outdoor_temperature"])
    num_timeslots = int(fixed["num_timeslots"])

    params = {
        "Pmax": Pmax,
        "Pvent": Pvent,
        "Tlow": Tlow,
        "TOK": TOK,
        "THigh": THigh,
        "HHigh": HHigh,
        "z_exch": z_exch,
        "z_loss": z_loss,
        "z_conv": z_conv,
        "z_cool": z_cool,
        "z_occ": z_occ,
        "eta_occ": eta_occ,
        "eta_vent": eta_vent,
        "Uvent": Uvent,
        "Tout": Tout,
        "num_timeslots": num_timeslots
    }

    # =========================================================
    # Candidate action set
    #
    # This is where the ADP policy is being tested:
    # every candidate action is simulated one step forward and
    # scored using immediate cost + approximate future value.
    # =========================================================
    heat_candidates = [
        0.0,
        0.25 * Pmax,
        0.50 * Pmax,
        0.75 * Pmax,
        Pmax
    ]

    ventilation_candidates = [0, 1]

    best_score = float("inf")
    best_heat1 = 0.0
    best_heat2 = 0.0
    best_ventilation = 0

    # =========================================================
    # ADP decision loop
    # =========================================================
    for heat1_candidate in heat_candidates:
        for heat2_candidate in heat_candidates:
            for ventilation_candidate in ventilation_candidates:

                heat1, heat2, ventilation = apply_overrules(
                    state,
                    heat1_candidate,
                    heat2_candidate,
                    ventilation_candidate,
                    Pmax,
                    Tlow,
                    TOK,
                    THigh,
                    HHigh,
                    Uvent
                )

                immediate_cost = safe_float(state.get("price_t", 4.0), 4.0) * (
                    heat1
                    + heat2
                    + Pvent * ventilation
                )

                next_state = simulate_next_state(
                    state,
                    heat1,
                    heat2,
                    ventilation,
                    params
                )

                future_cost_estimate = approximate_value(
                    next_state,
                    num_timeslots
                )

                score = immediate_cost + future_cost_estimate

                if score < best_score:
                    best_score = score
                    best_heat1 = heat1
                    best_heat2 = heat2
                    best_ventilation = ventilation

    # =========================================================
    # Return here-and-now action
    # =========================================================
    HereAndNowActions = {
        "HeatPowerRoom1": float(max(0.0, min(Pmax, best_heat1))),
        "HeatPowerRoom2": float(max(0.0, min(Pmax, best_heat2))),
        "VentilationON": 1 if float(best_ventilation) > 0.5 else 0
    }

    return HereAndNowActions
import matplotlib.pyplot as plt

def plot_adp_results(history):
    """
    Plot results from ADP simulation.

    history: list of dictionaries (one per timestep)
    """

    T = [h["time"] for h in history]

    T1 = [h["T1"] for h in history]
    T2 = [h["T2"] for h in history]
    H = [h["H"] for h in history]

    Heat1 = [h["Heat1"] for h in history]
    Heat2 = [h["Heat2"] for h in history]
    Vent = [h["Ventilation"] for h in history]

    Price = [h["price"] for h in history]
    Cost = [h["cost"] for h in history]

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    # ---- Temperature ----
    axes[0].plot(T, T1, label="Room 1 Temp")
    axes[0].plot(T, T2, label="Room 2 Temp")
    axes[0].axhline(18, linestyle="--", color="gray")
    axes[0].axhline(20, linestyle="--", color="gray")
    axes[0].set_ylabel("Temperature")
    axes[0].set_title("Temperatures")
    axes[0].legend()
    axes[0].grid(True)

    # ---- Heating ----
    axes[1].bar(T, Heat1, label="Heat Room 1")
    axes[1].bar(T, Heat2, bottom=Heat1, label="Heat Room 2")
    axes[1].set_ylabel("Heating Power")
    axes[1].set_title("Heating Decisions")
    axes[1].legend()
    axes[1].grid(True)

    # ---- Ventilation & Humidity ----
    axes[2].step(T, Vent, where="mid", label="Ventilation ON")
    axes[2].plot(T, H, label="Humidity")
    axes[2].axhline(60, linestyle="--", color="gray")
    axes[2].set_ylabel("Vent / Humidity")
    axes[2].set_title("Ventilation & Humidity")
    axes[2].legend()
    axes[2].grid(True)

    # ---- Price & Cost ----
    axes[3].plot(T, Price, label="Price")
    axes[3].bar(T, Cost, alpha=0.3, label="Cost")
    axes[3].set_ylabel("Price / Cost")
    axes[3].set_title("Price & Cost")
    axes[3].set_xlabel("Time")
    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()

# ============================================================
# Local test block
#
# This part is only for you.
# The teacher's environment will import select_action(state),
# so this block will NOT run during grading.
# ============================================================

#if __name__ == "__main__":

    import Data.v2_Checks as Checks

    fixed = SystemCharacteristics.get_fixed_data()

    Pmax = float(fixed["heating_max_power"])
    PowerMax = {1: Pmax, 2: Pmax}

    test_state = {
        "T1": float(fixed["T1"]),  # initial temperature at room 1
        "T2": float(fixed["T2"]),  # initial temperature at room 2
        "H": float(fixed["H"]),    # initial humidity
        "Occ1": 30.0,
        "Occ2": 20.0,
        "price_t": 4.0,
        "price_previous": 4.0,
        "vent_counter": 0,
        "low_override_r1": 0,
        "low_override_r2": 0,
        "current_time": 0
    }

    action = Checks.check_and_sanitize_action(
        policy=__import__(__name__),
        state=test_state,
        PowerMax=PowerMax
    )

    print("Test state:")
    print(test_state)

    print("\nSanitized ADP action:")
    print(action)

if __name__ == "__main__":

    fixed = SystemCharacteristics.get_fixed_data()

    Pmax = float(fixed["heating_max_power"])
    Pvent = float(fixed["ventilation_power"])
    num_timeslots = int(fixed["num_timeslots"])

    state = {
        "T1": float(fixed["T1"]),
        "T2": float(fixed["T2"]),
        "H": float(fixed["H"]),
        "Occ1": 30.0,
        "Occ2": 20.0,
        "price_t": 4.0,
        "price_previous": 4.0,
        "vent_counter": 0,
        "low_override_r1": 0,
        "low_override_r2": 0,
        "current_time": 0
    }

    total_cost = 0.0
    history = []

    for t in range(num_timeslots):

        action = select_action(state)

        heat1 = action["HeatPowerRoom1"]
        heat2 = action["HeatPowerRoom2"]
        ventilation = action["VentilationON"]

        hourly_cost = state["price_t"] * (
            heat1 + heat2 + Pvent * ventilation
        )

        total_cost += hourly_cost

        history.append({
            "time": t,
            "T1": state["T1"],
            "T2": state["T2"],
            "H": state["H"],
            "Occ1": state["Occ1"],
            "Occ2": state["Occ2"],
            "price": state["price_t"],
            "Heat1": heat1,
            "Heat2": heat2,
            "Ventilation": ventilation,
            "cost": hourly_cost
        })

        params = {
            "Pmax": Pmax,
            "Pvent": Pvent,
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
            "num_timeslots": num_timeslots
        }

        state = simulate_next_state(
            state,
            heat1,
            heat2,
            ventilation,
            params
        )

    print("\nFull-day ADP test")
    print("-----------------")
    print("Total daily cost:", total_cost)
    plot_adp_results(history)

    print("\nHourly results:")
    for row in history:
        print(row)