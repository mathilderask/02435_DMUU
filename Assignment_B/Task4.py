import numpy as np
import pandas as pd

import Data.v2_SystemCharacteristics as SystemCharacteristics
import Data.OccupancyProcessRestaurant as OccupancyProcessRestaurant



# Trained value function coefficients
THETA_BY_TIME = {
    0: np.array([
        -3.2733306783,
        -4.1522931171,
        -5.5422925995,
        0.9698570657,
        130.3150668361,
        0.8482829920,
        -3.0523084802,
        895.5304079061,
        33.9406924209,
        -19.0233675986,
        -32.7333067828,
    ]),
    1: np.array([
        -3.9432293544,
        -4.7870414720,
        -7.9611422493,
        1.4015449688,
        106.7485956199,
        1.1149449518,
        0.8334622347,
        601.3094427423,
        53.8096655197,
        -3.6007143954,
        -35.4890641899,
    ]),
    2: np.array([
        -2.0272788693,
        -5.5732312557,
        -4.4565963824,
        0.1639364443,
        87.5062055563,
        -0.9701861423,
        -0.2671399962,
        387.6532181572,
        4.0176347993,
        -12.3501936052,
        -16.2182309541,
    ]),
    3: np.array([
        -1.4576785949,
        -10.9925407460,
        -0.3335167587,
        0.2295686973,
        70.3534315677,
        0.1905740108,
        -0.1492409250,
        254.2377326662,
        -12.5554848578,
        1.2691628507,
        -10.2037501641,
    ]),
    4: np.array([
        3.4261438116,
        -9.0900752216,
        -8.5288458457,
        0.4146571540,
        54.7356223565,
        -0.7997515102,
        -0.4138019400,
        171.3864179958,
        -8.1314678550,
        10.6834151872,
        20.5568628697,
    ]),
    5: np.array([
        9.1212897131,
        -9.4990538103,
        -9.7393981787,
        0.1324170819,
        40.6872015820,
        -0.5621859998,
        -0.7188385257,
        105.6283497075,
        -4.9005606463,
        15.3875197596,
        45.6064485654,
    ]),
    6: np.array([
        12.6752182859,
        -8.1457650729,
        -9.4148582723,
        0.5840905845,
        28.6384296866,
        -0.2422520584,
        -0.3448333466,
        65.7831739775,
        5.1428850026,
        11.7884049942,
        50.7008731436,
    ]),
    7: np.array([
        24.0161035132,
        -8.0389712209,
        -7.3866796741,
        0.2859176446,
        18.8319010033,
        -0.3102354219,
        -0.4187398922,
        37.1652739144,
        17.2935222082,
        12.9271084001,
        72.0483105395,
    ]),
    8: np.array([
        31.9147238098,
        -4.9992592984,
        -5.0296637644,
        0.0475853128,
        10.9807315847,
        -0.0996332207,
        -0.0308856013,
        17.9757072424,
        22.6724901381,
        18.7814515327,
        63.8294476196,
    ]),
    9: np.array([
        16.9167058484,
        -1.3017123680,
        -1.6229686320,
        0.0389740896,
        5.1293166707,
        0.0400513137,
        -0.0910720508,
        6.4695575519,
        16.7327552891,
        14.1298303718,
        16.9167058484,
    ]),
}



# Basic helpers


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
    ])


def approximate_value(state, num_timeslots):
    t = int(round(safe_float(state.get("current_time", 0), 0)))
    t = max(0, min(t, num_timeslots - 1))

    if len(THETA_BY_TIME) == 0:
        return 0.0

    theta = THETA_BY_TIME[t]
    return float(np.dot(theta, feature_vector(state, num_timeslots)))



# Price and occupancy processes
def sample_next_price(current_price, previous_price):
    mean_price = 4.0
    reversion_strength = 0.12
    price_cap = 12.0
    price_floor = 0.0

    noise = np.random.normal(0, 0.5)

    next_price = (
        current_price
        + 0.6 * (current_price - previous_price)
        + reversion_strength * (mean_price - current_price)
        + noise
    )

    if next_price < 0:
        if np.random.rand() > 0.2:
            next_price = np.random.uniform(0, mean_price * 0.3)

    return float(np.clip(next_price, price_floor, price_cap))


def expected_next_price(current_price, previous_price):
    mean_price = 4.0
    reversion_strength = 0.12
    price_cap = 12.0
    price_floor = 0.0

    next_price = (
        current_price
        + 0.6 * (current_price - previous_price)
        + reversion_strength * (mean_price - current_price)
    )

    return float(np.clip(next_price, price_floor, price_cap))


def expected_next_occupancy(occ1, occ2, samples=5):
    vals = [
        OccupancyProcessRestaurant.next_occupancy_levels(occ1, occ2)
        for _ in range(samples)
    ]

    return (
        float(np.mean([v[0] for v in vals])),
        float(np.mean([v[1] for v in vals])),
    )



# Dynamics and overrules
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

    return (
        float(np.clip(heat1, 0.0, Pmax)),
        float(np.clip(heat2, 0.0, Pmax)),
        int(np.clip(ventilation, 0, 1)),
    )


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

    H_next = (
        Hum
        + params["eta_occ"] * (Occ1 + Occ2)
        - params["eta_vent"] * ventilation
    )

    if mode == "sample":
        price_next = sample_next_price(price_t, price_prev)
        Occ1_next, Occ2_next = OccupancyProcessRestaurant.next_occupancy_levels(Occ1, Occ2)
    else:
        price_next = expected_next_price(price_t, price_prev)
        Occ1_next, Occ2_next = expected_next_occupancy(Occ1, Occ2)

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



# ADP decision rule
def get_candidate_actions(Pmax):
    heat_candidates = [0.0, 0.25 * Pmax, 0.50 * Pmax, 0.75 * Pmax, Pmax]
    vent_candidates = [0, 1]

    return [
        (h1, h2, v)
        for h1 in heat_candidates
        for h2 in heat_candidates
        for v in vent_candidates
    ]


def evaluate_action(state, action, params):
    h1_raw, h2_raw, v_raw = action

    h1, h2, v = apply_overrules(state, h1_raw, h2_raw, v_raw, params)

    immediate_cost = safe_float(state.get("price_t", 4.0), 4.0) * (
        h1 + h2 + params["Pvent"] * v
    )

    next_state = simulate_next_state(
        state,
        h1,
        h2,
        v,
        params,
        mode="expected",
    )

    future_cost = approximate_value(next_state, params["num_timeslots"])

    return immediate_cost + future_cost, h1, h2, v


def select_action(state):
    params = get_fixed_params()
    actions = get_candidate_actions(params["Pmax"])

    best_score = float("inf")
    best_h1 = 0.0
    best_h2 = 0.0
    best_v = 0

    for action in actions:
        score, h1, h2, v = evaluate_action(state, action, params)

        if score < best_score:
            best_score = score
            best_h1 = h1
            best_h2 = h2
            best_v = v

    HereAndNowActions = {
        "HeatPowerRoom1": float(max(0.0, min(params["Pmax"], best_h1))),
        "HeatPowerRoom2": float(max(0.0, min(params["Pmax"], best_h2))),
        "VentilationON": int(best_v),
    }

    return HereAndNowActions



# Offline training: sampling-based backward induction
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

    for action in get_candidate_actions(params["Pmax"]):

        h1_raw, h2_raw, v_raw = action
        h1, h2, v = apply_overrules(state, h1_raw, h2_raw, v_raw, params)

        immediate_cost = safe_float(state.get("price_t", 4.0), 4.0) * (
            h1 + h2 + params["Pvent"] * v
        )

        future_values = []

        for _ in range(K_next):
            next_state = simulate_next_state(
                state,
                h1,
                h2,
                v,
                params,
                mode="sample",
            )

            future_values.append(
                value_with_theta(
                    next_state,
                    theta_next,
                    params["num_timeslots"],
                )
            )

        target = immediate_cost + float(np.mean(future_values))

        if target < best_value:
            best_value = target

    return best_value


def train_theta_backward_induction(N_states=300, K_next=10, seed=123):
    global THETA_BY_TIME

    np.random.seed(seed)

    params = get_fixed_params()
    H_day = params["num_timeslots"]

    n_features = len(feature_vector(sample_training_state(0), H_day))

    theta_by_time = {}
    theta_next = np.zeros(n_features)

    for t in reversed(range(H_day)):

        X = []
        y = []

        for _ in range(N_states):
            state = sample_training_state(t)
            target = bellman_target(state, theta_next, params, K_next)

            X.append(feature_vector(state, H_day))
            y.append(target)

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        theta_t, *_ = np.linalg.lstsq(X, y, rcond=None)

        theta_by_time[t] = theta_t
        theta_next = theta_t

        print(f"Finished theta training for t={t}")

    THETA_BY_TIME = theta_by_time

    print("\nCopy this block into the top of the file before submission:\n")
    print("THETA_BY_TIME = {")
    for t in range(H_day):
        print(f"    {t}: np.array([")
        for value in theta_by_time[t]:
            print(f"        {value:.10f},")
        print("    ]),")
    print("}")

    return theta_by_time


# ============================================================
# Evaluation on the 100 provided experiment days
# ============================================================

def evaluate_on_100_days():
    np.random.seed(123)
    params = get_fixed_params()

    price = pd.read_csv("Assignment_B/Data/v2_PriceData.csv").to_numpy()
    occ1 = pd.read_csv("Assignment_B/Data/OccupancyRoom1.csv").to_numpy()
    occ2 = pd.read_csv("Assignment_B/Data/OccupancyRoom2.csv").to_numpy()

    num_days = min(price.shape[0], occ1.shape[0], occ2.shape[0])
    H_day = min(
        price.shape[1],
        occ1.shape[1],
        occ2.shape[1],
        params["num_timeslots"]
    )

    print("Using number of days:", num_days)
    print("Using horizon length:", H_day)

    daily_costs = []

    for d in range(num_days):

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

        total_cost = 0.0

        for t in range(H_day):

            state["current_time"] = t
            state["price_t"] = float(price[d, t])
            state["Occ1"] = float(occ1[d, t])
            state["Occ2"] = float(occ2[d, t])

            if t > 0:
                state["price_previous"] = float(price[d, t - 1])

            action = select_action(state)

            h1 = float(action["HeatPowerRoom1"])
            h2 = float(action["HeatPowerRoom2"])
            v = int(action["VentilationON"])

            total_cost += state["price_t"] * (
                h1 + h2 + params["Pvent"] * v
            )

            state = simulate_next_state(
                state,
                h1,
                h2,
                v,
                params,
                mode="expected",
            )

        daily_costs.append(total_cost)

    daily_costs = np.array(daily_costs)

    print("\nADP policy evaluation on 100 experiment days")
    print("-------------------------------------------")
    print(f"Number of days: {num_days}")
    print(f"Average daily electricity cost: {np.mean(daily_costs):.4f}")
    print(f"Minimum daily cost: {np.min(daily_costs):.4f}")
    print(f"Maximum daily cost: {np.max(daily_costs):.4f}")
    print(f"Standard deviation: {np.std(daily_costs):.4f}")

    return daily_costs


# Run locally
if __name__ == "__main__":

    TRAIN_THETA = False
    EVALUATE_100_DAYS = True

    if TRAIN_THETA:
        train_theta_backward_induction(
            N_states=300,
            K_next=10,
            seed=123,
        )

    if EVALUATE_100_DAYS:
        evaluate_on_100_days()