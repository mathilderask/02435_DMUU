# -*- coding: utf-8 -*-
"""
Hybrid policy for Assignment B, Task 5.
Group 27.

Idea:
    Multi-stage stochastic lookahead MILP + soft comfort-risk penalties
    + robust fallback. The policy returns only the here-and-now action.

Expected local files:
    v2_SystemCharacteristics.py or SystemCharacteristics.py
    PriceProcessRestaurant.py
    OccupancyProcessRestaurant.py
"""

import sys
import math
import numpy as np
import pyomo.environ as pyo

# Make the policy work both with the v2 assignment files and with the original names.
try:
    import v2_SystemCharacteristics as SystemCharacteristics
    sys.modules.setdefault("SystemCharacteristics", SystemCharacteristics)
except Exception:
    import SystemCharacteristics

import PriceProcessRestaurant
import OccupancyProcessRestaurant


def select_action(state):
    # =========================================================
    # Small utilities
    # =========================================================
    def safe_float(x, default=0.0):
        try:
            if x is None:
                return float(default)
            val = float(x)
            if math.isnan(val) or math.isinf(val):
                return float(default)
            return val
        except Exception:
            return float(default)

    def safe_int(x, default=0):
        return int(round(safe_float(x, default)))

    def clip(x, lo, hi):
        return float(max(lo, min(hi, safe_float(x, lo))))

    # =========================================================
    # Fixed data
    # =========================================================
    params = SystemCharacteristics.get_fixed_data()

    Pmax = float(params["heating_max_power"])
    Pvent = float(params["ventilation_power"])

    Tlow = float(params["temp_min_comfort_threshold"])
    TOK = float(params["temp_OK_threshold"])
    THigh = float(params["temp_max_comfort_threshold"])
    HHigh = float(params["humidity_threshold"])

    z_exch = float(params["heat_exchange_coeff"])
    z_loss = float(params["thermal_loss_coeff"])
    z_conv = float(params["heating_efficiency_coeff"])
    z_cool = float(params["heat_vent_coeff"])
    z_occ = float(params["heat_occupancy_coeff"])

    eta_occ = float(params["humidity_occupancy_coeff"])
    eta_vent = float(params["humidity_vent_coeff"])

    Uvent = int(params["vent_min_up_time"])
    Tout = list(params["outdoor_temperature"])
    num_timeslots = int(params["num_timeslots"])

    # =========================================================
    # Read current state
    # =========================================================
    T1_0 = safe_float(state.get("T1", 21.0), 21.0)
    T2_0 = safe_float(state.get("T2", 21.0), 21.0)
    H_0 = safe_float(state.get("H", 40.0), 40.0)

    Occ1_0 = safe_float(state.get("Occ1", 30.0), 30.0)
    Occ2_0 = safe_float(state.get("Occ2", 20.0), 20.0)

    price_0 = safe_float(state.get("price_t", 4.0), 4.0)
    price_prev_0 = safe_float(state.get("price_previous", 4.0), 4.0)

    vent_counter_0 = safe_int(state.get("vent_counter", 0), 0)
    low_override_r1_0 = 1 if safe_float(state.get("low_override_r1", 0), 0) > 0.5 else 0
    low_override_r2_0 = 1 if safe_float(state.get("low_override_r2", 0), 0) > 0.5 else 0
    current_time = safe_int(state.get("current_time", 0), 0)

    # Deterministic seeding makes the same state produce the same sampled tree.
    seed = int(
        1009 * current_time
        + 37 * round(10 * T1_0)
        + 41 * round(10 * T2_0)
        + 43 * round(10 * H_0)
        + 47 * round(10 * price_0)
    ) % (2**32 - 1)
    np.random.seed(seed)

    # =========================================================
    # Hybrid tuning parameters
    # =========================================================
    remaining = max(1, num_timeslots - current_time)

    # Tree size: keep this compact to stay below the 15 second limit.
    LOOKAHEAD = min(4, remaining)
    INITIAL_SAMPLES = 24
    REDUCED_BRANCHES = 3

    BIG_M = 100.0
    SOLVER_TIME_LIMIT = 6
    MIP_GAP = 0.04

    TARGET_TEMP = 21.0

    # Hybrid additions: soft risk penalties. These make the policy less myopic
    # than pure expected energy minimization.
    TEMP_LOW_BUFFER = 0.4
    TEMP_HIGH_BUFFER = 0.4
    HUM_BUFFER = 2.0

    LOW_TEMP_PENALTY = 35.0
    HIGH_TEMP_PENALTY = 20.0
    HUMIDITY_PENALTY = 7.0
    TERMINAL_TEMP_PENALTY = 1.0 if remaining > 2 else 0.25
    START_VENT_PENALTY = 0.10

    # =========================================================
    # Robust fallback policy
    # =========================================================
    def fallback_action():
        avg_price = 4.0

        def heat_rule(T, low_override):
            if T >= THigh:
                return 0.0
            if low_override == 1 or T <= Tlow:
                return Pmax
            if T < TOK:
                return Pmax if price_0 <= 1.35 * avg_price else 0.75 * Pmax
            if T < TARGET_TEMP - 0.6 and price_0 <= avg_price:
                return 0.45 * Pmax
            return 0.0

        p1 = heat_rule(T1_0, low_override_r1_0)
        p2 = heat_rule(T2_0, low_override_r2_0)

        if H_0 > HHigh or vent_counter_0 in [1, 2]:
            v = 1
        elif H_0 > HHigh - 1.0 and max(T1_0, T2_0) < THigh - 0.5:
            v = 1
        else:
            v = 0

        return {
            "HeatPowerRoom1": clip(p1, 0.0, Pmax),
            "HeatPowerRoom2": clip(p2, 0.0, Pmax),
            "VentilationON": int(v),
        }

    # =========================================================
    # Scenario generation and reduction
    # =========================================================
    def weighted_distance(a, b):
        return (
            (a[0] - b[0]) ** 2
            + 0.08 * (a[1] - b[1]) ** 2
            + 0.08 * (a[2] - b[2]) ** 2
        )

    def reduce_samples_kmeans(samples, k, n_iter=7):
        n = len(samples)
        if n == 0:
            return [], []
        if n <= k:
            return samples[:], [1.0 / n for _ in range(n)]

        idx = np.random.choice(n, size=k, replace=False)
        centroids = [samples[i] for i in idx]

        for _ in range(n_iter):
            clusters = [[] for _ in range(k)]
            for s in samples:
                j = min(range(k), key=lambda c: weighted_distance(s, centroids[c]))
                clusters[j].append(s)

            new_centroids = []
            for j in range(k):
                if not clusters[j]:
                    new_centroids.append(samples[np.random.randint(0, n)])
                else:
                    arr = np.asarray(clusters[j], dtype=float)
                    new_centroids.append(tuple(np.mean(arr, axis=0)))
            centroids = new_centroids

        final_clusters = [[] for _ in range(k)]
        for s in samples:
            j = min(range(k), key=lambda c: weighted_distance(s, centroids[c]))
            final_clusters[j].append(s)

        centers, probs = [], []
        for cl in final_clusters:
            if cl:
                arr = np.asarray(cl, dtype=float)
                centers.append(tuple(np.mean(arr, axis=0)))
                probs.append(len(cl) / n)

        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        return centers, probs

    def simulate_one_step(price_t, price_prev, occ1_t, occ2_t):
        next_price = PriceProcessRestaurant.price_model(price_t, price_prev)
        next_occ1, next_occ2 = OccupancyProcessRestaurant.next_occupancy_levels(occ1_t, occ2_t)
        return float(next_price), float(next_occ1), float(next_occ2)

    try:
        # =====================================================
        # Build reduced scenario tree
        # =====================================================
        nodes, children, parent = {}, {}, {}
        next_node_id = 0
        root = next_node_id
        next_node_id += 1

        nodes[root] = {
            "stage": 0,
            "prob": 1.0,
            "price": price_0,
            "occ1": Occ1_0,
            "occ2": Occ2_0,
            "tout": Tout[current_time % num_timeslots],
            "price_prev": price_prev_0,
            "price_curr": price_0,
            "occ1_curr": Occ1_0,
            "occ2_curr": Occ2_0,
            "time_index": current_time,
        }
        children[root] = []
        parent[root] = None
        stage_nodes = {0: [root]}

        for stage in range(LOOKAHEAD - 1):
            stage_nodes[stage + 1] = []
            for n in stage_nodes[stage]:
                raw_samples = [
                    simulate_one_step(
                        nodes[n]["price_curr"],
                        nodes[n]["price_prev"],
                        nodes[n]["occ1_curr"],
                        nodes[n]["occ2_curr"],
                    )
                    for _ in range(INITIAL_SAMPLES)
                ]
                centers, probs = reduce_samples_kmeans(raw_samples, REDUCED_BRANCHES)

                for c_idx, center in enumerate(centers):
                    child = next_node_id
                    next_node_id += 1
                    c_price, c_occ1, c_occ2 = center
                    c_prob = float(probs[c_idx])
                    time_child = nodes[n]["time_index"] + 1

                    nodes[child] = {
                        "stage": stage + 1,
                        "prob": nodes[n]["prob"] * c_prob,
                        "price": float(c_price),
                        "occ1": float(c_occ1),
                        "occ2": float(c_occ2),
                        "tout": Tout[time_child % num_timeslots],
                        "price_prev": nodes[n]["price_curr"],
                        "price_curr": float(c_price),
                        "occ1_curr": float(c_occ1),
                        "occ2_curr": float(c_occ2),
                        "time_index": time_child,
                    }
                    parent[child] = n
                    children.setdefault(n, []).append(child)
                    children[child] = []
                    stage_nodes[stage + 1].append(child)

        all_nodes = list(nodes.keys())
        leaf_nodes = [n for n in all_nodes if len(children[n]) == 0]
        nonroot_nodes = [n for n in all_nodes if n != root]

        # =====================================================
        # Build stochastic MILP
        # =====================================================
        m = pyo.ConcreteModel()
        m.N = pyo.Set(initialize=all_nodes)
        m.R = pyo.Set(initialize=[1, 2])
        m.L = pyo.Set(initialize=leaf_nodes)
        m.NNR = pyo.Set(initialize=nonroot_nodes)

        m.q = pyo.Param(m.N, initialize={n: nodes[n]["prob"] for n in all_nodes})
        m.price = pyo.Param(m.N, initialize={n: nodes[n]["price"] for n in all_nodes})
        m.tout = pyo.Param(m.N, initialize={n: nodes[n]["tout"] for n in all_nodes})
        m.stage = pyo.Param(m.N, initialize={n: nodes[n]["stage"] for n in all_nodes})

        def occ_init(model, n, r):
            return nodes[n]["occ1"] if r == 1 else nodes[n]["occ2"]

        m.occ = pyo.Param(m.N, m.R, initialize=occ_init)

        # Commanded actions
        m.pc = pyo.Var(m.N, m.R, bounds=(0.0, Pmax))
        m.vb = pyo.Var(m.N, domain=pyo.Binary)

        # Effective actions after overrules/inertia
        m.pf = pyo.Var(m.N, m.R, bounds=(0.0, Pmax))
        m.ve = pyo.Var(m.N, domain=pyo.Binary)

        # States
        m.Temp = pyo.Var(m.N, m.R)
        m.Hum = pyo.Var(m.N)

        # Logic binaries
        m.y_low = pyo.Var(m.N, m.R, domain=pyo.Binary)
        m.y_ok = pyo.Var(m.N, m.R, domain=pyo.Binary)
        m.y_high = pyo.Var(m.N, m.R, domain=pyo.Binary)
        m.z_below_low = pyo.Var(m.N, m.R, domain=pyo.Binary)
        m.start_vent = pyo.Var(m.N, domain=pyo.Binary)

        # Hybrid soft-risk variables
        m.low_risk = pyo.Var(m.N, m.R, domain=pyo.NonNegativeReals)
        m.high_risk = pyo.Var(m.N, m.R, domain=pyo.NonNegativeReals)
        m.hum_risk = pyo.Var(m.N, domain=pyo.NonNegativeReals)
        m.term_dev = pyo.Var(m.L, m.R, domain=pyo.NonNegativeReals)

        # Root state fixing
        m.root_temp1 = pyo.Constraint(expr=m.Temp[root, 1] == T1_0)
        m.root_temp2 = pyo.Constraint(expr=m.Temp[root, 2] == T2_0)
        m.root_hum = pyo.Constraint(expr=m.Hum[root] == H_0)
        m.root_y_low1 = pyo.Constraint(expr=m.y_low[root, 1] == low_override_r1_0)
        m.root_y_low2 = pyo.Constraint(expr=m.y_low[root, 2] == low_override_r2_0)

        m.logic = pyo.ConstraintList()

        # Root threshold detection for OK/high.
        for r in [1, 2]:
            m.logic.add(m.Temp[root, r] >= TOK - BIG_M * (1 - m.y_ok[root, r]))
            m.logic.add(m.Temp[root, r] <= TOK + BIG_M * m.y_ok[root, r])
            m.logic.add(m.Temp[root, r] >= THigh - BIG_M * (1 - m.y_high[root, r]))
            m.logic.add(m.Temp[root, r] <= THigh + BIG_M * m.y_high[root, r])
            m.logic.add(m.Temp[root, r] <= Tlow + BIG_M * (1 - m.z_below_low[root, r]))
            m.logic.add(m.Temp[root, r] >= Tlow - BIG_M * m.z_below_low[root, r])

        # Threshold detection for non-root nodes.
        for n in nonroot_nodes:
            for r in [1, 2]:
                m.logic.add(m.Temp[n, r] >= THigh - BIG_M * (1 - m.y_high[n, r]))
                m.logic.add(m.Temp[n, r] <= THigh + BIG_M * m.y_high[n, r])

                m.logic.add(m.Temp[n, r] >= TOK - BIG_M * (1 - m.y_ok[n, r]))
                m.logic.add(m.Temp[n, r] <= TOK + BIG_M * m.y_ok[n, r])

                m.logic.add(m.Temp[n, r] <= Tlow + BIG_M * (1 - m.z_below_low[n, r]))
                m.logic.add(m.Temp[n, r] >= Tlow - BIG_M * m.z_below_low[n, r])

        # Low-temperature hysteresis.
        for n in nonroot_nodes:
            p = parent[n]
            for r in [1, 2]:
                m.logic.add(m.y_low[n, r] >= m.z_below_low[n, r])
                m.logic.add(m.y_low[n, r] >= m.y_low[p, r] - m.y_ok[n, r])
                m.logic.add(m.y_low[n, r] <= m.z_below_low[n, r] + m.y_low[p, r])
                m.logic.add(m.y_low[n, r] <= m.z_below_low[n, r] + (1 - m.y_ok[n, r]))

        # Effective heating action: high-temp priority, low-temp max, otherwise commanded power.
        for n in all_nodes:
            for r in [1, 2]:
                m.logic.add(m.pf[n, r] <= Pmax * (1 - m.y_high[n, r]))
                m.logic.add(m.pf[n, r] >= Pmax * (m.y_low[n, r] - m.y_high[n, r]))
                m.logic.add(m.pf[n, r] <= m.pc[n, r] + Pmax * (m.y_low[n, r] + m.y_high[n, r]))
                m.logic.add(m.pf[n, r] >= m.pc[n, r] - Pmax * (m.y_low[n, r] + m.y_high[n, r]))

        # Ventilation logic: humidity overrule, commanded ventilation, startup, and inertia.
        for n in all_nodes:
            m.logic.add(m.Hum[n] <= HHigh + BIG_M * m.ve[n])
            m.logic.add(m.ve[n] >= m.vb[n])

            if n == root:
                prev_on = 1 if vent_counter_0 > 0 else 0
                m.logic.add(m.start_vent[n] >= m.ve[n] - prev_on)
                m.logic.add(m.start_vent[n] <= m.ve[n])
                m.logic.add(m.start_vent[n] <= 1 - prev_on)
            else:
                p = parent[n]
                m.logic.add(m.start_vent[n] >= m.ve[n] - m.ve[p])
                m.logic.add(m.start_vent[n] <= m.ve[n])
                m.logic.add(m.start_vent[n] <= 1 - m.ve[p])

        if vent_counter_0 == 1:
            m.force_existing_vent_root = pyo.Constraint(expr=m.ve[root] == 1)
            for c in children[root]:
                m.logic.add(m.ve[c] == 1)
        elif vent_counter_0 == 2:
            m.force_existing_vent_root = pyo.Constraint(expr=m.ve[root] == 1)

        def descendants_with_depth(start_node, max_depth):
            result = []
            frontier = [(start_node, 0)]
            while frontier:
                curr, depth = frontier.pop(0)
                if depth == max_depth:
                    continue
                for ch in children[curr]:
                    result.append((ch, depth + 1))
                    frontier.append((ch, depth + 1))
            return result

        for n in all_nodes:
            for ch, _depth in descendants_with_depth(n, max(0, Uvent - 1)):
                m.logic.add(m.ve[ch] >= m.start_vent[n])

        # Dynamics from parent node to child node.
        m.dynamics = pyo.ConstraintList()
        for n in nonroot_nodes:
            p = parent[n]
            m.dynamics.add(
                m.Temp[n, 1]
                == m.Temp[p, 1]
                + z_exch * (m.Temp[p, 2] - m.Temp[p, 1])
                + z_loss * (m.tout[p] - m.Temp[p, 1])
                + z_conv * m.pf[p, 1]
                - z_cool * m.ve[p]
                + z_occ * m.occ[p, 1]
            )
            m.dynamics.add(
                m.Temp[n, 2]
                == m.Temp[p, 2]
                + z_exch * (m.Temp[p, 1] - m.Temp[p, 2])
                + z_loss * (m.tout[p] - m.Temp[p, 2])
                + z_conv * m.pf[p, 2]
                - z_cool * m.ve[p]
                + z_occ * m.occ[p, 2]
            )
            m.dynamics.add(
                m.Hum[n]
                == m.Hum[p]
                + eta_occ * (m.occ[p, 1] + m.occ[p, 2])
                - eta_vent * m.ve[p]
            )

        # Hybrid soft-risk constraints.
        for n in all_nodes:
            m.logic.add(m.hum_risk[n] >= m.Hum[n] - (HHigh - HUM_BUFFER))
            for r in [1, 2]:
                m.logic.add(m.low_risk[n, r] >= (Tlow + TEMP_LOW_BUFFER) - m.Temp[n, r])
                m.logic.add(m.high_risk[n, r] >= m.Temp[n, r] - (THigh - TEMP_HIGH_BUFFER))

        for n in leaf_nodes:
            for r in [1, 2]:
                m.logic.add(m.term_dev[n, r] >= m.Temp[n, r] - TARGET_TEMP)
                m.logic.add(m.term_dev[n, r] >= TARGET_TEMP - m.Temp[n, r])

        # Objective: expected energy + hybrid risk/regularization terms.
        energy_cost = sum(
            m.q[n] * m.price[n] * (Pvent * m.ve[n] + m.pf[n, 1] + m.pf[n, 2])
            for n in all_nodes
        )
        temp_risk_cost = sum(
            m.q[n]
            * (
                LOW_TEMP_PENALTY * (m.low_risk[n, 1] + m.low_risk[n, 2])
                + HIGH_TEMP_PENALTY * (m.high_risk[n, 1] + m.high_risk[n, 2])
            )
            for n in all_nodes
        )
        hum_risk_cost = sum(HUMIDITY_PENALTY * m.q[n] * m.hum_risk[n] for n in all_nodes)
        terminal_cost = sum(
            TERMINAL_TEMP_PENALTY * m.q[n] * (m.term_dev[n, 1] + m.term_dev[n, 2])
            for n in leaf_nodes
        )
        vent_start_cost = sum(START_VENT_PENALTY * m.q[n] * m.start_vent[n] for n in all_nodes)

        m.obj = pyo.Objective(
            expr=energy_cost + temp_risk_cost + hum_risk_cost + terminal_cost + vent_start_cost,
            sense=pyo.minimize,
        )

        # Solve.
        solver = pyo.SolverFactory("gurobi")
        solver.options["TimeLimit"] = SOLVER_TIME_LIMIT
        solver.options["MIPGap"] = MIP_GAP
        solver.options["OutputFlag"] = 0
        results = solver.solve(m, tee=False)

        # Extract here-and-now commanded action from the root node.
        p1 = pyo.value(m.pc[root, 1])
        p2 = pyo.value(m.pc[root, 2])
        v = pyo.value(m.vb[root])

        if p1 is None or p2 is None or v is None:
            raise RuntimeError("No valid root decision extracted.")

        action = {
            "HeatPowerRoom1": clip(p1, 0.0, Pmax),
            "HeatPowerRoom2": clip(p2, 0.0, Pmax),
            "VentilationON": 1 if float(v) > 0.5 else 0,
        }

        # Final safety overrides on the returned command. The environment should also
        # apply these, but doing it here prevents obviously bad returned values.
        if T1_0 >= THigh:
            action["HeatPowerRoom1"] = 0.0
        if T2_0 >= THigh:
            action["HeatPowerRoom2"] = 0.0
        if low_override_r1_0 == 1 and T1_0 < THigh:
            action["HeatPowerRoom1"] = Pmax
        if low_override_r2_0 == 1 and T2_0 < THigh:
            action["HeatPowerRoom2"] = Pmax
        if H_0 > HHigh or vent_counter_0 in [1, 2]:
            action["VentilationON"] = 1

        return action

    except Exception:
        return fallback_action()


# Optional local smoke test:
# if __name__ == "__main__":
#     test_state = {
#         "T1": 21, "T2": 21, "H": 40,
#         "Occ1": 30, "Occ2": 20,
#         "price_t": 4, "price_previous": 4,
#         "vent_counter": 0,
#         "low_override_r1": 0, "low_override_r2": 0,
#         "current_time": 0,
#     }
#     print(select_action(test_state))
