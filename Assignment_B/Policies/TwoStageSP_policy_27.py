"""Two-stage stochastic programming policy.

This module copies `SP_policy_27` but with a fixed two-stage
lookahead so the comparison runner can treat it as a separate policy.
"""


import numpy as np
import pyomo.environ as pyo

import Data.PriceProcessRestaurant
import Data.OccupancyProcessRestaurant
import Data.v2_SystemCharacteristics as SystemCharacteristics


def select_action(state):

    # =========================================================
    # Fixed problem data
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
    # Policy tuning parameters
    # =========================================================
    LOOKAHEAD = 2  # total number of stages including current stage
    INITIAL_SAMPLES = 30       # raw samples per node before reduction
    REDUCED_BRANCHES = 3      # reduced children per non-leaf node
    BIG_M = 100.0
    SOLVER_TIME_LIMIT = 6
    MIP_GAP = 0.03



    # =========================================================
    # Helpers
    # =========================================================
    def safe_float(x, default=0.0) -> float:
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
        

    def weighted_distance(a, b) -> float:
        """
        Distance for clustering uncertainty samples:
        a, b = (price, occ1, occ2)
        Weights keep occupancy and price on comparable scales.

        Parameters:
            a, b: tuples containing (price, occ1, occ2)

        Returns:
            float: weighted distance
        """
        price_range = 12.0
        occ1_range = 30.0  # 20 to 50
        occ2_range = 20.0  # 10 to 30

        return (
            ((a[0] - b[0]) / price_range) ** 2
            + ((a[1] - b[1]) / occ1_range) ** 2
            + ((a[2] - b[2]) / occ2_range) ** 2
        )
    

    def reduce_samples_kmeans(samples, k, n_iter=8) -> tuple[list[tuple[float, float, float]], list[float]]:
        """
        K-means clustering for sample reduction.

        Parameters:
            samples: list of tuples (price, occ1, occ2)
            k: number of clusters
            n_iter: number of iterations for k-means

        Returns:
            tuple: (centers, probs)
        """
        n = len(samples)
        if n == 0:
            return [], []
        if n <= k:
            prob = 1.0 / n
            return samples[:], [prob for _ in range(n)]

        # Initialize centroids from random samples
        idx = np.random.choice(n, size=k, replace=False)
        centroids = [samples[i] for i in idx]

        for _ in range(n_iter):
            clusters = [[] for _ in range(k)]

            for s in samples:
                j = min(range(k), key=lambda c: weighted_distance(s, centroids[c]))
                clusters[j].append(s)

            new_centroids = []
            for j in range(k):
                if len(clusters[j]) == 0:
                    new_centroids.append(samples[np.random.randint(0, n)])
                else:
                    arr = np.array(clusters[j], dtype=float)
                    new_centroids.append(tuple(np.mean(arr, axis=0)))
            centroids = new_centroids

        # Final assignment for probabilities
        final_clusters = [[] for _ in range(k)]
        for s in samples:
            j = min(range(k), key=lambda c: weighted_distance(s, centroids[c]))
            final_clusters[j].append(s)

        centers = []
        probs = []
        for j in range(k):
            if len(final_clusters[j]) == 0:
                continue
            arr = np.array(final_clusters[j], dtype=float)
            centers.append(tuple(np.mean(arr, axis=0)))
            probs.append(len(final_clusters[j]) / n)

        # Normalize just in case
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]

        return centers, probs


    def simulate_one_step(price_t, price_prev, occ1_t, occ2_t) -> tuple[float, float, float]:
        """
        One-step sampling from the provided stochastic processes.

        Parameters:
            price_t: current price
            price_prev: previous price
            occ1_t: current occupancy of room 1
            occ2_t: current occupancy of room 2

        Returns:
            tuple: (next_price, next_occ1, next_occ2)
        """
        next_price = Data.PriceProcessRestaurant.price_model(price_t, price_prev)
        next_occ1, next_occ2 = Data.OccupancyProcessRestaurant.next_occupancy_levels(occ1_t, occ2_t)
        return float(next_price), float(next_occ1), float(next_occ2)

    # =========================================================
    # Read observed current state
    # =========================================================
    T1_0 = safe_float(state.get("T1", 21.0), 21.0)
    T2_0 = safe_float(state.get("T2", 21.0), 21.0)
    H_0 = safe_float(state.get("H", 40.0), 40.0)

    Occ1_0 = safe_float(state.get("Occ1", 30.0), 30.0)
    Occ2_0 = safe_float(state.get("Occ2", 20.0), 20.0)

    price_0 = safe_float(state.get("price_t", 4.0), 4.0)
    price_prev_0 = safe_float(state.get("price_previous", 4.0), 4.0)

    vent_counter_0 = int(round(safe_float(state.get("vent_counter", 0), 0)))

    # The low-temperature overrule controller has memory:
    # if a room previously dropped below Tlow, the heater remains forced ON
    # until the room temperature reaches TOK. Therefore, this latch status
    # must be read from the environment state and cannot be inferred from
    # the current temperature alone when Tlow <= T < TOK.
    low_override_r1_0 = 1 if safe_float(state.get("low_override_r1", 0), 0) > 0.5 else 0
    low_override_r2_0 = 1 if safe_float(state.get("low_override_r2", 0), 0) > 0.5 else 0

    # Safety correction for inconsistent states:
    if T1_0 < Tlow:
        low_override_r1_0 = 1

    if T2_0 < Tlow:
        low_override_r2_0 = 1

    current_time = int(round(safe_float(state.get("current_time", 0), 0)))

    # =========================================================
    # Build reduced scenario tree
    #
    # Nodes store only exogenous information:
    # - price
    # - occupancy room 1
    # - occupancy room 2
    # - outdoor temperature
    # plus probability, parent, stage, and short histories needed
    # for future sampling.
    #
    # Endogenous states (Temp, Hum) are optimized in the MILP.
    # =========================================================
    nodes = {}
    children = {}
    parent = {}

    next_node_id = 0

    # Root node (current observed state)
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

    # For each non-leaf stage, generate raw samples then reduce to branches
    for stage in range(0, LOOKAHEAD - 1):
        stage_nodes[stage + 1] = []

        for n in stage_nodes[stage]:
            raw_samples = []

            for _ in range(INITIAL_SAMPLES):
                sample_price, sample_occ1, sample_occ2 = simulate_one_step(
                    nodes[n]["price_curr"],
                    nodes[n]["price_prev"],
                    nodes[n]["occ1_curr"],
                    nodes[n]["occ2_curr"],
                )
                raw_samples.append((sample_price, sample_occ1, sample_occ2))

            centers, probs = reduce_samples_kmeans(raw_samples, REDUCED_BRANCHES)

            for c_idx in range(len(centers)):
                child = next_node_id
                next_node_id += 1

                c_price, c_occ1, c_occ2 = centers[c_idx]
                c_prob = float(probs[c_idx])

                time_index_child = nodes[n]["time_index"] + 1

                nodes[child] = {
                    "stage": stage + 1,
                    "prob": nodes[n]["prob"] * c_prob,
                    "price": float(c_price),
                    "occ1": float(c_occ1),
                    "occ2": float(c_occ2),
                    "tout": Tout[time_index_child % num_timeslots],
                    "price_prev": nodes[n]["price_curr"],
                    "price_curr": float(c_price),
                    "occ1_curr": float(c_occ1),
                    "occ2_curr": float(c_occ2),
                    "time_index": time_index_child,
                    "branch_prob": c_prob,
                }

                parent[child] = n
                if n not in children:
                    children[n] = []
                children[n].append(child)
                children[child] = []

                stage_nodes[stage + 1].append(child)

    all_nodes = list(nodes.keys())
    leaf_nodes = [n for n in all_nodes if len(children[n]) == 0]
    nonroot_nodes = [n for n in all_nodes if n != root]



    # =========================================================
    # Build stochastic MILP on tree nodes
    # Node-based decisions => non-anticipativity by construction
    # =========================================================
    
    m = pyo.ConcreteModel()

    m.N = pyo.Set(initialize=all_nodes)
    m.R = pyo.Set(initialize=[1, 2])
    m.L = pyo.Set(initialize=leaf_nodes)
    m.NNR = pyo.Set(initialize=nonroot_nodes)

    # Scenario-tree node probabilities and exogenous data
    m.rho = pyo.Param(m.N, initialize={n: nodes[n]["prob"] for n in all_nodes})
    m.lam = pyo.Param(m.N, initialize={n: nodes[n]["price"] for n in all_nodes})
    m.occ = pyo.Param(
        m.N, m.R,
        initialize={(n, 1): nodes[n]["occ1"] for n in all_nodes} |
                    {(n, 2): nodes[n]["occ2"] for n in all_nodes}
    )
    m.Tout = pyo.Param(m.N, initialize={n: nodes[n]["tout"] for n in all_nodes})
    m.stage = pyo.Param(m.N, initialize={n: nodes[n]["stage"] for n in all_nodes})



    # -----------------------------------------------------
    # Decision variables at each node
    # -----------------------------------------------------
    m.q = pyo.Var(m.N, m.R, bounds=(0.0, Pmax))     # controllable heater power q_{r,n}
    m.vc = pyo.Var(m.N, domain=pyo.Binary)           # commanded ventilation v^c_n

    # Effective actions after overrules/inertia
    m.p = pyo.Var(m.N, m.R, bounds=(0.0, Pmax))
    m.v = pyo.Var(m.N, domain=pyo.Binary)

    # State variables
    m.T = pyo.Var(m.N, m.R)
    m.H = pyo.Var(m.N)

    # Binary logic
    m.u = pyo.Var(m.N, m.R, domain=pyo.Binary)
    m.b = pyo.Var(m.N, m.R, domain=pyo.Binary)
    m.w = pyo.Var(m.N, m.R, domain=pyo.Binary)
    m.a = pyo.Var(m.N, m.R, domain=pyo.Binary)
    m.s = pyo.Var(m.N, domain=pyo.Binary)



    # -----------------------------------------------------
    # Root state fixing
    # -----------------------------------------------------
    m.root_temp1 = pyo.Constraint(expr=m.T[root, 1] == T1_0)
    m.root_temp2 = pyo.Constraint(expr=m.T[root, 2] == T2_0)
    m.root_hum = pyo.Constraint(expr=m.H[root] == H_0)

    m.root_y_low1 = pyo.Constraint(expr=m.u[root, 1] == low_override_r1_0)
    m.root_y_low2 = pyo.Constraint(expr=m.u[root, 2] == low_override_r2_0)

    # Detect root OK / High conditions from observed temperature
    m.root_y_ok_lb = pyo.ConstraintList()
    m.root_y_ok_ub = pyo.ConstraintList()
    m.root_y_high_lb = pyo.ConstraintList()
    m.root_y_high_ub = pyo.ConstraintList()

    for r in [1, 2]:
        m.root_y_ok_lb.add(m.T[root, r] >= TOK - BIG_M * (1 - m.b[root, r]))
        m.root_y_ok_ub.add(m.T[root, r] <= TOK + BIG_M * m.b[root, r])

        m.root_y_high_lb.add(m.T[root, r] >= THigh - BIG_M * (1 - m.w[root, r]))
        m.root_y_high_ub.add(m.T[root, r] <= THigh + BIG_M * m.w[root, r])



    # -----------------------------------------------------
    # Threshold detection for non-root nodes
    # -----------------------------------------------------
    m.logic_cons = pyo.ConstraintList()

    for n in nonroot_nodes:
        for r in [1, 2]:
            # T >= THigh <=> y_high = 1
            m.logic_cons.add(m.T[n, r] >= THigh - BIG_M * (1 - m.w[n, r]))
            m.logic_cons.add(m.T[n, r] <= THigh + BIG_M * m.w[n, r])

            # T >= TOK <=> y_ok = 1
            m.logic_cons.add(m.T[n, r] >= TOK - BIG_M * (1 - m.b[n, r]))
            m.logic_cons.add(m.T[n, r] <= TOK + BIG_M * m.b[n, r])

            # T <= Tlow <=> z_below_low = 1
            m.logic_cons.add(m.T[n, r] <= Tlow + BIG_M * (1 - m.a[n, r]))
            m.logic_cons.add(m.T[n, r] >= Tlow - BIG_M * m.a[n, r])

    # -----------------------------------------------------
    # Low-temperature hysteresis update on the tree
    #
    # At node n:
    # u[n] = 1 if T[n] <= Tlow
    #         or if u[parent(n)] = 1 and T[n] < TOK
    #         else 0
    # -----------------------------------------------------
    for n in nonroot_nodes:
        p = parent[n]
        for r in [1, 2]:
            m.logic_cons.add(m.u[n, r] >= m.a[n, r])
            m.logic_cons.add(m.u[n, r] >= m.u[p, r] - m.b[n, r])
            m.logic_cons.add(m.u[n, r] <= m.a[n, r] + m.u[p, r])
            m.logic_cons.add(m.u[n, r] <= m.a[n, r] + (1 - m.b[n, r]))

    # -----------------------------------------------------
    # Effective heating action
    #
    # High-temp priority:
    # if w = 1 => p = 0
    # elif u = 1 => p = Pmax
    # else p = q
    # -----------------------------------------------------
    m.heat_logic = pyo.ConstraintList()
    for n in all_nodes:
        for r in [1, 2]:
            # high-temp forces off
            m.heat_logic.add(m.p[n, r] <= Pmax * (1 - m.w[n, r]))

            # low-temp override forces max unless high-temp also active
            m.heat_logic.add(m.p[n, r] >= Pmax * (m.u[n, r] - m.w[n, r]))

            # if neither override is active, pf should equal pc
            m.heat_logic.add(
                m.p[n, r] <= m.q[n, r] + Pmax * (m.u[n, r] + m.w[n, r])
            )
            m.heat_logic.add(
                m.p[n, r] >= m.q[n, r] - Pmax * (m.u[n, r] + m.w[n, r])
            )

    # -----------------------------------------------------
    # Ventilation logic
    #
    # Humidity overrule:
    # if H > HHigh => v = 1
    #
    # Inertia:
    # if ventilation starts at node n, it must remain ON
    # on descendants up to Uvent periods as allowed by the tree.
    # -----------------------------------------------------
    m.vnt_logic = pyo.ConstraintList()

    # humidity-triggered ON
    for n in all_nodes:
        m.vnt_logic.add(m.H[n] <= HHigh + BIG_M * m.v[n])

    # effective vent must be at least commanded vent
    for n in all_nodes:
        m.vnt_logic.add(m.v[n] >= m.vc[n])

    # startup detection
    for n in all_nodes:
        if n == root:
            prev_on = 1 if vent_counter_0 > 0 else 0
            m.vnt_logic.add(m.s[n] >= m.v[n] - prev_on)
            m.vnt_logic.add(m.s[n] <= m.v[n])
            m.vnt_logic.add(m.s[n] <= 1 - prev_on)
        else:
            p = parent[n]
            m.vnt_logic.add(m.s[n] >= m.v[n] - m.v[p])
            m.vnt_logic.add(m.s[n] <= m.v[n])
            m.vnt_logic.add(m.s[n] <= 1 - m.v[p])

    # existing inertia from current observed vent_counter
    # counter 1 => ON now and next step
    # counter 2 => ON now
    if vent_counter_0 == 1:
        m.force_counter_root = pyo.Constraint(expr=m.v[root] == 1)
        for c in children[root]:
            m.vnt_logic.add(m.v[c] == 1)
    elif vent_counter_0 == 2:
        m.force_counter_root = pyo.Constraint(expr=m.v[root] == 1)

    # minimum up-time along descendants
    def descendants_with_depth(start_node, max_depth):
        result = []
        frontier = [(start_node, 0)]
        while len(frontier) > 0:
            curr, depth = frontier.pop(0)
            if depth == max_depth:
                continue
            for ch in children[curr]:
                result.append((ch, depth + 1))
                frontier.append((ch, depth + 1))
        return result

    for n in all_nodes:
        desc = descendants_with_depth(n, Uvent - 1)
        for ch, depth in desc:
            m.vnt_logic.add(m.v[ch] >= m.s[n])

    # -----------------------------------------------------
    # Dynamics from parent node -> child node
    # -----------------------------------------------------
    m.dynamics = pyo.ConstraintList()

    for n in nonroot_nodes:
        p = parent[n]

        # room 1
        m.dynamics.add(
            m.T[n, 1] ==
            m.T[p, 1]
            + z_exch * (m.T[p, 2] - m.T[p, 1])
            + z_loss * (m.Tout[p] - m.T[p, 1])
            + z_conv * m.p[p, 1]
            - z_cool * m.v[p]
            + z_occ * m.occ[p, 1]
        )

        # room 2
        m.dynamics.add(
            m.T[n, 2] ==
            m.T[p, 2]
            + z_exch * (m.T[p, 1] - m.T[p, 2])
            + z_loss * (m.Tout[p] - m.T[p, 2])
            + z_conv * m.p[p, 2]
            - z_cool * m.v[p]
            + z_occ * m.occ[p, 2]
        )

        # humidity
        m.dynamics.add(
            m.H[n] ==
            m.H[p]
            + eta_occ * (m.occ[p, 1] + m.occ[p, 2])
            - eta_vent * m.v[p]
        )



    # -----------------------------------------------------
    # Objective: expected energy cost over nodes
    # plus small leaf penalty to reduce myopia
    # -----------------------------------------------------
    energy_cost = sum(
        m.rho[n] * m.lam[n] * (
            Pvent * m.v[n] + m.p[n, 1] + m.p[n, 2]
        )
        for n in all_nodes
    )

    m.obj = pyo.Objective(expr=energy_cost, sense=pyo.minimize)

    # -----------------------------------------------------
    # Solve
    # -----------------------------------------------------
    solver = pyo.SolverFactory("gurobi")
    solver.options["TimeLimit"] = SOLVER_TIME_LIMIT
    solver.options["MIPGap"] = MIP_GAP
    solver.options["OutputFlag"] = 0

    results = solver.solve(m, tee=False)

    term_cond = str(results.solver.termination_condition).lower()

    if ("optimal" not in term_cond) and ("feasible" not in term_cond):
        raise RuntimeError(f"Solver did not return usable solution: {term_cond}")

    # ----------------------------------------------
    # Extract here-and-now action from root node
    # -----------------------------------------------------
    p1 = pyo.value(m.p[root, 1])
    p2 = pyo.value(m.p[root, 2])

    # Return the commanded ventilation decision.
    v = pyo.value(m.v[root])

    if p1 is None or p2 is None or v is None:
        raise RuntimeError("No valid decision extracted from model.")

    HereAndNowActions = {
        "HeatPowerRoom1": float(max(0.0, min(Pmax, p1))),
        "HeatPowerRoom2": float(max(0.0, min(Pmax, p2))),
        "VentilationON": 1 if float(v) > 0.5 else 0
    }

    return HereAndNowActions
