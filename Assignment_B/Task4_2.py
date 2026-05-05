import numpy as np
import pyomo.environ as pyo

import Data.v2_SystemCharacteristics as SystemCharacteristics
import Data.PriceProcessRestaurant as PriceProcessRestaurant
import Data.OccupancyProcessRestaurant as OccupancyProcessRestaurant


# Paste trained coefficients from train_adp_value_function.py here.
# Example placeholder only:
ADP_THETA = {
    0: [0.0] * 16,
    1: [0.0] * 16,
    2: [0.0] * 16,
    3: [0.0] * 16,
    4: [0.0] * 16,
    5: [0.0] * 16,
    6: [0.0] * 16,
    7: [0.0] * 16,
    8: [0.0] * 16,
    9: [0.0] * 16,
}


def select_action(state):

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

    Tout = list(params["outdoor_temperature"])
    T_HORIZON = int(params["num_timeslots"])

    N_NEXT_SAMPLES = 10
    SOLVER_TIME_LIMIT = 4

    def safe_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return float(default)

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

    # Safety correction for the latch.
    if T1 < Tlow:
        low1 = 1
    if T2 < Tlow:
        low2 = 1

    theta_next = ADP_THETA.get(current_time + 1, [0.0] * 16)

    # ---------------------------------------------------------
    # Generate one-step exogenous samples
    # ---------------------------------------------------------
    samples = []
    for _ in range(N_NEXT_SAMPLES):
        next_price = PriceProcessRestaurant.price_model(price, price_prev)
        next_occ1, next_occ2 = OccupancyProcessRestaurant.next_occupancy_levels(occ1, occ2)
        samples.append((float(next_price), float(next_occ1), float(next_occ2)))

    prob = 1.0 / N_NEXT_SAMPLES

    try:
        m = pyo.ConcreteModel()

        m.K = pyo.RangeSet(0, N_NEXT_SAMPLES - 1)
        m.R = pyo.Set(initialize=[1, 2])

        # Commanded actions
        m.pc = pyo.Var(m.R, bounds=(0.0, Pmax))
        m.vb = pyo.Var(domain=pyo.Binary)

        # Effective actions after overrules/inertia
        m.pf = pyo.Var(m.R, bounds=(0.0, Pmax))
        m.ve = pyo.Var(domain=pyo.Binary)

        # Next physical states for each uncertainty sample
        m.T1_next = pyo.Var(m.K)
        m.T2_next = pyo.Var(m.K)
        m.H_next = pyo.Var(m.K)

        m.cons = pyo.ConstraintList()

        # Effective heating logic at the current state
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

        # Effective ventilation logic at the current state
        if H > HHigh or vent_counter > 0:
            m.cons.add(m.ve == 1)
        else:
            m.cons.add(m.ve >= m.vb)

        # One-step dynamics under each exogenous sample
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

        immediate_cost = price * (m.pf[1] + m.pf[2] + Pvent * m.ve)

        # Approximate future cost.
        # This uses a linear subset of the value-function features:
        # [constant, price_next, price_current, occ1_next, occ2_next,
        #  H_next, T1_next, T2_next].
        future_value = 0.0
        for k, (price_next, occ1_next, occ2_next) in enumerate(samples):
            future_value += prob * (
                theta_next[0]
                + theta_next[1] * price_next
                + theta_next[2] * price
                + theta_next[3] * occ1_next
                + theta_next[4] * occ2_next
                + theta_next[5] * m.H_next[k]
                + theta_next[7] * m.T1_next[k]
                + theta_next[8] * m.T2_next[k]
            )

        m.obj = pyo.Objective(expr=immediate_cost + future_value, sense=pyo.minimize)

        solver = pyo.SolverFactory("gurobi")
        solver.options["OutputFlag"] = 0
        solver.options["TimeLimit"] = SOLVER_TIME_LIMIT
        solver.solve(m, tee=False)

        p1 = pyo.value(m.pc[1])
        p2 = pyo.value(m.pc[2])
        v = pyo.value(m.vb)

        if p1 is None or p2 is None or v is None:
            raise RuntimeError("No valid ADP decision extracted.")

        HereAndNowActions = {
            "HeatPowerRoom1": float(max(0.0, min(Pmax, p1))),
            "HeatPowerRoom2": float(max(0.0, min(Pmax, p2))),
            "VentilationON": 1 if float(v) > 0.5 else 0,
        }

    except Exception:
        # Safe fallback
        p1_fb = Pmax if low1 == 1 and T1 < THigh else 0.0
        p2_fb = Pmax if low2 == 1 and T2 < THigh else 0.0
        v_fb = 1 if H > HHigh or vent_counter > 0 else 0

        HereAndNowActions = {
            "HeatPowerRoom1": float(p1_fb),
            "HeatPowerRoom2": float(p2_fb),
            "VentilationON": int(v_fb),
        }

    return HereAndNowActions