# -*- coding: utf-8 -*-
"""
Optimal in Hindsight solution for the restaurant HVAC problem.

This file is intended for Task 6 point 3:
    Python code implementing the Optimal in Hindsight Solution of Task 1.

It solves the full-day MILP with perfect knowledge of:
    - the full daily electricity price trajectory,
    - the full daily occupancy trajectory for Room 1,
    - the full daily occupancy trajectory for Room 2.

It is NOT an online policy for Tasks 3-5, because it needs the entire day of
future uncertainty before making decisions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import importlib
import numpy as np
import pandas as pd

import gurobipy as gp
from gurobipy import GRB


# =============================================================================
# Data and parameter loading
# =============================================================================

def load_raw_system_characteristics() -> Dict[str, Any]:
    """
    Load the fixed system data from Data.v2_SystemCharacteristics.

    The course files have sometimes used either get_fixed_data() or fetch_data(),
    so this accepts both.
    """
    module = importlib.import_module("Data.v2_SystemCharacteristics")

    if hasattr(module, "get_fixed_data"):
        return dict(module.get_fixed_data())

    if hasattr(module, "fetch_data"):
        return dict(module.fetch_data())

    raise AttributeError(
        "Data.v2_SystemCharacteristics must define get_fixed_data() or fetch_data()."
    )


def _first(params: Dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    """Return the first existing parameter from a list of possible aliases."""
    for name in names:
        if name in params:
            return params[name]
    return default


def _as_room_dict(value: Any, default: float) -> Dict[int, float]:
    """
    Convert a scalar/list/dict room parameter into a {1: value1, 2: value2} dict.
    """
    if value is None:
        return {1: float(default), 2: float(default)}

    if isinstance(value, dict):
        # Accept keys 1/2 or "1"/"2" or room-specific names if present.
        return {
            1: float(value.get(1, value.get("1", value.get("room1", default)))),
            2: float(value.get(2, value.get("2", value.get("room2", default)))),
        }

    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) >= 2:
            return {1: float(value[0]), 2: float(value[1])}
        if len(value) == 1:
            return {1: float(value[0]), 2: float(value[0])}

    return {1: float(value), 2: float(value)}


def build_oih_params(raw: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Normalize the SystemCharacteristics names into the names used by solve_day_milp().

    This makes the MILP robust to small naming differences between files/releases.
    """
    raw = load_raw_system_characteristics() if raw is None else dict(raw)

    num_timeslots = int(_first(raw, ["num_timeslots", "T", "H"], default=10))

    p_heater_raw = _first(
        raw,
        [
            "P_heater",
            "heating_max_power",
            "P_r",
            "Pr",
            "PowerMax",
            "max_heating_power",
        ],
        default=3.0,
    )
    p_heater = _as_room_dict(p_heater_raw, default=3.0)

    # Allow room-specific values to override scalar/list values.
    if "heating_max_power_room1" in raw:
        p_heater[1] = float(raw["heating_max_power_room1"])
    if "heating_max_power_room2" in raw:
        p_heater[2] = float(raw["heating_max_power_room2"])

    t_init_raw = _first(
        raw,
        ["T_init", "initial_temperature", "initial_temperatures"],
        default=20.0,
    )
    t_init = _as_room_dict(t_init_raw, default=20.0)

    tout = _first(
        raw,
        ["T_out", "Tout", "outdoor_temperature", "outdoor_temperatures"],
        default=[5.0] * num_timeslots,
    )
    if not isinstance(tout, (list, tuple, np.ndarray)):
        tout = [float(tout)] * num_timeslots
    tout = list(np.asarray(tout, dtype=float))
    if len(tout) < num_timeslots:
        tout = tout + [tout[-1]] * (num_timeslots - len(tout))

    return {
        "num_timeslots": num_timeslots,

        "P_heater": p_heater,
        "P_vent": float(_first(raw, ["P_vent", "Pvent", "ventilation_power", "P_ventilation"], default=1.0)),

        "z_exch": float(_first(raw, ["z_exch", "zeta_exch", "heat_exchange_coeff"], default=0.6)),
        "z_loss": float(_first(raw, ["z_loss", "zeta_loss", "thermal_loss_coeff"], default=0.1)),
        "z_conv": float(_first(raw, ["z_conv", "zeta_conv", "heating_efficiency_coeff"], default=1.0)),
        "z_cool": float(_first(raw, ["z_cool", "zeta_cool", "heat_vent_coeff"], default=0.7)),
        "z_occ": float(_first(raw, ["z_occ", "zeta_occ", "heat_occupancy_coeff"], default=0.02)),

        "eta_occ": float(_first(raw, ["eta_occ", "humidity_occupancy_coeff"], default=0.05)),
        "eta_vent": float(_first(raw, ["eta_vent", "humidity_vent_coeff"], default=5.0)),

        "T_low": float(_first(raw, ["T_low", "Tlow", "temperature_low"], default=18.0)),
        "T_ok": float(_first(raw, ["T_ok", "T_OK", "TOK", "temperature_ok"], default=20.0)),
        "T_high": float(_first(raw, ["T_high", "THigh", "temperature_high"], default=24.0)),
        "H_high": float(_first(raw, ["H_high", "Hhigh", "humidity_high"], default=60.0)),

        "T_out": tout,
        "T_init": t_init,
        "H_init": float(_first(raw, ["H_init", "initial_humidity"], default=45.0)),

        "vent_min_up_time": int(_first(raw, ["vent_min_up_time", "U_vent", "Uvent"], default=3)),
    }


def _read_csv_matrix(path: Path) -> np.ndarray:
    """Read a CSV file as a numeric matrix."""
    return pd.read_csv(path, header=None).to_numpy(dtype=float)


def load_historical_data(data_dir: str | Path = "Data") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load historical price and occupancy matrices.

    Returns:
        price, occ1, occ2 as arrays with shape (num_days, num_timeslots).

    The function tries common file names used in the assignment repository.
    """
    data_dir = Path(data_dir)

    candidates = {
        "price": ["v2_PriceData.csv"],
        "occ1": ["OccupancyRoom1.csv"],
        "occ2": ["OccupancyRoom2.csv"],
    }

    def find_file(kind: str) -> Path:
        for filename in candidates[kind]:
            path = data_dir / filename
            if path.exists():
                return path
        raise FileNotFoundError(
            f"Could not find {kind} data in {data_dir}. Tried: {candidates[kind]}"
        )

    price = _read_csv_matrix(find_file("price"))
    occ1 = _read_csv_matrix(find_file("occ1"))
    occ2 = _read_csv_matrix(find_file("occ2"))

    return price, occ1, occ2


# =============================================================================
# Optimal in Hindsight MILP
# =============================================================================

def solve_day_milp(
    prices: np.ndarray,
    occ1_day: np.ndarray,
    occ2_day: np.ndarray,
    params: Dict[str, Any],
    output_flag: int = 0,
) -> Dict[str, np.ndarray | float]:
    """
    Solve the daily Optimal-in-Hindsight MILP for the restaurant.

    Args:
        prices: array of prices for one day, length H.
        occ1_day: array of Room 1 occupancies for one day, length H.
        occ2_day: array of Room 2 occupancies for one day, length H.
        params: normalized parameter dictionary from build_oih_params().
        output_flag: Gurobi OutputFlag.

    Returns:
        Dictionary with objective value and time series:
        obj, p1, p2, v, T1, T2, H.
    """
    prices = np.asarray(prices, dtype=float)
    occ1_day = np.asarray(occ1_day, dtype=float)
    occ2_day = np.asarray(occ2_day, dtype=float)

    H = len(prices)
    R = [1, 2]
    T = range(H)

    m = gp.Model("Restaurant_OIH")
    m.Params.OutputFlag = output_flag

    # Variables
    p = m.addVars(R, T, lb=0.0, name="p")                  # heater power
    v = m.addVars(T, vtype=GRB.BINARY, name="v")           # ventilation ON/OFF

    Temp = m.addVars(R, T, lb=-GRB.INFINITY, name="Temp")  # room temperature
    Hum = m.addVars(T, lb=-GRB.INFINITY, name="Hum")       # humidity

    # Helper binaries
    start = m.addVars(T, vtype=GRB.BINARY, name="start")
    hum_hi = m.addVars(T, vtype=GRB.BINARY, name="hum_hi")
    temp_hi = m.addVars(R, T, vtype=GRB.BINARY, name="temp_hi")

    low_active = m.addVars(R, T, vtype=GRB.BINARY, name="low_active")
    low_trig = m.addVars(R, T, vtype=GRB.BINARY, name="low_trig")
    below_ok = m.addVars(R, T, vtype=GRB.BINARY, name="below_ok")

    # Parameters
    P_heater = params["P_heater"]
    P_vent = params["P_vent"]

    z_exch = params["z_exch"]
    z_loss = params["z_loss"]
    z_conv = params["z_conv"]
    z_cool = params["z_cool"]
    z_occ = params["z_occ"]

    eta_occ = params["eta_occ"]
    eta_vent = params["eta_vent"]

    Tlow = params["T_low"]
    Tok = params["T_ok"]
    Thigh = params["T_high"]
    Hhigh = params["H_high"]

    Tout = np.asarray(params["T_out"], dtype=float)
    Tinit = params["T_init"]
    Hinit = params["H_init"]

    U = int(params.get("vent_min_up_time", 3))

    # Big-M values
    M_T = 100.0
    M_H = 200.0

    # Heater power bounds
    for r in R:
        for t in T:
            m.addConstr(p[r, t] <= P_heater[r], name=f"pmax_{r}_{t}")

    # Initial conditions
    for r in R:
        m.addConstr(Temp[r, 0] == Tinit[r], name=f"Tinit_{r}")
    m.addConstr(Hum[0] == Hinit, name="Hinit")

    # Dynamics
    for t in range(1, H):
        for r in R:
            other = 2 if r == 1 else 1
            occ_rt_1 = occ1_day[t - 1] if r == 1 else occ2_day[t - 1]

            m.addConstr(
                Temp[r, t]
                == Temp[r, t - 1]
                + z_exch * (Temp[other, t - 1] - Temp[r, t - 1])
                + z_loss * (Tout[t - 1] - Temp[r, t - 1])
                + z_conv * p[r, t - 1]
                - z_cool * v[t - 1]
                + z_occ * occ_rt_1,
                name=f"Tdyn_{r}_{t}",
            )

        m.addConstr(
            Hum[t]
            == Hum[t - 1]
            + eta_occ * (occ1_day[t - 1] + occ2_day[t - 1])
            - eta_vent * v[t - 1],
            name=f"Hdyn_{t}",
        )

    # Ventilation start indicator, assuming v[-1] = 0
    for t in T:
        if t == 0:
            m.addConstr(start[0] >= v[0], name="start0_lb")
            m.addConstr(start[0] <= v[0], name="start0_ub")
        else:
            m.addConstr(start[t] >= v[t] - v[t - 1], name=f"start_lb_{t}")
            m.addConstr(start[t] <= v[t], name=f"start_ub1_{t}")
            m.addConstr(start[t] <= 1 - v[t - 1], name=f"start_ub2_{t}")

    # Ventilation inertia: if started, remain ON for U hours
    for t in T:
        for k in range(U):
            if t + k < H:
                m.addConstr(v[t + k] >= start[t], name=f"minup_{t}_{k}")

    # Humidity override: if humidity exceeds threshold, ventilation is forced ON.
    for t in T:
        m.addConstr(Hum[t] >= Hhigh - M_H * (1 - hum_hi[t]), name=f"humhi_lb_{t}")
        m.addConstr(Hum[t] <= Hhigh + M_H * hum_hi[t], name=f"humhi_ub_{t}")
        m.addConstr(v[t] >= hum_hi[t], name=f"hum_force_{t}")

    # High-temperature shutoff: if temperature exceeds upper threshold, heater is OFF.
    for r in R:
        for t in T:
            m.addConstr(Temp[r, t] >= Thigh - M_T * (1 - temp_hi[r, t]), name=f"thi_lb_{r}_{t}")
            m.addConstr(Temp[r, t] <= Thigh + M_T * temp_hi[r, t], name=f"thi_ub_{r}_{t}")
            m.addConstr(p[r, t] <= P_heater[r] * (1 - temp_hi[r, t]), name=f"p_off_hi_{r}_{t}")

    # Low-temperature override with hysteresis: if triggered, stay at max until T >= T_ok.
    for r in R:
        for t in T:
            # Cannot force max heating and force shutoff simultaneously.
            m.addConstr(low_active[r, t] + temp_hi[r, t] <= 1, name=f"mutual_excl_{r}_{t}")

            # low_trig indicates Temp <= Tlow.
            m.addConstr(Temp[r, t] <= Tlow + M_T * (1 - low_trig[r, t]), name=f"lowtrig_ub_{r}_{t}")
            m.addConstr(Temp[r, t] >= Tlow - M_T * low_trig[r, t], name=f"lowtrig_lb_{r}_{t}")

            # below_ok indicates Temp <= Tok.
            m.addConstr(Temp[r, t] <= Tok + M_T * (1 - below_ok[r, t]), name=f"belowok_ub_{r}_{t}")
            m.addConstr(Temp[r, t] >= Tok - M_T * below_ok[r, t], name=f"belowok_lb_{r}_{t}")

            if t == 0:
                m.addConstr(low_active[r, t] >= low_trig[r, t], name=f"lowact0_{r}")
                m.addConstr(low_active[r, t] <= low_trig[r, t], name=f"lowact0_ub_{r}")
            else:
                cont = m.addVar(vtype=GRB.BINARY, name=f"cont_{r}_{t}")
                m.addConstr(cont <= low_active[r, t - 1], name=f"cont1_{r}_{t}")
                m.addConstr(cont <= below_ok[r, t], name=f"cont2_{r}_{t}")
                m.addConstr(cont >= low_active[r, t - 1] + below_ok[r, t] - 1, name=f"cont3_{r}_{t}")

                m.addConstr(low_active[r, t] >= low_trig[r, t], name=f"lowact_trig_{r}_{t}")
                m.addConstr(low_active[r, t] >= cont, name=f"lowact_cont_{r}_{t}")
                m.addConstr(low_active[r, t] <= low_trig[r, t] + cont, name=f"lowact_ub_{r}_{t}")

            # If low_active = 1, force heater to max.
            m.addConstr(p[r, t] >= P_heater[r] * low_active[r, t], name=f"p_on_low_{r}_{t}")

    # Objective: total electricity cost
    m.setObjective(
        gp.quicksum(prices[t] * (p[1, t] + p[2, t] + P_vent * v[t]) for t in T),
        GRB.MINIMIZE,
    )

    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Model not optimal. Gurobi status = {m.Status}")

    return {
        "obj": float(m.ObjVal),
        "p1": np.array([p[1, t].X for t in T]),
        "p2": np.array([p[2, t].X for t in T]),
        "v": np.array([v[t].X for t in T]),
        "T1": np.array([Temp[1, t].X for t in T]),
        "T2": np.array([Temp[2, t].X for t in T]),
        "H": np.array([Hum[t].X for t in T]),
    }


def solve_all_days(
    price: np.ndarray,
    occ1: np.ndarray,
    occ2: np.ndarray,
    params: Dict[str, Any] | None = None,
    output_flag: int = 0,
) -> Tuple[list[Dict[str, np.ndarray | float]], np.ndarray, float]:
    """
    Solve the Optimal-in-Hindsight MILP for every historical day.

    Returns:
        solutions, daily_costs, average_daily_cost
    """
    params = build_oih_params() if params is None else params

    n_days = min(price.shape[0], occ1.shape[0], occ2.shape[0])
    solutions = []
    costs = np.zeros(n_days)

    for d in range(n_days):
        sol = solve_day_milp(price[d, :], occ1[d, :], occ2[d, :], params, output_flag=output_flag)
        solutions.append(sol)
        costs[d] = sol["obj"]

    return solutions, costs, float(np.mean(costs))


# =============================================================================
# Optional script entry point
# =============================================================================

if __name__ == "__main__":
    params = build_oih_params()
    price, occ1, occ2 = load_historical_data("Data")

    solutions, daily_costs, avg_cost = solve_all_days(price, occ1, occ2, params)

    print(f"Solved {len(daily_costs)} days.")
    print(f"Average daily Optimal-in-Hindsight cost: {avg_cost:.4f}")
