import gurobipy as gp
from gurobipy import GRB

def solve_day_milp(prices, occ1_day, occ2_day, params, output_flag=0) -> gp.Model:

    """
    Solve the daily MILP problem for the restaurant.

    Parameters:
    - prices: Daily prices for the restaurant
    - occ1_day: Occupancy for zone 1
    - occ2_day: Occupancy for zone 2
    - params: Model parameters
    - output_flag: Gurobi output flag

    Returns:
    - m: Gurobi model
    """

    import numpy as np

    H = len(prices)
    R = [1, 2]
    T = range(H)

    m = gp.Model("Restaurant_OIH")
    m.Params.OutputFlag = output_flag

    # Variables
    p = m.addVars(R, T, lb=0.0, name="p")                 # heater power
    v = m.addVars(T, vtype=GRB.BINARY, name="v")          # ventilation

    Temp = m.addVars(R, T, lb=-GRB.INFINITY, name="Temp") # temperature
    Hum  = m.addVars(T, lb=-GRB.INFINITY, name="Hum")     # humidity

    # Helper binaries
    start   = m.addVars(T, vtype=GRB.BINARY, name="start")     # ventilation start
    hum_hi  = m.addVars(T, vtype=GRB.BINARY, name="hum_hi")    # humidity above threshold
    temp_hi = m.addVars(R, T, vtype=GRB.BINARY, name="temp_hi")# temp above T_high

    # Low-temp override with hysteresis
    low_active = m.addVars(R, T, vtype=GRB.BINARY, name="low_active")
    low_trig   = m.addVars(R, T, vtype=GRB.BINARY, name="low_trig")
    below_ok   = m.addVars(R, T, vtype=GRB.BINARY, name="below_ok")

    # Params
    P_heater = params["P_heater"]
    P_vent   = params["P_vent"]

    z_exch = params["z_exch"]
    z_loss = params["z_loss"]
    z_conv = params["z_conv"]
    z_cool = params["z_cool"]
    z_occ  = params["z_occ"]

    eta_occ  = params["eta_occ"]
    eta_vent = params["eta_vent"]

    Tlow  = params["T_low"]
    Tok   = params["T_ok"]
    Thigh = params["T_high"]
    Hhigh = params["H_high"]

    Tout  = np.array(params["T_out"])
    Tinit = params["T_init"]
    Hinit = params["H_init"]

    U = int(params.get("vent_min_up_time", 3))

    # Big-M (safe; can be tightened later)
    M_T = 100.0
    M_H = 200.0

    # Bounds on heater power
    for r in R:
        for t in T:
            m.addConstr(p[r,t] <= P_heater[r], name=f"pmax_{r}_{t}")

    # Initial conditions
    for r in R:
        m.addConstr(Temp[r,0] == Tinit[r], name=f"Tinit_{r}")
    m.addConstr(Hum[0] == Hinit, name="Hinit")

    # Dynamics
    for t in range(1, H):
        for r in R:
            other = 2 if r == 1 else 1
            occ_rt_1 = occ1_day[t-1] if r == 1 else occ2_day[t-1]

            m.addConstr(
                Temp[r,t] ==
                Temp[r,t-1]
                + z_exch*(Temp[other,t-1] - Temp[r,t-1])
                + z_loss*(Tout[t] - Temp[r,t-1])
                + z_conv*p[r,t-1]
                - z_cool*v[t-1]
                + z_occ*occ_rt_1,
                name=f"Tdyn_{r}_{t}"
            )

        m.addConstr(
            Hum[t] ==
            Hum[t-1]
            + eta_occ*(occ1_day[t-1] + occ2_day[t-1])
            - eta_vent*v[t-1],
            name=f"Hdyn_{t}"
        )

    # Ventilation start indicator (assume v[-1]=0)
    for t in T:
        if t == 0:
            m.addConstr(start[t] >= v[t], name="start0")
        else:
            m.addConstr(start[t] >= v[t] - v[t-1], name=f"start_{t}")

    # Ventilation inertia (min up-time U)
    for t in T:
        for k in range(U):
            if t+k < H:
                m.addConstr(v[t+k] >= start[t], name=f"minup_{t}_{k}")

    # Humidity override: if humidity exceeds threshold -> ventilation ON
    for t in T:
        m.addConstr(Hum[t] >= Hhigh - M_H*(1 - hum_hi[t]), name=f"humhi_lb_{t}")
        m.addConstr(Hum[t] <= Hhigh + M_H*(hum_hi[t]), name=f"humhi_ub_{t}")
        m.addConstr(v[t] >= hum_hi[t], name=f"hum_force_{t}")

    # High temperature shutoff: if Temp > Thigh then p=0
    for r in R:
        for t in T:
            m.addConstr(Temp[r,t] >= Thigh - M_T*(1 - temp_hi[r,t]), name=f"thi_lb_{r}_{t}")
            m.addConstr(Temp[r,t] <= Thigh + M_T*(temp_hi[r,t]), name=f"thi_ub_{r}_{t}")
            m.addConstr(p[r,t] <= P_heater[r]*(1 - temp_hi[r,t]), name=f"p_off_hi_{r}_{t}")

    # Low temperature override with hysteresis: if triggered, stay at max until Temp >= Tok
    for r in R:
        for t in T:
            # low_trig indicates Temp below Tlow
            m.addConstr(Temp[r,t] <= Tlow + M_T*(1 - low_trig[r,t]), name=f"lowtrig_ub_{r}_{t}")
            m.addConstr(Temp[r,t] >= Tlow - M_T*(low_trig[r,t]), name=f"lowtrig_lb_{r}_{t}")

            # below_ok indicates Temp below Tok
            m.addConstr(Temp[r,t] <= Tok + M_T*(1 - below_ok[r,t]), name=f"belowok_ub_{r}_{t}")
            m.addConstr(Temp[r,t] >= Tok - M_T*(below_ok[r,t]), name=f"belowok_lb_{r}_{t}")

            if t == 0:
                m.addConstr(low_active[r,t] >= low_trig[r,t], name=f"lowact0_{r}")
            else:
                cont = m.addVar(vtype=GRB.BINARY, name=f"cont_{r}_{t}")
                m.addConstr(cont <= low_active[r,t-1], name=f"cont1_{r}_{t}")
                m.addConstr(cont <= below_ok[r,t],     name=f"cont2_{r}_{t}")
                m.addConstr(cont >= low_active[r,t-1] + below_ok[r,t] - 1, name=f"cont3_{r}_{t}")

                m.addConstr(low_active[r,t] >= low_trig[r,t], name=f"lowact_trig_{r}_{t}")
                m.addConstr(low_active[r,t] >= cont,          name=f"lowact_cont_{r}_{t}")
                m.addConstr(low_active[r,t] <= low_trig[r,t] + cont, name=f"lowact_ub_{r}_{t}")

            # If low_active=1, force heater to max (since p <= Pmax already)
            m.addConstr(p[r,t] >= P_heater[r]*low_active[r,t], name=f"p_on_low_{r}_{t}")

    # Objective: electricity cost
    m.setObjective(
        gp.quicksum(prices[t] * (p[1,t] + p[2,t] + P_vent*v[t]) for t in T),
        GRB.MINIMIZE
    )

    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Model not optimal. Status={m.Status}")

    return {
        "obj": m.ObjVal,
        "p1": np.array([p[1,t].X for t in T]),
        "p2": np.array([p[2,t].X for t in T]),
        "v":  np.array([v[t].X  for t in T]),
        "T1": np.array([Temp[1,t].X for t in T]),
        "T2": np.array([Temp[2,t].X for t in T]),
        "H":  np.array([Hum[t].X for t in T]),
    }


def plot_results(sol, price, occ1, occ2, d) -> None:

    """
    Plot the results of the optimization.

    Parameters:
    - sol: The solution dictionary containing the optimization results.
    - price: The electricity price data.
    - occ1: The occupancy data for room 1.
    - occ2: The occupancy data for room 2.
    - d: The day index (0-based) for which to plot the results.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    T = np.arange(len(sol["v"]))

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    # Room Temperatures
    axes[0].plot(T, sol["T1"], label='Room 1 Temp', marker='o')
    axes[0].plot(T, sol["T2"], label='Room 2 Temp', marker='s')
    axes[0].axhline(18, color='gray', linestyle='--', alpha=0.5)
    axes[0].axhline(20, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title("Room Temperatures for day {}".format(d+1))
    axes[0].legend()
    axes[0].grid(True)

    # Heater consumption
    axes[1].bar(T, sol["p1"], width=0.4, label='Room 1 Heater', alpha=0.7)
    axes[1].bar(T, sol["p2"], width=0.4, bottom=sol["p1"], label='Room 2 Heater', alpha=0.7)
    axes[1].set_ylabel("Heater Power (kW)")
    axes[1].set_title("Heater Consumption for day {}".format(d+1))
    axes[1].legend()
    axes[1].grid(True)

    # Ventilation and Humidity
    axes[2].step(T, sol["v"], where='mid', label='Ventilation ON', color='tab:blue')
    axes[2].plot(T, sol["H"], label='Humidity (%)', color='tab:orange', marker='o')
    axes[2].axhline(45, color='gray', linestyle='--', alpha=0.5)
    axes[2].axhline(60, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_ylabel("Ventilation / Humidity")
    axes[2].set_title("Ventilation Status and Humidity for day {}".format(d+1))
    axes[2].legend()
    axes[2].grid(True)

    # Electricity price and occupancy
    axes[3].plot(T, price[d, :], label='TOU Price (€/kWh)', color='tab:red', marker='x')
    axes[3].bar(T, occ1[d, :], label='Occupancy Room 1', alpha=0.5)
    axes[3].bar(T, occ2[d, :], bottom=occ1[d, :], label='Occupancy Room 2', alpha=0.5)
    axes[3].set_ylabel("Price / Occupancy")
    axes[3].set_xlabel("Time (hours)")
    axes[3].set_title("Electricity Price and Occupancy for day {}".format(d+1))
    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()