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
                + z_loss*(Tout[t-1] - Temp[r,t-1])
                + z_conv*p[r,t-1]
                - z_cool*v[t-1]
                + z_occ*occ_rt_1, # might change that name, so it shows it could be both rooms
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
            # v[-1] = 0  => start[0] = v[0]
            m.addConstr(start[0] >= v[0], name="start0_lb")
            m.addConstr(start[0] <= v[0], name="start0_ub")
        else:
            m.addConstr(start[t] >= v[t] - v[t - 1], name=f"start_lb_{t}")
            m.addConstr(start[t] <= v[t], name=f"start_ub1_{t}")
            m.addConstr(start[t] <= 1 - v[t - 1], name=f"start_ub2_{t}")
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
            # Mutual exclusivity: cannot force max heating and force shutoff simultaneously
            m.addConstr(
                low_active[r, t] + temp_hi[r, t] <= 1,
                name=f"mutual_excl_{r}_{t}"
            )

            # low_trig indicates Temp below Tlow
            m.addConstr(Temp[r, t] <= Tlow + M_T * (1 - low_trig[r, t]), name=f"lowtrig_ub_{r}_{t}")
            m.addConstr(Temp[r, t] >= Tlow - M_T * (low_trig[r, t]), name=f"lowtrig_lb_{r}_{t}")

            # below_ok indicates Temp below Tok
            m.addConstr(Temp[r, t] <= Tok + M_T * (1 - below_ok[r, t]), name=f"belowok_ub_{r}_{t}")
            m.addConstr(Temp[r, t] >= Tok - M_T * (below_ok[r, t]), name=f"belowok_lb_{r}_{t}")

            if t == 0:
                m.addConstr(low_active[r, t] >= low_trig[r, t], name=f"lowact0_{r}")
            else:
                cont = m.addVar(vtype=GRB.BINARY, name=f"cont_{r}_{t}")
                m.addConstr(cont <= low_active[r, t - 1], name=f"cont1_{r}_{t}")
                m.addConstr(cont <= below_ok[r, t], name=f"cont2_{r}_{t}")
                m.addConstr(cont >= low_active[r, t - 1] + below_ok[r, t] - 1, name=f"cont3_{r}_{t}")
                m.addConstr(low_active[r, t] >= low_trig[r, t], name=f"lowact_trig_{r}_{t}")
                m.addConstr(low_active[r, t] >= cont, name=f"lowact_cont_{r}_{t}")
                m.addConstr(low_active[r, t] <= low_trig[r, t] + cont, name=f"lowact_ub_{r}_{t}")

            # If low_active=1, force heater to max (since p <= Pmax already)
            m.addConstr(p[r, t] >= P_heater[r] * low_active[r, t], name=f"p_on_low_{r}_{t}")

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
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Color palette
    C_ROOM1 = "#7A1E14"   # deep petrol blue
    C_ROOM2 = "#1C97B6"   # Economist salmon (darker than before)
    C_VENT  = "#E3120B"   # Economist red accent
    C_HUM   = "#8CB1BC"   # soft grey-blue
    C_PRICE = "#2F2F2F"   # charcoal

    T = np.arange(len(sol["v"]))

    # Use a half-page sized figure (for 0.49\textwidth style LaTeX)
    fig, axes = plt.subplots(4, 1, figsize=(6, 8), sharex=True, constrained_layout=True)

    title_fs = 10
    label_fs = 9
    tick_fs  = 9
    legend_fs = 8

    print("DEBUG: PRINT NEW COLORS")

    # ---- 1) Temperatures ----
    axes[0].plot(T, sol["T1"], label="Room 1 Temp", marker="o", markersize=4, linewidth=1.5, color=C_ROOM1)
    axes[0].plot(T, sol["T2"], label="Room 2 Temp", marker="s", markersize=4, linewidth=1.5, color=C_ROOM2)
    axes[0].axhline(18, color="gray", linestyle="--", alpha=0.5)
    axes[0].axhline(20, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Temperature (°C)", fontsize=label_fs)
    axes[0].set_title(f"Room Temperatures for Day {d+1}", fontsize=title_fs)
    axes[0].legend(fontsize=legend_fs)
    axes[0].grid(True)

    # ---- 2) Heater consumption ----
    axes[1].bar(T, sol["p1"], width=0.6, label="Room 1 Heater", alpha=0.9, color=C_ROOM1)
    axes[1].bar(T, sol["p2"], width=0.6, bottom=sol["p1"], label="Room 2 Heater", alpha=0.9, color=C_ROOM2)
    axes[1].set_ylabel("Heater Power (kW)", fontsize=label_fs)
    axes[1].set_title(f"Heater Consumption for Day {d+1}", fontsize=title_fs)
    axes[1].legend(fontsize=legend_fs)
    axes[1].grid(True)

    # ---- 3) Ventilation + Humidity ----
    axes[2].step(T, sol["v"], where="mid", label="Ventilation ON", linewidth=1.5, color=C_VENT)
    axes[2].plot(T, sol["H"], label="Humidity (%)", marker="o", markersize=4, linewidth=1.5, color=C_HUM)
    axes[2].axhline(45, color="gray", linestyle="--", alpha=0.5)
    axes[2].axhline(60, color="gray", linestyle="--", alpha=0.5)
    axes[2].set_ylabel("Ventilation / Humidity", fontsize=label_fs)
    axes[2].set_title(f"Ventilation Status and Humidity for Day {d+1}", fontsize=title_fs)
    axes[2].legend(fontsize=legend_fs, loc="center right")
    axes[2].grid(True)

    # ---- 4) Occupancy + Price ----
    axes[3].bar(T, occ1[d, :], label="Occupancy Room 1", alpha=0.5, color=C_ROOM1)
    axes[3].bar(T, occ2[d, :], bottom=occ1[d, :], label="Occupancy Room 2", alpha=0.5, color=C_ROOM2)
    axes[3].set_ylabel("Occupancy", fontsize=label_fs)

    ax_price = axes[3].twinx()
    ax_price.plot(T, price[d, :], label="TOU Price (€/kWh)", marker="x", markersize=4, linewidth=1.5, color=C_PRICE)
    ax_price.set_ylabel("Price (€/kWh)", fontsize=label_fs)

    axes[3].set_xlabel("Time (hours)", fontsize=label_fs)
    axes[3].set_title(f"Electricity Price and Occupancy for Day {d+1}", fontsize=title_fs)

    # Combine legends
    l1, lab1 = axes[3].get_legend_handles_labels()
    l2, lab2 = ax_price.get_legend_handles_labels()
    axes[3].legend(l1 + l2, lab1 + lab2, fontsize=legend_fs, loc="upper left")
    axes[3].grid(True)

    # Tick sizes (do this at the end so nothing overrides it)
    for ax in axes:
        ax.tick_params(axis="both", labelsize=tick_fs)
    ax_price.tick_params(axis="y", labelsize=tick_fs)

    plots_dir = Path("Plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / f"day_{d+1}_c.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

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
    from pathlib import Path

    T = np.arange(len(sol["v"]))

    fig, axes = plt.subplots(4, 1, figsize=(3.4, 6.2), sharex=True, constrained_layout=True)

    def style_ax(ax):
        ax.tick_params(axis="both", which="major", labelsize=9)
        ax.title.set_fontsize(10)
        ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_size(9)

    for ax in axes:
        style_ax(ax)

    # Room Temperatures
    axes[0].plot(T, sol["T1"], label='Room 1 Temp', marker='o')
    axes[0].plot(T, sol["T2"], label='Room 2 Temp', marker='s')
    axes[0].axhline(18, color='gray', linestyle='--', alpha=0.5)
    axes[0].axhline(20, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title("Room Temperatures for Day {}".format(d+1))
    axes[0].legend()
    axes[0].grid(True)

    # Heater consumption
    axes[1].bar(T, sol["p1"], width=0.4, label='Room 1 Heater', alpha=0.7)
    axes[1].bar(T, sol["p2"], width=0.4, bottom=sol["p1"], label='Room 2 Heater', alpha=0.7)
    axes[1].set_ylabel("Heater Power (kW)")
    axes[1].set_title("Heater Consumption for Day {}".format(d+1))
    axes[1].legend()
    axes[1].grid(True)

    # Ventilation and Humidity
    axes[2].step(T, sol["v"], where='mid', label='Ventilation ON', color='tab:blue')
    axes[2].plot(T, sol["H"], label='Humidity (%)', color='tab:orange', marker='o')
    axes[2].axhline(45, color='gray', linestyle='--', alpha=0.5)
    axes[2].axhline(60, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_ylabel("Ventilation / Humidity")
    axes[2].set_title("Ventilation Status and Humidity for Day {}".format(d+1))
    axes[2].legend()
    axes[2].grid(True)

    # Electricity price and occupancy
    axes[3].bar(T, occ1[d, :], label='Occupancy Room 1', alpha=0.5)
    axes[3].bar(T, occ2[d, :], bottom=occ1[d, :], label='Occupancy Room 2', alpha=0.5)
    axes[3].set_ylabel("Occupancy")

    ax_price = axes[3].twinx()
    ax_price.tick_params(axis="y", labelsize=9, labelcolor="tab:red")
    ax_price.yaxis.label.set_size(9)    
    ax_price.plot(T, price[d, :], label='TOU Price (€/kWh)', color='tab:red', marker='x')
    ax_price.set_ylabel("Price (€/kWh)", color='tab:red')
    ax_price.tick_params(axis='y', labelcolor='tab:red')

    axes[3].set_xlabel("Time (hours)")
    axes[3].set_title("Electricity Price and Occupancy for Day {}".format(d+1))
    lines1, labels1 = axes[3].get_legend_handles_labels()
    lines2, labels2 = ax_price.get_legend_handles_labels()
    axes[3].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    axes[3].grid(True)

    #plt.tight_layout()
    #plt.show()
    plots_dir = Path("Plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / f"day_{d+1}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)