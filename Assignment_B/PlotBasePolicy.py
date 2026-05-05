# -*- coding: utf-8 -*-
"""
Diagnostic plotting script for BasePolicyRestaurant.py.

Run from the Assignment_B root folder, i.e. the folder that contains:
    BasePolicyRestaurant.py
    Data/v2_SystemCharacteristics.py
    Data/v2_PriceData.csv
    Data/OccupancyRoom1.csv
    Data/OccupancyRoom2.csv

This script simulates one full day with the base policy and saves a Part-A-style
plot to the Plots/ folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import BasePolicy as policy
import Data.v2_SystemCharacteristics as SystemCharacteristics


# -----------------------------------------------------------------------------
# Data / parameter helpers
# -----------------------------------------------------------------------------

def _params() -> Dict[str, Any]:
    if hasattr(SystemCharacteristics, "get_fixed_data"):
        return dict(SystemCharacteristics.get_fixed_data())
    if hasattr(SystemCharacteristics, "fetch_data"):
        return dict(SystemCharacteristics.fetch_data())
    raise AttributeError("Data.v2_SystemCharacteristics must define get_fixed_data() or fetch_data().")


PARAMS = _params()


def par(*names: str, default: Any) -> Any:
    for name in names:
        if name in PARAMS:
            return PARAMS[name]
    return default


def as_2d_csv(path: str | Path) -> np.ndarray:
    return pd.read_csv(path).to_numpy(dtype=float)


# Main fixed data, with aliases to survive small naming differences.
T_DAY = int(par("num_timeslots", "T", default=10))
P_VENT = float(par("ventilation_power", "P_vent", "Pvent", "PowerVent", default=1.0))
P_MAX = float(par("heating_max_power", "P_r", "Pr", "PowerMax", "max_heating_power", default=3.0))
P_MAX_R1 = float(par("heating_max_power_room1", "P1", "P_room1", default=P_MAX))
P_MAX_R2 = float(par("heating_max_power_room2", "P2", "P_room2", default=P_MAX))

T_LOW = float(par("T_low", "Tlow", "temperature_low", "TempLow", default=18.0))
T_OK = float(par("T_OK", "TOK", "temperature_ok", "TempOK", default=20.0))
T_HIGH = float(par("T_high", "THigh", "temperature_high", "TempHigh", default=24.0))
H_HIGH = float(par("H_high", "Hhigh", "humidity_high", "HumHigh", default=60.0))

T_INITIAL = float(par("initial_temperature", "T0", "initial_T", default=T_OK))
H_INITIAL = float(par("initial_humidity", "H0", "initial_H", default=45.0))

ZETA_EXCH = float(par("heat_exchange_coeff", "zeta_exch", "zeta_exchange", default=0.6))
ZETA_LOSS = float(par("thermal_loss_coeff", "zeta_loss", default=0.1))
ZETA_CONV = float(par("heating_efficiency_coeff", "zeta_conv", default=1.0))
ZETA_COOL = float(par("heat_vent_coeff", "zeta_cool", default=0.7))
ZETA_OCC = float(par("heat_occupancy_coeff", "zeta_occ", default=0.02))
ETA_OCC = float(par("humidity_occupancy_coeff", "eta_occ", default=0.15))
ETA_VENT = float(par("humidity_vent_coeff", "eta_vent", default=5.0))
OUTDOOR_TEMP = par("outdoor_temperature", "T_out", "Tout", default=[5.0] * T_DAY)


def outdoor_temperature(t: int) -> float:
    if isinstance(OUTDOOR_TEMP, (list, tuple, np.ndarray)):
        return float(OUTDOOR_TEMP[min(max(t, 0), len(OUTDOOR_TEMP) - 1)])
    return float(OUTDOOR_TEMP)


# -----------------------------------------------------------------------------
# Simple environment for diagnostics
# -----------------------------------------------------------------------------

def update_low_override(temp: float, old_flag: int) -> int:
    """Low-temperature hysteresis: active below T_LOW, stays active until T_OK."""
    if temp <= T_LOW:
        return 1
    if old_flag == 1 and temp < T_OK:
        return 1
    return 0


def apply_overrules(
    action: Dict[str, Any],
    T1: float,
    T2: float,
    H: float,
    low1: int,
    low2: int,
    vent_counter: int,
) -> Tuple[float, float, int]:
    """Map policy suggestions to actual actions under Part-A-style overrules."""
    v = int(float(action.get("VentilationON", 0)) >= 0.5)
    p1 = float(action.get("HeatPowerRoom1", 0.0))
    p2 = float(action.get("HeatPowerRoom2", 0.0))

    # Humidity overrule and ventilation inertia.
    if H > H_HIGH or vent_counter in (1, 2):
        v = 1

    # Temperature overrules. High-temperature shutoff has priority.
    if T1 > T_HIGH:
        p1 = 0.0
    elif low1 == 1 or T1 < T_LOW:
        p1 = P_MAX_R1

    if T2 > T_HIGH:
        p2 = 0.0
    elif low2 == 1 or T2 < T_LOW:
        p2 = P_MAX_R2

    p1 = float(np.clip(p1, 0.0, P_MAX_R1))
    p2 = float(np.clip(p2, 0.0, P_MAX_R2))
    return p1, p2, v


def simulate_day(day: int, price: np.ndarray, occ1: np.ndarray, occ2: np.ndarray) -> Dict[str, np.ndarray]:
    T1 = T_INITIAL
    T2 = T_INITIAL
    H = H_INITIAL
    low1 = update_low_override(T1, 0)
    low2 = update_low_override(T2, 0)
    vent_counter = 0

    sol = {"T1": [], "T2": [], "H": [], "p1": [], "p2": [], "v": [], "cost": []}

    for t in range(T_DAY):
        price_prev = float(price[day, t - 1]) if t > 0 else float(price[day, t])
        state = {
            "T1": T1,
            "T2": T2,
            "H": H,
            "Occ1": float(occ1[day, t]),
            "Occ2": float(occ2[day, t]),
            "price_t": float(price[day, t]),
            "price_previous": price_prev,
            "vent_counter": vent_counter,
            "low_override_r1": low1,
            "low_override_r2": low2,
            "current_time": t,
        }

        raw_action = policy.select_action(state)
        p1, p2, v = apply_overrules(raw_action, T1, T2, H, low1, low2, vent_counter)

        sol["T1"].append(T1)
        sol["T2"].append(T2)
        sol["H"].append(H)
        sol["p1"].append(p1)
        sol["p2"].append(p2)
        sol["v"].append(v)
        sol["cost"].append(float(price[day, t]) * (p1 + p2 + P_VENT * v))

        # Apply dynamics to get next state.
        T_out = outdoor_temperature(t)
        T1_next = (
            T1
            + ZETA_EXCH * (T2 - T1)
            + ZETA_LOSS * (T_out - T1)
            + ZETA_CONV * p1
            - ZETA_COOL * v
            + ZETA_OCC * float(occ1[day, t])
        )
        T2_next = (
            T2
            + ZETA_EXCH * (T1 - T2)
            + ZETA_LOSS * (T_out - T2)
            + ZETA_CONV * p2
            - ZETA_COOL * v
            + ZETA_OCC * float(occ2[day, t])
        )
        H_next = H + ETA_OCC * (float(occ1[day, t]) + float(occ2[day, t])) - ETA_VENT * v

        T1, T2, H = T1_next, T2_next, H_next
        low1 = update_low_override(T1, low1)
        low2 = update_low_override(T2, low2)
        vent_counter = min(vent_counter + 1, 3) if v == 1 else 0

    return {k: np.asarray(v, dtype=float) for k, v in sol.items()}


# -----------------------------------------------------------------------------
# Part-A-style plotting
# -----------------------------------------------------------------------------

def plot_results(sol: Dict[str, np.ndarray], price: np.ndarray, occ1: np.ndarray, occ2: np.ndarray, d: int) -> None:
    # Color palette
    C_ROOM1 = "#7A1E14"
    C_ROOM2 = "#1C97B6"
    C_VENT = "#E3120B"
    C_HUM = "#8CB1BC"
    C_PRICE = "#2F2F2F"

    T = np.arange(len(sol["v"]))
    fig, axes = plt.subplots(4, 1, figsize=(6, 8), sharex=True, constrained_layout=True)

    title_fs = 10
    label_fs = 9
    tick_fs = 9
    legend_fs = 8

    axes[0].plot(T, sol["T1"], label="Room 1 Temp", marker="o", markersize=4, linewidth=1.5, color=C_ROOM1)
    axes[0].plot(T, sol["T2"], label="Room 2 Temp", marker="s", markersize=4, linewidth=1.5, color=C_ROOM2)
    axes[0].axhline(T_LOW, color="gray", linestyle="--", alpha=0.5, label="T low / T OK")
    axes[0].axhline(T_OK, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Temperature (°C)", fontsize=label_fs)
    axes[0].set_title(f"Base Policy: Room Temperatures for Day {d + 1}", fontsize=title_fs)
    axes[0].legend(fontsize=legend_fs)
    axes[0].grid(True)

    axes[1].bar(T, sol["p1"], width=0.6, label="Room 1 Heater", alpha=0.9, color=C_ROOM1)
    axes[1].bar(T, sol["p2"], width=0.6, bottom=sol["p1"], label="Room 2 Heater", alpha=0.9, color=C_ROOM2)
    axes[1].set_ylabel("Heater Power (kW)", fontsize=label_fs)
    axes[1].set_title(f"Base Policy: Heater Consumption for Day {d + 1}", fontsize=title_fs)
    axes[1].legend(fontsize=legend_fs)
    axes[1].grid(True)

    axes[2].step(T, sol["v"], where="mid", label="Ventilation ON", linewidth=1.5, color=C_VENT)
    axes[2].plot(T, sol["H"], label="Humidity (%)", marker="o", markersize=4, linewidth=1.5, color=C_HUM)
    axes[2].axhline(H_HIGH, color="gray", linestyle="--", alpha=0.5)
    axes[2].set_ylabel("Ventilation / Humidity", fontsize=label_fs)
    axes[2].set_title(f"Base Policy: Ventilation Status and Humidity for Day {d + 1}", fontsize=title_fs)
    axes[2].legend(fontsize=legend_fs, loc="center right")
    axes[2].grid(True)

    axes[3].bar(T, occ1[d, :len(T)], label="Occupancy Room 1", alpha=0.5, color=C_ROOM1)
    axes[3].bar(T, occ2[d, :len(T)], bottom=occ1[d, :len(T)], label="Occupancy Room 2", alpha=0.5, color=C_ROOM2)
    axes[3].set_ylabel("Occupancy", fontsize=label_fs)

    ax_price = axes[3].twinx()
    ax_price.plot(T, price[d, :len(T)], label="TOU Price (€/kWh)", marker="x", markersize=4, linewidth=1.5, color=C_PRICE)
    ax_price.set_ylabel("Price (€/kWh)", fontsize=label_fs)

    axes[3].set_xlabel("Time (hours)", fontsize=label_fs)
    axes[3].set_title(f"Electricity Price and Occupancy for Day {d + 1}", fontsize=title_fs)
    l1, lab1 = axes[3].get_legend_handles_labels()
    l2, lab2 = ax_price.get_legend_handles_labels()
    axes[3].legend(l1 + l2, lab1 + lab2, fontsize=legend_fs, loc="upper left")
    axes[3].grid(True)

    for ax in axes:
        ax.tick_params(axis="both", labelsize=tick_fs)
    ax_price.tick_params(axis="y", labelsize=tick_fs)

    plots_dir = Path("Plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / f"base_policy_day_{d + 1}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    data_dir = Path(__file__).parent / "Data"
    price = as_2d_csv(data_dir / "v2_PriceData.csv")
    occ1 = as_2d_csv(data_dir / "OccupancyRoom1.csv")
    occ2 = as_2d_csv(data_dir / "OccupancyRoom2.csv")

    day = 10  # change to any 0-based day index
    sol = simulate_day(day, price, occ1, occ2)
    plot_results(sol, price, occ1, occ2, day)
    print(f"Day {day + 1} cost: {sol['cost'].sum():.2f}")
    print(f"Saved plot: Plots/base_policy_day_{day + 1}.png")
