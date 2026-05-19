

"""
Task 7: Distributed Decision-Making for Mall Energy Management

This script is adapted to the provided files:
- DataTask7.py
- Task7Occupancies.csv

It implements:
1. A centralized benchmark optimization problem.
2. A distributed dual-decomposition algorithm.
3. Plots required for Task 7:
   - objective value over 100 iterations for different step sizes,
   - multiplier evolution,
   - mall power-constraint violation evolution,
   - total energy consumption per store.

Run this file from the same folder as DataTask7.py and Task7Occupancies.csv.
"""

from __future__ import annotations

import os
import importlib.util
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as pyo


# ============================================================
# Settings
# ============================================================

N_STORES = 15
N_ITERATIONS = 100
STEP_SIZES = [0.001, 0.01, 0.1, 1.0, 10.0]
ADAPTIVE_ALPHA0 = 5.0

DATA_FILE = "Assignment_B/Data/DataTask7.py"
OCCUPANCY_FILE = "Assignment_B/Data/Task7Occupancies.csv"
OUTPUT_DIR = "task7_results"

SOLVER_NAME = "gurobi"
SOLVER_TEE = False


# ============================================================
# Data loading
# ============================================================

def load_fetch_data_module(path: str):
    """Import DataTask7.py without changing the original file."""
    spec = importlib.util.spec_from_file_location("DataTask7", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_task7_data(
    data_file: str = DATA_FILE,
    occupancy_file: str = OCCUPANCY_FILE,
) -> Dict:
    """Load all data needed for Task 7."""
    data_module = load_fetch_data_module(data_file)
    data = data_module.fetch_data()

    occ_raw = pd.read_csv(occupancy_file)

    # The supplied Task7Occupancies.csv has two rows:
    # row 0 = room 1 occupancy over hours 0,...,9
    # row 1 = room 2 occupancy over hours 0,...,9
    # The last column is only a room label and is ignored.
    H = int(data["num_timeslots"])
    occ_room_1 = occ_raw.iloc[0, :H].astype(float).to_numpy()
    occ_room_2 = occ_raw.iloc[1, :H].astype(float).to_numpy()

    data["occupancy"] = {
        1: occ_room_1,
        2: occ_room_2,
    }

    return data


# ============================================================
# Temperature dynamics
# ============================================================

def temperature_next(
    T_room: float,
    T_other: float,
    power: float,
    occ: float,
    T_out: float,
    data: Dict,
) -> float:
    """One-step temperature equation for one room."""
    return (
        T_room
        + data["heat_exchange_coeff"] * (T_other - T_room)
        - data["thermal_loss_coeff"] * (T_room - T_out)
        + data["heating_efficiency_coeff"] * power
        - data["heat_vent_coeff"] * 1.0  # ventilation is always ON
        + data["heat_occupancy_coeff"] * occ
    )


# ============================================================
# Objective evaluation
# ============================================================

def compute_system_objective(T: np.ndarray, data: Dict) -> float:
    T_ref = data["Temperature_reference"]
    H = int(data["num_timeslots"])

    objective = 0.0
    for n in range(N_STORES):
        w_n = n + 2
        for t in range(1, H + 1):
            T_store = 0.5 * (T[n, 0, t] + T[n, 1, t])
            objective += w_n * (T_store - T_ref) ** 2

    return float(objective)


# ============================================================
# Store subproblem
# ============================================================

def solve_store_subproblem(
    data: Dict,
    store_index: int,
    lambdas: np.ndarray,
    solver: pyo.SolverFactory,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve one store's local optimization problem for fixed multipliers.

    The local problem is:

        min sum_t sum_r w_n (T_{r,t+1} - T_ref)^2
            + sum_t lambda_t (p_{1,t} + p_{2,t})

    subject to local temperature dynamics and heater bounds.
    """
    H = int(data["num_timeslots"])
    w_n = store_index + 1

    m = pyo.ConcreteModel()

    m.R = pyo.Set(initialize=[1, 2])
    m.TS = pyo.RangeSet(0, H - 1)
    m.TS_full = pyo.RangeSet(0, H)

    m.p = pyo.Var(m.R, m.TS, domain=pyo.NonNegativeReals)
    m.Temp = pyo.Var(m.R, m.TS_full, domain=pyo.Reals)

    def heater_bound_rule(m, r, t):
        return m.p[r, t] <= data["heating_max_power"]

    m.heater_bound = pyo.Constraint(m.R, m.TS, rule=heater_bound_rule)

    def initial_temperature_rule(m, r):
        return m.Temp[r, 0] == data["initial_temperature"]

    m.initial_temperature = pyo.Constraint(m.R, rule=initial_temperature_rule)

    def temperature_dynamics_rule(m, r, t):
        other = 2 if r == 1 else 1
        return m.Temp[r, t + 1] == (
            m.Temp[r, t]
            + data["heat_exchange_coeff"] * (m.Temp[other, t] - m.Temp[r, t])
            - data["thermal_loss_coeff"] * (m.Temp[r, t] - data["outdoor_temperature"][t])
            + data["heating_efficiency_coeff"] * m.p[r, t]
            - data["heat_vent_coeff"] * 1.0
            + data["heat_occupancy_coeff"] * data["occupancy"][r][t]
        )

    m.temperature_dynamics = pyo.Constraint(m.R, m.TS, rule=temperature_dynamics_rule)

    def objective_rule(m):
        temperature_penalty = sum(
            w_n
            * (
                0.5 * (m.Temp[1, t + 1] + m.Temp[2, t + 1])
                - data["Temperature_reference"]
            ) ** 2
            for t in m.TS
        )

        multiplier_penalty = sum(
            lambdas[t] * sum(m.p[r, t] for r in m.R)
            for t in m.TS
        )

        return temperature_penalty + multiplier_penalty

    m.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    results = solver.solve(m, tee=SOLVER_TEE)
    termination = results.solver.termination_condition

    if termination not in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.locallyOptimal]:
        raise RuntimeError(f"Store {store_index} subproblem failed: {termination}")

    p_solution = np.zeros((2, H))
    T_solution = np.zeros((2, H + 1))

    for r in [1, 2]:
        for t in range(H):
            p_solution[r - 1, t] = pyo.value(m.p[r, t])
        for t in range(H + 1):
            T_solution[r - 1, t] = pyo.value(m.Temp[r, t])

    return p_solution, T_solution


# ============================================================
# Centralized benchmark
# ============================================================

def solve_centralized(data: Dict) -> Tuple[float, np.ndarray, np.ndarray]:
    """Solve the centralized benchmark optimization problem."""
    H = int(data["num_timeslots"])

    m = pyo.ConcreteModel()

    m.N = pyo.RangeSet(1, N_STORES)
    m.R = pyo.Set(initialize=[1, 2])
    m.TS = pyo.RangeSet(0, H - 1)
    m.TS_full = pyo.RangeSet(0, H)

    m.p = pyo.Var(m.N, m.R, m.TS, domain=pyo.NonNegativeReals)
    m.Temp = pyo.Var(m.N, m.R, m.TS_full, domain=pyo.Reals)

    def heater_bound_rule(m, n, r, t):
        return m.p[n, r, t] <= data["heating_max_power"]

    m.heater_bound = pyo.Constraint(m.N, m.R, m.TS, rule=heater_bound_rule)

    def initial_temperature_rule(m, n, r):
        return m.Temp[n, r, 0] == data["initial_temperature"]

    m.initial_temperature = pyo.Constraint(m.N, m.R, rule=initial_temperature_rule)

    def temperature_dynamics_rule(m, n, r, t):
        other = 2 if r == 1 else 1
        return m.Temp[n, r, t + 1] == (
            m.Temp[n, r, t]
            + data["heat_exchange_coeff"] * (m.Temp[n, other, t] - m.Temp[n, r, t])
            - data["thermal_loss_coeff"] * (m.Temp[n, r, t] - data["outdoor_temperature"][t])
            + data["heating_efficiency_coeff"] * m.p[n, r, t]
            - data["heat_vent_coeff"] * 1.0
            + data["heat_occupancy_coeff"] * data["occupancy"][r][t]
        )

    m.temperature_dynamics = pyo.Constraint(m.N, m.R, m.TS, rule=temperature_dynamics_rule)

    def mall_power_constraint_rule(m, t):
        return sum(m.p[n, r, t] for n in m.N for r in m.R) <= data["P_mall"]

    m.mall_power_constraint = pyo.Constraint(m.TS, rule=mall_power_constraint_rule)

    def objective_rule(m):
        return sum(
            (n + 1)
            * (
                0.5 * (m.Temp[n, 1, t + 1] + m.Temp[n, 2, t + 1])
                - data["Temperature_reference"]
            ) ** 2
            for n in m.N
            for t in m.TS
        )

    m.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    solver = pyo.SolverFactory(SOLVER_NAME)
    results = solver.solve(m, tee=SOLVER_TEE)
    termination = results.solver.termination_condition

    if termination not in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.locallyOptimal]:
        raise RuntimeError(f"Centralized problem failed: {termination}")

    p_solution = np.zeros((N_STORES, 2, H))
    T_solution = np.zeros((N_STORES, 2, H + 1))

    for n in range(1, N_STORES + 1):
        for r in [1, 2]:
            for t in range(H):
                p_solution[n - 1, r - 1, t] = pyo.value(m.p[n, r, t])
            for t in range(H + 1):
                T_solution[n - 1, r - 1, t] = pyo.value(m.Temp[n, r, t])

    objective = compute_system_objective(T_solution, data)

    return objective, p_solution, T_solution


# ============================================================
# Distributed dual-decomposition algorithm
# ============================================================

def run_distributed_algorithm(
    data: Dict,
    step_size: Union[float, str],
    n_iterations: int = N_ITERATIONS,
) -> Dict[str, np.ndarray]:
    """Run distributed dual decomposition for one step-size rule."""
    H = int(data["num_timeslots"])
    solver = pyo.SolverFactory(SOLVER_NAME)

    lambdas = np.zeros(H)

    objective_history = np.zeros(n_iterations)
    lambda_history = np.zeros((n_iterations + 1, H))
    violation_history = np.zeros((n_iterations, H))
    total_power_history = np.zeros((n_iterations, H))

    final_p = np.zeros((N_STORES, 2, H))
    final_T = np.zeros((N_STORES, 2, H + 1))

    lambda_history[0, :] = lambdas

    for k in range(n_iterations):
        p_all = np.zeros((N_STORES, 2, H))
        T_all = np.zeros((N_STORES, 2, H + 1))

        for store_index in range(1, N_STORES + 1):
            p_store, T_store = solve_store_subproblem(
                data=data,
                store_index=store_index,
                lambdas=lambdas,
                solver=solver,
            )
            p_all[store_index - 1, :, :] = p_store
            T_all[store_index - 1, :, :] = T_store

        total_power = p_all.sum(axis=(0, 1))
        violation = total_power - data["P_mall"]
        objective = compute_system_objective(T_all, data)

        if step_size == "adaptive":
            alpha_k = ADAPTIVE_ALPHA0 / (1 + k)
        else:
            alpha_k = float(step_size)

        # Projected subgradient ascent on the dual variables.
        lambdas = np.maximum(0.0, lambdas + alpha_k * violation)

        objective_history[k] = objective
        lambda_history[k + 1, :] = lambdas
        violation_history[k, :] = violation
        total_power_history[k, :] = total_power

        final_p = p_all
        final_T = T_all

        print(
            f"case={step_size}, iteration={k + 1:03d}, "
            f"objective={objective:.3f}, "
            f"max violation={violation.max():.3f}, "
            f"mean abs violation={np.mean(np.abs(violation)):.3f}"
        )

    return {
        "objective": objective_history,
        "lambda": lambda_history,
        "violation": violation_history,
        "total_power": total_power_history,
        "final_p": final_p,
        "final_T": final_T,
    }


# ============================================================
# Plotting
# ============================================================

def save_plot(filename: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.show()


def plot_objective_convergence(results: Dict[str, Dict], centralized_objective: float) -> None:
    plt.figure(figsize=(10, 6))

    for label, result in results.items():
        plt.plot(
            np.arange(1, N_ITERATIONS + 1),
            result["objective"],
            label=f"alpha = {label}",
        )

    plt.axhline(
        centralized_objective,
        linestyle="--",
        label="Optimal Objective",
    )

    plt.xlabel("Iteration")
    plt.ylabel("Objective value")
    plt.title("Objective value across distributed iterations")
    plt.grid(True)
    plt.legend()
    save_plot("objective_convergence_all_steps.png")


def plot_multiplier_evolution_combined(results: Dict[str, Dict]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    axes = axes.flatten()

    for ax, (label, result) in zip(axes, results.items()):
        lambdas = result["lambda"]
        H = lambdas.shape[1]

        for t in range(H):
            ax.plot(
                np.arange(0, N_ITERATIONS + 1),
                lambdas[:, t],
                label=f"t={t}"
            )

        ax.set_title(f"alpha = {label}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$\lambda_t$")
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=10)

    fig.suptitle("Multiplier evolution for all step-size choices", fontsize=16)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    plt.savefig(
        os.path.join(OUTPUT_DIR, "multiplier_evolution_combined.png"),
        dpi=300
    )
    plt.show()


def plot_violation_evolution_combined(results: Dict[str, Dict]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    axes = axes.flatten()

    for ax, (label, result) in zip(axes, results.items()):
        violations = result["violation"]
        H = violations.shape[1]

        for t in range(H):
            ax.plot(
                np.arange(1, N_ITERATIONS + 1),
                violations[:, t],
                label=f"t={t}"
            )

        ax.axhline(0.0, linestyle="--")
        ax.set_title(f"alpha = {label}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$\sum_n p_{n,t} - P^{mall}$")
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=10)

    fig.suptitle("Mall power-limit violation for all step-size choices", fontsize=16)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    plt.savefig(
        os.path.join(OUTPUT_DIR, "violation_evolution_combined.png"),
        dpi=300
    )
    plt.show()


def plot_energy_per_store(result: Dict, label: str) -> None:
    total_energy_per_store = result["final_p"].sum(axis=(1, 2))

    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(1, N_STORES + 1), total_energy_per_store)
    plt.xlabel("Store")
    plt.ylabel("Total heating energy over the day [kWh]")
    plt.title(f"Total heating energy per store, alpha = {label}")
    plt.xticks(np.arange(1, N_STORES + 1))
    plt.grid(True, axis="y")
    safe_label = str(label).replace(".", "p")
    save_plot(f"energy_per_store_alpha_{safe_label}.png")


def save_summary_tables(
    results: Dict[str, Dict],
    centralized_objective: float,
    centralized_p: np.ndarray,
) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary_rows = []
    for label, result in results.items():
        final_power = result["final_p"].sum(axis=(1, 2))
        summary_rows.append(
            {
                "case": label,
                "final_objective": result["objective"][-1],
                "centralized_objective": centralized_objective,
                "objective_gap": result["objective"][-1] - centralized_objective,
                "max_final_violation": result["violation"][-1, :].max(),
                "mean_abs_final_violation": np.mean(np.abs(result["violation"][-1, :])),
                "highest_energy_store": int(np.argmax(final_power) + 1),
                "lowest_energy_store": int(np.argmin(final_power) + 1),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(OUTPUT_DIR, "summary_results.csv"), index=False)

    energy = pd.DataFrame({"store": np.arange(1, N_STORES + 1)})
    energy["centralized_energy"] = centralized_p.sum(axis=(1, 2))

    for label, result in results.items():
        energy[f"distributed_energy_{label}"] = result["final_p"].sum(axis=(1, 2))

    energy.to_csv(os.path.join(OUTPUT_DIR, "energy_per_store.csv"), index=False)


# ============================================================
# Main script
# ============================================================

def main() -> None:
    print("Loading Task 7 data...")
    data = load_task7_data()

    print("Data loaded:")
    print(f"  Number of timeslots: {data['num_timeslots']}")
    print(f"  Mall power limit: {data['P_mall']} kW")
    print(f"  Reference temperature: {data['Temperature_reference']} °C")
    print(f"  Heating max power per heater: {data['heating_max_power']} kW")

    print("\nSolving centralized benchmark...")
    centralized_objective, centralized_p, centralized_T = solve_centralized(data)
    print(f"Centralized objective value: {centralized_objective:.4f}")

    results = {}

    for alpha in STEP_SIZES:
        label = str(alpha)
        print(f"\nRunning distributed algorithm with alpha = {alpha}...")
        results[label] = run_distributed_algorithm(data, step_size=alpha)

    print("\nRunning distributed algorithm with adaptive alpha_k = 5/(1+k)...")
    results["adaptive"] = run_distributed_algorithm(data, step_size="adaptive")

    print("\nCreating plots...")
    plot_objective_convergence(results, centralized_objective)
    plot_multiplier_evolution_combined(results)
    plot_violation_evolution_combined(results)
    plot_energy_per_store(results["adaptive"], label="adaptive")

    save_summary_tables(results, centralized_objective, centralized_p)

    print(f"\nFinished. All plots and CSV files are saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
