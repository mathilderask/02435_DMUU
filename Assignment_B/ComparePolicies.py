"""Compare HVAC policies across multiple experiments.

The script evaluates policies over the historical 100-day data set using the
shared simulation environment and saves a comparison figure in the Plots/
folder. The figure shows both the average daily cost and the histogram of daily
costs across experiments.

Usage:
    python ComparePolicies.py
    python ComparePolicies.py --experiments 100
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

import Policies.Dummy_policy_27 as Dummy_policy_27
import Policies.Hybrid_policy_27 as Hybrid_policy_27
import Policies.OIH_policy_27 as OIH_policy_27
import Policies.SP_policy_27 as SP_policy_27
import Policies.TwoStageSP_policy_27 as TwoStageSP_policy_27
from SimulationEnvironment import RestaurantSimulationEnvironment


class _ReplayPolicy:
    """Replay a precomputed hourly action schedule."""

    def __init__(self, actions: np.ndarray) -> None:
        self.actions = np.asarray(actions, dtype=float)
        self.index = 0

    def reset(self) -> None:
        self.index = 0

    def select_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if self.index >= len(self.actions):
            action = self.actions[-1]
        else:
            action = self.actions[self.index]
        self.index += 1
        return {
            "HeatPowerRoom1": float(action[0]),
            "HeatPowerRoom2": float(action[1]),
            "VentilationON": int(round(action[2])),
        }


class _OptimalInHindsightReplayPolicy:
    """Precompute the full-day MILP solution for a given experiment day."""

    def __init__(self, params: Dict[str, Any]) -> None:
        self.params = params
        self._replay = _ReplayPolicy(np.zeros((1, 3), dtype=float))

    def prepare_episode(self, price: np.ndarray, occ1: np.ndarray, occ2: np.ndarray) -> None:
        solution = OIH_policy_27.solve_day_milp(price, occ1, occ2, self.params, output_flag=0)
        actions = np.column_stack([solution["p1"], solution["p2"], solution["v"]])
        self._replay = _ReplayPolicy(actions)

    def select_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self._replay.select_action(state)


class _ExpectedValueReplayPolicy:
    """Deterministic lookahead policy based on the expected scenario across all days."""

    def __init__(self, params: Dict[str, Any], price: np.ndarray, occ1: np.ndarray, occ2: np.ndarray) -> None:
        self.params = params
        self.price_mean = np.mean(price, axis=0)
        self.occ1_mean = np.mean(occ1, axis=0)
        self.occ2_mean = np.mean(occ2, axis=0)
        self._replay = _ReplayPolicy(np.zeros((1, 3), dtype=float))
        self._solution = OIH_policy_27.solve_day_milp(
            self.price_mean,
            self.occ1_mean,
            self.occ2_mean,
            self.params,
            output_flag=0,
        )
        actions = np.column_stack([self._solution["p1"], self._solution["p2"], self._solution["v"]])
        self._replay = _ReplayPolicy(actions)

    def select_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self._replay.select_action(state)


def _load_experiment_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    env = RestaurantSimulationEnvironment()
    return env.price_data, env.occ1_data, env.occ2_data


def _policy_factories(price: np.ndarray, occ1: np.ndarray, occ2: np.ndarray) -> Dict[str, Any]:
    params = OIH_policy_27.build_oih_params()
    return {
        "Dummy": Dummy_policy_27,
        "Optimal in Hindsight": _OptimalInHindsightReplayPolicy(params),
        "SP": SP_policy_27,
        "TwoStageSP": TwoStageSP_policy_27,
        "Expected value": _ExpectedValueReplayPolicy(params, price, occ1, occ2),
        "Hybrid": Hybrid_policy_27,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare HVAC policies across multiple experiments.")
    parser.add_argument("--experiments", type=int, default=100, help="Number of experiments/days to evaluate.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "Plots" / "policy_comparison.png"),
        help="Path to the output figure.",
    )
    return parser.parse_args()


def _policy_label(name: str) -> str:
    return {
        "Dummy": "Dummy policy",
        "Optimal in Hindsight": "Optimal in hindsight",
        "SP": "Stochastic programming",
        "TwoStageSP": "Two-stage SP",
        "Expected value": "Deterministic lookahead",
        "Hybrid": "Hybrid policy",
    }.get(name, name)


def evaluate_policies(experiments: int) -> Dict[str, Dict[str, Any]]:
    env = RestaurantSimulationEnvironment()
    price, occ1, occ2 = env.price_data, env.occ1_data, env.occ2_data
    policies = _policy_factories(price, occ1, occ2)

    if experiments <= 0:
        raise ValueError("experiments must be positive")

    day_indices = list(range(min(experiments, env.num_days)))
    results: Dict[str, Dict[str, Any]] = {}

    for name, policy in policies.items():
        daily_costs = np.zeros(len(day_indices), dtype=float)

        for idx, day in enumerate(day_indices):
            if hasattr(policy, "prepare_episode"):
                policy.prepare_episode(price[day], occ1[day], occ2[day])
            if hasattr(policy, "reset"):
                try:
                    policy.reset()
                except TypeError:
                    pass

            episode = env.evaluate_policy(policy, day=day)
            daily_costs[idx] = float(episode["total_cost"])

        results[name] = {
            "day_indices": np.asarray(day_indices, dtype=int),
            "daily_costs": daily_costs,
            "average_daily_cost": float(np.mean(daily_costs)) if len(daily_costs) else 0.0,
        }

    return results


def plot_comparison(results: Dict[str, Dict[str, Any]], output_path: str, experiments: int) -> None:
    colors = {
        "Dummy": "#6B7280",
        "Optimal in Hindsight": "#111827",
        "SP": "#1C97B6",
        "TwoStageSP": "#4F46E5",
        "Expected value": "#8B5CF6",
        "Hybrid": "#E3120B",
    }

    names = list(results.keys())
    fig, axes = plt.subplots(2, 1, figsize=(11, 10), constrained_layout=True)

    avg_costs = [results[name]["average_daily_cost"] for name in names]
    axes[0].bar([_policy_label(name) for name in names], avg_costs, color=[colors[name] for name in names])
    axes[0].set_ylabel("Average daily cost")
    axes[0].set_title(f"Policy comparison across {experiments} experiments")
    axes[0].tick_params(axis="x", labelrotation=20)
    for label in axes[0].get_xticklabels():
        label.set_ha("right")
    axes[0].grid(True, axis="y", alpha=0.3)

    # Histogram outlines of daily costs (one per policy)
    all_costs = np.concatenate([results[name]["daily_costs"] for name in names if len(results[name]["daily_costs"])])
    if len(all_costs) > 0:
        handles: Dict[str, Any] = {}
        for name in names:
            data = results[name]["daily_costs"]
            if len(data) == 0:
                continue
            # plot histogram outline for this policy
            bins = np.histogram_bin_edges(data, bins="auto")
            _, _, patches = axes[1].hist(
                data,
                bins=bins,
                histtype="step",
                linewidth=2.0,
                label=_policy_label(name),
                color=colors[name],
            )
            handles[name] = patches[0]
    axes[1].set_xlabel("Daily electricity cost")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Histogram of daily costs")
    # Enforce specific legend order preferred by the user
    desired_order = [
        "Dummy",
        "Optimal in Hindsight",
        "SP",
        "TwoStageSP",
        "Expected value",
        "Hybrid",
    ]
    legend_handles = [handles[n] for n in desired_order if n in handles]
    legend_labels = [_policy_label(n) for n in desired_order if n in handles]
    if legend_handles:
        axes[1].legend(legend_handles, legend_labels, ncol=2)
    axes[1].grid(True, alpha=0.3)

    plots_dir = Path(output_path).parent
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    results = evaluate_policies(experiments=args.experiments)

    print("Policy comparison summary:")
    for name, result in results.items():
        print(f"  {_policy_label(name):<24} average daily cost = {result['average_daily_cost']:.2f}")

    plot_comparison(results, args.output, args.experiments)
    print(f"Saved comparison figure: {args.output}")


if __name__ == "__main__":
    main()