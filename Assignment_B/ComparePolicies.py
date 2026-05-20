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
import time

import Policies.Dummy_policy_27 as Dummy_policy_27
import Policies.Hybrid_policy_27 as Hybrid_policy_27
import Policies.OIH_policy_27 as OIH_policy_27
import Policies.SP_policy_27 as SP_policy_27
import Policies.TwoStageSP_policy_27 as TwoStageSP_policy_27
import Policies.ADP_policy_27 as ADP_policy_27
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
        "ADP": ADP_policy_27,
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
        "ADP": "ADP policy",
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
        start_time = time.perf_counter()
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
            "elapsed_seconds": float(time.perf_counter() - start_time),
        }

        print(f"Evaluated {_policy_label(name)} in {results[name]['elapsed_seconds']:.2f}s")

    return results

def plot_comparison(results: Dict[str, Dict[str, Any]], output_path: str, experiments: int) -> None:
    colors = {
        "Dummy": "#6B7280",
        "Optimal in Hindsight": "#111827",
        "SP": "#1C97B6",
        "TwoStageSP": "#4F46E5",
        "Expected value": "#BEA2FF",
        "ADP": "#F59E0B",
        "Hybrid": "#E3120B",
    }

    names = list(results.keys())

    # Wider figure, not taller
    fig = plt.figure(figsize=(16, 7.5), constrained_layout=True)

    # Outer layout: left = bar plot, right = stacked histograms
    outer_gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.85], wspace=0.08)

    # Left axis: top/bar plot
    ax_bar = fig.add_subplot(outer_gs[0, 0])

    # Right side: stacked small-multiple histograms
    right_gs = outer_gs[0, 1].subgridspec(len(names), 1, hspace=0.06)
    ax_hists = [fig.add_subplot(right_gs[i, 0]) for i in range(len(names))]

    # -------------------------------------------------
    # LEFT: Average daily costs
    # -------------------------------------------------
    avg_costs = [results[name]["average_daily_cost"] for name in names]

    ax_bar.bar(
        [_policy_label(name) for name in names],
        avg_costs,
        color=[colors[name] for name in names],
    )
    ax_bar.set_ylabel("Average daily cost")
    ax_bar.set_title(f"Policy comparison across {experiments} experiments")
    ax_bar.tick_params(axis="x", labelrotation=20)

    for label in ax_bar.get_xticklabels():
        label.set_ha("right")

    ax_bar.grid(True, axis="y", alpha=0.3)

    # -------------------------------------------------
    # RIGHT: Small-multiple histograms
    # -------------------------------------------------
    all_costs = np.concatenate([
        results[name]["daily_costs"]
        for name in names
        if len(results[name]["daily_costs"])
    ])

    if len(all_costs) > 0:
        # Use common bins across all policies
        bins = np.histogram_bin_edges(all_costs, bins="auto")

        # Common y-limit across all small plots
        max_count = 0
        for name in names:
            counts, _ = np.histogram(results[name]["daily_costs"], bins=bins)
            if len(counts):
                max_count = max(max_count, counts.max())

        for i, (ax, name) in enumerate(zip(ax_hists, names)):
            data = results[name]["daily_costs"]
            color = colors[name]

            ax.hist(
                data,
                bins=bins,
                histtype="stepfilled",
                alpha=0.18,
                color=color,
            )

            ax.hist(
                data,
                bins=bins,
                histtype="step",
                linewidth=2.0,
                color=color,
            )

            # Mean line
            ax.axvline(
                results[name]["average_daily_cost"],
                linestyle="--",
                linewidth=1.5,
                color=color,
                alpha=0.9,
            )

            ax.set_ylim(0, max_count * 1.15)

            # Policy name on each row
            ax.set_ylabel(
                _policy_label(name),
                rotation=0,
                ha="right",
                va="center",
                labelpad=30,
            )

            ax.grid(True, axis="x", alpha=0.25)
            ax.grid(True, axis="y", alpha=0.15)

            # Hide x labels except for last subplot
            if i < len(ax_hists) - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Daily electricity cost")

    # Title for the right panel
    ax_hists[0].set_title("Distribution of daily costs")

    # Shared y-label for the histogram column
    fig.text(0.545, 0.5, "Frequency", va="center", rotation=90)

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