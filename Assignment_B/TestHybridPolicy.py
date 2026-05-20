"""Test script to evaluate the Hybrid policy over 100 experiment days."""

from __future__ import annotations
import time

import numpy as np

import Policies.SP_policy_27 as Hybrid_policy_27
from SimulationEnvironment import RestaurantSimulationEnvironment


def test_hybrid_policy(experiments: int = 100) -> None:
    """Evaluate the hybrid policy over multiple experiment days.
    
    Args:
        experiments: Number of days to evaluate (default 100)
    """
    start_time = time.perf_counter()
    env = RestaurantSimulationEnvironment()
    
    if experiments <= 0:
        raise ValueError("experiments must be positive")
    
    day_indices = list(range(min(experiments, env.num_days)))
    daily_costs = np.zeros(len(day_indices), dtype=float)
    
    print(f"Evaluating Hybrid policy over {len(day_indices)} days...")
    
    for idx, day in enumerate(day_indices):
        if hasattr(Hybrid_policy_27, "reset"):
            try:
                Hybrid_policy_27.reset()
            except TypeError:
                pass
        
        episode = env.evaluate_policy(Hybrid_policy_27, day=day)
        daily_costs[idx] = float(episode["total_cost"])
        
        if (idx + 1) % 10 == 0:
            print(f"  Completed day {idx + 1}/{len(day_indices)}")
    
    average_cost = float(np.mean(daily_costs))
    std_cost = float(np.std(daily_costs))
    min_cost = float(np.min(daily_costs))
    max_cost = float(np.max(daily_costs))
    
    elapsed_seconds = float(time.perf_counter() - start_time)

    print("\n" + "=" * 50)
    print("Hybrid Policy Performance Summary")
    print("=" * 50)
    print(f"Number of experiments:  {len(day_indices)}")
    print(f"Average daily cost:     {average_cost:.2f}")
    print(f"Std dev of daily cost:  {std_cost:.2f}")
    print(f"Min daily cost:         {min_cost:.2f}")
    print(f"Max daily cost:         {max_cost:.2f}")
    print(f"Elapsed time:           {elapsed_seconds:.2f} seconds")
    print("=" * 50)


if __name__ == "__main__":
    test_hybrid_policy(experiments=100)
