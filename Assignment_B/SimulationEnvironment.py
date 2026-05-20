"""Reusable restaurant HVAC simulation environment.

This module packages the state update logic used in the plotting scripts into a
small environment class so policies can be evaluated.

The environment follows the same hourly dynamics as the existing diagnostic
scripts:
- low-temperature hysteresis for each room,
- high-temperature shutoff,
- humidity-triggered ventilation,
- ventilation inertia,
- linear temperature and humidity dynamics.

Example:
    from SimulationEnvironment import RestaurantSimulationEnvironment
    import DummyPolicy

    env = RestaurantSimulationEnvironment()
    result = env.evaluate_policy(DummyPolicy)
    print(result["total_cost"])
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

import Data.v2_SystemCharacteristics as SystemCharacteristics


def _load_params() -> Dict[str, Any]:
    if hasattr(SystemCharacteristics, "get_fixed_data"):
        return dict(SystemCharacteristics.get_fixed_data())
    if hasattr(SystemCharacteristics, "fetch_data"):
        return dict(SystemCharacteristics.fetch_data())
    raise AttributeError("Data.v2_SystemCharacteristics must define get_fixed_data() or fetch_data().")


PARAMS = _load_params()


def _par(*names: str, default: Any) -> Any:
    for name in names:
        if name in PARAMS:
            return PARAMS[name]
    return default


def _as_2d_array(data: Any, name: str) -> np.ndarray:
    array = np.asarray(data, dtype=float)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional.")
    return array


def _load_csv(path: Path) -> np.ndarray:
    return pd.read_csv(path).to_numpy(dtype=float)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


@dataclass
class StepResult:
    state: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class RestaurantSimulationEnvironment:
    """Hour-by-hour simulation environment for HVAC policy evaluation."""

    def __init__(
        self,
        price_data: Optional[Any] = None,
        occ1_data: Optional[Any] = None,
        occ2_data: Optional[Any] = None,
        *,
        day: int = 0,
        seed: Optional[int] = None,
        data_dir: Optional[Path | str] = None,
        initial_temperature: Optional[float] = None,
        initial_humidity: Optional[float] = None,
    ) -> None:
        self.params = PARAMS
        self.rng = np.random.default_rng(seed)

        base_dir = Path(data_dir) if data_dir is not None else Path(__file__).parent / "Data"

        if price_data is None:
            price_data = _load_csv(base_dir / "v2_PriceData.csv")
        if occ1_data is None:
            occ1_data = _load_csv(base_dir / "OccupancyRoom1.csv")
        if occ2_data is None:
            occ2_data = _load_csv(base_dir / "OccupancyRoom2.csv")

        self.price_data = _as_2d_array(price_data, "price_data")
        self.occ1_data = _as_2d_array(occ1_data, "occ1_data")
        self.occ2_data = _as_2d_array(occ2_data, "occ2_data")

        self.num_days = min(self.price_data.shape[0], self.occ1_data.shape[0], self.occ2_data.shape[0])
        self.num_timeslots = int(_par("num_timeslots", "T", default=self.price_data.shape[1]))

        self.P_max = float(_par("heating_max_power", "P_r", "Pr", "PowerMax", default=3.0))
        self.P_max_r1 = float(_par("heating_max_power_room1", "P1", "P_room1", default=self.P_max))
        self.P_max_r2 = float(_par("heating_max_power_room2", "P2", "P_room2", default=self.P_max))
        self.P_vent = float(_par("ventilation_power", "P_vent", "Pvent", default=1.0))

        self.T_low = float(_par("T_low", "Tlow", "temperature_low", default=18.0))
        self.T_ok = float(_par("T_OK", "TOK", "temperature_ok", default=20.0))
        self.T_high = float(_par("T_high", "THigh", "temperature_high", default=24.0))
        self.H_high = float(_par("H_high", "Hhigh", "humidity_high", default=60.0))

        self.z_exch = float(_par("heat_exchange_coeff", default=0.6))
        self.z_loss = float(_par("thermal_loss_coeff", default=0.1))
        self.z_conv = float(_par("heating_efficiency_coeff", default=1.0))
        self.z_cool = float(_par("heat_vent_coeff", default=0.7))
        self.z_occ = float(_par("heat_occupancy_coeff", default=0.02))
        self.eta_occ = float(_par("humidity_occupancy_coeff", default=0.15))
        self.eta_vent = float(_par("humidity_vent_coeff", default=5.0))

        self.T_initial = float(initial_temperature if initial_temperature is not None else _par("initial_temperature", default=self.T_ok))
        self.H_initial = float(initial_humidity if initial_humidity is not None else _par("initial_humidity", default=45.0))

        self.day = int(day)
        self.reset(day=day)

    def reset(self, *, day: Optional[int] = None) -> Dict[str, Any]:
        if day is not None:
            self.day = int(day)

        self.t = 0
        self.T1 = float(self.T_initial)
        self.T2 = float(self.T_initial)
        self.H = float(self.H_initial)
        self.low_override_r1 = 1 if self.T1 <= self.T_low else 0
        self.low_override_r2 = 1 if self.T2 <= self.T_low else 0
        self.vent_counter = 0
        self.total_cost = 0.0

        self.history: Dict[str, list[float]] = {
            "t": [],
            "T1": [],
            "T2": [],
            "H": [],
            "Occ1": [],
            "Occ2": [],
            "price": [],
            "p1": [],
            "p2": [],
            "v": [],
            "cost": [],
            "low_override_r1": [],
            "low_override_r2": [],
            "vent_counter": [],
        }

        return self.current_state()

    def _day_index(self) -> int:
        if self.num_days <= 0:
            raise ValueError("No exogenous data available.")
        return self.day % self.num_days

    def _time_index(self) -> int:
        return self.t % self.num_timeslots

    def _current_exogenous(self) -> tuple[float, float, float]:
        d = self._day_index()
        t = self._time_index()
        price = float(self.price_data[d, t])
        occ1 = float(self.occ1_data[d, t])
        occ2 = float(self.occ2_data[d, t])
        return price, occ1, occ2

    def _price_previous(self) -> float:
        d = self._day_index()
        t = self._time_index()
        if t == 0:
            return float(self.price_data[d, t])
        return float(self.price_data[d, t - 1])

    def current_state(self) -> Dict[str, Any]:
        price_t, occ1, occ2 = self._current_exogenous()
        return {
            "T1": float(self.T1),
            "T2": float(self.T2),
            "H": float(self.H),
            "Occ1": float(occ1),
            "Occ2": float(occ2),
            "price_t": float(price_t),
            "price_previous": float(self._price_previous()),
            "vent_counter": int(self.vent_counter),
            "low_override_r1": int(self.low_override_r1),
            "low_override_r2": int(self.low_override_r2),
            "current_time": int(self._time_index()),
        }

    def _sanitize_action(self, action: Dict[str, Any]) -> Dict[str, float | int]:
        return {
            "HeatPowerRoom1": float(np.clip(_to_float(action.get("HeatPowerRoom1", 0.0)), 0.0, self.P_max_r1)),
            "HeatPowerRoom2": float(np.clip(_to_float(action.get("HeatPowerRoom2", 0.0)), 0.0, self.P_max_r2)),
            "VentilationON": int(_to_float(action.get("VentilationON", 0.0)) > 0.5),
        }

    def _apply_overrules(self, action: Dict[str, Any]) -> Dict[str, float | int]:
        sanitized = self._sanitize_action(action)

        if self.H > self.H_high or self.vent_counter in (1, 2):
            sanitized["VentilationON"] = 1

        if self.T1 > self.T_high:
            sanitized["HeatPowerRoom1"] = 0.0
        elif self.low_override_r1 == 1 or self.T1 < self.T_low:
            sanitized["HeatPowerRoom1"] = self.P_max_r1

        if self.T2 > self.T_high:
            sanitized["HeatPowerRoom2"] = 0.0
        elif self.low_override_r2 == 1 or self.T2 < self.T_low:
            sanitized["HeatPowerRoom2"] = self.P_max_r2

        return sanitized

    def _update_low_override(self, temp: float, previous_flag: int) -> int:
        if temp <= self.T_low:
            return 1
        if previous_flag == 1 and temp < self.T_ok:
            return 1
        return 0

    def _outdoor_temperature(self, t: int) -> float:
        if "outdoor_temperature" in self.params:
            out = self.params["outdoor_temperature"]
            if isinstance(out, (list, tuple, np.ndarray)) and len(out) > 0:
                return float(out[min(max(t, 0), len(out) - 1)])
            return float(out)
        return 5.0

    def step(self, action: Dict[str, Any]) -> StepResult:
        if self.t >= self.num_timeslots:
            raise RuntimeError("Episode has already finished. Call reset() before stepping again.")

        actual_action = self._apply_overrules(action)
        price_t, occ1, occ2 = self._current_exogenous()

        p1 = float(actual_action["HeatPowerRoom1"])
        p2 = float(actual_action["HeatPowerRoom2"])
        v = int(actual_action["VentilationON"])

        cost = float(price_t) * (p1 + p2 + self.P_vent * v)
        reward = -cost
        self.total_cost += cost

        self.history["t"].append(float(self._time_index()))
        self.history["T1"].append(float(self.T1))
        self.history["T2"].append(float(self.T2))
        self.history["H"].append(float(self.H))
        self.history["Occ1"].append(float(occ1))
        self.history["Occ2"].append(float(occ2))
        self.history["price"].append(float(price_t))
        self.history["p1"].append(p1)
        self.history["p2"].append(p2)
        self.history["v"].append(float(v))
        self.history["cost"].append(cost)
        self.history["low_override_r1"].append(float(self.low_override_r1))
        self.history["low_override_r2"].append(float(self.low_override_r2))
        self.history["vent_counter"].append(float(self.vent_counter))

        t_out = self._outdoor_temperature(self._time_index())
        T1_next = (
            self.T1
            + self.z_exch * (self.T2 - self.T1)
            + self.z_loss * (t_out - self.T1)
            + self.z_conv * p1
            - self.z_cool * v
            + self.z_occ * occ1
        )
        T2_next = (
            self.T2
            + self.z_exch * (self.T1 - self.T2)
            + self.z_loss * (t_out - self.T2)
            + self.z_conv * p2
            - self.z_cool * v
            + self.z_occ * occ2
        )
        H_next = self.H + self.eta_occ * (occ1 + occ2) - self.eta_vent * v

        self.T1 = float(T1_next)
        self.T2 = float(T2_next)
        self.H = float(H_next)
        self.low_override_r1 = self._update_low_override(self.T1, self.low_override_r1)
        self.low_override_r2 = self._update_low_override(self.T2, self.low_override_r2)
        self.vent_counter = min(self.vent_counter + 1, 3) if v == 1 else 0
        self.t += 1

        done = self.t >= self.num_timeslots
        return StepResult(state=self.current_state(), reward=reward, done=done, info={"action": actual_action, "cost": cost})

    def _policy_action(self, policy: Any, state: Dict[str, Any]) -> Dict[str, Any]:
        if hasattr(policy, "select_action"):
            return policy.select_action(state)
        if callable(policy):
            return policy(state)
        raise TypeError("policy must be callable or expose select_action(state)")

    def run_episode(self, policy: Any, *, day: Optional[int] = None, max_steps: Optional[int] = None) -> Dict[str, np.ndarray]:
        self.reset(day=day)
        steps = self.num_timeslots if max_steps is None else min(int(max_steps), self.num_timeslots)

        for _ in range(steps):
            state = self.current_state()
            try:
                action = self._policy_action(policy, state)
            except Exception as e:
                print("POLICY ERROR:", repr(e))
                action = {"HeatPowerRoom1": 0.0, "HeatPowerRoom2": 0.0, "VentilationON": 0}
            self.step(action)

        return {key: np.asarray(values, dtype=float) for key, values in self.history.items()}

    def evaluate_policy(self, policy: Any, *, day: Optional[int] = None, max_steps: Optional[int] = None) -> Dict[str, Any]:
        history = self.run_episode(policy, day=day, max_steps=max_steps)
        total_cost = float(np.sum(history["cost"]))
        return {
            "history": history,
            "total_cost": total_cost,
            "mean_cost": float(np.mean(history["cost"])) if len(history["cost"]) else 0.0,
            "final_temperature_room1": float(self.T1),
            "final_temperature_room2": float(self.T2),
            "final_humidity": float(self.H),
        }

    def evaluate_policy_over_days(
        self,
        policy: Any,
        *,
        days: Optional[Iterable[int]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a policy on multiple experiment days and return daily costs."""
        if days is None:
            day_indices = list(range(self.num_days))
        else:
            day_indices = [int(day) for day in days]

        daily_costs = np.zeros(len(day_indices), dtype=float)
        histories: list[Dict[str, np.ndarray]] = []

        for idx, day in enumerate(day_indices):
            history = self.run_episode(policy, day=day)
            histories.append(history)
            daily_costs[idx] = float(np.sum(history["cost"]))

        return {
            "day_indices": np.asarray(day_indices, dtype=int),
            "daily_costs": daily_costs,
            "average_daily_cost": float(np.mean(daily_costs)) if len(daily_costs) else 0.0,
            "histories": histories,
        }


def evaluate_policy(policy: Any, *, day: int = 0, seed: Optional[int] = None) -> Dict[str, Any]:
    """Convenience wrapper for one-off policy evaluation."""
    env = RestaurantSimulationEnvironment(day=day, seed=seed)
    return env.evaluate_policy(policy, day=day)


if __name__ == "__main__":
    import Dummy_policy_27

    env = RestaurantSimulationEnvironment(day=0, seed=0)
    result = env.evaluate_policy(Dummy_policy_27)
    print(f"Total cost: {result['total_cost']:.2f}")
    print(f"Final temperatures: {result['final_temperature_room1']:.2f}, {result['final_temperature_room2']:.2f}")