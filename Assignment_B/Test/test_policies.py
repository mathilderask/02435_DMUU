

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import SP_policy_27 as SP_policy
import Hybrid_policy_27 as hybrid_policy

from v2_Checks import check_and_sanitize_action

state = {
    "T1": 17.0,
    "T2": 17.0,
    "H": 50.0,
    "Occ1": 30.0,
    "Occ2": 20.0,
    "price_t": 5.0,
    "price_previous": 4.5,
    "vent_counter": 0,
    "low_override_r1": 1,
    "low_override_r2": 1,
    "current_time": 0,
}

PowerMax = {1: 3.0, 2: 3.0}

action_SP = check_and_sanitize_action(SP_policy, state, PowerMax)
action_hybrid = check_and_sanitize_action(hybrid_policy, state, PowerMax)

print(" ")
print("Sanitized action SP:")
print(action_SP)

print(" ")
print("Sanitized action Hybrid:")
print(action_hybrid)