"""Two-stage stochastic programming policy wrapper.

This module reuses `SP_policy_27.select_action` with a fixed two-stage
lookahead so the comparison runner can treat it as a separate policy.
"""

from __future__ import annotations

from typing import Any, Dict

from . import SP_policy_27


def select_action(state: Dict[str, Any]) -> Dict[str, Any]:
    return SP_policy_27.select_action(state, lookahead=2)
