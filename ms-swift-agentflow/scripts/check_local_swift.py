#!/usr/bin/env python3
"""Validate this plugin against an installed local ms-swift checkout."""

from swift.rewards import orms
from swift.rollout.multi_turn import multi_turns

import swift_plugin  # noqa: F401

assert "mat_agentflow" in multi_turns
assert "mat_agentflow_reward" in orms
print("ms-swift plugin registration OK")
