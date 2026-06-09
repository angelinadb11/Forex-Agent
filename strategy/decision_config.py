from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionConfig:
    """Controls consensus scoring and primary-agent vote rules."""

    use_zone_cluster: bool = False
    use_rsi_gate: bool = False


LEGACY_DECISION_CONFIG = DecisionConfig(use_zone_cluster=False, use_rsi_gate=False)
ZONE_RSI_DECISION_CONFIG = DecisionConfig(use_zone_cluster=True, use_rsi_gate=True)
