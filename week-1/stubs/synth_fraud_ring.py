"""D4 Generator: Synthetic Continuous-Time Payment Fraud Ring Stream."""

from __future__ import annotations

from typing import Any
import numpy as np

from stubs.splits_and_metrics import TemporalStream


def generate_synthetic_fraud_ctdg(
    num_accounts: int = 5_000,
    num_edges: int = 150_000,
    fraud_rate: float = 0.01,
    horizon_seconds: int = 30 * 86_400,
    edge_dim: int = 16,
    seed: int = 42,
) -> tuple[TemporalStream, dict[str, Any]]:
    """Generate D4: an unipartite directed payment CTDG with smurfing/burst rings.

    Required Properties (per FraudRadar Spec):
    1. Unipartite directed graph: any account in `[0, num_accounts)` can send (`src`)
       and receive (`dst`), with `src[i] != dst[i]`.
    2. Background commerce (~`1 - fraud_rate` edges, `y = 0`): recurring accounts with
       skewed hub popularity and realistic timestamps across `[0, horizon_seconds]`.
    3. Planted smurfing / burst rings (~`fraud_rate` edges, `y = 1`): coordinated fan-out /
       fan-in or multi-hop mule bursts within tight temporal windows (`60s - 900s`), with
       some bad actors recurring across multiple rings.
    4. Mid-stream regime shift around `0.5 * horizon_seconds` (e.g., shifting split-amount
       distributions and channel features so static feature thresholds degrade).
    5. Metadata dict returning `"ring_ids"` (per-edge ring ID or `-1` for benign) and
       `"shift_timestamp"` so Week 4 camouflage & error analyses can slice by ring!

    Returns:
        stream: Chronologically sorted `TemporalStream` (`int64` src/dst/t/y, `float32` msg).
        metadata: Dict containing ground-truth ring annotations and shift timestamp.
    """
    raise NotImplementedError("Implement generate_synthetic_fraud_ctdg in Week 1 (Days 1-2).")
