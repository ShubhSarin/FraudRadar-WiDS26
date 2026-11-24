"""Chronological boundary-snapped splits, SHA-256 manifests, and imbalanced fraud metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class TemporalStream:
    """Framework-agnostic container for a continuous-time interaction stream.

    Maps 1-to-1 with `torch_geometric.data.TemporalData(src, dst, t, msg, y)`.
    """

    src: np.ndarray  # shape: (E,), int64
    dst: np.ndarray  # shape: (E,), int64
    t: np.ndarray  # shape: (E,), int64 (elapsed seconds, monotonic non-decreasing)
    msg: np.ndarray  # shape: (E, D), float32
    y: np.ndarray  # shape: (E,), int64 in {0, 1}

    @property
    def num_events(self) -> int:
        return int(len(self.t))

    @property
    def num_nodes(self) -> int:
        if self.num_events == 0:
            return 0
        return int(max(self.src.max(), self.dst.max()) + 1)

    def slice_events(self, start: int, end: int) -> "TemporalStream":
        return TemporalStream(
            src=self.src[start:end].copy(),
            dst=self.dst[start:end].copy(),
            t=self.t[start:end].copy(),
            msg=self.msg[start:end].copy(),
            y=self.y[start:end].copy(),
        )


def strict_chronological_split(
    stream: TemporalStream,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[TemporalStream, TemporalStream, TemporalStream, dict[str, Any]]:
    """Split `stream` into (train, val, test, manifest) with strict timestamp separation.

    Requirements:
    1. Stable-sort `stream` by `t` if not already monotonic non-decreasing.
    2. Compute nominal split indices at `1 - val_ratio - test_ratio` (70%) and
       `1 - test_ratio` (85%), then snap each boundary forward so that:
           train.t.max() < val.t.min()   AND   val.t.max() < test.t.min()
       even when multiple interactions share the exact same integer-second timestamp!
    3. Build and return a `manifest` dictionary containing at least:
       - `"num_events"`: int
       - `"val_start_idx"`: int
       - `"test_start_idx"`: int
       - `"t_train_max"`: int
       - `"t_val_min"`: int
       - `"t_val_max"`: int
       - `"t_test_min"`: int
       - `"test_sha256"`: str (hex SHA-256 digest over test `src`, `dst`, `t`, `y`)
       - `"seen_train_nodes"`: set[int] or sorted list[int] of all nodes in `train`
    """
    raise NotImplementedError("Implement strict_chronological_split in Week 1 (Days 1-2).")


def verify_test_manifest(test_stream: TemporalStream, manifest: dict[str, Any]) -> None:
    """Verify `test_stream` matches `manifest['test_sha256']` and `manifest['t_test_min']`.

    Raises:
        ValueError: If `test_stream` timestamps or SHA-256 digest do not match `manifest`.
    """
    raise NotImplementedError("Implement verify_test_manifest in Week 1 (Days 1-2).")


def compute_fraud_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    seen_train_nodes: set[int],
    top_alert_frac: float = 0.01,
) -> dict[str, float]:
    """Compute imbalanced anomaly detection metrics without ROC-AUC theater.

    Requirements:
    - `"pr_auc"`: Overall step-wise Average Precision (`sklearn.metrics.average_precision_score`).
      NEVER use trapezoidal `auc(recall, precision)`.
    - `"recall_at_1pct"`: Fraction of all true positives (`y_true == 1`) captured inside the
      highest-scoring `max(1, int(top_alert_frac * len(y_score)))` predictions.
    - `"pr_auc_seen"`: `average_precision_score` on transductive events where BOTH `src[i]`
      AND `dst[i]` belong to `seen_train_nodes` (`float("nan")` if 0 positives in subset).
    - `"pr_auc_unseen"`: `average_precision_score` on inductive events where `src[i]` OR
      `dst[i]` was NEVER seen in `seen_train_nodes` (`float("nan")` if 0 positives in subset).
    """
    raise NotImplementedError("Implement compute_fraud_metrics in Week 1 (Days 1-2).")
