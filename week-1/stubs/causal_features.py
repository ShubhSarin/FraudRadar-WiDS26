"""O(E) single-pass strictly causal (< t) rolling graph feature extractor."""

from __future__ import annotations

import numpy as np

# Elapsed seconds matching JODIEDataset("wikipedia" | "reddit" | "mooc") and D4
WINDOWS_SEC: dict[str, int] = {
    "1h": 3_600,
    "24h": 86_400,
    "7d": 604_800,
}


def extract_causal_rolling_features(
    src: np.ndarray,
    dst: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Compute strictly causal rolling graph statistics in O(E) amortized time.

    Causality & Complexity Contract:
    1. `t` must be monotonic non-decreasing (`np.all(np.diff(t) >= 0)`). Raise `ValueError` if not.
    2. STRICT `< t` CAUSALITY (Two-Phase Timestamp Commit):
       If multiple events share the exact same timestamp `tau` (`t[i] == t[i+1] == tau`),
       NONE of the events at `tau` may see each other's updates! Read features for the entire
       `tau` slice from history `< tau` first, and commit all events at `tau` into your
       per-node `collections.deque` structures ONLY after advancing past `tau`.
    3. AMORTIZED O(E) RUNTIME:
       Maintain sliding `deque` windows per node so each event enters and exits each window
       (`1h`, `24h`, `7d`) at most once. Must process 672k edges in a few seconds on CPU.
    4. Recommended feature set (at least 8 columns, `float32`, zero `NaN` or `inf`):
       - Source & destination log-recency (`log1p(t - last_seen_t)`, plus cold-start indicators)
       - Source & destination causal rolling event counts across `1h`, `24h`, `7d`
       - Short-vs.-long activity surge ratio (`deg_1h / (deg_24h / 24.0 + 1.0)`)
       - Goh-Barabasi inter-event burstiness `B = (sigma - mu) / (sigma + mu + eps)`
       - Prior `(src, dst)` repeat interaction count and/or unique 24h counterparty count

    Returns:
        features: `np.ndarray` of shape `(len(t), F)` with `dtype=np.float32`.
        feature_names: `list[str]` of length `F` naming every column for interpretability.
    """
    raise NotImplementedError("Implement extract_causal_rolling_features in Week 1 (Days 5-6).")
