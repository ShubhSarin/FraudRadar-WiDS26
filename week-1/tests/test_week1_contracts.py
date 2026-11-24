"""Self-grading Week 1 contract tests for FraudRadar.

Run from repository root:
    pytest week-1/tests/ -v
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

WEEK1_DIR = Path(__file__).resolve().parents[1]
if str(WEEK1_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK1_DIR))

from stubs.causal_features import extract_causal_rolling_features
from stubs.leakage_guard import LeakageGuard, LeakageGuardError
from stubs.splits_and_metrics import (
    TemporalStream,
    compute_fraud_metrics,
    strict_chronological_split,
    verify_test_manifest,
)
from stubs.synth_fraud_ring import generate_synthetic_fraud_ctdg


def _make_toy_stream(n: int = 200, seed: int = 42) -> TemporalStream:
    rng = np.random.default_rng(seed)
    # Deliberately create repeated integer-second timestamps around 70% (idx 140) and 85% (idx 170)
    t = np.sort(rng.integers(0, 50, size=n, dtype=np.int64)) * 100
    t[138:144] = t[140]
    t[168:174] = t[170]
    src = rng.integers(0, 30, size=n, dtype=np.int64)
    dst = (src + rng.integers(1, 15, size=n, dtype=np.int64)) % 30
    msg = rng.standard_normal((n, 8)).astype(np.float32)
    y = (rng.random(n) < 0.10).astype(np.int64)
    y[0] = 1
    y[-1] = 1
    return TemporalStream(src=src, dst=dst, t=t, msg=msg, y=y)


class TestChronologicalSplitsAndManifest:
    def test_boundary_snapping_eliminates_timestamp_ties(self) -> None:
        stream = _make_toy_stream()
        train, val, test, manifest = strict_chronological_split(stream, 0.15, 0.15)

        assert train.num_events + val.num_events + test.num_events == stream.num_events
        assert train.t.max() < val.t.min(), "Train and Val share boundary timestamps!"
        assert val.t.max() < test.t.min(), "Val and Test share boundary timestamps!"
        assert manifest["t_train_max"] == int(train.t.max())
        assert manifest["t_test_min"] == int(test.t.min())

    def test_manifest_detects_test_tampering(self) -> None:
        stream = _make_toy_stream()
        _, _, test, manifest = strict_chronological_split(stream, 0.15, 0.15)

        # Clean verification passes
        verify_test_manifest(test, manifest)

        # Flipping a single test label MUST trigger ValueError
        tampered = test.slice_events(0, test.num_events)
        tampered.y[0] = 1 - tampered.y[0]
        with pytest.raises(ValueError):
            verify_test_manifest(tampered, manifest)


class TestImbalancedMetrics:
    def test_pr_auc_recall_at_1pct_and_inductive_split(self) -> None:
        y_true = np.zeros(200, dtype=np.int64)
        y_true[[5, 25, 100, 150]] = 1  # 4 true anomalies (2% prevalence)

        y_score = np.linspace(0.01, 0.20, 200)
        # Put 2 anomalies in the top 1% (top 2 alerts out of 200)
        y_score[5] = 0.99
        y_score[150] = 0.95
        y_score[25] = 0.50
        y_score[100] = 0.40

        src = np.arange(200, dtype=np.int64) % 20
        dst = (src + 1) % 20
        # Mark node 999 on index 150 as an unseen inductive node
        src[150] = 999
        seen_nodes = set(range(20))

        metrics = compute_fraud_metrics(y_true, y_score, src, dst, seen_nodes, top_alert_frac=0.01)
        assert np.isclose(metrics["recall_at_1pct"], 0.50), "2 of 4 anomalies are in top-2 alerts!"
        assert 0.50 < metrics["pr_auc"] <= 1.0
        assert not np.isnan(metrics["pr_auc_seen"])
        assert not np.isnan(metrics["pr_auc_unseen"])


class TestCausalRollingFeatures:
    def test_same_second_events_do_not_leak_into_each_other(self) -> None:
        # Three events involving node 0 at the EXACT same timestamp t=1000
        src = np.array([0, 0, 0, 0], dtype=np.int64)
        dst = np.array([1, 2, 3, 4], dtype=np.int64)
        t = np.array([100, 1000, 1000, 2000], dtype=np.int64)

        feats_a, names = extract_causal_rolling_features(src, dst, t)
        assert feats_a.shape[0] == 4
        assert feats_a.shape[1] == len(names) and len(names) >= 8
        assert np.isfinite(feats_a).all()

        # Swapping or removing event index 2 (also at t=1000) MUST NOT alter features of event index 1!
        feats_b, _ = extract_causal_rolling_features(src[:2], dst[:2], t[:2])
        np.testing.assert_array_equal(
            feats_a[1],
            feats_b[1],
            err_msg="Intra-second leakage: event at t=1000 saw sibling event at t=1000!",
        )

    def test_amortized_linear_scaling_speed(self) -> None:
        rng = np.random.default_rng(123)
        n = 50_000
        src = rng.integers(0, 1_000, size=n, dtype=np.int64)
        dst = rng.integers(1_000, 1_500, size=n, dtype=np.int64)
        t = np.cumsum(rng.integers(0, 30, size=n, dtype=np.int64))

        t0 = time.perf_counter()
        feats, _ = extract_causal_rolling_features(src, dst, t)
        elapsed = time.perf_counter() - t0

        assert feats.shape[0] == n
        assert elapsed < 3.0, f"Rolling extractor took {elapsed:.2f}s on 50k edges (likely O(E^2))!"


class TestLeakageGuardInvariants:
    def test_pre_forward_timestamp_guard_vetoes_future_memory(self) -> None:
        batch_src = np.array([1, 2], dtype=np.int64)
        batch_dst = np.array([3, 4], dtype=np.int64)
        batch_t = np.array([500, 510], dtype=np.int64)
        node_last_update = np.array([0, 490, 505, 100, 200], dtype=np.int64)  # node 2 has 505 > 500!

        with pytest.raises(LeakageGuardError):
            LeakageGuard.check_pre_forward_timestamps(
                batch_src=batch_src,
                batch_dst=batch_dst,
                batch_t=batch_t,
                node_last_update_t=node_last_update,
                sampled_neighbor_eids=np.array([10, 11], dtype=np.int64),
                sampled_neighbor_t=np.array([400, 450], dtype=np.int64),
                batch_start_eid=20,
            )

    def test_future_perturbation_invariance_passes_causal_and_vetoes_leaker(self) -> None:
        stream = _make_toy_stream(n=120, seed=7)
        t_star = int(np.median(stream.t))

        # 1. Strictly causal mock predictor (cumulative mean of past msg[:, 0] strictly < t)
        def clean_causal_predictor(s: TemporalStream) -> np.ndarray:
            out = np.zeros(s.num_events, dtype=np.float64)
            running_sum = 0.0
            running_cnt = 0
            i = 0
            while i < s.num_events:
                j = i
                while j < s.num_events and s.t[j] == s.t[i]:
                    j += 1
                mean_before = running_sum / running_cnt if running_cnt > 0 else 0.0
                out[i:j] = 1.0 / (1.0 + np.exp(-(s.msg[i:j, 0] - mean_before)))
                running_sum += float(s.msg[i:j, 0].sum())
                running_cnt += j - i
                i = j
            return out

        LeakageGuard.check_future_perturbation_invariance(clean_causal_predictor, stream, t_star)

        # 2. Subtly leaked predictor (standardizes msg[:, 0] using global full-stream mean!)
        def leaked_global_scaler_predictor(s: TemporalStream) -> np.ndarray:
            global_mean = float(s.msg[:, 0].mean())  # Peeks at t > t_star!
            return 1.0 / (1.0 + np.exp(-(s.msg[:, 0] - global_mean)))

        with pytest.raises(LeakageGuardError):
            LeakageGuard.check_future_perturbation_invariance(
                leaked_global_scaler_predictor, stream, t_star
            )


class TestSyntheticFraudRingGenerator:
    def test_d4_generator_structure_and_imbalance(self) -> None:
        stream, meta = generate_synthetic_fraud_ctdg(
            num_accounts=500,
            num_edges=10_000,
            fraud_rate=0.02,
            seed=42,
        )
        assert stream.num_events == 10_000
        assert (np.diff(stream.t) >= 0).all(), "D4 timestamps must be sorted chronologically!"
        assert (stream.src != stream.dst).all(), "Self-loops should not appear in payment transfers!"
        pos_rate = float(stream.y.mean())
        assert 0.005 <= pos_rate <= 0.05, f"Unexpected fraud rate {pos_rate:.4f}"
        assert "ring_ids" in meta and "shift_timestamp" in meta
