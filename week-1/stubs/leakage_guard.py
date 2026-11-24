"""Deterministic LeakageGuard invariants and D1 planted leakage trap injectors."""

from __future__ import annotations

from typing import Callable
import numpy as np

from stubs.splits_and_metrics import TemporalStream


class LeakageGuardError(AssertionError):
    """Raised whenever a temporal causality or evaluation invariant is violated."""


class LeakageGuard:
    """Non-negotiable deterministic referee with veto power over every reported score."""

    @staticmethod
    def check_split_monotonicity(
        train: TemporalStream,
        val: TemporalStream,
        test: TemporalStream,
    ) -> None:
        """Invariant #1: Verify internal timestamp monotonicity, non-empty splits, and
        strict cross-split ordering (`train.t.max() < val.t.min()` and `val.t.max() < test.t.min()`).

        Raises:
            LeakageGuardError: On any ordering or overlap violation.
        """
        raise NotImplementedError("Implement Invariant #1 in Week 1 (Days 3-4).")

    @staticmethod
    def check_pre_forward_timestamps(
        batch_src: np.ndarray,
        batch_dst: np.ndarray,
        batch_t: np.ndarray,
        node_last_update_t: np.ndarray,
        sampled_neighbor_eids: np.ndarray,
        sampled_neighbor_t: np.ndarray,
        batch_start_eid: int,
    ) -> None:
        """Invariant #2: Verify before a batch forward pass that:
        - No touched node in `batch_src` or `batch_dst` has `node_last_update_t > batch_t.min()`.
        - Every sampled neighbor edge ID is strictly `< batch_start_eid` (from a prior batch)
          and has timestamp `<= batch_t.min()`.

        Raises:
            LeakageGuardError: If memory or neighbor history already contains current/future events.
        """
        raise NotImplementedError("Implement Invariant #2 in Week 1 (Days 3-4).")

    @staticmethod
    def check_future_perturbation_invariance(
        predict_stream_fn: Callable[[TemporalStream], np.ndarray],
        stream: TemporalStream,
        t_star: int,
        seed: int = 42,
    ) -> None:
        """Invariant #3 (Bit-for-Bit Future-Perturbation Invariance):
        1. Run `predict_stream_fn` on a clean copy of `stream` with fixed `seed`.
        2. Create a corrupted copy of `stream` where EVERY event with `t > t_star` has
           scrambled `dst` node IDs and randomized `msg` features.
        3. Re-run `predict_stream_fn` with the exact same `seed`.
        4. Assert predictions for all events with `t <= t_star` are bit-for-bit identical
           (`np.array_equal(preds_clean[past_mask], preds_corrupted[past_mask])`).

        Raises:
            LeakageGuardError: If any prediction at `t <= t_star` changes by even 1 ULP,
                or if `t_star` does not partition `stream` into non-empty past and future subsets.
        """
        raise NotImplementedError("Implement Invariant #3 in Week 1 (Days 3-4).")

    @staticmethod
    def check_shuffled_label_collapse(
        train_and_score_val_fn: Callable[[TemporalStream, TemporalStream, int], float],
        train: TemporalStream,
        val: TemporalStream,
        tolerance_multiplier: float = 3.0,
        seed: int = 42,
    ) -> float:
        """Invariant #4: Permute `train.y` randomly, train via `train_and_score_val_fn`,
        and verify validation `PR-AUC` collapses near the validation positive base rate
        (`<= max(base_rate * tolerance_multiplier, base_rate + 0.02)`).

        Returns:
            The shuffled-label validation `PR-AUC`.

        Raises:
            LeakageGuardError: If validation `PR-AUC` remains suspiciously elevated under random labels.
        """
        raise NotImplementedError("Implement Invariant #4 in Week 1 (Days 3-4).")


# ---------------------------------------------------------------------------
# Planted Leakage Traps on D1 (Used to quantify `inflation` and test the Guard)
# ---------------------------------------------------------------------------


def inject_trap_random_split(
    stream: TemporalStream,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[TemporalStream, TemporalStream, TemporalStream]:
    """Planted Trap #1: Split `stream` randomly (shuffling across time) instead of chronologically."""
    raise NotImplementedError("Implement Planted Trap #1 in Week 1 (Days 3-4).")


def inject_trap_pre_update_features(
    src: np.ndarray,
    dst: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Planted Trap #2: Simulate pre-predict state update by updating node event/anomaly counters
    BEFORE reading features for the current interaction (leaking current batch/event state!).
    """
    raise NotImplementedError("Implement Planted Trap #2 in Week 1 (Days 3-4).")


def inject_trap_full_timeline_stats(stream: TemporalStream) -> np.ndarray:
    """Planted Trap #3: Compute node degrees, hub scores/PageRank, or historical rates across
    100% of `stream` (train + val + test combined) instead of strictly causal windows.
    """
    raise NotImplementedError("Implement Planted Trap #3 in Week 1 (Days 3-4).")
