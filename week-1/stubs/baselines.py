"""Week 1 Baseline Interfaces: Config A, Config B (Optuna), and Config C (Static GraphSAGE)."""

from __future__ import annotations

from typing import Any
import numpy as np

from stubs.splits_and_metrics import TemporalStream


def run_config_a_feature_only_lgbm(
    train: TemporalStream,
    val: TemporalStream,
    seed: int = 42,
) -> dict[str, Any]:
    """Config A (Lower Bound): LightGBM trained strictly on raw edge features `msg`.

    Returns:
        Dict containing `"val_probs"` (`np.ndarray`), `"val_metrics"` (`dict[str, float]`),
        `"model"`, and `"train_seconds"` (`float`).
    """
    raise NotImplementedError("Implement Config A (Feature-Only LightGBM) in Week 1 (Days 5-6).")


def run_config_b_causal_rolling_lgbm_optuna(
    train: TemporalStream,
    val: TemporalStream,
    n_trials: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """Config B (Primary Tabular Baseline): LightGBM + Optuna (`n_trials=50`) trained on
    raw edge features `msg` concatenated with strictly causal `1h / 24h / 7d` rolling stats.

    Causality Note:
        When computing causal rolling features for `val`, stream `train` + `val` in order
        so `val` nodes retain their true prior `train` history without ever peeking forward!

    Returns:
        Dict containing `"val_probs"`, `"val_metrics"`, `"best_params"`, `"feature_importances"`,
        `"model"`, and `"train_seconds"`.
    """
    raise NotImplementedError("Implement Config B (Causal Rolling LightGBM + Optuna) in Week 1 (Days 5-6).")


def run_config_c_static_graphsage(
    train: TemporalStream,
    val: TemporalStream,
    hidden_dim: int = 64,
    max_epochs: int = 25,
    patience: int = 5,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    """Config C (Structural Bound): 2-layer Static GraphSAGE + edge anomaly MLP.

    Causality Rule:
        The static message-passing adjacency (`edge_index`) MUST be constructed exclusively
        from `train` edges (`t <= t_train_max`). Never include validation or test edges in
        the static graph structure! Optimize Focal Loss or weighted BCE with early stopping
        on validation `PR-AUC`.

    Returns:
        Dict containing `"val_probs"`, `"val_metrics"`, `"best_epoch"`, and `"train_seconds"`.
    """
    raise NotImplementedError("Implement Config C (Static GraphSAGE) in Week 1 (Days 5-6).")
