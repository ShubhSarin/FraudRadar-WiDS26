# FraudRadar: Real-Time Fraud & Anomaly Detection

[![Program](https://img.shields.io/badge/WiDS_2026-Mentorship_Project-blue)](#4-week-curriculum-roadmap)
[![Stack](https://img.shields.io/badge/PyTorch_Geometric-TGN_%7C_LightGBM_%7C_Optuna-ee4c2c)](#stack--compute-limits)
[![Compute](https://img.shields.io/badge/Compute-Free_Colab_T4_%2B_CPU-f9ab00)](#stack--compute-limits)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> *Catch scammers, vandals, and hidden payment fraud rings as timestamped interactions unfold in real time—without ever peeking into the future.*

**FraudRadar** is a 4-week hands-on machine learning engineering & research project investigating continuous-time **Temporal Graph Neural Networks (TGNs)** versus strictly causal **Gradient Boosted Decision Trees (LightGBM)** on extreme-imbalance dynamic interaction streams (`0.05%` to `1.0%` positive anomaly rates).

---

## Core Research Question

> *At matched causal information and free-Colab compute, does continuous-time node memory (`TGN`) genuinely beat a tuned `LightGBM` with strictly causal rolling graph features on imbalanced dynamic anomaly detection—and how much of each model's gain survives adversarial edge and feature camouflage?*

---

## 4-Week Curriculum Roadmap

Resources, mathematical notes, and verification tests unlock week by week after each Friday checkpoint review.

| Week | Focus & Build | Deliverable Gate | Status |
| :--- | :--- | :--- | :--- |
| **[Week 1](./week-1/README.md)** | Chronological `70/15/15` loaders (`D1–D4`), deterministic `LeakageGuard` + 3 planted D1 traps, `Feature-Only LightGBM` (A), `Causal Rolling LightGBM + Optuna` (B), `Static GraphSAGE` (C) | 1-page D1 leakage inflation report + locked validation baseline leaderboard | **Unlocked** |
| **Week 2** | Continuous-time `TGNMemory` + `LastNeighborLoader` + `TransformerConv` pipeline with Drive/SQLite checkpointing; `Vanilla TGN` (Config D) | Verified `LeakageGuard` pass (4/4 invariants, 0/3 traps leaked) + Config D table + 3 forensic error analyses | *Unlocks Week 2* |
| **Week 3** | Two-stage link-prediction pretraining with historical & inductive hard negatives, `Focal Loss` + time-decay (`Full TGN`, E), `Hybrid TGN + Defense` (F) | Full validation ablation (`A–F`) + PR curves + Seen (transductive) vs. Unseen (inductive) breakdown | *Unlocks Week 3* |
| **Week 4** | Adversarial camouflage stress-test (edge $k$, feature $\alpha$, temporal smoothing), defense ablation, single-shot locked test evaluation, final report | Reproducible repo & Colab notebooks, 5–6 page report, 15-minute live demo on a surprise stream | *Unlocks Week 4* |

---

## Benchmark Streams (`D1–D4`)

| ID | Stream Type | Source | Scale & Imbalance | Core Challenge |
| :--- | :--- | :--- | :--- | :--- |
| **D1** | Rich-feature stream | `JODIEDataset("wikipedia")` | 157k edges, `~0.14%` pos, `172-d` | Extreme imbalance; host for planted leakage & camouflage traps |
| **D2** | Ultra-imbalanced bursty stream | `JODIEDataset("reddit")` | 672k edges, `~0.05%` pos, `172-d` | 1-in-2,000 anomaly rate; $O(E)$ feature scaling & hard-negative mining |
| **D3** | Weak-feature structural stream | `JODIEDataset("mooc")` | 412k edges, `~0.99%` pos, `4-d` | Minimal raw features; topology & inter-event timing must carry signal |
| **D4** | Synthetic payment fraud ring | Generated CTDG | ~150k edges, `~1.00%` pos, `16-d` | Planted smurfing/burst rings + mid-stream regime shift + camouflage |

---

## Non-Negotiable Ground Rules

1. **Zero `ROC-AUC` Theater:** On a `0.05%` anomaly stream, a trivial classifier can score `0.95 ROC-AUC` while drowning human analysts in 99% false positives. Every experiment reports **`PR-AUC`** (`average_precision_score`), **`Recall@Top-1% Alerts`**, and **Seen (Transductive) vs. Unseen (Inductive) `PR-AUC`**.
2. **Strict Chronological Causality ($t_{\text{train}} < t_{\text{val}} < t_{\text{test}}$):** Predictions on event $e_i$ at timestamp $t_i$ may **only** use information strictly prior to $t_i$. Every pipeline must pass the deterministic `LeakageGuard`—including bit-for-bit **Future-Perturbation Invariance**.
3. **Locked Test Window:** The final 15% chronological test window is touched **once per final configuration** via an isolated evaluator verified by a `SHA-256` split-timestamp manifest. Never tune hyperparameters, thresholds, or defenses on the test split.
4. **Honest Small-$N$ Comparisons:** Across 4 datasets and 2 seeds (`42`, `123`), report per-dataset scores, seed spread, and **Mean Rank**—treating gaps smaller than the seed spread as ties rather than running underpowered significance tests.

---

## Stack & Compute Limits

* **Stack:** Python 3.10+, PyTorch, PyTorch Geometric (`TGNMemory`, `LastNeighborLoader`, `TransformerConv`, `SAGEConv`), LightGBM, Optuna, scikit-learn, SQLite.
* **Compute Budget:** `1x Free Google Colab T4 (15 GB VRAM)` + local CPU for baselines and tests. Max `25 epochs` (early stopping patience `5`), `batch_size = 200`, $\le 10$ minutes per TGN run on T4.

### Local Quickstart (Week 1)

```bash
# Clone the repository
git clone https://github.com/ShubhSarin/FraudRadar-WiDS26.git
cd FraudRadar-WiDS26

# Create virtual environment & install dependencies (using uv or pip)
python -m venv .venv
# Windows: .venv\Scripts\activate | macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"

# Run the Week 1 causality & metric contract test suite
pytest week-1/tests/ -v
```

Head over to **[`week-1/README.md`](./week-1/README.md)** to begin Week 1!
