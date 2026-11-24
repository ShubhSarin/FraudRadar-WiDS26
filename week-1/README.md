# Week 1: Strict Causality, Leakage Traps & Causal Graph Baselines

> **Week 1 Rule:** **Zero continuous-time TGN code this week.** Before training recurrent graph memory on a GPU, we build an unbreakable causal evaluation harness, prove quantitatively how easily subtle temporal bugs fake `0.90+` scores, and engineer a tuned `Causal Rolling LightGBM + Optuna` baseline that our neural models will have to earn the right to beat.

---

## 1. Learning Objectives

By the end of Week 1, you will be able to:
1. Load continuous-time dynamic interaction streams (`D1–D3` via PyTorch Geometric's `JODIEDataset`) and synthesize a realistic **smurfing / burst payment fraud stream (`D4`)** with a mid-stream regime shift.
2. Enforce strict chronological splits (`70% train / 15% val / 15% locked test`) with **timestamp-boundary snapping**, `SHA-256` split-manifest locking, and **Seen (Transductive) vs. Unseen (Inductive)** node breakdowns.
3. Build a deterministic **`LeakageGuard`** enforcing 4 invariants—including **bit-for-bit Future-Perturbation Invariance**—and measure `inflation = PR-AUC_leaked - PR-AUC_causal` across **3 planted leakage traps** on `D1`.
4. Engineer an **$O(E)$ single-pass streaming feature extractor** computing strictly causal (`t' < t`) `1h / 24h / 7d` rolling node/edge statistics across 672,447 edges in seconds on CPU.
5. Benchmark **Config A** (`Feature-Only LightGBM`), **Config B** (`Causal Rolling LightGBM + Optuna`, 50 trials, seeds `42` & `123`), and **Config C** (`Static GraphSAGE` on train-only edges).

---

## 2. Day-by-Day Execution Plan

| Days | Milestone | Key Files to Implement |
| :--- | :--- | :--- |
| **Days 1–2** | Read [`NOTES_MATH_AND_CAUSALITY.md`](./NOTES_MATH_AND_CAUSALITY.md) + core readings. Implement chronological boundary snapping, `SHA-256` manifest locking, `PR-AUC` / `Recall@Top-1%` / inductive metrics, and the `D4` synthetic fraud-ring generator. | [`stubs/splits_and_metrics.py`](./stubs/splits_and_metrics.py)<br>[`stubs/synth_fraud_ring.py`](./stubs/synth_fraud_ring.py) |
| **Days 3–4** | Implement the 4 deterministic `LeakageGuard` checks + the 3 planted leakage traps on `D1` (`wikipedia`). Make all unit tests in `week-1/tests/` pass! | [`stubs/leakage_guard.py`](./stubs/leakage_guard.py)<br>[`tests/test_week1_contracts.py`](./tests/test_week1_contracts.py) |
| **Days 5–6** | Implement the $O(E)$ `deque`-based causal rolling feature engine (`1h / 24h / 7d`). Train & log **Config A**, **Config B** (50 Optuna trials, 2 seeds), and **Config C** (`Static GraphSAGE`, train-only adjacency) across `D1–D4`. | [`stubs/causal_features.py`](./stubs/causal_features.py)<br>[`stubs/baselines.py`](./stubs/baselines.py) |
| **Day 7** | Generate the **D1 Leakage Inflation Bar Chart**, compile the **Locked Validation Baseline Leaderboard (`Configs A–C`)**, and write your **1-Page Week 1 Diagnostic Report**. | Checkpoint 1 PR / Submission |

---

## 3. Curated Week 1 Reading List

Read these in order (~95 minutes total). Every paper and doc page maps directly to a Week 1 deliverable:

1. **[First-Principles Math & Causality Notes (`NOTES_MATH_AND_CAUSALITY.md`)](./NOTES_MATH_AND_CAUSALITY.md)** *(25 mins — Start Here!)*
   * Walks through the exact math of `PR-AUC` vs. `ROC-AUC` under `0.05%` imbalance, integer-second tie snapping, $O(E)$ amortized sliding-window feature extraction, and Future-Perturbation Invariance.
2. **[The Precision-Recall Plot Is More Informative than the ROC Plot on Imbalanced Datasets](https://doi.org/10.1371/journal.pone.0118432)** — *Saito & Rehmsmeier (PLOS ONE, 2015)* + **[Scikit-Learn `average_precision_score` Docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)** *(20 mins)*
   * **What to extract:** Why False Positive Rate $\frac{\text{FP}}{\text{FP}+\text{TN}}$ hides thousands of false alarms when $\text{TN} \gg \text{TP}$, and why step-wise `average_precision_score` must be used instead of trapezoidal `auc(recall, precision)`.
3. **[JODIE: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks](https://arxiv.org/abs/1908.01207)** — *Kumar et al. (KDD 2019)* + **[PyG `JODIEDataset` Docs](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.JODIEDataset.html)** *(20 mins)*
   * **What to extract:** Section 4's description of `wikipedia`, `reddit`, and `mooc` dynamic state-change labels (`y=1` when an action triggers a ban or course dropout), and how PyG represents streams via `TemporalData(src, dst, t, msg, y)`.
4. **[Leakage and the Reproducibility Crisis in ML-based Science](https://arxiv.org/abs/2207.07048)** — *Kapoor & Narayanan (Patterns, 2023)* *(15 mins — Skim Section 2)*
   * **What to extract:** Temporal leakage, non-causal aggregation across full timelines, and why validation hygiene must be enforced programmatically rather than by honor system.
5. **[Inductive Representation Learning on Large Graphs (GraphSAGE)](https://arxiv.org/abs/1706.02216)** — *Hamilton et al. (NeurIPS 2017)* + **[PyG `SAGEConv` Docs](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html)** *(15 mins)*
   * **What to extract:** How static neighborhood aggregation works—and why collapsing a timestamped stream into a static `train`-only graph loses burst cadence and temporal ordering (`Config C`).
6. **[LightGBM Parameters Tuning Guide](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)** & **[Optuna Docs](https://optuna.readthedocs.io/)** *(Reference while building Config B)*
   * **Practical tip:** On `0.05%–0.14%` positive rates, avoid blindly setting `scale_pos_weight = 2000`—extreme positive weights often destabilize Newton leaf steps and ruin probability ranking! Let Optuna search moderate `scale_pos_weight` $\in [1.0, 25.0]$ alongside `learning_rate`, `num_leaves`, `min_child_samples`, `subsample`, and `colsample_bytree` optimizing validation `PR-AUC`.

---

## 4. Dataset Reference & Important Quirks (`D1–D4`)

| ID | Dataset | Loader / Source | Edges | Nodes | Pos Rate (`y=1`) | Raw Edge Dim (`msg`) | Structural Note |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **D1** | **Wikipedia** | `JODIEDataset(root, "wikipedia")` | 157,474 | 9,227 (8,227 users + 1,000 pages) | `~0.14%` | `172-d` | **Bipartite:** `dst` IDs are *already* offset by `8,227` in PyG! Timestamps `t` are integer seconds (`~30 days`). |
| **D2** | **Reddit** | `JODIEDataset(root, "reddit")` | 672,447 | 10,984 (10,000 users + 984 subs) | `~0.05%` | `172-d` | **Bipartite:** Largest stream. Naive $O(E^2)$ pandas windowing will hang for hours—use $O(E)$ `deque` pointers! |
| **D3** | **MOOC** | `JODIEDataset(root, "mooc")` | 411,749 | 7,144 (7,047 users + 97 items) | `~0.99%` | `4-d` | **Bipartite:** Weak edge features—graph topology & $\Delta t$ timing carry the entire anomaly signal! |
| **D4** | **Synthetic Fraud Ring** | `synth_fraud_ring.py` | ~150,000 | ~5,000 accounts | `~1.00%` | `16-d` | **Unipartite Directed:** Any account `0..4999` can act as *both* sender (`src`) and receiver (`dst`)! |

### Colab Setup Note (Cache Processed Datasets to Drive!)
When running `Config C (Static GraphSAGE)` on Google Colab T4, install PyG cleanly in ~15 seconds (no C++ source compilation required):
```bash
pip install -q torch_geometric lightgbm optuna
```
Mount Google Drive and point `JODIEDataset(root="/content/drive/MyDrive/FraudRadar/data_cache", name=...)` so processed `.pt` tensors load in **<2 seconds** on future sessions.

---

## 5. Week 1 Deliverable Checklist (Checkpoint 1 Gate)

Before unlocking Week 2 (`TGN`), verify every box below:

- [ ] **All Contract Tests Green:** `pytest week-1/tests/ -v` passes with zero errors (replace `NotImplementedError` bodies in `week-1/stubs/` or import your implementations from `fraudradar/`).
- [ ] **Split-Timestamp Manifests Locked:** `70/15/15` split manifests (`SHA-256` hashes + strict $t_{\text{train\_max}} < t_{\text{val\_min}} < t_{\text{test\_min}}$) generated for `D1–D4`. **Zero** evaluation runs touch the 15% locked test split in Week 1!
- [ ] **3/3 Planted Leakage Traps Quantified & Caught on `D1`:**
  1. *Trap 1:* Random `70/15/15` edge split.
  2. *Trap 2:* Pre-predict state/history update (updating node history *before* scoring the current event/batch).
  3. *Trap 3:* Full-timeline graph statistics (computing degree, PageRank, or historical user rates across the entire dataset before splitting).
  * `LeakageGuard` deterministically raises `LeakageGuardError` on all 3 traps AND passes **bit-for-bit Future-Perturbation Invariance** + **Shuffled-Label Collapse** on your clean causal pipeline.
- [ ] **Validation Baseline Leaderboard (`Configs A, B, C`, Seeds `42` & `123`):**
  * Reports Validation `PR-AUC` (mean $\pm$ seed spread), `Recall@Top-1% Alerts`, `Seen (Transductive) PR-AUC`, `Unseen (Inductive) PR-AUC`, and wall-clock runtime across `D1–D4`.
  * `Config B (Causal Rolling LightGBM + Optuna)` cleanly beats `Config A (Feature-Only LightGBM)`.
- [ ] **1-Page Diagnostic Report:** Summarizes the D1 leakage inflation bar chart (`inflation = PR-AUC_leaked - PR-AUC_causal`), explains which causal rolling features drove `Config B`'s gains (especially on `D3` MOOC and `D4` fraud rings), and analyzes where `Static GraphSAGE (C)` falls short without continuous-time ordering.
