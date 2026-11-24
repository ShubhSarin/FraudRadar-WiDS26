# Week 1 Deep Dive: Metrics, Causality & $O(E)$ Streaming Graph Features

> This walkthrough derives the mathematical and algorithmic foundations you need for Week 1—so every design decision in your codebase has a clear *why* behind it.

---

## 1. Why `ROC-AUC` Fails Catastrophically at `0.05%` Anomaly Rates

Suppose we evaluate an anomaly detector on **Reddit (`D2`)**'s validation window of $N = 100,000$ interactions containing **`0.05%` true anomalies**:
* Positive events ($y = 1$, banned actions): $P = 50$
* Negative events ($y = 0$, normal posts): $N_- = 99,950$

Now imagine a noisy model flags **5,045 interactions** as high-risk alerts, catching $45$ of the $50$ true anomalies alongside **$5,000$ false alarms**:

1. **True Positive Rate (Recall):**
   $$\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{45}{50} = 0.90$$
2. **False Positive Rate:**
   $$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}} = \frac{5,000}{99,950} \approx 0.050$$

On an ROC curve ($\text{TPR}$ vs. $\text{FPR}$), hitting $(0.05, 0.90)$ produces a dazzling **`ROC-AUC` $\approx 0.96$**! Yet look at what happens to the human fraud analyst opening that alert queue:

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{45}{45 + 5,000} \approx 0.0089 \quad (\text{99.1\% False Alarms!})$$

### Our Two Primary Operational Metrics

1. **`PR-AUC` via Step-Wise Average Precision (`average_precision_score`):**
   $$\text{AP} = \sum_{n} (R_n - R_{n-1}) P_n$$
   * **Expected value of a random guesser:** Equals the base prevalence $\pi = \frac{P}{P + N_-}$ (`0.0014` on Wikipedia, `0.0005` on Reddit, `0.0099` on MOOC).
   * **Implementation Rule:** Always call `sklearn.metrics.average_precision_score(y_true, y_score)`. **Never** compute `sklearn.metrics.auc(recall, precision)`—linear trapezoidal interpolation between sparse recall jumps artificially inflates `PR-AUC`!
2. **`Recall@Top-1% Alerts`:**
   If an operations team has budget to review the highest-scoring $K = \lfloor 0.01 \cdot E_{\text{eval}} \rfloor$ events in a window, what fraction of all true anomalies in that window were caught?
   $$\text{Recall@1\%} = \frac{1}{\sum_{i=1}^{E_{\text{eval}}} y_i} \sum_{i \in \text{Top-}K(\hat{\mathbf{p}})} y_i$$

---

## 2. Continuous-Time Dynamic Graphs & The Integer-Second Split Trap

A **Continuous-Time Dynamic Graph (CTDG)** is an ordered stream of timestamped directed interaction events:

$$\mathcal{G} = \left\{ e_i = (u_i, v_i, t_i, \mathbf{m}_i, y_i) \right\}_{i=1}^{E}, \qquad 0 \le t_1 \le t_2 \le \dots \le t_E$$

where $u_i \in \mathcal{V}$ is the source node, $v_i \in \mathcal{V}$ is the destination node, $t_i \in \mathbb{R}_{\ge 0}$ is the timestamp (in **seconds**), $\mathbf{m}_i \in \mathbb{R}^d$ is the raw interaction feature vector, and $y_i \in \{0, 1\}$ is the ground-truth anomaly state-change label.

### Subtlety #1: Simultaneous-Second Ties at Split Boundaries
Notice the $\le$ (non-strict!) inequality $t_i \le t_{i+1}$. In `wikipedia`, `reddit`, and `mooc`, timestamps are **integer seconds**. If events $e_{110230}, e_{110231}, e_{110232}$ all occur at $t = 1,842,500$ and a naive `int(0.70 * E)` slice cuts between $e_{110230}$ and $e_{110231}$, then:

$$\max_{e \in \mathcal{D}_{\text{train}}} t_e = \min_{e \in \mathcal{D}_{\text{val}}} t_e = 1,842,500$$

violating strict chronological separation ($t_{\text{train}} < t_{\text{val}} < t_{\text{test}}$)!

**The Fix (Boundary Snapping):** After computing the nominal `70%` and `85%` indices, advance the split index with `np.searchsorted(t, t[raw_idx], side="right")` to the first event whose timestamp is **strictly greater** than the boundary timestamp:

$$\max_{e \in \mathcal{D}_{\text{train}}} t_e < \min_{e \in \mathcal{D}_{\text{val}}} t_e \qquad \text{and} \qquad \max_{e \in \mathcal{D}_{\text{val}}} t_e < \min_{e \in \mathcal{D}_{\text{test}}} t_e$$

### Subtlety #2: Transductive (Seen) vs. Inductive (Unseen) Events
Define the set of training nodes $\mathcal{V}_{\text{train}} = \bigcup_{e_i \in \mathcal{D}_{\text{train}}} \{u_i, v_i\}$. During validation or testing, an interaction $e_j = (u_j, v_j, t_j, \mathbf{m}_j, y_j)$ is:
* **Seen (Transductive):** $u_j \in \mathcal{V}_{\text{train}} \;\land\; v_j \in \mathcal{V}_{\text{train}}$ (both endpoints have historical training trajectories).
* **Unseen (Inductive):** $u_j \notin \mathcal{V}_{\text{train}} \;\lor\; v_j \notin \mathcal{V}_{\text{train}}$ (at least one cold-start node never observed in training!).

Reporting `PR-AUC` separately on `Seen` and `Unseen` subsets reveals whether a model relies purely on memorizing recurring bad actors or generalizes structurally to brand-new accounts.

---

## 3. Synthesizing `D4` (Payment Smurfing & Burst Fraud Rings)

While `D1–D3` are **bipartite** user–item streams (`u` $\in$ users, `v` $\in$ pages/subreddits/course-units), real financial transaction networks are **unipartite directed graphs** where any account $a \in \{0, \dots, |\mathcal{V}|-1\}$ can both send and receive funds—allowing multi-hop money laundering loops!

Your `D4` generator (~`150,000` edges, ~`5,000` accounts, `~1.0%` positive fraud edges, elapsed timestamps $t \in [0, 2.592 \times 10^6]$ seconds = 30 days, `16-d` edge features) should combine three ingredients:

1. **Background Benign Commerce (~99% of edges):**
   * Accounts have power-law/log-normal activity weights (creating realistic merchant hubs alongside occasional retail users) and diurnal rate modulation $\lambda(t) \propto 1 + 0.5 \sin(2\pi t / 86400)$.
   * Edge features $\mathbf{m}_i \in \mathbb{R}^{16}$ encode log-amount, amount z-score vs. global priors, payment channel one-hots, device/geo indicators, and benign noise dimensions.
2. **Planted Smurfing / Fan-Out–Fan-In Burst Rings (~1% of edges, `y = 1`):**
   * **Structure:** An orchestrator account $s$ rapidly splits funds just below reporting thresholds across $m \in \{4, \dots, 8\}$ mule accounts $\{r_1, \dots, r_m\}$, which within a tight window $\Delta t_{\text{ring}} \in [60\text{s}, 900\text{s}]$ forward the funds into a collector/cash-out account $c$ ($s \to r_k \to c$).
   * Recurring fraudsters participate in multiple rings before going dormant, giving temporal memory models a real state-tracking signal.
3. **Mid-Stream Regime Shift (at $t = 0.5 T_{\max}$):**
   * Halfway through the 30-day timeline (inside `train`, carrying into `val` and `test`), ring operators adapt—for example, shifting transaction split sizes and preferred payment channel features—testing whether models rely on brittle static feature thresholds or invariant burst-graph dynamics!

---

## 4. $O(E)$ Causal Rolling Graph Features (`1h / 24h / 7d`)

To give `LightGBM` (`Config B`) matched causal temporal context without peeking into the future, we compute rolling graph summaries for every event $e_i = (u_i, v_i, t_i)$ over strictly prior history:

$$\mathcal{H}_{< t_i} = \left\{ e_j \in \mathcal{G} \;\middle|\; t_j < t_i \right\}$$

### Why Naive Pandas/NumPy Filtering Explodes
On `reddit` ($E = 672,447$), checking `(t < t_i) & (t >= t_i - 604800) & (src == u_i)` per row requires $O(E^2) \approx 4.5 \times 10^{11}$ comparisons—hours of CPU wall time!

### The Amortized $O(E)$ Two-Phase `deque` Algorithm
Because timestamps $t_1 \le t_2 \le \dots \le t_E$ are monotonically non-decreasing, we maintain per-node double-ended queues (`collections.deque`) for each window $W \in \{3600, 86400, 604800\}$:

1. **Group Simultaneous-Second Slices:** Advance pointer $j$ across the contiguous block $[i, j)$ where $t_i = t_{i+1} = \dots = t_{j-1} = \tau$.
2. **Phase A — Evict Expired & Read Features (Strictly $< \tau$):**
   * For each event $k \in [i, j)$ and each window $W$: pop timestamps from the left of `dq[W][u_k]` while `dq[W][u_k][0] < tau - W`.
   * Read $O(1)$ summaries from the queues **without** inserting any event from $[i, j)$ yet! Every event at second $\tau$ sees strictly $t' < \tau$, guaranteeing zero intra-second leakage even if events at $\tau$ arrive in arbitrary order.
3. **Phase B — Commit Slice $[i, j)$ to History:**
   * After all features for $[i, j)$ are written, append $\tau$ to the queues and update `last_seen`, inter-event $\Delta t$ buffers, pair interaction counts, and distinct neighbor counters.

**Complexity Proof:** Each of the $E$ events is appended to each `deque` **once** and popped `popleft()` **at most once**. Total work across the entire dataset is **$O(E)$** (~5–8 seconds on CPU for 672k edges!).

### High-Signal Causal Rolling Features to Extract
* **Recency:** $\log(1 + \Delta t_u)$ and $\log(1 + \Delta t_v)$, where $\Delta t_u = t_i - t_{\text{last}}(u)$ (use a clean sentinel or `is_first_interaction` binary flag when $u$ has no prior events!).
* **Multi-Scale Causal Degrees (`1h / 24h / 7d`):** $\log(1 + \text{deg}_{W}^{\text{out}}(u))$, $\log(1 + \text{deg}_{W}^{\text{in}}(v))$, and total activity across `1h`, `24h`, and `7d`.
* **Short-vs.-Long Velocity / Surge Ratio:**
  $$\text{Surge}_{1\text{h}/24\text{h}}(u) = \frac{\text{deg}_{1\text{h}}(u)}{\frac{1}{24}\text{deg}_{24\text{h}}(u) + 1.0}$$
* **Goh–Barabási Inter-Event Burstiness:** Over the last $K$ (e.g., $K=20$) causal inter-event intervals $\Delta t$ of node $u$ with mean $\mu_{\Delta t}$ and standard deviation $\sigma_{\Delta t}$:
  $$B(u) = \frac{\sigma_{\Delta t} - \mu_{\Delta t}}{\sigma_{\Delta t} + \mu_{\Delta t} + \epsilon} \in [-1, 1]$$
  ($B \to 1$ indicates violent micro-bursts; $B \approx 0$ indicates memoryless Poisson traffic; $B \to -1$ indicates robotic periodic timers!)
* **Local Pair & Hub Context:** Prior $(u, v)$ repeat interaction count and unique counterparty fan-out/fan-in ratio in the last `24h`.

---

## 5. The 4 Deterministic `LeakageGuard` Invariants

Your `LeakageGuard` is a non-LLM, deterministic referee with veto power. Any pipeline failing even one check is immediately rejected:

1. **Strict Split Monotonicity & Disjoint Edge IDs:**
   $$\max t_{\text{train}} < \min t_{\text{val}} \le \max t_{\text{val}} < \min t_{\text{test}}, \qquad \mathcal{E}_{\text{train}} \cap \mathcal{E}_{\text{val}} = \emptyset, \quad \mathcal{E}_{\text{val}} \cap \mathcal{E}_{\text{test}} = \emptyset$$
2. **Pre-Forward Timestamp Invariant:**
   Before predicting batch $B$ starting at timestamp $t_{\min}(B)$ and global edge index $i_{\text{start}}(B)$, every node memory timestamp and sampled historical neighbor edge must come strictly from prior batches ($e_{\text{id}} < i_{\text{start}}(B)$ and $t \le t_{\min}(B)$, strictly $< t_{\min}(B)$ whenever timestamps don't tie across batch boundaries).
3. **Bit-for-Bit Future-Perturbation Invariance:**
   Choose an interior timestamp $t^*$ inside the evaluation stream $\mathcal{S}$. Let $\tilde{\mathcal{S}}_{> t^*}$ be an identical copy of $\mathcal{S}$ up to $t^*$, with **every** destination node $v_j$ and feature vector $\mathbf{m}_j$ for $t_j > t^*$ replaced by random noise. Under fixed deterministic RNG seeds, a genuinely causal model $f$ **must** satisfy:
   $$\forall e_i \text{ with } t_i \le t^*: \qquad f(\mathcal{S})_i \equiv f(\tilde{\mathcal{S}}_{> t^*})_i \quad (\text{exact `0.0` max absolute difference!})$$
   *Why this test is magic:* Any accidental `StandardScaler.fit(full_X)`, global `PageRank`, full-stream `groupby('src').mean()`, bidirectional sequence leak, or `closed='right'` future window changes $f(\tilde{\mathcal{S}}_{>t^*})_i$ and gets caught red-handed!
4. **Shuffled-Label Null Collapse:**
   Train the pipeline on $\tilde{\mathbf{y}}_{\text{train}} = \pi_{\text{rand}}(\mathbf{y}_{\text{train}})$. Validation `PR-AUC` must collapse near the positive base rate $\bar{y}_{\text{val}}$ (confirming labels never leaked into features or target encodings).
