# Semantic Entropy for VLA Action Spaces

**Status of this document.** Sections 1–5 are established published work with citations. Section 6 is a standard architecture refresher. Sections 7–9 are a *proposed adaptation* — no paper defines "ε-pose clustering." Section 9a is the committed experimental plan (Stages A–D); it is a decision record, not a literature claim. Where I claim something is published, there is a citation. Where there is no citation, it is argument, not literature.

**Confidence tags:** [Certain] hard evidence. [Likely] strong inference. [Guessing] filling gaps.

**Revision 2026-10-07.** Corrected against `thesis-import/thesis-brief.md` and re-checked sources. Changes:
- §9a: the A.2.2 loading target is **0.20** (LIBERO-PRO, arXiv 2510.03827 Table 3), not 45.0%. The Lee & Har numbers were re-read from the current arXiv HTML and differ from the Aug 27 copy (see §9a.2).
- §9a/§9b: checkpoint facts. `pi05_libero_finetuned` = `pi05_libero_finetuned_v044` (byte-identical). Issue #2533 is in huggingface/lerobot, not openpi. Chunk length is 50, so H is now an open choice (§8.5).
- §8.5 (new): implementation contract checked in the LeRobot source (image flip, state vector, sampling hook, action units).
- §9a.2 (new): what "baseline" can and cannot mean for this checkpoint.
- §9a Stage A: per-task and length-matched AUROC; Pos mechanism flagged as unresolved.
- §9d (new): the π₀.₅ visual-nuisance ablation (side test S.5), code in `thesis-docs/ablation_pi05/`.
- §9c (new): candidate solutions for camera moves that go out of distribution (Stage C).
- §14: next actions updated. Status as of the last chat (2026-08-30): **nothing has run on π₀.₅ yet.**

---

## 0. What you are being asked to believe

Three claims, in dependency order:

1. Token-level entropy measures the wrong thing when many different outputs mean the same thing. **[Certain]** — published, replicated.
2. The fix is to cluster outputs by meaning first, then compute entropy over clusters. **[Certain]** — Kuhn et al. ICLR 2023, Farquhar et al. Nature 2024.
3. Robot action chunks have an equivalent of "meaning," namely the physical outcome they produce, and can be clustered by a pose-space distance threshold ε. **[Guessing]** — this is the proposal. Nobody has published it for VLAs.

Claim 3 is your contribution and your risk. Claims 1–2 are your foundation.

---

## 1. Prerequisites

You need these before Section 3 makes sense:

- **Shannon entropy.** $H(p) = -\sum_i p_i \log p_i$ over a categorical distribution. Maximised by uniform, zero for a point mass. Units: nats with natural log, bits with log base 2.
- **A categorical distribution is not the same as a metric space.** Entropy treats categories as unordered labels. It has no notion of "these two categories are close." This is the crack everything else grows out of.
- **Autoregressive sequence probability.** For a token sequence $y = (y_1 \dots y_L)$, the model gives $p(y|x) = \prod_{l=1}^{L} p(y_l \mid y_{<l}, x)$. You get this free from a forward pass.
- **Monte Carlo estimation.** If you can sample from a distribution but not evaluate its density, you can still estimate expectations by sampling and averaging.
- **Flow matching.** Refresher in Section 6 — read it if the ODE formulation is hazy.

---

## 2. The problem semantic entropy solves

Ask an LLM "What is the capital of France?" and sample ten times. Suppose it returns:

```
"Paris"                    ×4
"It's Paris"               ×3
"France's capital is Paris" ×2
"Lyon"                     ×1
```

Naive token-level entropy sees **four distinct sequences** with a fairly spread distribution and reports high uncertainty. But <cite index="20-1">naive entropy-based uncertainty measures variation in the exact answers, treating 'Paris', 'It's Paris' and 'France's capital Paris' as different — which is unsuitable for language tasks where different answers mean the same thing</cite>.

The model is actually 90% confident. Nine of ten samples mean *Paris*. Only one dissents. The entropy is inflated by **surface-form variation that carries no semantic content**.

**This is the exact failure mode you will find in your OpenVLA token entropy, and it has a specific mechanism I will name in Section 10.**

---

## 3. Semantic entropy, formally

### 3.1 Setup

Semantic entropy measures how uncertain a model is about the *meaning* of its output rather than the *surface form*. It does this by partitioning sampled outputs into equivalence classes before applying the entropy formula.

### 3.2 Variable legend

| Symbol | Read as | What it is | What it impacts |
|---|---|---|---|
| $x$ | "x" | the input prompt / observation | what the model conditions on |
| $y$ | "y" | one sampled output sequence | an individual generation |
| $\mathcal{Y}$ | "calligraphic Y" | the set of $M$ sampled outputs | your empirical view of the model's distribution |
| $M$ | "M" | number of samples drawn | estimator variance — the single most important knob |
| $c$ | "c" | one semantic equivalence class | the grouping level; bridges sequences to meanings |
| $\mathcal{C}$ | "calligraphic C" | the set of all classes found | the "vocabulary of meanings" for this input |
| $n_c$ | "n sub c" | count of samples falling in class $c$ | numerator of the discrete estimator |
| $p(y\mid x)$ | "p of y given x" | model likelihood of sequence $y$ | raw sequence-level confidence |
| $p(c\mid x)$ | "p of c given x" | total probability mass on class $c$ | confidence in a *meaning* |
| $\hat p(c\mid x)$ | "p-hat of c given x" | empirical estimate of the above | what you compute when likelihoods are unavailable |
| $\text{SE}(x)$ | "S E of x" | semantic entropy | the final uncertainty score |
| $\mathcal{E}(y_i, y_j)$ | "script E of y-i, y-j" | equivalence predicate | decides cluster membership |

### 3.3 Step-by-step

---

**Step 1: Sample.**

```
𝒴 = {y⁽¹⁾, …, y⁽ᴹ⁾},   y⁽ⁱ⁾ ~ p(· | x)
```

**In words:** Take the input $x$. Draw a sample from the model's output distribution — call it $y^{(1)}$. Draw another, independently — $y^{(2)}$. Repeat until you have $M$ of them. Collect them into the set $\mathcal{Y}$.

**Why this step?** You cannot enumerate every possible output. Sampling gives you a finite window onto the distribution.

**Hidden assumption:** The samples are i.i.d. draws from the *same* conditional distribution. If you vary temperature between draws, or the model has hidden state carrying over, this breaks.

**Where this breaks:** <cite index="8-1">Responses are typically generated at high temperature to encourage output diversity</cite>. That is a deliberate distortion of $p$. The entropy you compute is the entropy of the *tempered* distribution, not the model's actual belief. You must report the temperature and hold it fixed.

---

**Step 2: Define equivalence.**

```
𝓔(yᵢ, yⱼ) = 1  ⟺  yᵢ ⊨ yⱼ  ∧  yⱼ ⊨ yᵢ
```

**In words:** Take two sampled outputs, $y_i$ and $y_j$. Ask a natural-language-inference model whether $y_i$ logically entails $y_j$. Then ask, separately, whether $y_j$ entails $y_i$. If **both** answers are yes, the predicate $\mathcal{E}$ returns 1 and the two outputs count as meaning the same thing. If either direction fails, return 0.

**Why this step?** One-directional entailment is too weak. "It's Paris, the largest city in France" entails "It's Paris" but not conversely — they carry different information. Requiring both directions forces genuine equivalence.

**How it's implemented:** <cite index="7-1">Pretrained models such as DeBERTa-Large-MNLI or general-purpose language models are used to assess entailment between sentence pairs. When confronted with a new output, check for bidirectional entailment with a representative from each existing cluster; if no entailment is found with any of them, create a new cluster</cite>.

Note this is a **greedy, sequential** algorithm, not a full pairwise clustering. It compares each new item to one representative per cluster, which is $O(M \cdot |\mathcal{C}|)$ instead of $O(M^2)$. It is also **order-dependent** — a different shuffle of $\mathcal{Y}$ can yield a different partition.

**Hidden assumption:** Equivalence is transitive. The greedy algorithm *enforces* transitivity by construction, but bidirectional entailment as judged by a neural NLI model is **not actually transitive**. This is a known soft spot.

---

*Where we are now: we have $M$ samples partitioned into classes. We still need to turn class membership into probabilities.*

---

**Step 3a: Class probability — the likelihood-based estimator.**

```
p(c | x) = ∑_{y ∈ c} p(y | x)
```

**In words:** Pick one class $c$. Look at every sampled output $y$ that landed in $c$. For each, take the model's likelihood $p(y|x)$ — obtained by multiplying together the conditional probabilities of each token in the sequence. Sum those likelihoods across all members of $c$. Call the result $P_c$. That is the total probability mass the model places on meaning $c$.

**Why this step?** Probability mass scattered across paraphrases belongs to the meaning, not to any one wording. Pooling it recovers the model's actual confidence in the meaning.

**Hidden assumption:** Classes are **disjoint** — every $y$ lives in exactly one $c$. Overlap would double-count mass.

**Where this breaks:** This requires $p(y|x)$. **Remember this requirement. It is the exact thing that kills the π₀ plan in Section 7.**

---

**Step 3b: Class probability — the discrete estimator.**

```
p̂(c | x) = n_c / M
```

**In words:** Count how many of your $M$ samples landed in class $c$ — call that count $n_c$. Divide $n_c$ by $M$. That fraction is your estimate of the class probability. No likelihoods anywhere.

**Why this step?** It is a plain Monte Carlo estimate: if you sample from $p$ and count how often you land in $c$, the frequency converges to $p(c)$. It works for any model you can sample from, including black-box APIs and continuous generative heads.

**Is this legitimate?** Yes, and it is published. <cite index="5-1">The discrete variant of the semantic entropy estimator outperforms baselines on both AUROC and AURAC metrics</cite>, and <cite index="5-1">both semantic entropy and its discrete approximation outperform the best baselines</cite>. Farquhar et al., Nature 2024, Fig. 2. **[Certain]**

**Hidden assumption:** Samples are i.i.d. and $M$ is large enough that observed frequencies approximate true probabilities.

**Where this breaks:** Hard. Small $M$ makes this estimator badly biased — see Section 9.

---

**Step 4: Entropy over classes.**

```
SE(x) = -∑_{c ∈ 𝒞} p(c|x) · log p(c|x)
```

**In words:** Take the set of classes $\mathcal{C}$. For each class $c$: take its probability $P_c$. Compute the natural log of $P_c$ — a negative number, since $P_c \le 1$. Multiply $P_c$ by that log. You now have one negative number per class. Sum them all — call the sum $S$. Put a minus sign in front of $S$. That is $\text{SE}(x)$.

**Why this step?** Standard Shannon entropy, applied to the distribution over meanings instead of the distribution over sequences. Same formula, different input distribution. That substitution *is* the whole method.

**Where this breaks:** $\mathcal{C}$ is estimated from a finite sample. Meanings the model would produce with probability $< 1/M$ never appear, so $\text{SE}$ is systematically **underestimated**.

---

### 3.4 Plain words, no symbols

Sample the model many times. Group the samples by what they mean, not what they say. Ask what fraction of samples fell into each meaning-group. Measure how spread out that distribution over groups is. All samples in one group means the model is confident. Samples scattered across many groups means it does not know which meaning is right.

---

## 4. Which estimator, and why it matters for you

| | Likelihood-based (3a) | Discrete (3b) |
|---|---|---|
| Needs $p(y\mid x)$ | Yes | No |
| Works on autoregressive LLMs | Yes | Yes |
| Works on diffusion / flow matching | **No** (see §7) | **Yes** |
| Sample efficiency | Better | Worse |
| Published validation | Kuhn 2023, Farquhar 2024 | Farquhar 2024 |

**[Certain] For a flow-matching policy, the discrete estimator is your only option.** Gemini's plan used the likelihood-based formula $p(c|x) = \sum_{a \in c} p(a|x)$ while recommending π₀, which cannot supply the term. That is the bug.

---

## 5. Related published variants worth knowing

- **Kernel Language Entropy (KLE), Nikitin et al. 2024.** <cite index="1-1">Generalises SE to a kernel-based uncertainty measure that captures both intra-cluster spread and inter-cluster distance without hard partitions</cite>. **This is the principled answer to "what threshold do I pick?"** — it removes the threshold. If your ε-sensitivity ablation comes back ugly, KLE is your escape hatch.
- **SNNE, Nguyen et al. ACL 2025.** <cite index="1-1">A nearest-neighbour entropy estimator with provable generalisation over SE</cite>.
- **Semantic Entropy Probes, Kossen et al. 2024.** <cite index="6-1">Approximate SE from the hidden states of a single generation, reducing the 5-to-10-fold inference overhead to almost zero</cite>. Relevant if your per-step sampling cost blows the control loop.
- **Bayesian budget estimation, arXiv 2504.03579.** <cite index="4-1">Achieves the same hallucination-detection AUROC using 53% of the samples, and adaptively tunes the number of samples so harder contexts get more</cite>. Directly attacks your $N$ problem.
- **Miller–Madow bias correction, Miller 1955.** <cite index="1-1">Applied to semantic entropy in recent work</cite>. See §9.

---

## 6. Flow matching refresher

**The object.** A flow matching model learns a time-dependent velocity field $v_\theta(a, \tau, o)$ where $a$ is a point in action space, $\tau \in [0,1]$ is flow time (not robot time), and $o$ is the conditioning observation.

**Sampling.** Draw $a_0 \sim \mathcal{N}(0, I)$. Integrate the ODE

$$\frac{da_\tau}{d\tau} = v_\theta(a_\tau, \tau, o)$$

from $\tau = 0$ to $\tau = 1$, typically with 5–10 Euler steps. The endpoint $a_1$ is your sampled action chunk. Different noise seeds $a_0$ give different $a_1$.

**Training.** Conditional flow matching regresses $v_\theta$ against the velocity of a straight path between a noise sample and a data sample. No likelihood ever appears in the loss.

**In π₀ specifically.** A ~3B VLM backbone encodes the image and instruction once. A lightweight action expert runs the flow ODE. This asymmetry is why batched sampling is cheap: the expensive part runs once, the cheap part runs $N$ times.

---

## 7. Why π₀ gives you samples but not likelihoods

**[Certain]** For a continuous normalizing flow, the log-density of a generated point follows the instantaneous change-of-variables formula:

$$\log p(a_1 \mid o) = \log p_0(a_0) - \int_0^1 \nabla \cdot v_\theta(a_\tau, \tau, o) \, d\tau$$

**In words:** Start with the log-density of your noise sample under the standard Gaussian — that part is closed-form and free. Then, at every point along the integration path, compute the **divergence** of the velocity field: the trace of its Jacobian with respect to the action. Integrate that trace over the whole path. Subtract the integral from the noise log-density.

**Why this is expensive:** The Jacobian of $v_\theta$ with respect to a $H \times 7$ action chunk is a matrix of size $(7H)^2$. Its exact trace costs $7H$ backward passes *per ODE step*. Hutchinson's estimator reduces this to a few stochastic probes per step, but you still pay multiple backward passes per sample, and you now have estimator noise sitting inside your entropy estimate.

**Consequence:**

> π₀ hands you **samples**, cheaply. It does not hand you $p(a|x)$, cheaply. Any semantic entropy formula that sums likelihoods is not implementable at control frequency. The discrete estimator, which only counts, is.

This is why Section 3b is the load-bearing part of this document.

---

## 8. The proposal: ε-pose clustering

**[Guessing] — no citation exists for this. It is an adaptation, and the argument for it is below, not in a paper.**

### 8.1 The analogy, and where it holds

| LLM semantic entropy | VLA analogue | Does the analogy hold? |
|---|---|---|
| Output = token sequence | Output = action chunk $A \in \mathbb{R}^{H \times 7}$ | Yes |
| Surface form varies | Exact joint trajectory varies | Yes |
| Meaning is invariant | Physical outcome is invariant | **Partially** — see §12 |
| Equivalence via NLI model | Equivalence via pose distance | **This is the substitution** |
| Bidirectional entailment | Symmetric metric $d(A,B) \le \varepsilon$ | Metric is *automatically* symmetric — an improvement |

The metric substitution is actually **cleaner than NLI**: a distance function is symmetric and reflexive by construction, so bidirectionality comes free, and unlike a neural entailment model it is deterministic and cheap.

### 8.2 The behavioral distance

Given two action chunks $A, B \in \mathbb{R}^{H \times 7}$, apply **pose integration** to get end-effector poses, then:

$$d(A,B) = \underbrace{\lVert \mathbf{p}_A - \mathbf{p}_B \rVert_2}_{\text{position, metres}} + \lambda_R \underbrace{d_{\text{geo}}(R_A, R_B)}_{\text{rotation, radians}} + \lambda_g \underbrace{\lvert g_A - g_B \rvert}_{\text{gripper, binary}}$$

where $\mathbf{p}$ is terminal end-effector position, $R$ is terminal orientation with $d_{\text{geo}}$ the geodesic distance on $SO(3)$, and $g$ is the terminal gripper command.

**Pose integration, not forward or inverse kinematics.** Neither term applies: FK maps joint angles → pose; IK maps target pose → joint angles. This operation never touches joint angles at all — it accumulates $H$ sequential *end-effector-space* deltas onto a known current pose, entirely within task space. Position integrates by plain summation, $\mathbf{p}_{\text{term}} = \mathbf{p}_0 + \sum_h \Delta\mathbf{p}_h$. Orientation does **not** sum — $SO(3)$ is curved, so each rotation delta must be *composed* (quaternion or matrix multiplication) onto the running orientation: $R_{\text{term}} = \Delta R_H \circ \cdots \circ \Delta R_1 \circ R_0$. Gripper takes the chunk's final value with no integration; the implementation uses its sign (open/close), so small continuous noise in the gripper output cannot split clusters by itself. Getting this wrong (treating orientation as summable) is silent, not a crash.

**Design choices you must defend in the thesis:**

- **Terminal pose, not full path.** Two chunks that curve differently but end in the same grasp are behaviorally equivalent. Path-integrated distance would separate them, which defeats the purpose. **[Guessing]** — but it is the choice consistent with "meaning = outcome."
- **Gripper weight $\lambda_g$ must dominate.** Same position with the gripper open versus closed is a *completely* different outcome. Set $\lambda_g$ large enough that gripper disagreement always forces separate clusters. This is the single most important weight.
- **$\lambda_R$ units.** Position is in metres, rotation in radians. Without calibration you are adding incommensurable quantities. Set $\lambda_R$ so that a rotation error that would break the grasp costs about as much as a position error that would break the grasp.

### 8.3 The algorithm

```
Input:  observation o_t, model π, sample count N, threshold ε
Output: SE_t

1.  Encode o_t once through the VLM backbone.
2.  Sample N action chunks A⁽¹⁾…A⁽ᴺ⁾ by running the flow ODE
    from N independent noise seeds. Batch these.
3.  Clusters ← ∅
4.  for i = 1…N:
5.      assigned ← false
6.      for each cluster c in Clusters:
7.          if d(A⁽ⁱ⁾, representative(c)) ≤ ε:
8.              add A⁽ⁱ⁾ to c;  assigned ← true;  break
9.      if not assigned:
10.         create new cluster with A⁽ⁱ⁾ as representative
11. for each cluster c:  p̂_c ← n_c / N
12. SE_t ← -∑_c p̂_c · log p̂_c
13. Apply Miller–Madow correction (§9).
```

Steps 4–10 mirror Farquhar's greedy sequential clustering exactly — same structure, entailment predicate swapped for a metric threshold. Keeping the algorithm identical is deliberate: it makes the method easy to defend as "Farquhar's discrete SE with a robotic equivalence relation."

**Order dependence:** like the original, this is shuffle-sensitive. Fix the seed and report it. If a reviewer objects, single-linkage agglomerative clustering at the same $\varepsilon$ is order-invariant and costs $O(N^2)$ distance evaluations — trivial for $N \le 100$.

### 8.4 Why this is the right shape — the central argument

**This is the strongest justification for the whole method, and it is what Gemini never articulated.**

Your problem, throughout this thesis, has been separating *aleatoric* uncertainty (many valid ways to do the task — irreducible, do not act on it) from *epistemic* uncertainty (the model does not know what is going on — reducible by looking again).

ε-pose clustering performs that separation **structurally**:

- The model samples ten paths that all end in the same grasp → **one cluster** → $\text{SE} = 0$. Multimodality in *how* is collapsed. That is aleatoric variation, correctly discarded.
- The model samples six paths ending at the cup and four ending at the plate → **two clusters** → $\text{SE} \approx 0.67$ nats. That is genuine disagreement about *what*. Epistemic, and exactly what should trigger a camera move.

Raw action variance cannot do this — it reports a large number in both cases. **Clustering by outcome is an aleatoric filter, and that is the thesis argument in one sentence.**

### 8.5 Implementation contract for π₀.₅ in LeRobot (added 2026-10-07)

Checked in the LeRobot 0.6.1 source (the cluster has 0.6.2 editable; re-check there):

| Item | Fact | Confidence |
|---|---|---|
| Checkpoint | `lerobot/pi05_libero_finetuned`. `_v044` is byte-identical (same git hash for every file). Model tree on the card: `pi05_base` → `pi05_libero_base` → `pi05_libero_finetuned`, i.e. LeRobot's own LIBERO fine-tune. | [Certain] |
| Chunk | `chunk_size` 50, `num_inference_steps` 10 (Euler), `max_action_dim` 32 (padded). Pass `n_action_steps=10`; the config default of 50 runs the whole chunk open-loop. | [Certain] |
| Images | `observation.images.image` = agentview, `image2` = wrist, 256×256. LeRobot's `LiberoProcessorStep` **rotates both images 180°** itself. Send raw renders through that step; never flip by hand as well. | [Certain] |
| State | Same step builds `observation.state` (8) = eef pos (3) + eef axis-angle from the quaternion (3) + gripper qpos (2). π₀.₅ tokenises the normalised state into the prompt. | [Certain] |
| Order | `lerobot-eval` runs `preprocess_observation` → `LiberoProcessorStep` → policy preprocessor → `select_action` → postprocessor. An own loop must follow the same order. | [Certain] |
| Sampling hook | `policy.predict_action_chunk(batch, noise=...)` passes `noise` to `sample_actions`; noise shape `(B, 50, 32)`. N samples = batch of N with N noise tensors. | [Certain] |
| Encode once | `sample_actions` runs the backbone for every batch element. Batching N copies costs N backbone passes in one call, not 1 + N as §6 and Scenario B assume. Encoding once needs a KV-cache reuse patch. Time it (T4) before deciding. | [Certain] about the code, [Guessing] about the cost |
| Action units | Env actions are deltas in [−1, 1]; robosuite OSC scales them to about 0.05 m and 0.5 rad per unit. Convert before computing $d$, or state ε in action units. | [Likely]; verify in the controller config |
| Success | `env.check_success()`, step cap 280 for LIBERO-Spatial (LeRobot `TASK_SUITE_MAX_STEPS`). | [Certain] |

**Open: the horizon H.** Earlier drafts assumed H = 10 (openpi `action_horizon`). This checkpoint emits 50 actions and executes 10. Until a decision, compute SE for H = 10 and H = 50 from the same samples (no extra model passes) and report both.

---

## 9. How many samples, really

The plug-in entropy estimator is **biased downward**. With $C$ true clusters and $N$ samples, the bias is approximately $-(C-1)/(2N)$.

**Miller–Madow correction:**

$$\hat H_{\text{MM}} = \hat H_{\text{plug-in}} + \frac{\hat C - 1}{2N}$$

where $\hat C$ is the number of clusters actually observed. Apply it. It costs nothing.

**Concrete numbers.** Suppose the truth is a 60/40 split over two clusters, $H = 0.673$ nats.

| $N$ | Bias | Roughly what you can resolve |
|---|---|---|
| 5 | $-0.10$ | nothing — noise dominates |
| 10 | $-0.05$ | presence vs. absence of disagreement, barely |
| 20 | $-0.025$ | coarse ordering |
| 50 | $-0.010$ | differences of ~0.1 nats |
| 100 | $-0.005$ | differences of ~0.05 nats |

**[Likely] Gemini's ablation grid of $N \in \{5, 10, 20\}$ tops out below where the estimator becomes usable.** Run $N \in \{10, 20, 50, 100\}$ and let the data show you where it stabilises. Also note the ceiling: with $N$ samples, $\text{SE} \le \log N$, so small $N$ compresses your dynamic range. And a hidden trap — $\hat C$ itself grows with $N$, so a rising SE across the sweep may be estimator artifact rather than signal. Plot $\hat C$ against $N$ alongside SE.

The Bayesian estimator (arXiv 2504.03579) is worth reading before you commit compute; it reports comparable detection quality at roughly half the samples.

**Naming note:** $\hat C$ (cluster count) was previously written $\hat K$ in earlier drafts of this document. Renamed here to free up $K$ for the experimental-plan step count introduced in §9a below, and because the perturbation-uncertainty paper (arXiv 2606.20754, §12) independently uses $K$ for *its* sample count — the role $N$ plays in this document. Three different meanings on one letter was a live risk; this document now uses $N$ = samples per reading, $\hat C$ = clusters observed, $K$ = readings per episode (§9a) or experimental-plan step number, depending on context — the two $K$ uses don't overlap in scope so this is not reintroducing the collision.

---

## 9a. Experimental plan: Stage A (does the signal exist) and Stage B (a live trigger)

Sections 1–9 establish the *estimator*. This section establishes the *experiment* — what gets run, on what benchmark, in what order, and what each stage is and is not allowed to conclude. Everything below assumes the model is **π₀.₅** (`lerobot/pi05_libero_finetuned`; §8.5), not π₀-plain — see §9b for why the model changed from earlier drafts of this document, which assumed π₀.

**The boundary that matters:** Stage A never moves a camera. It asks *does SE correlate with failure at all*, offline, retrospectively, on data already collected. Stage B still never moves a camera — it asks a different question, *can SE be turned into a per-step decision live*, and that question is not answered by finishing Stage A. An AUROC computed over a whole rollout (Stage A's output) is a **retrospective, rollout-level** statistic; a controller needs a **per-timestep** trigger threshold. These are different objects. Stage C (camera movement, confronting the OOD wall of §13 attack #3) and Stage D (the closed-loop active-perception controller, §11's $S_t$) come after, and are out of scope for this section.

### Stage A — establish the signal (fixed camera, offline)

**A.1 — Vanilla LIBERO-Spatial, π₀.₅ (pipeline validation only)**

| Step | Action |
|---|---|
| A.1.1 | Launch full 24GB allocation. Load `pi05_libero_finetuned`, not a partial-VRAM fraction sized to the ">8GB inference floor" figure — that floor ignores MuJoCo/LIBERO renderer overhead sharing the same node. |
| A.1.2 | Run LIBERO-Spatial. Reproduce the published number (~97–99%, per LeRobot's own reproduction table). |
| A.1.3 | If it doesn't match, check `n_action_steps=10` first (LeRobot's reproduction note: "matching the original OpenPI implementation"), then the image/state contract in §8.5, and the norm-stats loading failure reported for `pi05_libero_base` in huggingface/lerobot issue #2533, before assuming a deeper bug. |
| A.1.3b | Own loop on `OffScreenRenderEnv` + `PI05Policy` must match `lerobot-eval` success on the same episodes before any SE number is trusted. |
| A.1.4 | Same run, offline: for each step, sample $N$ action chunks, pose-integrate each to a terminal pose (§8.2 — this is pose integration, not forward kinematics; position sums, orientation composes on $SO(3)$), cluster, compute $\text{SE}_t$. |
| A.1.5 | Sanity-check the $\text{SE}_t$ values: bounded by $\log N + (N-1)/(2N)$ after Miller–Madow, not uniformly zero, not uniformly identical, no crashes across the full episode set. |
| A.1.6 | **Do not** attempt to correlate $\text{SE}$ with failure here. At ~97–99% success, LIBERO-Spatial gives roughly 6–16 failures per 500 episodes — too few for any correlation claim to mean anything. This step proves the *pipeline* runs; it makes no claim about the *method*. |

**A.2 — LIBERO-PRO Spatial-Pos, π₀.₅ (the actual test)**

Benchmark choice: **Spatial-Pos only**, not Object-Pos, Goal-Pos, or any Env-shift suite, and not multiple suites at once.

- *Why Pos, not Env.* PRO's Env shift changes the scene context (e.g. kitchen table → living-room table) while preserving the task. That uncertainty is not reducible by moving a camera — there is no viewpoint from which a living room becomes a kitchen. Pos shift (object placement, spatial arrangement) creates occlusion and viewpoint-dependent ambiguity, which a camera move can genuinely resolve. Using Env-shift data at this stage would validate a signal that has no bearing on Stage C/D.
- *Why Spatial, not Object/Goal/10.* LIBERO-PRO's own paper reports π₀.₅ at **0.20** on Spatial-Pos (arXiv 2510.03827, Table 3; about 400 failures per 500). Lee & Har report 43.8% on their Spatial-Pos setup (arXiv 2606.20754, Table I, current HTML). Either way there are hundreds of failures, enough for AUROC. The two numbers disagree by 2×, which [Likely] means different position settings or checkpoints (see the warning below and §9a.2). It also keeps you on the same task family as your existing OpenVLA Spatial baseline.
- ⚠️ *Which Pos setting is unresolved.* `libero_pro_task_catalog.md` maps Position to `use_swap` (an object moved into another's region). The Aug 27 chat described shift levels x0.1–x0.5 via `.init`/`.bddl` folders. Decide which one, and record it, before A.2.1.
- *Why one suite, not several at once.* Running multiple PRO suites before establishing the method works on any single one multiplies compute and multiplies the chance of a cross-suite discrepancy you can't yet explain — this project has already produced exactly that kind of discrepancy once (the Dataset A/B task-7 mismatch, 72% vs. 92%, documented in the baseline log). One clean result first; decide whether to expand only after.

| Step | Action |
|---|---|
| A.2.1 | Same checkpoint, same code as A.1, pointed at PRO Spatial-Pos scenes instead of classic Spatial. |
| A.2.2 | Success should land near **0.20** (2510.03827 Table 3) for the setting that paper used, as confirmation PRO loaded correctly. A loading sanity check, not the experiment. If it lands near 0.44 instead, you are probably on Lee & Har's setting; say which. |
| A.2.3 | Compute $\text{SE}_t$ at **every environment step** across all 500 episodes (not chunk-boundary-only — see the note below on why this stage needs the dense reading). Save the raw N sampled chunks per step so ε, λ, H and the raw-variance baseline (§9a.2) can be recomputed offline without new rollouts. |
| A.2.4 | Compute AUROC: does the rollout-level $\text{SE}$ signal (e.g. its max or mean over the episode) separate success from failure? This is the real result of Stage A. Report it **pooled and per task**: the suite average hides tasks from near 0 to near 1, so a pooled AUROC can track task identity. Also report a **length-matched** version (e.g. first 100 steps only): failures run to the step cap, which inflated the OpenVLA AUROC. |
| A.2.5 | Compare with **our own raw-variance baseline on the same episodes** (their "Sampling" method, computed from the saved chunks), then against Lee & Har Table I (Spatial-Pos: Sampling 0.943, PFD 0.961, current HTML). Their table is a reference, not apples-to-apples: their checkpoint and Pos setting are not stated (§9a.2). |

*Why every step, not chunk-boundary-only, at this stage:* chunk-boundary triggering (§13 attack #5) is a deployment-time efficiency move — skip readings between decisions a live controller has already committed to. There is no controller yet. Computing $\text{SE}$ only at chunk boundaries here would throw away roughly 4 of every 5 readings before you have any evidence they're not needed, which is the wrong place to economize before the correlation itself is established.

**A.3 — only after A.2 shows a real signal**

| Step | Action |
|---|---|
| A.3.1 | $\varepsilon$-sensitivity sweep (§13 attack #1) — mandatory. Sweep across two orders of magnitude and demonstrate a plateau before trusting any single $\varepsilon$. |
| A.3.2 | $N$ sweep (§9): $N \in \{10, 20, 50, 100\}$, tracking $\hat C$ alongside $\text{SE}$ per the trap noted in §9. |
| A.3.3 | Only then decide whether to add further PRO suites, or proceed to Stage B. |

### Stage B — a live per-step trigger (still fixed camera)

Stage A's AUROC is computed after the fact, over an entire completed rollout. A controller needs to decide, at the current timestep, whether uncertainty is high enough to act — without knowing yet how the episode ends. Converting one into the other is not automatic, and needs its own steps.

| Step | Action |
|---|---|
| B.1 | **Reframe the label.** Stage A's ground truth was one label per *episode* (success/failure). A per-step trigger needs a per-step target. The nearest available proxy is: was $\text{SE}_t$ elevated during the steps immediately preceding failure, versus the equivalent steps in matched successful episodes? This is a weaker, noisier label than Stage A's — say so explicitly when reporting it, not as a like-for-like replacement for the rollout AUROC. |
| B.2 | **Apply Layer-2 aggregation (§11).** Layer 1 ($\text{SE}_t$ itself) is too noisy step-to-step to threshold directly — this is the same reason token entropy was never thresholded raw in the OpenVLA pipeline. Compute $S_t = \max_{w \in \mathcal{W}} \text{pool}_w[z((1+\lambda \cdot \text{ATR}_t)\cdot \text{SE}_t)]$ exactly as specified in §11. The ATR gate and nested-window max-pool carry over unchanged from the token-entropy pipeline; the $\beta$-weighted DoF sum does not apply (SE is already scalar); z-scoring is mandatory here specifically because $\text{SE}$'s $\log N$ ceiling makes raw thresholds not comparable across different $N$ choices from the A.3.2 sweep. |
| B.3 | **Choose a threshold-selection method, and pre-register it before looking at held-out data.** Youden's index (maximizing sensitivity + specificity − 1) is the standard choice and is what Tang et al.'s baseline uses for their own rollout-level threshold — reusing their method here keeps the comparison honest, but remember it was designed for the rollout-level object in Stage A, not the per-step object being built here. State explicitly that the threshold is being repurposed, and expect it to need recalibration. |
| B.4 | **Split data before touching the threshold.** Fit the trigger threshold on one partition of PRO Spatial-Pos episodes; evaluate detection latency and false-trigger rate on a held-out partition. Reusing A.2's episodes for both fitting and evaluation would overstate performance. |
| B.5 | **Report three numbers, not one.** (a) Detection latency — how many steps before episode-end does the trigger fire on episodes that go on to fail; (b) false-trigger rate — how often it fires on episodes that go on to succeed; (c) the rollout-level AUROC from Stage A, so the reader can see how much was lost converting a retrospective statistic into a live one. Expect (c) to be the ceiling B.3–B.4 approach, not a floor they exceed. |
| B.6 | **Stop here.** Stage B produces a threshold and a firing decision, evaluated entirely on a fixed camera. It does not decide what the robot does when the trigger fires — that is Stage C (confronting camera-movement OOD, §13 attack #3, starting from the matched-viewpoint differential already proposed as escape #2) and Stage D (the closed-loop controller). Do not fold camera movement into Stage B's evaluation; conflating "does the trigger fire correctly" with "does moving the camera help" makes a failure in either step impossible to diagnose separately. |

### 9a.1 The full arc, for orientation

| Stage | Camera | Question answered | Status |
|---|---|---|---|
| **A** | Fixed | Does SE exist as a real, failure-correlated signal? | This section |
| **B** | Fixed | Can SE become a live per-step trigger? | This section |
| **C** | Moves | Confront the viewpoint-OOD wall directly (§13 attack #3) | Not started — candidate solutions in §9c; the §9d ablation measures how far the camera can move before SE rises on its own |
| **D** | Moves | The closed-loop active-perception controller: $S_t$ from §11, ATR gate, robot acting on its own uncertainty | Not started |

Finishing A does not hand you B's threshold; finishing B does not hand you C's OOD-robust trigger. Each stage answers a different question and has its own open problems already documented elsewhere in this file (§12 for the OOD wall, §13 for the reviewer attacks each stage will face).

### 9a.2 What "baseline" means for this checkpoint (added 2026-10-07)

**The uncomfortable part first: no published entropy baseline exists for LIBERO-PRO, for any π₀.₅ checkpoint.** [Certain for the two papers below; Likely for the field]

| Source | What it gives | What it does not give |
|---|---|---|
| LIBERO-PRO, arXiv 2510.03827 | Success rates (π₀.₅ Spatial-Pos 0.20, Table 3, as recorded on Aug 27). Says π₀/π₀.₅ were run "with a single official checkpoint". | Which checkpoint (openpi or LeRobot). Any uncertainty numbers. |
| Lee & Har, arXiv 2606.20754 (current HTML, re-read 2026-10-07) | Table I, Spatial-Pos: success 43.8%; AUROC PCA-KMeans 0.782, RND 0.815, ACE 0.837, Sampling 0.943, PFD 0.961. | Checkpoint, Pos setting, code link. No semantic entropy. Their "Sampling" is raw action variance. |
| `lerobot/pi05_libero_finetuned` card | Lineage: LeRobot fine-tune from `pi05_base` via `pi05_libero_base`. | Any LIBERO success number on the card. |

⚠️ The Aug 27 copy of this document quoted 2606.20754 as 45.0% / Sampling 0.935 / perturbation 0.963. The current HTML reads 43.8% / 0.943 / 0.961. [Likely] the paper was revised. Cite the version you read, with its version number.

**Decision recorded: keep `lerobot/pi05_libero_finetuned` (= `_v044`).** [Likely] the papers used Physical Intelligence's openpi checkpoint, but neither says so, so switching models would not buy a verified match either; it would add a second software stack. What makes the comparison fair is generating every baseline ourselves, on the same episodes and the same samples:

1. **Success baseline:** A.1 (classic Spatial, should be about 97–99%) and A.2.2 (PRO Spatial-Pos).
2. **Raw-variance baseline** (Lee & Har's "Sampling"): variance of the same N chunks used for SE. Free, from the saved samples.
3. **SE** (ours), pooled, per task and length-matched.
4. Lee & Har's numbers as an external reference only, with the setup difference stated.

If an examiner insists on exact comparability, the fallback is to run openpi's `pi05_libero` for A.2.2 only and check that its success rate matches theirs. [Guessing] whether that checkpoint loads in LeRobot without conversion.

### 9b. Why π₀.₅ and PRO, not π₀ and classic LIBERO

Earlier drafts of this document, and Sections 1–8 above, are written generically against "π₀." The experimental plan in this section commits to **π₀.₅** on **LIBERO-PRO Spatial-Pos**, for reasons that only became decisive once the benchmark question was worked through — recorded here so the choice doesn't look unmotivated against the rest of the document:

- **Ceiling problem, solved by the benchmark, not the model.** π₀.₅ scores about 97–99% on classic LIBERO-Spatial (a handful of failures per 500) — too few for failure detection, which is why π₀-plain looked preferable early on. But PRO Spatial-Pos brings π₀.₅ down to 0.20 (2510.03827) or 43.8% (2606.20754) — the ceiling problem was a classic-LIBERO artifact, not a π₀.₅ artifact.
- **Published reference numbers exist for π₀.₅ on PRO and not for π₀.** arXiv 2606.20754's Table I gives success rate *and* two failure-detection AUC baselines (sampling, perturbation) for π₀.₅ across all eight PRO suites. No equivalent has been located for π₀-plain on PRO — using π₀ here would mean generating novel numbers with nothing to check them against, the same situation that produced an unsourced, later-retracted 96.8% figure earlier in this project's search history.
- **Baseline comparability, weaker than first thought.** Matching model family, benchmark and 500-episode protocol to arXiv 2606.20754 makes A.2.5 a close comparison, but not apples-to-apples: neither paper names its π₀.₅ checkpoint (§9a.2). The fair comparison is SE vs raw variance on our own episodes.
- **Checkpoint risk, noted not resolved.** `lerobot/pi05_libero_base` scores 0% on LIBERO in huggingface/lerobot issue #2533 (probably a norm-stats loading problem; its card describes it as a base to fine-tune). Use `pi05_libero_finetuned` (= `_v044`), and still run A.1.2–A.1.3 before trusting it.

### 9c. When the camera moves out of distribution: candidate solutions (added 2026-10-07)

**The problem.** Every LIBERO checkpoint was trained with the agentview camera in one place. Move it and SE rises because the image is unfamiliar, not because the scene is ambiguous. In the decomposition H = f + g + h (f = genuine scene ambiguity, g = viewpoint OOD, h = visual nuisance), a camera move raises g, while the controller needs it to lower f. On a fixed-camera model, [Likely] SE is lowest at the training view, so a naive "move until SE drops" controller would always return home.

None of these is tested. Ordered from cheapest to most expensive:

| # | Idea | How it works | Cost | Main risk |
|---|---|---|---|---|
| 1 | **Stay inside the in-distribution envelope** | Use the §9d camera-jitter results (and a finer sweep) to find how far the camera can move before SE rises on nominal scenes. Allow only moves inside that radius. | Cheap; data comes from §9d | The envelope may be too small to see around anything |
| 2 | **Viewpoint-calibrated SE** | Measure SE on nominal (unoccluded) scenes at each candidate camera pose: a baseline table g(v). Trigger on SE(v) − g(v), not raw SE. This is the matched-viewpoint differential made into a lookup. | Moderate: one calibration sweep per task | g may depend on the scene, not only the pose |
| 3 | **Paired occlusion gate** (pre-registered) | At the training pose, compare target occluded vs visible. Pass if ΔH > 0 in ≥70% of ≥20 pairs per task, Wilcoxon p < 0.05. Shows f exists separately from g. | Moderate | Fails → SE does not see occlusion, and C/D are off |
| 4 | **Use the wrist camera as the active camera** | The wrist camera already moved through many viewpoints in training, so a "look" motion of the arm stays closer to the training distribution than moving agentview. Active perception = an information-gathering arm motion. | Moderate: needs a look-motion primitive | π₀.₅ fuses both views; the agentview still dominates [Guessing] |
| 5 | **Look, then return** | Move the camera to resolve ambiguity, then return to the training pose before acting, so the policy only acts on in-distribution views. The look step needs some other consumer of the new view (e.g. re-querying with the wrist view, or a separate detector). | Moderate | The policy itself never benefits from the extra view |
| 6 | **Viewpoint-randomised fine-tuning** | Fine-tune on demos re-rendered from jittered camera poses (or LIBERO-plus camera variants). | High: GPU time and a training pipeline | May flatten the very SE response the controller needs |
| 7 | **Re-render to the training view** | Reconstruct the scene (depth / 3D) from the moved camera and render a virtual training-pose image (cf. ActiveVLA's virtual views). | High | Large engineering effort; artefacts become a new OOD source |

**Suggested order:** 3 (does f exist at all?) → 1 and 2 (bound and subtract g, using §9d data) → 4 (the most thesis-friendly move). 6 and 7 only if the supervisor accepts the time. LIBERO-plus (arXiv 2510.13626) perturbs camera pose and can quantify the viewpoint drop for this model without new code; it needs its own env because it replaces `hf-libero`.

### 9d. Side test: the π₀.₅ visual-nuisance ablation (S.5, added 2026-10-07)

**Question.** How much does SE move under visual changes that should not matter (the h term), and how far can the camera move before SE rises by itself (feeds §9c idea 1)? Run after A.1 passes; it does not block A.2.

**Code:** `thesis-docs/ablation_pi05/` (`model_server_ablation.py`, `run_perturbation_ablation.py`, `run_ablation.sh`) plus the unchanged `perturbations.py` from the OpenVLA version. Written 2026-10-07, never run.

| Condition | Change | Level |
|---|---|---|
| nominal | none | — |
| brightness_dim / bright | HSV value × 0.5 / × 1.6, both cameras | 2 levels |
| texture_mild / aggressive | hue ±5° / ±20°, saturation 0.85–1.15 / 0.6–1.4, pixel noise σ 5 / 15, both cameras | 2 levels |
| camera_jitter_small / large | agentview moved once per episode: ±1 cm, ±2° / ±4 cm, ±8° | 2 levels |

Protocol: LIBERO-Spatial tasks 4 and 7, 10 episodes per condition (140 episodes), canonical init states, fixed seeds, step cap 280. Every step: N = 10 chunks, SE for H = 10 and 50 (ε = 0.02, λ_R = 0.1, λ_g = 10, all [Guessing] first values), raw chunks saved so ε, λ, H and raw variance can be redone offline. Ten actions executed per chunk.

| Check before reading results | Pass when |
|---|---|
| Nominal success vs `lerobot-eval` on the same tasks | Close; if much lower, the §8.5 contract is broken |
| Action scale (0.05 m, 0.5 rad) | Matches the LIBERO controller config |
| SE on nominal | Not always 0; ≤ ln 10 + 0.45 |

**Reading the result.** If SE rises clearly under brightness or texture, h is large and Stage B needs a nuisance filter. If SE stays flat under small jitter but rises under large jitter, that radius is the first estimate of the §9c envelope. With 10 episodes per cell this is a direction check, not a significance test. The old analysis script (`step_4c_analyze_ablation.py`) expects per-DoF token entropy and must be rewritten for scalar SE.

---

## 10. Compared to your current OpenVLA token entropy

### 10.1 What you compute now

Single forward pass. Slice logits at `[0, 31744:32000]`, softmax over 256 bins, entropy per DoF, giving `entropy_per_step` of shape `(T, 7)`. Perplexity as $1/p_{\max}$.

### 10.2 The flaw, named

**Softmax entropy over ordinal action bins is metric-blind.**

Consider two distributions over the 256 bins for a single DoF:

- **Case A:** mass split 50/50 between bin 100 and bin 101.
- **Case B:** mass split 50/50 between bin 10 and bin 250.

Both give $H = \log 2 = 0.693$ nats. **Identical entropy.**

But bins 100 and 101 are adjacent — they decode to nearly the same displacement. Case A is a model that is *certain*, merely straddling a quantisation boundary. Case B is a model torn between moving sharply left and sharply right. These are opposite epistemic states and your current metric cannot tell them apart.

This is precisely the "Paris" / "It's Paris" problem, transposed. Action bins carry an **order and a metric** that the softmax entropy formula discards, exactly as token entropy discards the fact that two strings mean the same thing.

**[Likely] This may be the mechanism behind your σ=3 blur observation** — the 62% H_sum collapse with a non-monotone entropy response. Under degraded input the model may be shifting from a spread-across-neighbours distribution to a peaked one without the *behavioral* uncertainty changing in the same direction. Worth checking directly: for your existing saved `.npz` files, compute the bin-index variance (metric-aware) alongside $H$ (metric-blind) and see whether they diverge where you saw the non-monotonicity. **This is a cheap, high-value experiment on data you already have, and it can be run this week without any pivot.**

### 10.3 Side by side

| | Token entropy (current) | ε-pose SE (proposed) |
|---|---|---|
| Forward passes per step | 1 | 1 backbone + $N$ action-expert |
| Output shape | `(T, 7)` per-DoF | `(T,)` scalar |
| Respects action metric | **No** | Yes |
| Distinguishes adjacent vs. distant modes | **No** | Yes |
| Separates aleatoric from epistemic | No | Structurally, yes |
| Free parameters | none | $\varepsilon$, $N$, $\lambda_R$, $\lambda_g$ |
| Compute per step | ~50 ms | $N \times$ ODE integration |
| Published precedent for VLA | Karli, Valle | **none** |

**The cost is real: you lose per-DoF resolution.** SE is one scalar per timestep. A per-DoF variant is constructible — cluster on each DoF's terminal value independently — but it forfeits the outcome-equivalence argument that justifies the method. Pick one and defend it; do not have it both ways.

---

## 11. Compared to your composite signal $S_t$

Your current pipeline has two layers, and it matters that you see them as separate:

**Layer 1 — instantaneous uncertainty.** Token entropy $H_t \in \mathbb{R}^7$, one forward pass.

**Layer 2 — temporal aggregation.** ATR multiplicative gate $\tilde H = (1 + \lambda \cdot \text{ATR}) H$, β-weighted sum across DoF, max-pool over nested windows.

**ε-pose SE replaces Layer 1 only.** Layer 2 is an aggregation scheme over whatever scalar stream Layer 1 emits, and most of it survives the swap:

| Component | Survives? | Why |
|---|---|---|
| ATR gate | **Yes, unchanged** | ATR is computed over *action history*, not logits. It is architecture-agnostic. Keep the $\varepsilon \approx 0.02$ noise floor. |
| β-weighted DoF sum | **No** | SE is already a scalar. There are no per-DoF components to weight. This layer disappears. |
| Nested-window max-pool | **Yes** | Operates on a scalar time series. Transfers directly. |
| Z-score normalisation | **Yes, and now mandatory** | You already found the Karli perplexity-vs-entropy timing gap is a Jensen artifact that vanishes under z-scoring. SE has a different scale and a hard ceiling of $\log N$ — normalise before thresholding or your ablation-2 threshold sweep is uninterpretable. |

Composite form after the swap:

$$S_t = \max_{w \in \mathcal{W}} \; \text{pool}_w \Big[ z\big( (1 + \lambda \cdot \text{ATR}_t) \cdot \text{SE}_t \big) \Big]$$

**Losing the β layer is not purely a loss.** Every β was a tuned hyperparameter you had to justify. Dropping them removes free parameters from the aggregation — though you spend that budget on $\varepsilon$, $\lambda_R$, $\lambda_g$ instead. Net parameter count is roughly flat; the difference is that the new parameters have *physical* interpretations (metres, radians) and can be set from task geometry rather than swept blindly.

---

## 12. Prior art, and what is actually novel

**Search results as of Aug 2026. Verify before writing the related-work section.**

### Nobody has published semantic entropy for VLA action spaces. [Likely]

The gap is real. <cite index="21-1">In the broader context of large language and vision-language models, uncertainty quantification often relies on token-level probabilities, semantic entropy, or generation consistency. Inspired by these approaches, initial efforts toward confidence estimation for VLAs have predominantly followed an ensemble-based paradigm</cite> — semantic entropy is named as LLM background, and the VLA work goes elsewhere.

### But the neighbourhood is filling in fast. Read these before you commit.

**1. "Perturbation-Based Uncertainty for Failure Detection in VLA Models" (arXiv 2606.20754, Lee & Har, KAIST)** — read in full; the closest published method, and the one your evaluation plan (§9a) is deliberately benchmarked against. Confirmed mechanism: Gaussian noise $\epsilon^{(k)} \sim \mathcal{N}(0,\sigma^2 I)$ injected into intermediate transformer hidden activations (for π0.5, into the vision-language backbone before the action expert), $K$ perturbed forward passes per step, uncertainty = variance of the resulting actions across perturbations (their Eq. 5) — no clustering anywhere. Tested on π0.5 and OpenVLA-OFT, on LIBERO and LIBERO-PRO.

Their **sampling baseline is raw action variance, not semantic entropy** — this is the comparison §8.4 already argues against ("raw action variance cannot do this — it reports a large number in both cases"), and it means their critique of sampling does not touch the method in this document; it was never tested.

**But their Fig. 2 is a real, sharper problem than "sampling reads low."** They document a case where the policy repeatedly fails while stochastic samples stay tightly self-consistent — confidently wrong. Worked through numerically: when $N$ samples land within $\varepsilon$ of each other, ε-pose clustering produces exactly one cluster, and $\text{SE} = -1 \cdot \log 1 = 0$ **exactly**. Raw variance would at least report a small nonzero number in this case. Clustering *by design* discards that residual spread — it is an aleatoric filter (§8.4), and the confidently-wrong failure mode is precisely where filtering aleatoric variation removes the last remaining signal. This needs a direct answer in this section, not a gesture: the honest position is that $\text{SE}$ alone detects *ambiguity*, not *wrongness*, and these are different failure modes that Fig. 2 demonstrates coming apart. Their own Table I supports the same split — the one PRO suite where sampling truly collapses (Object-Pos, AUC 0.484 in the current HTML; 0.482 in the Aug 27 copy) is where perturbation gains most (→0.640; earlier copy 0.651), consistent with perturbation and clustering catching different things rather than one subsuming the other.

**2. "Shifting Uncertainty to Critical Moments" (arXiv 2603.18342)** — read this first. They <cite index="23-1">build on existing uncertainty metrics and introduce aggregation strategies that better reflect the temporal and kinematic structure of robotic manipulation</cite>. **That description overlaps your $S_t$ design — nested windows, kinematic gating, temporal pooling.** Check whether your aggregation contribution is already taken.

**3. VLAConf (arXiv 2605.29605)** — states your problem precisely and then routes around it: <cite index="21-1">repeated sampling and ensemble inference are computationally expensive, making them difficult to reconcile with high-frequency real-time control; and relying on action-token probabilities restricts these methods to discrete-action VLAs. For state-of-the-art VLAs that output continuous actions via diffusion or flow-matching, disagreement-based uncertainty is often unstable and difficult to interpret</cite>.

Their conclusion is to abandon the action-output space for the internal representation space. **A reviewer who knows this paper will ask why you did not.** Your answer should be that outcome-clustering is what makes disagreement interpretable — which is a real answer, but you need it ready.

**4. ConfidenceVLA** — <cite index="24-1">derives confidence through predictive disagreement across prompt ensembles and action-token probabilities</cite>.

**5. Karli, Valle** — already in your bibliography. <cite index="24-1">Karli et al. used the entropy or perplexity of a VLA's output tokens as a proxy for decision uncertainty; Valle et al. systematically evaluated a wide range of uncertainty metrics on VLAs, including token-level indicators and higher-level metrics of action-sequence dynamics such as action position instability</cite>.

### Honest novelty statement

What survives as yours:

1. Transposing Farquhar's discrete semantic entropy estimator to continuous robot action chunks via a behavioral equivalence relation. **[Likely] novel.**
2. Using outcome-clustering as an *aleatoric filter* to make sampled disagreement epistemically interpretable — the direct answer to VLAConf's objection. **[Likely] novel, and it is your strongest single claim.**
3. Driving camera repositioning from that signal. **[Likely] novel.**

What is not yours: the entropy formula, the discrete estimator, temporal aggregation of uncertainty over manipulation episodes (possibly), and the observation that VLAs are overconfident under shift.

---

## 13. What a reviewer will attack

Ordered by how much damage each does.

**1. "ε is a free parameter and your result is a function of it."**
Too small → every sample is its own cluster → $\text{SE} = \log N$ always. Too large → one cluster → $\text{SE} = 0$ always. The transition between these degenerate regimes may be sharp. **Mandatory ablation:** sweep $\varepsilon$ across two orders of magnitude and show a plateau. If there is no plateau, the method has no operating point and you switch to KLE, which removes the threshold entirely.

**2. "Sample diversity is not calibrated uncertainty."**
π₀.₅'s spread reflects the multimodality of its training demonstrations plus whatever the flow learned — not a posterior. A model trained on demonstrators who disagreed will sample diversely while being perfectly certain. §8.4 mitigates this but does not eliminate it; the perturbation paper's Fig. 2 (§12, item 1) shows the sharp version of this failure — a confidently-wrong rollout produces $\text{SE}=0$ exactly. Be upfront about both.

**3. "Viewpoint OOD conflation."**
`pi05_libero_finetuned` is trained on fixed camera poses. Move the camera and samples disagree because the observation is out of distribution, not because the scene is ambiguous. **[Certain] This affects semantic entropy exactly as it affects token entropy. Switching models does not solve it.** Pose-randomised fine-tuning or a matched-viewpoint differential design is required either way — this is Stage C in §9a, not started, and is a wall regardless of which flow-matching checkpoint is used. This is the same wall as before, wearing new clothes. Candidate answers, cheapest first, are in §9c.

**4. "Terminal-pose equivalence throws away safety-relevant information."**
Two chunks ending at the same grasp, one of which sweeps through the obstacle, are *not* behaviorally equivalent even though your metric says they are. Either defend terminal-pose as the right abstraction for grasp tasks specifically, or add a collision-check term.

**5. "Your compute budget does not close."**
$N$ ODE integrations per control step. Measure it before you design around it. If it exceeds your control period, the honest fallback is triggering at action-chunk boundaries rather than per step — which is arguably the correct granularity anyway, since a chunk is the unit of behavior.

---

## 14. Next actions (updated 2026-10-07)

1. **Find out what ran after 2026-08-30.** The last chat ends at `pip install hf-libero`.
2. **Stage 0 + A.1:** separate envs, `hf-libero`, 2-episode smoke test, then reproduce LIBERO-Spatial with `lerobot-eval` and match it with the own loop (§8.5).
3. **Time N = 10 and N = 50** batched sampling (T4). The batched code runs the backbone N times (§8.5); decide whether the encode-once patch is needed.
4. **Decide the PRO Pos setting** (§9a warning) and **H** (§8.5) before A.2.
5. **A.2** with per-task and length-matched AUROC and our own raw-variance baseline (§9a.2).
6. **A.3.1 ε sweep** stays a primary experiment: it decides whether the method has an operating point.
7. **S.5 ablation (§9d)** any time after A.1; its jitter results start §9c.
8. **Write the confidently-wrong (Fig. 2) paragraph** (§12, item 1) before submission.
9. **Metric-blindness check on the saved OpenVLA `.npz`** (§10.2): cheap, independent of everything above.

---

## References

- Kuhn, Gal, Farquhar. *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation.* ICLR 2023.
- Farquhar, Kossen, Kuhn, Gal. *Detecting hallucinations in large language models using semantic entropy.* Nature 630, 625–630 (2024). — **discrete estimator is here**
- Nikitin et al. *Kernel Language Entropy.* 2024. — soft clustering, no threshold
- Kossen et al. *Semantic Entropy Probes.* arXiv 2406.15927.
- *Hallucination Detection on a Budget: Efficient Bayesian Estimation of Semantic Entropy.* arXiv 2504.03579.
- Miller. Note on the bias of information estimates. 1955.
- Lee, Har. *Perturbation-Based Uncertainty for Failure Detection in Vision-Language-Action Models.* arXiv 2606.20754.
- Zhou et al. *LIBERO-PRO: Towards Robust and Fair Evaluation of Vision-Language-Action Models Beyond Memorization.* arXiv 2510.03827.
- Black et al. *π₀.₅: A Vision-Language-Action Model with Open-World Generalization.* arXiv 2504.16054.
- *Shifting Uncertainty to Critical Moments.* arXiv 2603.18342.
- *VLAConf: Calibrated Task-Success Confidence for VLA Models.* arXiv 2605.29605.
- Valle et al. *Evaluating Uncertainty and Quality of VLA-enabled Robots.* arXiv 2507.17049.
- Karli et al. — entropy/perplexity as VLA decision uncertainty.
- Lipman et al. *Flow Matching for Generative Modeling.* ICLR 2023.
- Black et al. *π₀: A Vision-Language-Action Flow Model for General Robot Control.* 2024.
