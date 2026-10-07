# VLA Entropy Thesis: decision log, 2026-03-26 to 2026-08-30

Distilled from the five exported chats. Dates are the chat timestamps. Each row records a decision that was actually stated in a chat ([Certain] that it was decided). The **Status** column is as of the last chat, 2026-08-30; nothing after the 2026-09-04 export is visible. Companion file: [thesis-brief.md](thesis-brief.md).

## Proposal and framing (Mar to Jul)

| Date | Decision | Why | Status at Aug 30 |
|---|---|---|---|
| 03-26 | Proposal (Hebrew): logit entropy as the trigger for repositioning the camera of a VLA (active perception). | Original thesis idea. | Core idea kept. The signal changed to semantic entropy (Aug 19 and 27). |
| 06-03 | Proposal V2, update 1: trigger signal S_t = sliding-window max of β-weighted per-DoF entropy, after Tang et al. | Tang showed that aggregated entropy predicts failure and that a global mean is weak. | Kept as Layer 2. A z-scored SE replaces the β-weighted sum. |
| 06-03 | V2, update 3: separate epistemic from aleatoric in two ways. (a) Motion test: move the camera and see whether entropy drops. (b) MC Dropout with an MI decomposition. Run both if time allows. | (a) answers "did moving help?" and (b) answers "should we move at all?". | (a) survives as Reduction-on-Trigger. (b) was dropped once the method moved to sampled flow-matching chunks [Likely]. |
| 06-03 | V2, update 4: Isaac Sim replaced by LIBERO. | Tang, ActiveVLA and most VLA papers report on LIBERO. | Kept. Isaac Sim was raised again on Aug 14 and deferred again. |
| 06-03 | V2, update 5: new metric Reduction-on-Trigger, ΔS = S_t before minus after the camera move. | Tang never acts, so this measures what is new. | A Stage D metric. Not reached. |
| 06-03 | Threshold comparison with three configurations: Karli mean+1σ (naive), Tang SW+ATR+BO, and own rolling z-score. Perplexity added as a secondary signal. | A controlled ablation that justifies the design. | Perplexity was tested on Aug 11 and lost to entropy. |
| 06-03 | Wrote the abstract, introduction and related work (about 17 pages, .docx and .pdf). | Early writing. | Status unknown. |
| 06-17 to 07-09 | MC Dropout only at trigger time, never at every control step. | MC Dropout at full control rate is infeasible, the same argument as Karli's 30 Hz point. | Moot after the pivot [Likely]. |
| 06-22 | SigLIP toy tests: a radius-40 blur raised entropy 12.7× when the candidates competed; cropping barely moved it. | Sanity check that visual degradation shows up in entropy. | Informational only. |

## OpenVLA pipeline (Jul to Aug 11)

| Date | Decision | Why | Status at Aug 30 |
|---|---|---|---|
| 07-12 to 07-16 | Run on the BGU Run:AI cluster (L40S), GPFS home, OSMesa rendering, and two processes: a sim orchestrator plus `model_server.py`, joined by a pickle pipe. | EGL is unavailable in the container. OSMesa, TensorFlow and torch segfault in one process. | Still the infrastructure. |
| 07-13 to 07-20 | Action-token slice = IDs [31744, 32000). Verified gen == full argmax == slice. | Logits are 32064 wide, so `[-256:]` would read padding. | Done (OpenVLA only). |
| 07-20 | Blur test on OpenVLA at σ = 3 and σ = 10. | Does visual degradation move action entropy? | Non-monotone, n = 1. An anecdote. |
| By 08-02 | LIBERO-Spatial as the OpenVLA suite. | Its difficulty is perceptual (two identical bowls that differ only in location), which is the kind of ambiguity a camera move should resolve. | Done. |
| 08-02 | `USE_TF=0` so transformers stops auto-importing TensorFlow. | That import segfaulted the model server. | Fixed. |
| 08-10 | Baseline reproduced: 424/500 = 84.8% (published 84.7%). | Validates the harness end to end. | Done. |
| 08-11 | Tier-1 analysis. Entropy AUROC 0.801 (w = 100, Tang β, no ATR), perplexity 0.743, combined 0.796. Keep entropy alone; ATR is next. | Combining did not help. | ATR and BO were never run. The length confound was never tested. |

## Doubts, method rewrite and pivot (Aug 14 to 27)

| Date | Decision | Why | Status at Aug 30 |
|---|---|---|---|
| 08-14 | Isaac Sim deferred again. Instead, a MuJoCo perturbation ablation: brightness (dim 0.5, bright 1.6), texture (mild, aggressive), camera jitter (0.01 m and 2°, 0.04 m and 8°); tasks 4 and 7; 10 episodes per condition; 120 episodes. | Measure what drives entropy before buying photorealism. | Half-finished. env/vla can no longer run it (transformers 5.5.4). |
| 08-18 | OpenVLA-OFT rejected. | Its L1 regression head gives no token distribution, so no entropy. | Final. |
| 08-18 | Entropy modelled as H = f(task ambiguity) + g(viewpoint novelty) + h(nuisance); only f is useful. Pre-registered paired-occlusion gate: ΔH > 0 in at least 70% of at least 20 pairs per task, Wilcoxon p < 0.05. | On a fixed-camera checkpoint, entropy is lowest at the training viewpoint by construction, so any camera move raises it. | Gate not run. The camera wall is still open. |
| 08-18 | Target: finish about 6 months from Aug 18 (mid-Feb 2027). Send Shapira a memo with three framings: (A) trigger improves success, (B) abstain or ask for help, (C) characterise what entropy measures. | Jacob's own target. Abstention is the fallback floor, not the plan. | Memo status unknown. |
| 08-18 to 08-19 | Gemini's π₀ semantic-entropy plan rejected as written. Replaced by discrete semantic entropy (empirical cluster frequencies) with ε-pose clustering. `semantic_entropy_for_vla.md` written. | Gemini's plan needs p(a given x), which flow matching does not give cheaply. Samples alone suffice for the discrete version. | Became the method. |
| 08-23 | "Forward kinematics" corrected to pose integration: sum the positions, compose the rotations on SO(3), take the last gripper value. | Actions are end-effector deltas, not joint angles. | In the Aug 27 spec. |
| 08-23 | `pi0_libero` needs no fine-tuning. Classic LIBERO is near its ceiling (about 97 to 99%), which leaves too few failures. | Failure detection needs failures. | Led to LIBERO-PRO on Aug 27. |
| 08-23 | "Lean on the wrist view" option dropped. | π₀ fuses agentview and wrist jointly, with no per-view knob. | Final. |
| 08-26 | ActiveVLA read: 4 fixed RGB-D cameras plus synthetic virtual views; no camera ever moves. | Positions the thesis: physical repositioning is still open. | Related work. |
| 08-26 | Scenario B adopted: N sampled chunks per reading, 11 passes at N = 10. Scenario A (N full rollouts, 114× the cost) rejected. Cluster count renamed K̂ → Ĉ. | Cost, and to free K and N for loop count and sample count. | Final. |
| 08-27 | Perturbation paper (arXiv 2606.20754) changes nothing in the plan. Its sampling baseline is raw variance, not semantic entropy. | Different mechanism and different gap. | Open item: its Fig. 2 confidently-wrong case gives SE = 0. Needs a written answer. |
| 08-27 | **Pivot:** π₀ → π₀.₅ (`lerobot/pi05_libero_finetuned`), classic → LIBERO-PRO Spatial-Pos. Pos rather than Env. One suite first. | PRO gives enough failures. An Env shift cannot be fixed by moving a camera, and Env has a documented object-drift bug. π₀ scores 0.00 on Spatial-Pos, so it gives no successes to separate. | Current plan. |
| 08-27 | Stage A computes SE at every step (Scenario B). Chunk-boundary readings (Scenario C, K ≈ 25) only for deployment. | Stage A measures; it does not control. | Current plan. |
| 08-27 | Staged arc A to D written into the spec as §9a and §9b. A: offline, fixed camera. B: live trigger, fixed camera. C: viewpoint OOD. D: closed-loop controller. | Prove the signal before anything moves. | Current plan. |
| 08-27 | Correction: π₀.₅ on PRO Spatial-Pos is **0.20** (arXiv 2510.03827, Table 3), not 45.0% (arXiv 2606.20754). | Two papers ran the same nominal condition and disagree by more than 2×. | **Never applied to the spec.** §9a still says 45.0% (lines 354, 360, 403). |
| 08-27 | Open: the PRO position magnitude (x0.1 to x0.5) is unchosen. Semantic (paraphrase) perturbation exists in PRO and was flagged as worth its own step. | `use_task` cannot be combined with other perturbations; the others can. | Unchosen. |

## π₀.₅ setup (Aug 30)

| Date | Decision | Why | Status at Aug 30 |
|---|---|---|---|
| 08-30 | Use `lerobot-eval` only to validate the observation contract, then write an own loop on `OffScreenRenderEnv` + `PI05Policy`. | The thesis needs access to N samples and camera control. | Not started. |
| 08-30 | Do not carry over OpenVLA preprocessing (180° rotation, 0.9 centre crop, prompt template, gripper normalize/invert). | The most likely way to get 60% instead of 97%. | Standing rule. |
| 08-30 | Three separate conda envs: OpenVLA (to finish the ablation), vanilla LIBERO with LeRobot, and LIBERO-plus. | env/vla now has transformers 5.5.4, which breaks OpenVLA (needs 4.40.1). Installing LIBERO-plus uninstalls `hf-libero`. | Not done as of the export. |
| 08-30 | Lead for Stage C: LeRobot's `--env.type=libero_plus` has camera pose as a perturbation axis, which could quantify the viewpoint drop from published numbers. | Would answer the viewpoint-OOD objection without a new confound study. | Unverified lead. LIBERO-plus (arXiv 2510.13626) is a different benchmark from LIBERO-PRO. |
| 08-30 | Replace the cloned `libero` with `hf-libero`, and assert `libero.__file__` in every launcher. | The PYTHONPATH clone silently shadows `hf-libero`. | `egl_probe` built (needs `--no-build-isolation`). `pip install hf-libero` was the next command when the chat ends. |

## Checked after import (2026-10-07)

- [Certain] `lerobot/pi05_libero_finetuned` and `lerobot/pi05_libero_finetuned_v044` on Hugging Face are byte-identical: every file has the same git hash. The checkpoint name does not matter.
