# VLA Entropy Thesis: distilled brief

Imported 2026-10-07 from Jacob's export of his earlier Claude account (5 chat PDFs, Mar to Aug 30 2026, plus the plan files in `Downloads/thesis/claude`). The raw exports stay on his PC. **Nothing after the export date (2026-09-04) is visible here**, so "current status" below means "as of the last chat, 2026-08-30".

Confidence tags: [Certain] seen directly in the chats or files. [Likely] strong inference. [Guessing] filling gaps.

---

## 0. Read this first: the uncomfortable parts

1. **No experiment has run on the current method.** [Certain as of 2026-08-30] The method (π₀.₅ + ε-pose semantic entropy on LIBERO-PRO) was committed on Aug 27. The last chat (Aug 30) ends mid-install of `hf-libero`. No π₀.₅ LIBERO number was reproduced and no semantic entropy was computed. The 6-month target set on Aug 18 is **mid-Feb 2027**, about 19 weeks from Oct 7. The plan keeps 8 weeks for writing, which leaves roughly 11 weeks of experiments if nothing ran in September.
2. **The only real result (OpenVLA, AUROC 0.801) is weaker than it looks.**
   - [Likely] Tang et al.'s *naive global-mean* baseline scores AUROC 0.845 on the same suite, so 0.801 is below their simplest baseline, not "approaching their SW-only". The Aug 11 summary mislabelled 0.845 as SW-only; the Aug 18 retrieval of the paper corrected it.
   - [Certain] Every failed episode runs to the 220-step cap (mean T = 220.0), while successes average 106.8 steps. This length confound was flagged on Aug 18 and never tested.
   - [Likely] The AUROC rises steadily as the window grows (0.771 at w=30, 0.801 at w=100). That is what a length detector would also show.
3. **The plan file's Stage A.2 target is wrong.** [Certain] `semantic_entropy_for_vla.md` §9a still says π₀.₅ scores 45.0% on Spatial-Pos (from arXiv 2606.20754). LIBERO-PRO's own paper (arXiv 2510.03827, Table 3) gives **0.20**. The PRO position-shift magnitude (x0.1 to x0.5) is also still unchosen, so A.2 is underspecified.
4. **Semantic entropy is exactly 0 in the most dangerous case.** [Certain] Take a policy that is confidently wrong, where the samples agree (Fig. 2 of 2606.20754). ε-clustering then gives one cluster and SE = 0, while raw variance at least stays nonzero. This needs a written answer before submission.
5. **The camera wall is unsolved.** [Certain] Every LIBERO checkpoint was fine-tuned on a fixed camera, so moving the camera is out-of-distribution (OOD) for every model considered. Stages C and D (camera moves) have no concrete method yet beyond the matched-viewpoint differential idea. [Guessing] Pose-randomised fine-tuning might also flatten the very entropy response the controller needs.

---

## 1. The thesis in one paragraph

MSc thesis at BGU. The supervisor is called Shapira in the chats; the cluster department is `ashapiro`. The question: can a VLA's own uncertainty decide *when* to gather more visual information, by repositioning a camera (active perception), instead of acting on an uncertain view? It started from a Hebrew proposal (reviewed 2026-03-26) that used logit entropy as the trigger.

Tang et al. (arXiv 2603.18342) showed that properly aggregated entropy predicts VLA failure. They left "active uncertainty-aware planning and closed-loop error recovery" as future work, and that sentence is the thesis motivation.

The defensible contribution, as of Aug 27 [Likely]:
1. Transposing Farquhar's *discrete* semantic entropy to continuous action chunks via a behavioural (pose-space) equivalence relation.
2. Using outcome-clustering as an aleatoric filter, which answers VLAConf's objection that sampled disagreement is uninterpretable.
3. Driving camera repositioning from that signal.

---

## 2. History in four phases

| Phase | Dates | What happened |
|---|---|---|
| Learning + proposal V2 | Mar to Jul 2026 | Studied VLAs (RT-2, π₀, OpenVLA), uncertainty (Kendall & Gal, Depeweg, MC Dropout), Tang and Karli. Proposal V2 adopted Tang-style S_t, moved from Isaac Sim to LIBERO, and added a threshold comparison and a Reduction-on-Trigger metric. Ran SigLIP entropy toy tests. |
| OpenVLA instrumentation | Jul 12 to Aug 11 | Built the pipeline on the BGU Run:AI cluster. Reproduced OpenVLA LIBERO-Spatial at 84.8% (published 84.7%). Logged per-step token entropy and perplexity for 500 episodes. Tier-1 AUROC 0.801. |
| Doubts + pivot | Aug 14 to Aug 27 | Designed a MuJoCo perturbation ablation (never completed). Camera-limitation debate. Reviewed a Gemini π₀ plan and corrected it into ε-pose discrete SE. Pivoted to π₀.₅ + LIBERO-PRO Spatial-Pos with a staged plan A to D. |
| π₀.₅ setup | Aug 30 | LeRobot `PI05Policy` single-frame inference worked. `lerobot-eval` was blocked by a LIBERO package collision; fixed `egl_probe` build; next step was `pip install hf-libero`. Status since then unknown. |

---

## 3. Current method (committed 2026-08-27)

Authoritative spec: `source/semantic_entropy_for_vla.md`, which is the Aug 27 revision. The file Jacob flagged, `source/Our Semantic Entropy Plan by Claude.md`, is the **Aug 19 original and is superseded**: it lacks §9a/§9b, still says "forward kinematics", and still uses K̂ for the cluster count.

- **Model:** π₀.₅ via LeRobot, checkpoint `lerobot/pi05_libero_finetuned`.
  - [Certain, checked on Hugging Face 2026-10-07] `lerobot/pi05_libero_finetuned` and `lerobot/pi05_libero_finetuned_v044` are byte-identical: every file (7.47 GB `model.safetensors`, `config.json`, normalizer stats) has the same git hash. Either name loads the same model. The Aug 30 guess that `_v044` was a lerobot-0.4.4 config pin is wrong.
  - What ran on Aug 30: `PI05Policy.from_pretrained("lerobot/pi05_libero_finetuned_v044")` plus `select_action` on one recorded frame of the `lerobot/libero` dataset. That is a single forward pass, not a simulator rollout.
  - Avoid `pi05_libero_base`: lerobot issue #2533 reports 0% LIBERO success with it (probably a norm-stats loading problem), and per the Aug 27 chat its card describes it as a base model to fine-tune, not a LIBERO policy.
  - Set `n_action_steps=10`.
- **Benchmark:** LIBERO-PRO **Spatial-Pos only**.
  - Pos, not Env: an Env shift is not reducible by moving a camera, and the Env perturbation has a documented object-drift bug.
  - One suite before any others.
  - Classic LIBERO-Spatial is only for pipeline validation, because π₀.₅ scores about 97 to 99% there and leaves too few failures.
- **Estimator:** discrete semantic entropy, p̂_c = n_c / N, then SE = −Σ p̂_c ln p̂_c, plus the Miller–Madow correction (Ĉ−1)/(2N).
  - Flow matching gives samples but not cheap likelihoods, so the likelihood-based SE is not computable.
- **Equivalence:** each of N sampled chunks A ∈ ℝ^(H×7) is pose-integrated to a terminal pose. Position is a running sum, orientation composes on SO(3), and the gripper is the last value. This is not FK and not IK.
  - Distance: d = ‖p_A − p_B‖ + λ_R·d_geo(R_A, R_B) + λ_g·|g_A − g_B|. λ_g must dominate.
  - Greedy Farquhar-style clustering at threshold ε, with a fixed, reported seed.
- **Cost (Scenario B):** 1 backbone pass + N action-expert passes per reading, which is 11 passes at N=10.
  - Full-rollout clustering (Scenario A) was rejected as 114× more expensive.
  - Stage A computes SE at every step; chunk-boundary triggering (Scenario C, K≈25) is a deployment optimisation for later.
- **Layer 2 (from the OpenVLA phase):** S_t = max over windows of pool_w[z((1+λ·ATR_t)·SE_t)]. The β-weighted DoF sum disappears and z-scoring becomes mandatory.
- **Staged arc:**
  - A: fixed camera, offline. Does SE correlate with failure? A.1 vanilla Spatial validation, A.2 PRO Spatial-Pos AUROC against the 2606.20754 Table I baselines (sampling 0.935, perturbation 0.963), A.3 ε sweep (mandatory, must show a plateau) and N ∈ {10, 20, 50, 100} with Ĉ tracked.
  - B: fixed camera, live per-step trigger. Pre-registered threshold, held-out split, and three reported numbers: detection latency, false-trigger rate, and the rollout AUROC ceiling.
  - C: confront viewpoint OOD.
  - D: the closed-loop controller.
- **Observation contract (LeRobot LIBERO)** [Certain from docs quoted in chat]:
  - `observation.state` is 8-dim: eef position, axis-angle orientation, gripper qpos.
  - `observation.images.image` is agentview; `observation.images.image2` is the wrist camera. Both are HWC uint8.
  - Actions are Box(−1, 1, (7,)).
  - Plan: use `lerobot-eval` only to validate this contract, then write a custom loop on `OffScreenRenderEnv` + `PI05Policy`. That loop is needed for N-sample access and camera control.
  - N-sample hook: [Likely] LeRobot added a noise argument to the policy's action-prediction methods. Find that signature before writing the sampler, rather than patching the ODE.
- **Do not carry over the OpenVLA preprocessing:** 180° rotation, 0.9 center crop, prompt template, gripper normalize/invert. [Certain] The chat flagged this as the most likely way to get 60% instead of 97%.

---

## 4. Established results (OpenVLA-7B, LIBERO-Spatial, agentview only)

- **Instrument.**
  - Action tokens are IDs [31744, 32000). The vocab is 32000 but the logits are 32064 wide (64 padding), so **never slice `[-256:]`**.
  - Verified gen == full-argmax == slice on about 1,400 rows, including the edge bins.
- **Baseline (dataset A):** 424/500 = **84.8%** (published 84.7%).
  - Per task 0 to 9: 100, 92, 92, 88, 72, 88, 90, 72, 78, 76.
  - File: `~/runs/baseline/libero_spatial_baseline.txt`.
- **Entropy run (dataset B):** 421/500 = 84.2%, 79 failures.
  - Per task: 92, 88, 82, 88, 64, 88, 100, 92, 72, 76.
  - File: `~/runs/stage4_entropy/*.npz`.
  - **A and B disagree on 7 of 10 tasks (task 7: 72 vs 92). This is unresolved.** [Likely] The two runs used different server or preprocessing code.
- **Blur sanity check:**
  - At σ=10, the summed entropy H_sum rose from 5.63 to 7.00, but DoF4 fell from 1.34 to 0.00 (confidently wrong).
  - At σ=3, H_sum fell 62% (5.63 to 2.12), so the response is non-monotone. That is n=1, an anecdote.
- **Tier-1 (Aug 11):**
  - Entropy AUROC 0.801 (sliding-window max, w=100, Tang β, no ATR). Perplexity 0.743. Unweighted combination 0.796, so combining did not help.
  - Precision is at most 0.36 at any z-threshold; the failure base rate is about 16%.
  - Cohen's d ≈ 0.92 for failure vs success episode-mean H, but see the length confound in §0.
- **Pooled stats:**
  - H mean 0.646 nats, median 0.530.
  - Gripper is the most confident DoF (median 0.215) and dx the least.
  - Applying Tang's β to raw H gives the gripper little weight; per-DoF normalisation is probably needed.
- **Tang hyperparameters (Table A1, Spatial):** β = [1, 2.6, 10, 1, 1, 1, 10], w = 93, α = 0.95.
  - Full SW+ATR+BO AUROC 0.936.
  - Tang's own OpenVLA Spatial success was 79.6%.
- **Not done:** ATR and BO stages; length-matched analysis; bin-index-variance vs H "metric-blindness" check on the saved npz; the perturbation ablation (h term).

---

## 5. Open problems, ranked

1. **π₀.₅ environment + A.1 reproduction.** Without this, nothing downstream exists.
2. **Fix plan §9a.** Make A.2.2 0.20 (2510.03827 Table 3), keep 45.0% as the 2606.20754 measurement, and choose the PRO Pos level.
   - [Likely] Two different "Pos" mechanisms are in play. `libero_pro_task_catalog.md` maps Position to `use_swap` (move an object into another object's region), while the Aug 27 chat described five magnitude levels (x0.1 to x0.5) chosen through `.init`/`.bddl` folders. Check which one the 0.20 was measured with before choosing.
   - [Likely] The 0.20 is an average over tasks that range from near 0 up to 0.98. A pooled AUROC can then be driven by *which task* an episode is, not by uncertainty within a task. Report per-task or task-stratified AUROC alongside the pooled one.
3. **Time N=50 batched flow samples** before committing 500 dense-SE episodes.
4. **The confidently-wrong (SE = 0) paragraph** for the proposal.
5. **Length confound** in any OpenVLA result that goes into the thesis. Use length-matched prefixes such as the first 100 steps.
6. **Dataset A/B mismatch.**
7. **Viewpoint OOD (Stage C).** Candidates discussed:
   - Matched-viewpoint differential: the paired-occlusion design, with the pre-registered gate ΔH > 0 in ≥70% of ≥20 pairs per task and Wilcoxon p < 0.05.
   - Pose-randomised fine-tuning, with the risk that it erases the signal.
   - Choosing between in-distribution views is weakened because π₀ fuses agentview and wrist with no per-view knob.
   - LIBERO-plus (LeRobot `--env.type=libero_plus`) perturbs camera pose, so published numbers might quantify the viewpoint drop directly. Unverified; it needs its own env because it uninstalls `hf-libero`.
8. **Prior-art overlap.** Tang's aggregation overlaps the old S_t design; Lee & Har's perturbation paper is the closest method; VLAConf's objection needs the §8.4 answer.
9. **The three-framings memo to the supervisor** (A: trigger improves success; B: abstain/ask for help; C: characterise what entropy measures). Status unknown.

---

## 6. Infrastructure facts that cost days to rediscover

- **Cluster:** BGU HPC, Run:AI, L40S nodes.
  - Home is GPFS at `/gpfs0/bgu-ashapiro/users/yaacubov` and survives container death.
  - Conda envs are at `$HOME/env/{vla,myenv}`; the HF cache is `$HOME/data/hf_cache`.
- **Jobs:** submit batch work from the login node with `runai-bgu submit cmd -n NAME -c 4 -m 24G --gpu-memory 22G --conda vla --working-dir ... -e MUJOCO_GL=osmesa -- "python ..."`.
  - Use **G, not Gi** (32Gi exceeds the 32G limit).
  - The interactive quota is 0.17 GPU (about 8 GB); 24 GB is 0.5 of an L40S.
  - The job memory cgroup is 32 GB. Orphaned model servers (one 7B model each) cause OOM kills.
- **Rendering:**
  - EGL is unavailable because the container has `NVIDIA_DRIVER_CAPABILITIES=compute,utility`, so use **OSMesa**.
  - OSMesa, TensorFlow and torch collide in one process (segfault). That is why the pipeline is **split into two processes**: the sim orchestrator and `model_server.py`, talking over a stdin/stdout pickle pipe.
  - TensorFlow auto-imported by transformers also segfaults the server; fixed by `USE_TF=0` or uninstalling TF.
- **OpenVLA stack:** transformers 4.40.1, tokenizers 0.19.1, timm 0.9.10, robosuite 1.4.0, mujoco 2.3.7, LIBERO commit `8f1084e`, imported through PYTHONPATH because the editable install maps nothing.
- **env/vla is broken for OpenVLA.** [Certain] As of Aug 30 it has transformers 5.5.4 and an editable lerobot 0.6.2 (`~/thesis/lerobot`). Plan: three envs, one each for OpenVLA (to finish the ablation), vanilla LIBERO with LeRobot, and LIBERO-plus. Where LIBERO-PRO goes was not decided.
- **Two LIBERO packages, one namespace.**
  - The original clone, via PYTHONPATH, silently wins over `hf-libero`, which `lerobot[libero]` needs.
  - Add `assert` lines on `libero.__file__` in each launcher.
  - Build fix: `pip install egl_probe --no-build-isolation`, then `pip install hf-libero`.
- **Code:** `github.com/Jacob170/vla-active-perception`, cloned at `~/thesis/vla-active-perception`.
  - Key scripts: `eval_split.py`, `model_server.py`, `step_4_entropy_logger.py`, `step_4_offline_analysis.py`, `ablations/mujoco_perturbation_ablation/`.
  - A separate manuscript repo has a LaTeX conversion and a `RESEARCH_LOG.md` (Entries 001 to 004).
  - Google Drive "תזה" folder: Hebrew docs 01 to 03 (theory, journey, work plan) and a "baseline log" doc.

---

## 7. Key literature (verify before citing)

- Kuhn, Gal, Farquhar, ICLR 2023 (semantic entropy). Farquhar et al., Nature 2024 (the discrete estimator).
- Tang et al., "Shifting Uncertainty to Critical Moments", arXiv 2603.18342 (OpenVLA, SW+ATR+BO, LIBERO).
- Karli, Kurumisawa, Fitzgerald, "Ask Before You Act", RSS 2025 workshop (π₀-FAST, entropy + perplexity).
- Valle et al., arXiv 2507.17049 (uncertainty metrics on VLAs).
- Lee & Har, "Perturbation-Based Uncertainty for Failure Detection in VLA Models", arXiv 2606.20754: π₀.₅ on LIBERO-PRO, Table I; its sampling baseline is raw variance.
- VLAConf, arXiv 2605.29605.
- Zhou et al., LIBERO-PRO, arXiv 2510.03827. LIBERO-Plus, arXiv 2510.13626 (camera viewpoint is the top vulnerability; base OpenVLA 0.8% under camera perturbation).
- ActiveVLA, arXiv 2601.08325 (4 fixed RGB-D cameras + virtual views; no physical camera moves).
- Kossen et al., SE probes, arXiv 2406.15927. Nikitin et al., KLE (the threshold-free fallback). Bayesian SE budget, arXiv 2504.03579.
- Black et al. π₀ (2024) and π₀.₅ (arXiv 2504.16054). Lipman et al., flow matching. Bajcsy 1988 (active perception). Kendall & Gal 2017. Depeweg decomposition.

---

## 8. Source files copied into this project

`/mnt/project-files/thesis-import/source/`:
- `semantic_entropy_for_vla.md`: the current method spec (Aug 27).
- `Our Semantic Entropy Plan by Claude.md`: the Aug 19 original, superseded.
- `scenario_B_chunk_sampling.html`: the actual algorithm loop and cost (11 passes per reading; 1,375 per episode every step, 275 at chunk boundaries).
- `scenario_A_full_rollout.html`: the rejected full-rollout reading (114× cost).
- `libero_pro_task_catalog.md`: every LIBERO-PRO perturbation per classic task.

The OpenVLA ablation scripts and the raw chat PDFs stay in `Downloads/thesis/claude` on Jacob's PC.
