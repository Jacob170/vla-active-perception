# CLAUDE.md: VLA active-perception thesis

Read this before doing any work on this project. Human overview with step statuses: `docs/README.md` (copies of the brief, decision log and spec are also in `docs/`). History: `thesis-import/thesis-brief.md` and `thesis-import/decision-log.md`. **Source of truth for the method:** `thesis-import/source/semantic_entropy_for_vla.md` (§8 method, §9a/§9b stage plan, §12 prior art, §13 reviewer attacks).

## 1. Project

- Jacob's MSc thesis at BGU (supervisor: Shapira). Question: can a VLA's own uncertainty decide when to reposition the camera (active perception) instead of acting on an uncertain view?
- Method: π0.5 (LeRobot `PI05Policy`) on LIBERO-PRO Spatial-Pos. Discrete semantic entropy (SE) over N sampled flow-matching action chunks, clustered by terminal end-effector pose (ε-pose clustering).
- Stages: **A** offline, fixed camera (does SE predict failure?) → **B** live per-step trigger, fixed camera → **C** viewpoint OOD → **D** closed-loop controller (end goal).
- Deadline: mid-Feb 2027. From 2026-10-07 that is about 19 weeks: 8 for writing, about 11 for experiments.

## 2. State and next tasks

- Known state ends **2026-08-30** (export 2026-09-04). Nothing has run on the current method. One `PI05Policy.select_action` on one `lerobot/libero` dataset frame worked; `lerobot-eval` failed with "no libero"; the next command was `pip install hf-libero`.
- 2026-10-07: Jacob does not remember anything running after 2026-09-04 beyond maybe a few-episode example. No output files dated after 2026-09-04 exist in the repo, `~/runs`, `~/videos`, `~/data`, `~/scripts`, `~/thesis/lerobot` or `~/LIBERO` (checked 2026-10-07). Treat it as nothing ran.
- 2026-10-07, Stage 0 partly done in `env/vla`: `~/.libero/config.yaml` now points at hf-libero; OpenCV fixed to headless; no-model smoke test passes (`scripts/pi_0.5_fintuned/smoke_libero_lerobot.py`, details in §4.4). Not done yet: separate OpenVLA env, 2-episode run with the model.
- **First action in a new session:** check README §7 and §8 match the state above.
- Task queue, in order (details in README §8):
  1. Stage 0: separate envs, swap the LIBERO clone for `hf-libero`, 2-episode smoke test.
  2. A.1: reproduce LIBERO-Spatial (~97–99%) with `lerobot-eval`; write down the observation and action contract.
  3. Own loop on `OffScreenRenderEnv` + `PI05Policy`; must match `lerobot-eval` on the same episodes.
  4. SE sampler, pose integration and clustering; sanity checks; time N = 50 batched.
  5. A.2 on PRO Spatial-Pos, only after the Pos setting is resolved (§4.1).
- Out of scope until Jacob says otherwise: camera moves (Stages C and D), extra PRO suites, fine-tuning.

## 3. Working with Jacob

- Advisor, not assistant. The first sentence challenges an assumption or names the gap. Uncomfortable truth first. No warm-up.
- Tag claims **[Certain]** / **[Likely]** / **[Guessing]**. Say up front if most of a reply is guessing.
- Never write: "Great question", "You're absolutely right", "That makes a lot of sense", "Absolutely", "Definitely".
- Disagree as: "I disagree because X. Here's what I'd do instead: Y. The risk in your approach is Z." Hold the position unless he gives new information.
- **Do only what he asked.** He has objected to unrequested work. Name anything else in one line and let him decide.
- If he asks for a short answer, give a few lines. When teaching, define jargon, build from basics, and use visuals.
- Theory and summary documents for him have been in Hebrew, right-to-left (math and code left-to-right). Chat is in English.
- Cite the source of every number: paper and table, or run and file. He checks.

## 4. Hard rules

### 4.1 Model and benchmark

- Checkpoint `lerobot/pi05_libero_finetuned`. `lerobot/pi05_libero_finetuned_v044` is byte-identical (every file has the same git hash, checked 2026-10-07). Never use `lerobot/pi05_libero_base`: lerobot issue #2533 reports 0% on LIBERO. (The spec calls it an openpi issue; it is in huggingface/lerobot.)
- Always pass `--policy.n_action_steps=10`. The checkpoint config defaults to 50, which executes the whole chunk open-loop.
- Checkpoint config (Hugging Face `config.json`, read 2026-10-07): `chunk_size` 50, `num_inference_steps` 10; inputs `observation.images.image` (agentview) and `observation.images.image2` (wrist), both 3×256×256, `observation.state` (8), plus one empty camera; action dim 7; MEAN_STD normalisation for state and action; PaliGemma `gemma_2b` backbone with a `gemma_300m` action expert.
- Benchmark: LIBERO-PRO **Spatial-Pos only** until A.2 shows a signal. Classic LIBERO-Spatial is for pipeline validation only: about 97–99% success leaves too few failures for AUROC.
- π0.5 on PRO Spatial-Pos = **0.20** (arXiv 2510.03827, Table 3). The spec's 45.0% (lines 354, 360, 403) comes from arXiv 2606.20754 and is not the A.2.2 loading-check target.
- ⚠️ Unresolved: which Pos mechanism. `libero_pro_task_catalog.md` maps Position to `use_swap`; the Aug 27 chat described shift levels x0.1–x0.5 via `.init`/`.bddl` folders. Ask Jacob before A.2.
- **Never carry OpenVLA preprocessing into π0.5:** no 0.9 centre crop, no `In: What action should the robot take to ...?` prompt template, no gripper normalise/invert. Use LeRobot's `make_pre_post_processors`.
- **Image flip:** π0.5 *does* see 180°-rotated images, but LeRobot applies the rotation itself in `LiberoProcessorStep` (`lerobot/processor/env_processor.py`, `torch.flip(img, dims=[2, 3])`, checked in the LeRobot 0.6.1 wheel on 2026-10-07). Own loops must send raw renders and run that step, as `lerobot-eval` does; flipping by hand as well gives upside-down images. The same step builds `observation.state` = eef pos (3) + eef axis-angle (3) + gripper qpos (2).

### 4.2 Estimator

- Discrete SE only: p̂_c = n_c/N, SE = −Σ p̂_c ln p̂_c, plus Miller–Madow (Ĉ−1)/(2N). Never a likelihood-based SE: flow matching gives no cheap p(a|x).
- Pose integration, not FK or IK. Position: p0 + Σ Δp. Orientation: compose on SO(3) by quaternion or matrix product, never sum. Gripper: last value. Take p0 and R0 from the observation (LIBERO `robot0_eef_pos` / `robot0_eef_quat`, or `observation.state`).
- [Likely] LIBERO actions are normalised deltas in [−1, 1] that robosuite's OSC controller scales. Convert them to metres and radians before computing d, or define ε in action units and say so. Verify against the controller config.
- Distance d = ‖Δp‖ + λ_R·d_geo(R_A, R_B) + λ_g·|Δg|. λ_g must dominate, so gripper disagreement always splits clusters.
- Greedy clustering against one representative per cluster. Fix and record the shuffle seed.
- Sample N independent noise seeds per reading and encode the backbone once. [Likely] LeRobot's action-sampling method takes a noise argument; use it instead of patching the ODE. Resample fresh seeds every reading (Lee & Har reuse one noise tensor; state the difference).
- ⚠️ Open: H. The spec assumed H = 10 (openpi `action_horizon`); the LeRobot chunk is 50 with 10 executed. Until Jacob decides, compute SE for both H = 10 and H = 50 from the same samples (no extra model passes) and log both.
- Log per reading: N, ε, λ_R, λ_g, H, seed, Ĉ, SE.

### 4.3 Experiment protocol

- Stage A computes SE at **every environment step**, not only at chunk boundaries. **No camera moves in Stages A and B.**
- No AUROC on classic LIBERO-Spatial.
- Report AUROC pooled **and per task**, plus a length-matched version: failed episodes run to the step cap.
- ε sweep over two orders of magnitude (must plateau; otherwise fall back to KLE). N ∈ {10, 20, 50, 100}, with Ĉ plotted next to SE.
- Stage B: pre-register the threshold method (Youden) and evaluate on a held-out split. Report detection latency, false-trigger rate and the Stage A AUROC ceiling.
- Stage C gate (pre-registered): camera fixed at the training pose, target occluded vs visible; ΔH > 0 in ≥70% of ≥20 pairs per task, Wilcoxon p < 0.05.

### 4.4 Cluster (BGU Run:AI)

- Submit from the login node: `runai-bgu submit cmd -n NAME -c 4 -m 24G --gpu-memory 22G --conda <env> --working-dir ... -e MUJOCO_GL=osmesa -- "python ..."`. Use **G, never Gi** (32Gi exceeds the 32G limit). For π0.5 request the full 24 GB GPU.
- **`MUJOCO_GL=osmesa`.** EGL is unavailable (`NVIDIA_DRIVER_CAPABILITIES=compute,utility`). Commands copied from docs that set `egl` will fail; change them.
- Set `USE_TF=0`. The image is tf-docker, and transformers auto-importing TensorFlow segfaults alongside OSMesa.
- The OpenVLA pipeline is split-process: sim orchestrator plus `model_server.py`, pickle over stdin/stdout. Test whether π0.5 needs the same split before porting it.
- Before launching, run `pgrep -af model_server`. Orphaned 7B servers cause OOM in the 32 GB cgroup.
- Envs: never install LeRobot into the OpenVLA env. Plan: three envs (OpenVLA; LeRobot + `hf-libero`; LIBERO-plus). `env/vla` is now Python 3.12 + transformers 5.5.4 + LeRobot 0.6.2 (editable at `~/thesis/lerobot`), so OpenVLA is broken there. Rebuild OpenVLA from `PINNED.md` / `requirements-frozen.txt`: transformers 4.40.1, tokenizers 0.19.1, timm 0.9.10, robosuite 1.4.0, mujoco 2.3.7 (not 3.x), LIBERO commit `8f1084e` via `PYTHONPATH`.
- LIBERO import clash: a clone on `PYTHONPATH` silently shadows `hf-libero`. Every launcher asserts `libero.__file__`. OpenVLA side: `'yaacubov/LIBERO' in libero.__file__`. LeRobot side: run under `env -u PYTHONPATH` and assert `'site-packages' in libero.__file__`. 
- LIBERO config (done 2026-10-07): `~/.libero/config.yaml` now points every key at hf-libero in `~/env/vla/lib/python3.12/site-packages/libero/libero` (bddl_files, init_files). The old config (pointing at `~/LIBERO/libero/libero`) is kept at `~/.libero_openvla/config.yaml` and `~/.libero/config.yaml.openvla.bak`; OpenVLA launchers must set `LIBERO_CONFIG_PATH=~/.libero_openvla` (both LIBERO versions read that variable). Never delete `~/.libero/config.yaml`: if it is missing, hf-libero's import calls `input()` and a batch job hangs.
- hf-libero ships no meshes. It ignores the config `assets` key and downloads them on first env creation to `~/.cache/libero/assets` (586 files, about 1 min; done 2026-10-07). libero_spatial BDDL and init files are byte-identical between hf-libero 0.1.4 and the `~/LIBERO` clone (`diff -rq`, 2026-10-07).
- `env/vla` OpenCV: hf-libero and robosuite pull `opencv-python` (GUI build), which overwrote cv2 and needs `libGL.so.1`, absent in the image; `import robosuite` failed. Fixed 2026-10-07 with `pip install --force-reinstall --no-deps opencv-python-headless==4.13.0.92` (cv2 4.13.0). Any reinstall of hf-libero or robosuite can bring the GUI build back; rerun that command.
- Stage 0 smoke test, no model (passed 2026-10-07 on interactive-dev-0-9): `env -u PYTHONPATH USE_TF=0 MUJOCO_GL=osmesa ~/env/vla/bin/python scripts/pi_0.5_fintuned/smoke_libero_lerobot.py`. (A) hf-libero `OffScreenRenderEnv`, libero_spatial task 0, init state 0, 10 dummy steps, 256×256 renders. (B) LeRobot `make_env(LiberoEnv)` → `preprocess_observation` → env preprocessor: `observation.images.image` and `image2` (1,3,256,256) float in [0,1], `observation.state` (1,8). Saved frames in `scripts/pi_0.5_fintuned/smoke_out/` confirm the raw render is upside-down and the processor output upright. `lerobot-eval` is at `~/env/vla/bin/lerobot-eval`.
- Versions in `env/vla` (2026-10-07): lerobot 0.6.2 (editable, `~/thesis/lerobot` at bf31dd79), hf-libero 0.1.4, robosuite 1.4.0, mujoco 3.8.1, torch 2.11.0, gymnasium 1.3.0, numpy 2.2.6. The interactive pod has an L40S slice of only about 7.6 GB, so π0.5 (7.5 GB of weights) must go through `runai-bgu submit`.
- `egl_probe` build fix: `pip install egl_probe --no-build-isolation`.

### 4.5 Legacy OpenVLA facts

- Action tokens are IDs [31744, 32000). Logits are 32064 wide: never slice `[-256:]`. Perplexity = 1/p_max per DoF.
- Data: `~/runs/baseline/libero_spatial_baseline.txt` (84.8%) and `~/runs/stage4_entropy/*.npz` (500 episodes; Tier-1 AUROC 0.801).
- New episode files: save `task_id` as int and use `np.savez_compressed`. The ablation saved a string, which broke `step_4d_baseline_stats.py`. The ablation judges success as `info.get("success", reward > 0)` while the baseline uses `done`; align them before comparing.

## 5. File map

| Path | What |
|---|---|
| `docs/README.md` (repo) | Human overview, plan with statuses, tests, risks |
| `docs/` (repo) | Copies of thesis-brief, decision-log, spec, LIBERO-PRO catalog, scenario pages |
| `thesis-import/source/semantic_entropy_for_vla.md` | Method spec (Aug 27) |
| `thesis-import/source/scenario_B_chunk_sampling.html` | SE loop and costs |
| `thesis-import/source/libero_pro_task_catalog.md` | LIBERO-PRO perturbations per task, with config flags |
| `thesis-import/thesis-brief.md`, `decision-log.md` | History and decisions |
| Cluster `~/thesis/vla-active-perception/` | Code repo (GitHub `Jacob170/vla-active-perception`): `model_server.py`, `scripts/`, `signals/entropy.py`, `ablations/` |
| Cluster `~/thesis/lerobot/` | LeRobot 0.6.2, editable |
| Cluster `/gpfs0/bgu-ashapiro/users/yaacubov/` | `$HOME` on GPFS; survives container death |
| Cluster `~/LIBERO/` | Upstream LIBERO clone at `8f1084e`, OpenVLA side only |
| Cluster `~/env/vla/lib/python3.12/site-packages/libero/` | hf-libero 0.1.4 (LeRobot side); assets in `~/.cache/libero/assets` |
| Cluster `~/data/hf_cache/hub/` (`HF_HOME=~/data/hf_cache`) | `models--lerobot--pi05_libero_finetuned_v044`: complete, revision 8e174154, `model.safetensors` 7,473,096,344 B, sha256 877b3ec1… matches the Hub LFS hash (checked 2026-10-07); all 6 runtime files match Hub hashes (`train_config.json` and README not downloaded). `models--lerobot--pi05_libero_finetuned` has only `config.json`. PaliGemma tokenizer in `models--google--paligemma-3b-pt-224`. |
| Repo `scripts/openvla/` | OpenVLA scripts with an `_openVLA` suffix, moved from `scripts/` |
| Repo `scripts/pi_0.5_fintuned/` | `pi_0.5_load_model_test1.py` (one-frame load test, 2026-08-27), `smoke_libero_lerobot.py` (Stage 0 no-model test) |

π0.5 code so far: `thesis-docs/ablation_pi05/` (side task S.5, written 2026-10-07, never run; README §8.1). Its server follows the `lerobot-eval` chain (`preprocess_observation` → `LiberoProcessorStep` → policy pre/post-processors) and samples N chunks in one batch via `predict_action_chunk(batch, noise=...)`. Its SE functions can seed `signals/` for Stage A. Suggested placement, following the repo: runners in `scripts/`, the SE estimator in `signals/` next to `signals/entropy.py`.

## 6. Done means

- The command, env name, git commit and output paths are recorded.
- Numbers are checked against the source they should match, and the source is cited.
- README §7 and §8 statuses are updated.

## 7. Prior art to position against

- Tang et al., arXiv 2603.18342: temporal aggregation; overlaps Layer 2. Naive global-mean baseline 0.845 on Spatial.
- Lee & Har, arXiv 2606.20754: perturbation-based uncertainty; their sampling baseline is raw variance. Fig. 2 (confidently wrong) gives SE = 0 for us; the thesis must answer it.
- VLAConf, arXiv 2605.29605: says sampled disagreement is uninterpretable. Answer: outcome clustering as an aleatoric filter (spec §8.4).
- ActiveVLA, arXiv 2601.08325: 4 fixed RGB-D cameras plus virtual views; no physical camera motion.
- Novelty claims (spec §12): discrete SE for continuous action chunks via behavioural equivalence; outcome clustering as an aleatoric filter; camera repositioning driven by that signal.
