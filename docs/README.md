# Active Perception for VLAs, Driven by Semantic Entropy

MSc thesis at BGU (supervisor: Shapira). Owner: Jacob.

> **Status date: 2026-10-07.** Everything here comes from the old Claude chats up to **2026-08-30** (exported 2026-09-04). If anything ran after that, update section 7 ("Where we stopped") first.
>
> Confidence tags: **[Certain]** seen in the chats, files or papers. **[Likely]** strong inference. **[Guessing]** filling a gap.
>
> Short on time? Read sections **1**, **7** and **8**.

---

## 1. Read this first: where things really stand

| # | The hard truth | What it means for you |
|---|---|---|
| 1 | **Nothing has run on the current method yet.** [Certain] | The thesis result does not exist yet. Job one is getting π0.5 to run inside LIBERO. |
| 2 | **The only real result is weak.** OpenVLA entropy AUROC = 0.801. | It is below Tang et al.'s simplest baseline (0.845) [Likely]. Also, every failed episode ran to the 220-step limit, so the score may partly measure episode length [Certain]. |
| 3 | **The two reference papers disagree.** [Certain] | LIBERO-PRO's own paper gives π0.5 **20%** on Spatial-Pos; Lee & Har report 43.8%. Neither names its checkpoint, so there is no ready-made baseline for our model: we must produce our own (spec §9a.2). The spec was fixed on 2026-10-07. |
| 4 | **Semantic entropy is 0 when the robot is confidently wrong.** [Certain] | If all sampled plans agree on a wrong move, the score says "sure". The thesis needs a written answer to this. |
| 5 | **Moving the camera is not solved.** [Certain] | Every model was trained with a fixed camera, so any new view confuses it. Stages C and D have no method yet. |
| 6 | **Time is tight.** | Target: mid-Feb 2027, about 19 weeks away. 8 weeks are for writing, so about 11 weeks remain for 4 stages of experiments. |

---

## 2. The idea in one picture

```
TODAY'S VLA
  image + instruction --> VLA picks one action --> robot acts, even when confused

OUR VLA
  image + instruction --> VLA samples N possible plans
                                      |
                                      v
                      group the plans by where the hand ends up
                                      |
                 +--------------------+--------------------+
                 |                                         |
                 v                                         v
        one group (low SE)                       several groups (high SE)
        "I know what to do"                      "I am torn between options"
                 |                                         |
                 v                                         v
                act                             move the camera, look again
                                                           |
                                      +--------------------+--------------------+
                                      |                                         |
                                      v                                         v
                              now one group: act                    still torn: stop or ask for help
```

**SE = semantic entropy**: a number that says how much the sampled plans disagree about *what* to do.

---

## 3. The problem we solve, and how

### 3.1 What is wrong with today's VLAs

| # | Problem | Evidence |
|---|---|---|
| 1 | They act even when they are confused. They give no "I am not sure" signal. | Tang et al. (arXiv 2603.18342) and Lee & Har (arXiv 2606.20754) both build failure detectors because the policy itself gives none. |
| 2 | Small changes in the scene break them. | π0.5 gets 98% on LIBERO-Spatial but **20%** when objects are moved (LIBERO-PRO, arXiv 2510.03827, Table 3). [Certain] |
| 3 | They see from one fixed camera. If the view is bad, they cannot look again. | LIBERO-plus (arXiv 2510.13626): camera viewpoint is the top weakness. ActiveVLA needs 4 fixed RGB-D cameras to get around it. |
| 4 | Current uncertainty scores do not fit modern VLAs. | Token entropy only works for token models like OpenVLA, and it ignores how far apart two actions are. π0 and π0.5 (flow matching) give no probabilities at all. Raw spread between samples cannot tell harmless variety from real confusion (the VLAConf objection, arXiv 2605.29605). |

### 3.2 How our method answers each problem

| # | Our answer | Why it helps | Honest limit |
|---|---|---|---|
| 1 | Measure SE at every step. | High SE means the model is torn between different outcomes, so it is a warning before failure. | SE = 0 when the model is confidently wrong. |
| 2 | Test on LIBERO-PRO Spatial-Pos, where π0.5 fails often. | About 80% failures gives plenty of cases to test the warning on. | Success per task ranges from about 0 to 98%, so results must be reported per task. |
| 3 | Move the camera only when SE is high (active perception). | One camera, used smartly, instead of many cameras all the time. | A new view is out-of-distribution for the model. Unsolved (Stage C). |
| 4 | Sample N plans, group them by where the hand ends up, count the groups. | Needs only samples, so it works on flow-matching VLAs. Ignores harmless variety (see 3.3). | Depends on the grouping distance ε, which needs a stable range. |

### 3.3 The key idea in one example

```
Case 1: all 10 plans end at the same grasp      -> 1 group  -> SE = 0.00 -> act
        (the paths differ a bit: harmless variety)

Case 2: 6 plans end at the bowl, 4 at the plate -> 2 groups -> SE = 0.67 -> look again
        (real confusion about WHAT to do)

Raw spread between plans is high in BOTH cases. SE is high only in case 2.
```

This is the main thesis argument: **grouping by outcome filters out harmless variety (aleatoric) and keeps real confusion (epistemic)**, which is the confusion that looking again can fix.

---

## 4. End goal and hypotheses

### 4.1 End goal

A π0.5 robot policy in LIBERO that watches its own SE while it works. When SE crosses a threshold, it moves the camera and checks again. If SE drops, it continues. If it does not, it stops or asks for help. Success is measured as **higher task success than the same policy without the trigger**, plus a positive **Reduction-on-Trigger** (SE before the camera move minus SE after).

### 4.2 Main hypothesis

> When a VLA's semantic entropy is high, it is likely to fail. Looking again at those moments lowers the entropy and raises the success rate.

### 4.3 Hypotheses, one per stage

| ID | Hypothesis | Stage | How we test it | Expected result | Status |
|---|---|---|---|---|---|
| H0 | Token entropy predicts failure (old OpenVLA pipeline). | Before the pivot | AUROC over 500 LIBERO-Spatial episodes | Above 0.5 | **Tested:** 0.801. Weak: below Tang's 0.845 baseline and possibly an episode-length effect. |
| H1 | SE separates failed from successful episodes (fixed camera). | A | AUROC over 500 PRO Spatial-Pos episodes | Clearly above 0.5 and above our own raw-variance baseline; Lee & Har report 0.943 (sampling) and 0.961 (perturbation), arXiv 2606.20754 Table I, current version | Not tested |
| H2 | SE is stable: a plateau over ε and N. | A.3 | ε sweep over 2 orders of magnitude; N in {10, 20, 50, 100} | A plateau exists | Not tested |
| H3 | SE works as a live warning. | B | Per-step trigger on held-out episodes | Fires before failure, with few false alarms | Not tested |
| H4 | Looking again lowers SE when the confusion comes from the view. | C | Paired-occlusion test | ΔH > 0 in ≥70% of ≥20 pairs per task, Wilcoxon p < 0.05 | Not tested |
| H5 | The closed loop raises success. | D | Same tasks, trigger on vs off | Higher success, Reduction-on-Trigger > 0 | Not tested |

Small checks already done: a SigLIP toy test showed blur raises entropy 12.7× when the text choices compete [Certain]. An OpenVLA blur test gave a non-monotone response, but on only 1 episode.

### 4.4 What the entropy we measure is made of

```
 entropy we measure  =  f  (the task is truly ambiguous)      <- the part we want
                     +  g  (the camera view is new to model)  <- noise from moving the camera
                     +  h  (lighting, texture, ...)           <- noise from the scene
```

Each planned test isolates one part. The MuJoCo ablation measures h, a viewpoint sweep measures g, and the paired-occlusion test measures f.

---

## 5. How the method works

### 5.1 One SE reading

```
 one SE reading (repeats every control step)
 --------------------------------------------------------------------------
 1  OBSERVE     agentview image + wrist image + robot state
 2  ENCODE      big backbone runs ONCE                        (1 pass)
 3  SAMPLE      small action expert runs N times, new seed    (N = 10 passes)
                -> N plans, each H steps x 7 numbers
 4  INTEGRATE   add up each plan's moves -> final hand pose
 5  GROUP       plans that end within ε of each other = one group
 6  COUNT       p = share of plans in each group;  SE = -Σ p·ln p  (+ bias fix)
 7  DECIDE      smooth over time -> compare with threshold -> act, or look again
 --------------------------------------------------------------------------
 cost: 1 + N = 11 passes per reading
```

| | Every step (Stage A) | Only when a new chunk starts (deployment, later) |
|---|---|---|
| Readings per episode | about 125 | about 25 |
| Passes per episode | 1,375 | 275 (5× cheaper) |

Rejected: running N full episodes per reading ("Scenario A"), which costs 114× more.

### 5.2 The formulas

| What | Formula | Note |
|---|---|---|
| Share of group c | p̂_c = n_c / N | Just counting. No probabilities needed. |
| Semantic entropy | SE = −Σ p̂_c ln p̂_c + (Ĉ − 1)/(2N) | The second term is the Miller–Madow bias fix. Ĉ = groups found. SE ≤ ln N. |
| Distance between plans A and B | d = ‖p_A − p_B‖ + λ_R·angle(R_A, R_B) + λ_g·\|g_A − g_B\| | p = hand position, R = hand rotation, g = gripper. λ_g must be the largest. |
| Final hand pose | position: sum the steps; rotation: multiply the step rotations; gripper: last value | This is "pose integration", not forward or inverse kinematics. |
| Smoothed signal (Layer 2) | S_t = max over windows of pool[z((1 + λ·ATR_t)·SE_t)] | ATR = how jumpy recent actions are. z = standardise. |

---

## 6. What we did so far

| When (2026) | What we did | Result |
|---|---|---|
| Mar 26 | Wrote the research proposal (Hebrew): entropy as the trigger to move the camera. | Proposal reviewed. |
| Mar to Jun | Learned VLAs (RT-2, OpenVLA, π0) and uncertainty (Kendall & Gal, MC Dropout). | Background. |
| Jun 3 | Proposal V2: Tang-style trigger signal, LIBERO instead of Isaac Sim, a new metric (Reduction-on-Trigger), Karli as a baseline. Wrote the abstract, introduction and related work. | V2 plan and a 17-page draft. |
| Jun 22 | Toy test with SigLIP (an image-text model). | Blur raised entropy 12.7× when choices competed. |
| Jul 12–16 | Built the pipeline on the BGU cluster (Run:AI, L40S GPU). Simulator and model run as two separate processes. | Works without crashes. |
| Jul 20 | Checked that we read the right 256 action scores. Blur test on OpenVLA. | Verified. Blur effect not monotone (1 episode). |
| Aug 10 | Ran OpenVLA on LIBERO-Spatial, 500 episodes. | **84.8%** (paper: 84.7%). Pipeline validated. |
| Aug 11 | Logged entropy for 500 episodes and computed AUROC. | Entropy **0.801**, perplexity 0.743, both together 0.796. |
| Aug 14–18 | Designed a MuJoCo test: brightness, texture, small camera shifts (120 episodes). | Half finished. |
| Aug 18 | Camera discussion: a moved camera confuses fixed-camera models. Rejected OpenVLA-OFT (it has no entropy). Target set: mid-Feb 2027. | The "camera wall" was found. |
| Aug 19 | Rewrote the method as semantic entropy with ε-pose grouping. | `semantic_entropy_for_vla.md` |
| Aug 23–27 | Fixed details (pose integration). Switched to π0.5 + LIBERO-PRO Spatial-Pos. Wrote the stage plan A to D. | **The current plan.** |
| Aug 30 | Started π0.5 setup with LeRobot. | One forward pass worked. The simulator test was blocked by a LIBERO install clash. |

### 6.1 OpenVLA results in detail

| Measure | Value |
|---|---|
| Success, baseline run (A) | 84.8% (424/500) |
| Success, entropy run (B) | 84.2% (421/500) |
| Runs A and B disagree | on 7 of 10 tasks (task 7: 72% vs 92%). Not explained. |
| Entropy AUROC (window 100, Tang weights, no ATR) | 0.801 |
| Perplexity AUROC | 0.743 |
| Both combined | 0.796 (no gain) |
| Best precision | 0.36 (failure rate is about 16%) |
| Length of failed episodes | always 220 steps (the cap) vs 106.8 for successes |
| Not done | ATR and BO stages, a same-length test, the metric-blindness check |

### 6.2 What we built on the cluster

```
 +--------------------------+     pickle pipe     +--------------------------+
 | simulator process        | ------------------> | model_server.py          |
 | MuJoCo + LIBERO          |                     | OpenVLA on the GPU       |
 | OSMesa rendering         | <------------------ | returns action + entropy |
 +--------------------------+                     +--------------------------+
```

Two processes, because MuJoCo rendering, TensorFlow and torch crash when loaded together.

---

## 7. Where we stopped (as of 2026-08-30)

**The last command in the chats:** `pip install hf-libero` (not run yet).

| Area | File or thing | Where | State |
|---|---|---|---|
| π0.5 setup | Test script | none saved | The LeRobot docs example was pasted inline: load `PI05Policy`, run one frame of the `lerobot/libero` dataset. **It worked.** |
| π0.5 setup | `lerobot-eval` | cluster | **Failed: "no libero".** The old LIBERO clone clashes with `hf-libero`. |
| π0.5 setup | `egl_probe` build | cluster, `env/vla` | Fixed with `pip install egl_probe --no-build-isolation`. |
| π0.5 model | `lerobot/pi05_libero_finetuned` (same files as `..._v044`) | Hugging Face cache on the cluster [Likely] | Downloaded and loads. Checked 2026-10-07: both names are byte-identical. |
| Environment | `env/vla` | `/gpfs0/bgu-ashapiro/users/yaacubov/env/vla` | Now Python 3.12 (it was 3.10 in July), transformers 5.5.4, LeRobot 0.6.2. **OpenVLA no longer runs here.** |
| LeRobot | editable install | `~/thesis/lerobot` | v0.6.2 |
| Plan | `semantic_entropy_for_vla.md` | `thesis-import/source/` | Method spec. Revised 2026-10-07: numbers fixed, plus §8.5 (LeRobot contract), §9a.2 (baselines), §9c (camera out of distribution), §9d (ablation). |
| Plan | `scenario_B_chunk_sampling.html` | `thesis-import/source/` | The algorithm loop and costs. |
| Plan | `libero_pro_task_catalog.md` | `thesis-import/source/` | Every LIBERO-PRO change for every task. |
| OpenVLA code | `vla-active-perception` repo | `~/thesis/vla-active-perception` (GitHub: Jacob170) | Done. Key files: `model_server.py`, `scripts/eval_split.py`, `scripts/step_4_entropy_logger.py`, `scripts/step_4_offline_analysis.py`, `signals/entropy.py`, `PINNED.md` (pinned versions). |
| OpenVLA data | Baseline log | `~/runs/baseline/libero_spatial_baseline.txt` | 84.8% |
| OpenVLA data | Entropy episodes | `~/runs/stage4_entropy/*.npz` + plots | 500 episodes, analysed |
| Ablation | `ablations/mujoco_perturbation_ablation/` | in the repo, outputs in `outputs/ablation/` | Half finished. Cannot run in `env/vla` any more. |
| Writing | Draft chapters, Hebrew docs 01–03, `RESEARCH_LOG.md` (entries 001–004) | Google Drive folder "תזה", manuscript repo | Drafts. |

---

## 8. The plan, step by step

Legend: ✅ done · ⬜ to do · ⚠️ needs a decision first

```
  Stage A                Stage B               Stage C               Stage D
  fixed camera           fixed camera          camera moves          camera moves
  offline                live                  (open problem)        closed loop
  "Is SE a real    --->  "Can SE warn    --->  "Does looking   --->  "Does the robot
   failure signal?"       in time?"             again help?"          succeed more?"
                                                                      = END GOAL
```

Rule: **no camera moves in Stages A and B.** Each stage answers a different question.

### Stage 0: get π0.5 running in LIBERO

| # | Step | Status |
|---|---|---|
| 0.1 | Keep separate conda envs: one for OpenVLA, one for LeRobot + LIBERO (π0.5), one for LIBERO-plus later. Never install LeRobot into the OpenVLA env. | ⬜ |
| 0.2 | In the π0.5 env: `pip uninstall -y libero`, unset `PYTHONPATH`, and move `~/.libero/config.yaml` to a backup (the old one points at the clone). | ⬜ |
| 0.3 | Build fix: `pip install egl_probe --no-build-isolation` | ✅ Aug 30 (in `env/vla`) |
| 0.4 | `pip install hf-libero` | ⬜ **← stopped here** |
| 0.5 | Check: `env -u PYTHONPATH python -c "from libero.libero import benchmark; print(list(benchmark.get_benchmark_dict()))"` lists 5 suites, and `libero.__file__` is under `site-packages`. Put this check in every launcher (`run_openvla.sh`, `run_lerobot.sh`). | ⬜ |
| 0.6 | Smoke test: 2 episodes (command below). | ⬜ |
| 0.7 | Fix the plan file: 45% → 20% in §9a. | ✅ 2026-10-07 |

```bash
export MUJOCO_GL=osmesa        # EGL does not work on this cluster
lerobot-eval --output_dir=./eval_logs/smoke \
  --env.type=libero --env.task=libero_spatial --env.task_ids='[0]' \
  --eval.batch_size=1 --eval.n_episodes=2 --env.max_parallel_tasks=1 \
  --policy.path=lerobot/pi05_libero_finetuned --policy.n_action_steps=10
```

`--policy.n_action_steps=10` is required. The checkpoint's own config says 50, which would run the whole chunk without re-planning.

### Stage A.1: plain LIBERO-Spatial (pipeline check only)

| # | Step | Status |
|---|---|---|
| A.1.1 | Ask for the full 24 GB GPU, not a fraction. Load `lerobot/pi05_libero_finetuned`. | ⬜ |
| A.1.2 | Run LIBERO-Spatial, 500 episodes. Expect about 97–99% (LeRobot got 97.0, Physical Intelligence 98.8). | ⬜ |
| A.1.3 | If lower: check `n_action_steps=10`, then the norm-stats loading problem (lerobot issue #2533), then that no OpenVLA preprocessing slipped in. | ⬜ |
| A.1.4 | Write our own loop on `OffScreenRenderEnv` + `PI05Policy`. We need it for N samples now and for camera control later. Check it matches `lerobot-eval`. | ⬜ |
| A.1.5 | ⚠️ Decide the plan length for pose integration. The model predicts 50 steps per chunk but executes 10. The plan assumed 10. Default until decided: compute SE for both 10 and 50 from the same samples (no extra model passes), then choose after A.2. | ⬜ |
| A.1.6 | At every step: sample N plans, pose-integrate, group, compute SE. | ⬜ |
| A.1.7 | Sanity check: SE ≤ ln N, not all zero, not all equal, no crashes. | ⬜ |
| A.1.8 | Time N = 50 batched samples. Is 500 PRO episodes at every step affordable? | ⬜ |
| — | **Do not compute AUROC here.** Only about 6–16 failures per 500. | — |

### Stage A.2: LIBERO-PRO Spatial-Pos (the real test)

Why Spatial-Pos: it is the only Spatial change that gives a useful mix of success and failure *and* can be fixed by looking again.

| π0.5 on LIBERO-PRO Spatial (arXiv 2510.03827) | Success | Useful for us? |
|---|---|---|
| No change | 0.98 | Too few failures |
| Object changed | 0.97 | Too few failures |
| Instruction reworded (Semantic) | 0.97 | Too few failures |
| Task changed | 0.01 | Too few successes |
| Environment changed | 0.46 | Cannot be fixed by moving a camera; has an object-drift bug |
| **Position changed (Pos)** | **0.20** | **Yes** |

| # | Step | Status |
|---|---|---|
| A.2.0 | ⚠️ Choose the Pos setting. The task catalog maps Pos to `use_swap` (move an object into another object's place). The Aug 27 chat described shift sizes x0.1 to x0.5. Check which one gave 0.20. | ⬜ |
| A.2.1 | Same model and code as A.1, on PRO Spatial-Pos scenes. | ⬜ |
| A.2.2 | Loading check: success should be near **20%**. | ⬜ |
| A.2.3 | Compute SE at every step for 500 episodes (about 400 failures expected). | ⬜ |
| A.2.4 | AUROC: does episode-level SE (max or mean) separate success from failure? Report pooled **and per task**. | ⬜ |
| A.2.5 | Compare SE with our own raw-variance baseline on the same episodes, then with arXiv 2606.20754 Table I (sampling 0.943, perturbation 0.961, 43.8% success). Their checkpoint is not stated, so it is a reference, not a like-for-like test. | ⬜ |

### Stage A.3: is the result stable? (only if A.2 shows a signal)

| # | Step | Status |
|---|---|---|
| A.3.1 | Sweep ε over 2 orders of magnitude. There must be a plateau. If not, switch to KLE (a method with no threshold). | ⬜ |
| A.3.2 | Sweep N: 10, 20, 50, 100. Plot Ĉ next to SE (Ĉ grows with N and can fake a trend). | ⬜ |
| A.3.3 | Decide: more PRO suites, or go to Stage B. | ⬜ |

### Stage B: a live warning (still a fixed camera)

| # | Step | Status |
|---|---|---|
| B.1 | Make a per-step label: was SE high in the steps before a failure, compared with the same steps in matched successes? | ⬜ |
| B.2 | Smooth with Layer 2: S_t from section 5.2. | ⬜ |
| B.3 | Choose the threshold method before looking at test data (Youden's index). | ⬜ |
| B.4 | Split the episodes: fit the threshold on one part, test on the other. | ⬜ |
| B.5 | Report 3 numbers: warning time before failure, false alarms on successes, and the Stage A AUROC (the ceiling). | ⬜ |
| B.6 | Stop. No camera moves in Stage B. | — |

### Stage C: moving the camera (open problem)

| # | Step | Status |
|---|---|---|
| C.1 | Build and check a camera setter: set the camera pose after every reset, render 3 frames (az0 and ±10°), compare them. | ⬜ |
| C.2 | **Paired-occlusion test (the gate):** camera fixed at the training pose; two start states per task, target hidden vs visible. Pass: ΔH > 0 in ≥70% of ≥20 pairs per task, Wilcoxon p < 0.05. | ⬜ |
| C.3 | Matched-viewpoint test: record SE from a few fixed alternate camera poses (offline, not live). All are equally new to the model, so differences show real information. | ⬜ |
| C.4 | Look up LIBERO-plus camera-pose results (LeRobot `--env.type=libero_plus`) to see how much new views hurt π0.5. Needs its own env. | ⬜ |
| C.5 | Only if needed: fine-tune with random camera poses. Risk: it may erase the signal. | ⬜ |

### Stage D: the closed loop (end goal)

| # | Step | Status |
|---|---|---|
| D.1 | Trigger fires → move the camera → re-measure SE → act if it dropped, otherwise stop or ask. | ⬜ |
| D.2 | Compare success with trigger on vs off, same tasks and seeds. | ⬜ |
| D.3 | Report Reduction-on-Trigger: SE before minus after each camera move. | ⬜ |
| D.4 | Move SE readings to chunk boundaries (5× cheaper) for deployment speed. | ⬜ |

### Side tasks (cheap, can run any time)

| # | Task | Why | Status |
|---|---|---|---|
| S.1 | Write the "confidently wrong" paragraph (SE = 0 case). | A reviewer will ask. | ⬜ |
| S.2 | Same-length AUROC on the OpenVLA data (for example the first 100 steps only). | Removes the episode-length effect. | ⬜ |
| S.3 | Metric-blindness check on the saved OpenVLA `.npz`: bin-index variance vs entropy. | Cheap; uses existing data. | ⬜ |
| S.4 | Explain why runs A and B disagree on 7 tasks. | Any OpenVLA number in the thesis depends on it. | ⬜ |
| S.5 | **π0.5 visual-nuisance ablation.** Run `thesis-docs/ablation_pi05/run_ablation.sh` (nominal + 6 conditions × 10 episodes × tasks 4 and 7) after A.1 passes. Check its nominal success against `lerobot-eval` first. Details in §8.1. | Measures h (scene noise): does SE rise under brightness, texture or small camera shifts? Supports H4. | ⬜ Code written 2026-10-07, never run |
| S.6 | Send Shapira the three-framings memo: (A) the trigger improves success, (B) abstain or ask for help, (C) describe what entropy measures. | Agree on the fallback early. | Status unknown |

### 8.1 The π0.5 ablation (side task S.5)

The old OpenVLA ablation was ported to π0.5 on 2026-10-07. The code is in `thesis-docs/ablation_pi05/`. It has never run.

| File | What changed from the OpenVLA version |
|---|---|
| `model_server_ablation.py` | Loads `lerobot/pi05_libero_finetuned` with LeRobot. Each step it draws N = 10 action plans, groups them by where the hand ends up, and returns the semantic entropy for H = 10 and H = 50. It executes 10 actions per plan, like `n_action_steps=10`. |
| `run_perturbation_ablation.py` | Sends both cameras and the robot state, not one image. Images go raw: LeRobot rotates them itself. No OpenVLA prompt or gripper fix. Adds a **nominal** condition. Step cap 280 (LeRobot's). Saves every sampled plan so ε, λ and H can be changed later without rerunning. |
| `run_ablation.sh` | Same "refuse if already running" guard. Adds `USE_TF=0` and `env -u PYTHONPATH`. Run it in the LeRobot + `hf-libero` env. |
| `perturbations.py` | Unchanged. Use the original. |

```
for each task (4, 7) × condition (nominal + 6) × episode (10):
    reset → settle 10 steps → (camera condition: move agentview camera once)
    every step: perturb both images → server samples 10 plans → SE(H=10), SE(H=50)
                → execute next action → stop on success or step 280
```

Before trusting the numbers:

| Check | Pass when |
|---|---|
| Nominal success vs `lerobot-eval` on the same tasks | Close. If much lower, the image or state handling is wrong. |
| Units of the action scale (`POS_SCALE` 0.05 m, `ROT_SCALE` 0.5 rad) | Match the LIBERO controller config on the cluster |
| SE on nominal | Not always 0, never above ln 10 + 0.45 |
| Speed of one step with N = 10 | 140 episodes fit the GPU budget |

The old analysis script (`step_4c_analyze_ablation.py`) reads per-joint entropy and will not work on these files. It needs a rewrite for one SE number per step.


---

## 9. Tests to run (checklist)

| Test | Data | Pass when | Stage |
|---|---|---|---|
| T0 Smoke test | 2 episodes, LIBERO-Spatial task 0 | Runs end to end without errors | 0 |
| T1 Reproduce π0.5 | 500 episodes, LIBERO-Spatial | About 97–99% success | A.1 |
| T2 Own loop matches | Same episodes, own loop vs `lerobot-eval` | Same success rate | A.1 |
| T3 SE sanity | A.1 episodes | SE ≤ ln N, not constant, no crashes | A.1 |
| T4 Speed | N = 50, batched | 500 dense episodes fit the GPU budget | A.1 |
| T5 PRO loads correctly | 500 episodes, PRO Spatial-Pos | Success near 20% | A.2 |
| T6 **Main result** | Same 500 episodes | AUROC clearly above 0.5, reported per task; compared with our raw-variance baseline, then 0.943 and 0.961 | A.2 |
| T7 ε plateau | ε over 2 orders of magnitude | A flat region exists | A.3 |
| T8 N stability | N = 10, 20, 50, 100 | SE settles; Ĉ tracked | A.3 |
| T9 Live warning | Held-out split | Warning time, false-alarm rate, AUROC ceiling | B |
| T10 Gate | ≥20 occluded/visible pairs per task | ΔH > 0 in ≥70%, Wilcoxon p < 0.05 | C |
| T11 Closed loop | Trigger on vs off | Higher success, Reduction-on-Trigger > 0 | D |

---

## 10. Risks and what to do about them

| Risk | Why it matters | What to do |
|---|---|---|
| Confidently wrong (SE = 0) | Misses failures where all plans agree on a wrong move. | Say it plainly: SE detects ambiguity, not wrongness. Compare with the perturbation method, which catches different cases. |
| Camera wall | Any new view raises entropy, useful or not. | Paired-occlusion gate first, then matched-viewpoint test, LIBERO-plus numbers, and fine-tuning only if needed. |
| No ε plateau | The result would depend on a hand-picked number. | Mandatory sweep. Fallback: KLE. |
| Too slow | 1,375 passes per episode × 500 episodes. | Time it early (T4). Use chunk-boundary readings later. |
| Task identity drives AUROC | Spatial-Pos tasks range from about 0 to 98% success. | Report per-task AUROC. |
| Episode length | Failed episodes last longer. | Use same-length windows. |
| Two different "Pos" settings | The wrong one makes numbers incomparable. | Decide in A.2.0. |
| Package mess | A wrong LIBERO silently gives wrong numbers. | Separate envs; check `libero.__file__` in every launcher. |
| Prior work overlap | Tang (smoothing), Lee & Har (perturbation), VLAConf (objection). | The novelty is outcome grouping as an aleatoric filter plus the camera trigger. |
| Time | About 11 weeks for 4 stages. | Finish A and B first. Agree on a fallback with Shapira. |

---

## 11. Time check

| Item | Weeks |
|---|---|
| Today (2026-10-07) to mid-Feb 2027 | about 19 |
| Writing (the Aug 18 plan: do not compress) | 8 |
| **Left for experiments** | **about 11** |

[Likely] Stages A and B alone support framing B: "SE tells the robot when to stop or ask for help". The Aug 18 chat called that the floor, not the plan. Stages C and D fit only if A and B go smoothly.

---

## 12. Key numbers

| What | Number | Source |
|---|---|---|
| OpenVLA LIBERO-Spatial success (ours) | 84.8% (424/500) | Our run, Aug 10 |
| OpenVLA published | 84.7% | OpenVLA paper |
| OpenVLA entropy AUROC (ours) | 0.801 | Our run, Aug 11 |
| Tang full method AUROC, Spatial | 0.936 | arXiv 2603.18342, Table A1 |
| Tang naive global-mean baseline, Spatial | 0.845 | arXiv 2603.18342 |
| π0.5 on LIBERO-Spatial | 98.8% (Physical Intelligence), 97.0% (LeRobot) | π0.5 paper; LeRobot docs |
| π0.5 on PRO Spatial-Pos | **0.20** | arXiv 2510.03827, Table 3 |
| π0.5 on PRO Spatial-Pos (another paper) | 43.8% (45.0% in the Aug 27 copy) | arXiv 2606.20754, Table I, current HTML |
| Failure-detection AUC on Spatial-Pos: sampling / perturbation | 0.943 / 0.961 (0.935 / 0.963 in the Aug 27 copy) | arXiv 2606.20754, Table I, current HTML |
| SE cost | 11 passes per reading, 1,375 per episode (275 at chunk boundaries) | Scenario B |

---

## 13. Files in this project

| File | What it is |
|---|---|
| `thesis-docs/README.md` | This file |
| `thesis-docs/CLAUDE.md` | Rules and context for Claude agents |
| `thesis-import/thesis-brief.md` | Detailed brief distilled from all old chats |
| `thesis-import/decision-log.md` | Every decision from Mar 26 to Aug 30, with dates |
| `thesis-import/source/semantic_entropy_for_vla.md` | **The method spec** (Aug 27). Sections 9a and 9b are the stage plan. |
| `thesis-import/source/Our Semantic Entropy Plan by Claude.md` | Old version (Aug 19). Superseded. |
| `thesis-import/source/scenario_B_chunk_sampling.html` | Picture of the SE loop and its cost |
| `thesis-import/source/scenario_A_full_rollout.html` | The rejected expensive version |
| `thesis-import/source/libero_pro_task_catalog.md` | All LIBERO-PRO changes per task |
| `thesis-docs/ablation_pi05/` | π0.5 version of the ablation scripts (§8.1). Not run yet. |

---

## 14. Words used here

| Word | Meaning |
|---|---|
| VLA | Vision-Language-Action model: sees images, reads an instruction, outputs robot actions. |
| OpenVLA | A 7B VLA that outputs actions as tokens (256 bins per number). |
| π0, π0.5 | VLAs from Physical Intelligence that make actions with flow matching. |
| Flow matching | Start from random noise and refine it in about 10 steps into a plan. Different noise gives a different plan. |
| Action chunk / plan | A short list of future actions. Each action is 7 numbers: x, y, z, roll, pitch, yaw, gripper. |
| LIBERO | A simulated robot benchmark (MuJoCo) with 4 suites of 10 tasks each. |
| LIBERO-PRO | LIBERO with changes (moved objects, reworded instructions, ...) to test whether models only memorised. |
| Spatial-Pos | LIBERO-Spatial tasks with object positions changed. |
| Entropy | A number for how spread out the choices are. 0 means sure. |
| Semantic entropy (SE) | Entropy over groups of outputs that mean the same thing. |
| ε (epsilon) | The distance below which two plans count as the same. |
| N, K, Ĉ | N = plans sampled per reading. K = readings per episode. Ĉ = groups found. |
| AUROC | The chance that a random failed episode scores higher than a random successful one. 0.5 = coin flip, 1.0 = perfect. |
| Aleatoric uncertainty | Uncertainty that cannot be removed: many valid ways to do the task. |
| Epistemic uncertainty | Uncertainty from missing information: looking again can fix it. |
| OOD | Out of distribution: something the model never saw in training. |
| ATR | Average true range: how jumpy recent actions are. Used as a gate. |
| Reduction-on-Trigger | How much SE drops after the camera moves. |
