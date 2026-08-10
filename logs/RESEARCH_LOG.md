# Research Log — VLA Active Perception (MSc, BGU)

One entry per finding, written the same day as the result. Format: date, finding,
raw numbers, provenance (script + data file), caveats. Thesis chapters are written
FROM this log; nothing goes into the manuscript that isn't logged here first.

---

## Entry 001 — 2026-07-17 — Overconfidence under degraded input (blur sanity test)

**Setup:** LIBERO-Spatial task 0, seed 0, frame 0; `openvla-7b-finetuned-libero-spatial`;
greedy decoding; Gaussian blur σ=10, 31×31 kernel. Entropy slice verified same day
(gen==full==slice over ~1,400 rows incl. boundary bins 31744/31999; V=32000).

**Numbers:**
- sharp:   H_sum = 5.63, per-DoF = [1.40 0.76 0.98 0.94 1.34 0.20 0.00]
- blurred: H_sum = 7.00, per-DoF = [2.11 1.70 1.80 1.39 0.00 0.00 0.00]

**Finding:** Aggregate entropy rose under blur, concentrated in translation DoFs — but
DoF4 (rotation) collapsed 1.34 → 0.00 nats: the model became *maximally confident*
on one axis precisely when visual information was removed. Consistent with the thesis
premise (confidence ≠ perceptual adequacy); motivates per-DoF treatment and
max-pooling in the composite signal.

**Provenance:** `scripts/blur_test.py`, `model_server.py`.
**Caveats:** n=1 frame/task/seed — motivating anecdote and design justification only.
Draft thesis paragraph exists (see 05_composite_signal section / former Drive log).

---

## Entry 002 — 2026-07-20 — Entropy is non-monotone in degradation severity (blur sweep)

**Setup:** Same frame as Entry 001; σ ∈ {0, 3, 10, 20}, 31×31 kernel.

**Numbers (H_sum):** 5.63 / 2.12 / 7.00 / 6.74
- σ=3 per-DoF = [1.04 0.14 0.28 0.00 0.09 0.56 0.00] — three DoFs ≈ 0

**Finding:** Mild blur (σ=3) induced a *global* confidence collapse (−62% H_sum) before
heavy blur raised entropy. Promotes Entry 001 from single-DoF anecdote to whole-policy
phenomenon: uncertainty does not track degradation monotonically.

**Provenance:** `scripts/blur_sweep.py` → `~/runs/blur_sweep/blur_sweep_task0_seed0.npz`.
**Caveats:** n=1 frame. Before figure use: repeat over ~5 tasks/frames.

---

## Entry 003 — 2026-07-20 — First SUCCESS after preprocessing alignment

**Setup:** Task 0, seed 0, 220-step cap. Fixes applied as a package: full 180° image
rotation ([::-1, ::-1]), 10 settle steps ([0,0,0,0,0,0,-1]) post-reset, 0.9-area center
crop → 224×224 (official geometry), gripper normalize→binarize→invert (verbatim
official `robot_utils.py`).

**Numbers:** SUCCESS in 77 steps. H_sum mean 4.85 / min 0.67 / max 9.05
(pre-fix FAIL baseline: 5.78 / 1.36 / 10.66 over 200 steps).

**Finding:** Pipeline validated end-to-end. Entropy *decreased* once input matched the
training distribution — third directional confirmation that H tracks input quality.
No per-fix attribution possible (four changes, one run).

**Provenance:** `scripts/rollout_one.py`, `model_server.py` →
`~/runs/first_episode/ep_0000_fixed.npz`, video `first_episode_fixed.mp4`.

---

## Entry 004 — [date] — LIBERO-Spatial baseline established
   Setup: official protocol, 10 tasks × 50 episodes = 500 rollouts, openvla-7b-finetuned-libero-spatial, split-process harness (eval_split.py + model_server.py), agentview camera, greedy decoding.
   Result: 424/500 = 84.8% (published OpenVLA: 84.7% — match within 0.1%).
   Significance: split-pipeline harness validated against literature; baseline is the קו-האפס for all active-perception comparisons. Data: ~/runs/baseline/libero_spatial_baseline.txt.


### per-taks breakdown:

``` awk '/^task /{split($0,a," "); t[a[2]]++; if(/True/)s[a[2]]++} END{for(k=0;k<10;k++)printf "task %d: %d/%d\n", k, s[k], t[k]}' ~/runs/baseline/libero_spatial_baseline.txt ```

Output:
' ~/runs/baseline/libero_spatial_baseline.txt
task 0: 50/50
task 1: 46/50
task 2: 46/50
task 3: 44/50
task 4: 36/50
task 5: 44/50
task 6: 45/50
task 7: 36/50
task 8: 39/50
task 9: 38/50

---

