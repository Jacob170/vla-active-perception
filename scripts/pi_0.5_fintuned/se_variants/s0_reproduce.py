#!/usr/bin/env python3
"""Step 0 - reproduce the stored SE from the raw plans. Gate: if this fails, nothing downstream is trusted.

Checks
  a. The new code (se_lib) recomputes the run's own SE (endpoint pose + old gripper term, at the eps stored in the
     npz) and agrees with the stored se_h10 / se_h50. Plans are stored in float16, so a small share of steps flips
     a cluster (a2_analysis.py saw 98.9% identical at H10 and 99.9% at H50 on 10 episodes).
  b. The task split is the pre-registered one: calibration = all-success tasks (0, 2, 5, 8), test = mixed (1, 4).
  c. Descriptive: how often any plan switches its gripper inside H10 / H50 (the new gripper term only acts there).

PASS rule (fixed before running): for each H, identical steps >= 98% AND every episode's mean SE within 0.02 of
the stored mean AND the split matches. Exit code 1 on FAIL so run_all.sh stops.

Usage: python s0_reproduce.py RUN_DIR --out OUT_DIR [--allow_other_split]
"""
import argparse, os, sys
import numpy as np
import se_lib as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow_other_split", action="store_true", help="only for a different run, never for A.2")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = L.episode_files(a.run_dir)
    print(f"[s0] {len(files)} episodes in {a.run_dir}", flush=True)

    ver = {H: {"steps": 0, "identical": 0, "max_abs_diff": 0.0, "worst_episode_mean_diff": 0.0} for H in L.HORIZONS}
    switch = {H: [0, 0] for H in L.HORIZONS}
    meta = []
    for i, f in enumerate(files):
        e = L.load(f)
        meta.append({k: e[k] for k in ("file", "task", "ep", "success", "steps")})
        se = {H: np.zeros(e["steps"]) for H in L.HORIZONS}
        for t in range(e["steps"]):
            P, R, G = L.step_poses(e["plans"][t], e["eef_pos"][t], e["eef_mat"][t], e["ps"], e["rs"])
            gaps = L.pose_gaps(P, R)
            g_ref = L.gripper_ref(e["actions"], t)
            for H in L.HORIZONS:
                se[H][t], _ = L.variant_se("V1_endpoint_g0", L.condensed(gaps[H - 1]), G, g_ref, H, e["eps_stored"])
                switch[H][0] += int((L.switch_times(G, g_ref, H) > 0).any())
                switch[H][1] += 1
        for H in L.HORIZONS:
            if H not in e["se_stored"]:
                continue
            d = np.abs(se[H] - e["se_stored"][H])
            v = ver[H]
            v["steps"] += len(d)
            v["identical"] += int((d < 1e-9).sum())
            v["max_abs_diff"] = max(v["max_abs_diff"], float(d.max()))
            v["worst_episode_mean_diff"] = max(v["worst_episode_mean_diff"], abs(float(se[H].mean() - e["se_stored"][H].mean())))
        if (i + 1) % 10 == 0:
            print(f"[s0] {i + 1}/{len(files)}", flush=True)

    calib, test, per_task = L.split_from_meta(meta)
    ok = True
    for H, v in ver.items():
        v["share_identical"] = v["identical"] / max(1, v["steps"])
        v["pass"] = v["steps"] > 0 and v["share_identical"] >= 0.98 and v["worst_episode_mean_diff"] <= 0.02
        ok &= v["pass"]
    split_ok = calib == L.CALIB_TASKS and test == L.TEST_TASKS
    ok &= split_ok or a.allow_other_split
    res = {"n_episodes": len(files), "verify": ver, "per_task_success": per_task,
           "calib_tasks": calib, "test_tasks": test, "split_matches_preregistered": split_ok,
           "share_of_readings_with_any_gripper_switch": {H: s[0] / max(1, s[1]) for H, s in switch.items()},
           "PASS": bool(ok)}
    L.write_json(os.path.join(a.out, "s0_reproduce.json"), res)

    print("\n=== STEP 0: reproduce the stored SE ===")
    for H, v in ver.items():
        print(f"H{H}: {v['identical']}/{v['steps']} steps identical ({100 * v['share_identical']:.2f}%), "
              f"worst episode-mean diff {v['worst_episode_mean_diff']:.4f}, max step diff {v['max_abs_diff']:.4f} "
              f"-> {'pass' if v['pass'] else 'FAIL'}")
    print(f"split: calibration {calib}, test {test} (pre-registered {L.CALIB_TASKS} / {L.TEST_TASKS}) "
          f"-> {'pass' if split_ok else ('allowed by flag' if a.allow_other_split else 'FAIL')}")
    print(f"per-task successes: {per_task}")
    print("readings where at least one plan switches its gripper inside the window: "
          + ", ".join(f"H{H} {100 * r:.1f}%" for H, r in res["share_of_readings_with_any_gripper_switch"].items()))
    print(f"STEP 0 {'PASS' if ok else 'FAIL - stop here'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
