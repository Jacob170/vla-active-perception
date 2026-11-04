#!/usr/bin/env python3
"""Step 1 - the per-step scale curve and the success norm. Needs: s0 PASS.

What it does
  1. One integration pass over all episodes. Writes a cache that s2 and s3 read, so they never re-integrate:
       s1_cache.npz    gaps (R, 50, 45) float32  pose gap per reading, per step s = 1..50, per plan pair (no gripper)
                       G (R, 50, 10) float32     raw gripper value per step and plan
                       gref (R,) int8            gripper command in effect at the reading
                       rawvar10, rawvar50 (R,)   spread of the N end positions (the raw-variance comparator)
                       ep (R,), t (R,)           which episode and step each reading row belongs to
       s1_episodes.json  one row per episode: file, task, ep, success, steps, first row in the cache
  2. The success norm (Jacob, 2026-11-04): for each step s, the median pose gap over every reading and every pair
     of the CALIBRATION episodes only (all-success tasks 0, 2, 5, 8). No test-task episode touches it.
       s1_normalizer.json  m[s] for s = 1..50, plus the 10/25/75/90th percentiles per s
  3. Descriptive, calibration episodes only (no labels of the test tasks are looked at):
       - does the typical gap grow with s, and how fast (m_s / m_1)
       - the share of pairs already below eps 0.01 at step 1 (why a minimum over steps cannot separate anything)
       - argmax histograms: at which step s does a pair disagree most. Raw gaps: is the max just the endpoint
         (argmax at s = H)? Normalized gaps: are there real early disagreements (Jacob's "anomaly mid-window")?
     The argmax is no longer a gate: the gripper is out of the per-step distance, so its pace artifact cannot
     drive the max any more, and every variant stays in the s4 panel regardless.

Usage: python s1_scale.py RUN_DIR --out OUT_DIR
"""
import argparse, json, os, sys
import numpy as np
import se_lib as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s0 = os.path.join(a.out, "s0_reproduce.json")
    if not os.path.exists(s0) or not json.load(open(s0)).get("PASS"):
        sys.exit("[s1] step 0 has not passed - run s0_reproduce.py first")
    files = L.episode_files(a.run_dir)

    meta, gaps, Gs, gref, rv10, rv50, epi, tt = [], [], [], [], [], [], [], []
    row = 0
    for i, f in enumerate(files):
        e = L.load(f)
        meta.append({"file": e["file"], "task": e["task"], "ep": e["ep"], "success": e["success"],
                     "steps": e["steps"], "row0": row})
        for t in range(e["steps"]):
            P, R, G = L.step_poses(e["plans"][t], e["eef_pos"][t], e["eef_mat"][t], e["ps"], e["rs"])
            gaps.append(L.condensed(L.pose_gaps(P, R)).astype(np.float32))
            Gs.append(G.astype(np.float32))
            gref.append(L.gripper_ref(e["actions"], t))
            rv10.append(float(P[9].var(axis=0, ddof=1).sum()))
            rv50.append(float(P[49].var(axis=0, ddof=1).sum()))
            epi.append(i)
            tt.append(t)
        row += e["steps"]
        if (i + 1) % 10 == 0:
            print(f"[s1] integrated {i + 1}/{len(files)}", flush=True)
    gaps = np.stack(gaps)
    np.savez(os.path.join(a.out, "s1_cache.npz"), gaps=gaps, G=np.stack(Gs), gref=np.array(gref, np.int8),
             rawvar10=np.array(rv10, np.float32), rawvar50=np.array(rv50, np.float32),
             ep=np.array(epi, np.int32), t=np.array(tt, np.int32))
    with open(os.path.join(a.out, "s1_episodes.json"), "w") as fh:
        json.dump(meta, fh, indent=0)

    calib_eps = [i for i, m in enumerate(meta) if m["task"] in L.CALIB_TASKS and m["success"]]
    rows = np.isin(np.array(epi), calib_eps)
    m = L.normalizer(gaps, rows)
    G_cal = gaps[rows].astype(np.float64)                      # (Rc, 50, 45)
    q = {p: np.percentile(G_cal, p, axis=(0, 2)).tolist() for p in (10, 25, 75, 90)}
    L.write_json(os.path.join(a.out, "s1_normalizer.json"),
                 {"m": m.tolist(), "percentiles": q, "calib_tasks": L.CALIB_TASKS,
                  "n_calib_episodes": len(calib_eps), "n_calib_readings": int(rows.sum()),
                  "definition": "median pose gap (||dp|| + 0.1 * geodesic angle) per step s over all calibration readings and all 45 pairs"})

    # descriptive, calibration only
    desc = {"m_s": {int(s): float(m[s - 1]) for s in (1, 2, 3, 5, 10, 20, 30, 50)},
            "growth": {"m10_over_m1": float(m[9] / m[0]), "m50_over_m1": float(m[49] / m[0]),
                       "m50_over_m10": float(m[49] / m[9])},
            "share_pairs_below_eps0.01_at_step1": float((G_cal[:, 0, :] <= 0.01).mean()),
            "share_pairs_below_eps0.01_at_step10": float((G_cal[:, 9, :] <= 0.01).mean())}
    argmax = {}
    for H in L.HORIZONS:
        raw = G_cal[:, :H, :].argmax(axis=1) + 1
        nrm = (G_cal[:, :H, :] / m[:H, None]).argmax(axis=1) + 1
        argmax[H] = {"raw_hist": np.bincount(raw.ravel(), minlength=H + 1)[1:].tolist(),
                     "norm_hist": np.bincount(nrm.ravel(), minlength=H + 1)[1:].tolist(),
                     "raw_share_at_last_step": float((raw == H).mean()),
                     "norm_share_at_last_step": float((nrm == H).mean()),
                     "norm_share_in_first_third": float((nrm <= max(1, H // 3)).mean())}
    desc["argmax"] = argmax
    L.write_json(os.path.join(a.out, "s1_scale.json"), desc)

    print("\n=== STEP 1: scale curve and success norm (calibration tasks only) ===")
    print(f"calibration: tasks {L.CALIB_TASKS}, {len(calib_eps)} episodes, {int(rows.sum())} readings, 45 pairs each")
    print("typical (median) pose gap by step s: " + ", ".join(f"s{s} {v:.4f}" for s, v in desc["m_s"].items()))
    g = desc["growth"]
    print(f"growth: m10/m1 = {g['m10_over_m1']:.1f}, m50/m1 = {g['m50_over_m1']:.1f}  "
          f"(about 1 would mean a flat curve: then a step-dependent tolerance is unnecessary)")
    print(f"pairs already within eps 0.01: at step 1 {100 * desc['share_pairs_below_eps0.01_at_step1']:.1f}%, "
          f"at step 10 {100 * desc['share_pairs_below_eps0.01_at_step10']:.1f}%")
    for H, d in argmax.items():
        print(f"argmax H{H}: raw gaps peak at the last step in {100 * d['raw_share_at_last_step']:.1f}% of pairs; "
              f"normalized gaps: last step {100 * d['norm_share_at_last_step']:.1f}%, first third {100 * d['norm_share_in_first_third']:.1f}%")
        if H == 10:
            print(f"   raw  histogram s=1..10: {d['raw_hist']}")
            print(f"   norm histogram s=1..10: {d['norm_hist']}")
    print(f"wrote s1_cache.npz ({gaps.nbytes / 1e6:.0f} MB of gaps), s1_episodes.json, s1_normalizer.json, s1_scale.json")
    print("STEP 1 DONE")


if __name__ == "__main__":
    main()
