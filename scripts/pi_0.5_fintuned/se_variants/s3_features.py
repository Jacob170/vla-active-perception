#!/usr/bin/env python3
"""Step 3 - the SE series of every variant for every episode, and the per-episode feature. Needs: s1, s2.

Variants (declared 2026-11-04, before any of them was computed on real data)
  V1_endpoint_g0   the current metric: pose after H steps + the run's gripper term, frozen eps (the baseline)
  V2_endpoint_g1   pose after H steps + the new gripper term (switch time); isolates the gripper change
  V3_rawmax_g1     largest raw pose gap over steps 1..H + new gripper term
  V4_normmax_g1    largest NORMALIZED pose gap over steps 1..H + new gripper term   <- PRIMARY at H10 (Jacob)
  V5_normlate_g1   mean normalized pose gap over the last third of the window + new gripper term
  V6_dispersion    no clustering, no eps: per step, the mean normalized gap over the 45 pairs; max over steps
  RAWVAR           spread of the N end positions (comparator, as in the clean test)
Normalized = divided by the success norm m[s] of step 1 (calibration tasks only).

Stability check (not part of the test family): V4-V6 again with a leave-one-out norm built from ALL successes
except the episode being scored (suffix _loo). The eps from step 2 is reused, so the operating point may drift a bit.

Feature per episode = mean over its first K readings, K = steps of the shortest successful episode (79 in A.2),
exactly the equal-length window of the clean test. Full-episode means are written too, for plots only: they are
length-confounded and are not evidence.

Outputs: s3_features.csv (one row per episode), s3_series.npz (per-reading series, for plots)
Usage: python s3_features.py --out OUT_DIR
"""
import argparse, csv, json, os, sys
import numpy as np
import se_lib as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s2 = json.load(open(os.path.join(a.out, "s2_eps.json")))
    if not s2.get("PASS"):
        sys.exit("[s3] step 2 has not passed")
    eps = {int(H): {v: d.get("eps") for v, d in s2["by_H"][H].items() if isinstance(d, dict)} for H in s2["by_H"]}
    z, meta = L.load_cache(a.out)
    gaps, G, gref, ep = z["gaps"], z["G"], z["gref"], z["ep"]
    rawvar = {10: z["rawvar10"], 50: z["rawvar50"]}
    m_cal = np.array(json.load(open(os.path.join(a.out, "s1_normalizer.json")))["m"])

    # leave-one-out norms over ALL successes (only needed for test-task episodes)
    succ = [i for i, mm in enumerate(meta) if mm["success"]]
    succ_rows = np.isin(ep, succ)
    m_all = L.normalizer(gaps, succ_rows)
    m_loo = {}
    for i, mm in enumerate(meta):
        if mm["task"] in L.TEST_TASKS:
            m_loo[i] = L.normalizer(gaps, succ_rows & (ep != i)) if mm["success"] else m_all

    K = min(mm["steps"] for mm in meta if mm["success"])
    print(f"[s3] {len(meta)} episodes, window K = {K}", flush=True)
    names = []
    for H in L.HORIZONS:
        names += [f"{v}_H{H}" for v in L.VARIANTS] + [f"RAWVAR_H{H}"]
        names += [f"{v}_loo_H{H}" for v in L.NORM_VARIANTS]
    series = {n: [] for n in names}
    rows_out = []
    for i, mm in enumerate(meta):
        r0, T = mm["row0"], mm["steps"]
        ser = {n: np.full(T, np.nan) for n in names}
        for t in range(T):
            r = r0 + t
            gc = gaps[r].astype(np.float64)
            Gr, gr = G[r].astype(np.float64), int(gref[r])
            for H in L.HORIZONS:
                parts = L.pose_parts(gc, H, m_cal)
                for v in L.CLUSTERED:
                    ser[f"{v}_H{H}"][t] = L.variant_se(v, parts[v], Gr, gr, H, eps[H][v])[0]
                ser[f"V6_dispersion_H{H}"][t] = L.dispersion(gc, H, m_cal)
                ser[f"RAWVAR_H{H}"][t] = rawvar[H][r]
                if i in m_loo:
                    pl = L.pose_parts(gc, H, m_loo[i])
                    for v in ("V4_normmax_g1", "V5_normlate_g1"):
                        ser[f"{v}_loo_H{H}"][t] = L.variant_se(v, pl[v], Gr, gr, H, eps[H][v])[0]
                    ser[f"V6_dispersion_loo_H{H}"][t] = L.dispersion(gc, H, m_loo[i])
        row = {"task": mm["task"], "ep": mm["ep"], "success": int(mm["success"]), "steps": T}
        for n in names:
            x = ser[n]
            row[f"{n}_K"] = float(np.mean(x[:K])) if not np.isnan(x[0]) else ""
            row[f"{n}_full"] = float(np.mean(x)) if not np.isnan(x[0]) else ""
            series[n].append(x.astype(np.float32))
        rows_out.append(row)
        if (i + 1) % 10 == 0:
            print(f"[s3] {i + 1}/{len(meta)}", flush=True)

    cols = ["task", "ep", "success", "steps"] + [f"{n}_{w}" for n in names for w in ("K", "full")]
    with open(os.path.join(a.out, "s3_features.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in sorted(rows_out, key=lambda r: (r["task"], r["ep"])):
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
    np.savez_compressed(os.path.join(a.out, "s3_series.npz"), K=K,
                        **{n: np.concatenate(series[n]) for n in names},
                        steps=np.array([mm["steps"] for mm in meta]), task=np.array([mm["task"] for mm in meta]),
                        success=np.array([mm["success"] for mm in meta]))
    L.write_json(os.path.join(a.out, "s3_meta.json"), {"K": K, "eps": eps, "variants": L.VARIANTS, "PASS": True})
    print(f"\n=== STEP 3: features written (window K = {K}) ===\nSTEP 3 DONE")


if __name__ == "__main__":
    main()
