#!/usr/bin/env python3
"""Step 2 - pick eps for each new variant by matching the operating point. Needs: s1 outputs.

Why: a variant with a smaller effective eps splits more and its SE rises everywhere; comparing it with the
current metric would then compare resolutions, not information. So every new variant gets the eps at which it
produces the SAME MEAN NUMBER OF CLUSTERS as the current metric (V1 at the frozen eps 0.01 / 0.03) on the
calibration episodes. Mean, not median: the cluster count is an integer, so its median is flat over a wide eps
range and cannot pin eps down. No labels are used: only calibration readings, only cluster counts.

The switch-time weight is not a separate knob: LAMBDA_T = 0.4 * eps, fixed in advance, so a 1-2 step difference
in gripper timing merges and 3 or more splits, at whatever eps is chosen.

Outputs
  s2_eps.json     chosen eps per variant and horizon, the cluster count achieved, the target
  s2_curves.json  mean cluster count against eps for every variant (to look for a plateau; informational only)
Exit code 1 if a variant cannot reach the target inside the searched range.

Usage: python s2_anchor.py --out OUT_DIR [--max_readings 0]
"""
import argparse, json, os, sys
import numpy as np
import se_lib as L

GRID = {"V2_endpoint_g1": (1e-4, 1.0), "V3_rawmax_g1": (1e-4, 1.0),
        "V4_normmax_g1": (1e-2, 100.0), "V5_normlate_g1": (1e-2, 100.0)}


def mean_C(variant, items, H, eps):
    return float(np.mean([L.variant_se(variant, pc, G, gr, H, eps)[1] for pc, G, gr in items]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_readings", type=int, default=0, help="0 = every calibration reading")
    ap.add_argument("--n_grid", type=int, default=30)
    a = ap.parse_args()
    z, meta = L.load_cache(a.out)
    m = np.array(json.load(open(os.path.join(a.out, "s1_normalizer.json")))["m"])
    gaps, G, gref, ep = z["gaps"], z["G"], z["gref"], z["ep"]
    calib_eps = [i for i, mm in enumerate(meta) if mm["task"] in L.CALIB_TASKS and mm["success"]]
    rows = np.flatnonzero(np.isin(ep, calib_eps))
    if a.max_readings and len(rows) > a.max_readings:
        rows = rows[np.linspace(0, len(rows) - 1, a.max_readings).astype(int)]
    print(f"[s2] {len(rows)} calibration readings", flush=True)

    out, curves, ok = {}, {}, True
    for H in L.HORIZONS:
        parts = [L.pose_parts(gaps[r].astype(np.float64), H, m) for r in rows]
        items = lambda v: [(p[v], G[r].astype(np.float64), int(gref[r])) for p, r in zip(parts, rows)]
        target = mean_C("V1_endpoint_g0", items("V1_endpoint_g0"), H, L.FROZEN_EPS[H])
        out[H] = {"target_mean_clusters": target,
                  "V1_endpoint_g0": {"eps": L.FROZEN_EPS[H], "mean_clusters": target, "note": "frozen, the anchor"}}
        curves[H] = {}
        print(f"[s2] H{H}: target mean cluster count {target:.3f} (V1 at eps {L.FROZEN_EPS[H]})", flush=True)
        for v, (lo, hi) in GRID.items():
            it = items(v)
            grid = np.geomspace(lo, hi, a.n_grid)
            cs = [mean_C(v, it, H, e) for e in grid]
            curves[H][v] = {"eps": grid.tolist(), "mean_clusters": cs}
            # cluster count falls as eps grows; find the bracket around the target and bisect in log space
            br = [k for k in range(len(grid) - 1) if (cs[k] - target) * (cs[k + 1] - target) <= 0]
            if not br:
                ok = False
                out[H][v] = {"eps": None, "mean_clusters": None, "error": f"target {target:.3f} outside [{min(cs):.3f}, {max(cs):.3f}]"}
                print(f"[s2]   {v}: FAIL, target not reachable in eps [{lo}, {hi}]", flush=True)
                continue
            elo, ehi = grid[br[0]], grid[br[0] + 1]
            for _ in range(25):
                mid = float(np.sqrt(elo * ehi))
                if mean_C(v, it, H, mid) > target:
                    elo = mid
                else:
                    ehi = mid
            cands = [elo, ehi]
            vals = [mean_C(v, it, H, e) for e in cands]
            k = int(np.argmin([abs(x - target) for x in vals]))
            out[H][v] = {"eps": cands[k], "lambda_t": L.LAMBDA_T_RATIO * cands[k], "mean_clusters": vals[k]}
            print(f"[s2]   {v}: eps {cands[k]:.5g} -> mean clusters {vals[k]:.3f}", flush=True)
        out[H]["V6_dispersion"] = {"eps": None, "note": "no clustering"}

    L.write_json(os.path.join(a.out, "s2_eps.json"), {"by_H": out, "PASS": ok,
                 "rule": "eps matched so the mean cluster count on calibration readings equals V1 at the frozen eps"})
    L.write_json(os.path.join(a.out, "s2_curves.json"), curves)
    print("\n=== STEP 2: eps anchored to the current operating point ===")
    for H in L.HORIZONS:
        print(f"H{H} (target {out[H]['target_mean_clusters']:.3f} clusters): "
              + ", ".join(f"{v} eps={out[H][v]['eps']:.4g}" for v in GRID if out[H][v].get("eps") is not None))
    print(f"STEP 2 {'PASS' if ok else 'FAIL - a variant could not be anchored'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
