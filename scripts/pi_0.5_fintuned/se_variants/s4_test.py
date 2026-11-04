#!/usr/bin/env python3
"""Step 4 - the within-task test of every variant, with the rules fixed before running. Needs: s3.

Only (failure, success) pairs from the SAME task count: task 1 (2 ok / 8 fail) and task 4 (6 ok / 4 fail) give
16 + 24 = 40 pairs. The calibration tasks contribute no pairs and were the only data the norm and eps saw.

Per variant: within-task AUROC of the per-episode mean over the first K readings (failure = positive class),
exact permutation p (labels shuffled inside each task, 9450 arrangements), family-wise p (max-statistic over
the 6 declared variants of that horizon), bootstrap CI95, per-task AUROC.

PASS rule (pre-registered 2026-11-04): family p < 0.05 AND CI95 lower bound > 0.5 AND AUROC above the
raw-variance comparator at the same horizon. Primary: V4_normmax_g1 at H10. H50 is a secondary family.
Every variant is reported; none is dropped after the fact.

Even a PASS is a candidate, not a result: 40 pairs from 20 episodes. It earns a confirming run on new t1/t4
episodes with this exact script and these exact eps values.

Usage: python s4_test.py --out OUT_DIR
"""
import argparse, csv, json, os
import numpy as np
import se_lib as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args()
    E = []
    with open(os.path.join(a.out, "s3_features.csv")) as fh:
        for r in csv.DictReader(fh):
            e = {"task": int(r["task"]), "ep": int(r["ep"]), "success": bool(int(r["success"]))}
            e["x"] = {k[:-2]: float(v) for k, v in r.items() if k.endswith("_K") and v != ""}
            E.append(e)
    T = [e for e in E if e["task"] in L.TEST_TASKS]
    perm = L.WithinTaskPerm(T)
    print(f"[s4] test tasks {perm.mixed_tasks}: {perm.n_pairs} within-task pairs, "
          f"{'exact' if perm.exact else 'Monte Carlo'} null over {len(perm.Y)} arrangements", flush=True)

    res = {"n_pairs": perm.n_pairs, "test_tasks": perm.mixed_tasks, "families": {}, "loo_stability": {}}
    for H in L.HORIZONS:
        names = [f"{v}_H{H}" for v in L.VARIANTS]
        cols = {n: np.array([e["x"][n] for e in T]) for n in names}
        pv = {n: perm.perm_values(cols[n]) for n in names}
        dmax = np.max(np.stack([np.abs(pv[n] - 0.5) for n in names]), axis=0)
        crit = float(0.5 + np.percentile(dmax, 95))
        rv = np.array([e["x"][f"RAWVAR_H{H}"] for e in T])
        rv_auc = perm.observed(rv)
        fam = {"critical_auroc_for_family_p_0.05": crit,
               "critical_pairs_of_40": int(np.ceil(crit * perm.n_pairs)),
               "RAWVAR": {"auroc": rv_auc, "p_two_sided": perm.pvalue(np.abs(perm.perm_values(rv) - 0.5), abs(rv_auc - 0.5))}}
        for n in names:
            ao = perm.observed(cols[n])
            d = abs(ao - 0.5)
            lo, hi = L.bootstrap_ci(T, lambda e, n=n: e["x"][n], n_boot=a.n_boot)
            per_task = {}
            for t in perm.mixed_tasks:
                f = [e["x"][n] for e in T if e["task"] == t and not e["success"]]
                s = [e["x"][n] for e in T if e["task"] == t and e["success"]]
                per_task[t] = {"auroc": L.auroc(f, s), "n_fail": len(f), "n_succ": len(s)}
            p_fam = perm.pvalue(dmax, d)
            passed = bool(p_fam < 0.05 and lo is not None and lo > 0.5 and ao > rv_auc)
            fam[n] = {"auroc": ao, "p_two_sided": perm.pvalue(np.abs(pv[n] - 0.5), d), "p_family": p_fam,
                      "ci95": [lo, hi], "per_task": per_task, "above_rawvar": bool(ao > rv_auc), "PASS": passed}
        res["families"][f"H{H}"] = fam
        for v in L.NORM_VARIANTS:
            n = f"{v}_loo_H{H}"
            if all(n in e["x"] for e in T):
                res["loo_stability"][n] = {"auroc": perm.observed(np.array([e["x"][n] for e in T])),
                                           "auroc_with_calibration_norm": fam[f"{v}_H{H}"]["auroc"]}
    pv_name = f"{L.PRIMARY[0]}_H{L.PRIMARY[1]}"
    res["PRIMARY"] = {"variant": pv_name, **res["families"][f"H{L.PRIMARY[1]}"][pv_name]}
    L.write_json(os.path.join(a.out, "s4_results.json"), res)

    def f3(x): return "n/a" if x is None else f"{x:.3f}"
    print("\n=== STEP 4: within-task test, failure = positive, 0.5 = chance ===")
    print(f"pairs: {perm.n_pairs} from tasks {perm.mixed_tasks}")
    for H in L.HORIZONS:
        fam = res["families"][f"H{H}"]
        tag = "PRIMARY FAMILY" if H == L.PRIMARY[1] else "secondary family"
        print(f"\nH{H} ({tag}). To pass the family correction a variant needs AUROC >= {fam['critical_auroc_for_family_p_0.05']:.3f} "
              f"(about {fam['critical_pairs_of_40']}/{perm.n_pairs} pairs). Raw variance: {f3(fam['RAWVAR']['auroc'])}")
        print(f"  {'variant':18s} {'AUROC':>6s} {'CI95':>15s} {'p':>7s} {'p_fam':>7s}  {'t1':>5s} {'t4':>5s}  verdict")
        for v in L.VARIANTS:
            d = fam[f"{v}_H{H}"]
            pt = d["per_task"]
            print(f"  {v:18s} {f3(d['auroc']):>6s} [{f3(d['ci95'][0])}, {f3(d['ci95'][1])}] {d['p_two_sided']:7.4f} {d['p_family']:7.4f}  "
                  f"{f3(pt.get(1, {}).get('auroc')):>5s} {f3(pt.get(4, {}).get('auroc')):>5s}  {'PASS' if d['PASS'] else 'no'}"
                  + ("   <- primary" if (v, H) == L.PRIMARY else ""))
    print("\nleave-one-out norm (stability only, not tested):")
    for n, d in res["loo_stability"].items():
        print(f"  {n:28s} AUROC {f3(d['auroc'])}  (calibration norm: {f3(d['auroc_with_calibration_norm'])})")
    p = res["PRIMARY"]
    print(f"\nPRIMARY {pv_name}: AUROC {f3(p['auroc'])}, family p {p['p_family']:.4f} -> {'PASS (a candidate for a confirming run)' if p['PASS'] else 'no'}")


if __name__ == "__main__":
    main()
