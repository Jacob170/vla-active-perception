#!/usr/bin/env python3
"""Pre-registered comparison classic (in-distribution) vs PRO-swap, successes only. See PLAN_triple_he.md.

Question: does SE rise because of the perturbation itself, even when the robot succeeds?
Leg A (primary, tasks 0,2,5,8): classic successes vs PRO successes, within task.
Leg B (descriptive, tasks 3,6,7,9): classic successes vs PRO failures.
t1, t4 (descriptive, NOT independent of the earlier 40-pair test): three cells.

Everything is recomputed from the stored plans with se_lib (V1 = endpoint pose + old gripper, the run's metric)
at the frozen eps (H10 0.01, H50 0.03). Stored SE is used only to verify the code (gate 0b).
Usage: python se_triple.py --classic RUN_DIR --pro RUN_DIR --out OUT_DIR
"""
import argparse, csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import se_lib as L

LEG_A = (0, 2, 5, 8)
LEG_A_SENS = (0, 2, 8)
LEG_B = (3, 6, 7, 9)
THREE_CELL = (1, 4)
K_MIN = 30
N_PERM, N_BOOT, SEED = 20_000, 2000, 0
GATE_MATCH = 0.98
SD_BOUND = 0.5                       # "bounded" if CI95 of pooled d excludes +0.5 pooled within-group SD


# ---- recompute ----------------------------------------------------------------------------------------
def recompute(run_dir, label):
    """One dict per episode with SE series at the frozen eps (and at the stored eps for H10, for the check)."""
    out = []
    for i, f in enumerate(L.episode_files(run_dir)):
        e = L.load(f)
        T = e["steps"]
        se = {H: np.zeros(T) for H in L.HORIZONS}
        se_chk = np.zeros(T)                                    # H10 at the stored eps -> compare with stored
        same = abs(e["eps_stored"] - L.FROZEN_EPS[10]) < 1e-12
        for t in range(T):
            P, R, G = L.step_poses(e["plans"][t], e["eef_pos"][t], e["eef_mat"][t], e["ps"], e["rs"])
            gaps = L.pose_gaps(P, R)
            g_ref = L.gripper_ref(e["actions"], t)
            for H in L.HORIZONS:
                se[H][t], _ = L.variant_se("V1_endpoint_g0", L.condensed(gaps[H - 1]), G, g_ref, H, L.FROZEN_EPS[H])
            if same:
                se_chk[t] = se[10][t]
            else:
                se_chk[t], _ = L.variant_se("V1_endpoint_g0", L.condensed(gaps[9]), G, g_ref, 10, e["eps_stored"])
        out.append({"run": label, "file": e["file"], "task": e["task"], "ep": e["ep"], "success": e["success"],
                    "steps": T, "eps_stored": e["eps_stored"], "se": se, "se_chk10": se_chk,
                    "se_stored10": e["se_stored"].get(10)})
        if (i + 1) % 10 == 0:
            print(f"[triple] {label} {i + 1} episodes", flush=True)
    return out


def match_rate(E):
    tot = ident = 0
    for e in E:
        if e["se_stored10"] is None:
            continue
        d = np.abs(e["se_chk10"] - e["se_stored10"])
        tot += len(d)
        ident += int((d < 1e-9).sum())
    return (ident / tot if tot else None), tot


# ---- statistics ---------------------------------------------------------------------------------------
def cells(E, tasks, sel_c, sel_p, key):
    """Per task: (classic values, PRO values) under the two selectors."""
    out = {}
    for t in tasks:
        c = np.array([key(e) for e in E if e["task"] == t and e["run"] == "classic" and sel_c(e)])
        p = np.array([key(e) for e in E if e["task"] == t and e["run"] == "pro" and sel_p(e)])
        out[t] = (c, p)
    return out


def pooled_d(cl):
    ds = [p.mean() - c.mean() for c, p in cl.values() if len(c) and len(p)]
    return float(np.mean(ds)), [float(x) for x in ds]


def within_sd(cl):
    num = den = 0.0
    for c, p in cl.values():
        for g in (c, p):
            if len(g) > 1:
                num += g.var(ddof=1) * (len(g) - 1)
                den += len(g) - 1
    return float(np.sqrt(num / den)) if den else float("nan")


def perm_test(cl, n=N_PERM, seed=SEED):
    """Stratified permutation: shuffle the run label inside each task. Two-sided Monte Carlo p."""
    rng = np.random.default_rng(seed)
    d_obs, _ = pooled_d(cl)
    vals = np.zeros(n)
    tasks = [(np.concatenate([c, p]), len(c)) for c, p in cl.values() if len(c) and len(p)]
    for r in range(n):
        acc = 0.0
        for x, nc in tasks:
            xs = x[rng.permutation(len(x))]
            acc += xs[nc:].mean() - xs[:nc].mean()
        vals[r] = acc / len(tasks)
    p = (1 + int((np.abs(vals) >= abs(d_obs) - 1e-12).sum())) / (1 + n)
    return d_obs, float(p), vals


def boot_ci(cl, n=N_BOOT, seed=SEED):
    """Stratified bootstrap (within each cell) of the pooled d and of each d_t."""
    rng = np.random.default_rng(seed)
    keys = [t for t, (c, p) in cl.items() if len(c) and len(p)]
    pooled, per = np.zeros(n), {t: np.zeros(n) for t in keys}
    for r in range(n):
        acc = 0.0
        for t in keys:
            c, p = cl[t]
            dt = p[rng.integers(0, len(p), len(p))].mean() - c[rng.integers(0, len(c), len(c))].mean()
            per[t][r] = dt
            acc += dt
        pooled[r] = acc / len(keys)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return q(pooled), {t: q(v) for t, v in per.items()}


def leg(cl):
    d, ds = pooled_d(cl)
    sd = within_sd(cl)
    ci, ci_t = boot_ci(cl)
    res = {"pooled_d_pro_minus_classic": d, "ci95": ci, "within_group_sd": sd,
           "effect_size_sd": d / sd if sd and sd > 0 else None,
           "per_task": {int(t): {"n_classic": int(len(c)), "n_pro": int(len(p)),
                                 "mean_classic": float(c.mean()) if len(c) else None,
                                 "mean_pro": float(p.mean()) if len(p) else None,
                                 "d": float(p.mean() - c.mean()) if len(c) and len(p) else None,
                                 "ci95": ci_t.get(t)} for t, (c, p) in cl.items()}}
    return res


# ---- plot ---------------------------------------------------------------------------------------------
def mean_curve(series):
    if not series:
        return None
    Lmax = max(len(s) for s in series)
    M = np.full((len(series), Lmax), np.nan)
    for i, s in enumerate(series):
        M[i, :len(s)] = s
    with np.errstate(all="ignore"):
        return np.nanmean(M, axis=0)


def plot(E, K, H, out):
    fig, axes = plt.subplots(2, 4, figsize=(22, 9), dpi=100)
    for r, tasks in enumerate((LEG_A, LEG_B)):
        for c, t in enumerate(tasks):
            ax = axes[r, c]
            groups = (("classic success", "classic", True, "#1b7f4c", "-"),
                      ("PRO success", "pro", True, "#1f5f8b", "-"),
                      ("PRO failure", "pro", False, "#c0392b", "--"))
            for lab, run, ok, col, ls in groups:
                ser = [e["se"][H] for e in E if e["task"] == t and e["run"] == run and e["success"] == ok]
                mu = mean_curve(ser)
                if mu is not None:
                    ax.plot(np.arange(len(mu)), mu, color=col, ls=ls, lw=2, label=f"{lab} (n={len(ser)})")
            ax.axvline(K, color="#555", ls="--", lw=1)
            ax.axhline(L.SE_CEIL, color="#999", ls=":", lw=1)
            ax.set_title(f"task {t} ({'leg A' if r == 0 else 'leg B'})", fontsize=10)
            ax.set_xlabel("reading (episode step)")
            if c == 0:
                ax.set_ylabel("SE (nats)")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
    fig.suptitle(f"Mean SE per reading, V1 endpoint metric, H={H}, eps={L.FROZEN_EPS[H]} (dashed = K={K}; dotted = ceiling)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out, f"triple_curves{'' if H == 10 else '_H50'}.png"), dpi=300)
    plt.close(fig)


# ---- main -----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classic", required=True)
    ap.add_argument("--pro", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    E = recompute(a.classic, "classic") + recompute(a.pro, "pro")
    res = {"frozen_eps": L.FROZEN_EPS, "n_classic": sum(e["run"] == "classic" for e in E),
           "n_pro": sum(e["run"] == "pro" for e in E)}

    # gate 0a: stored eps
    res["eps_stored"] = {run: sorted({round(e["eps_stored"], 6) for e in E if e["run"] == run}) for run in ("classic", "pro")}
    print(f"[gate 0a] stored eps: classic {res['eps_stored']['classic']}, pro {res['eps_stored']['pro']} "
          f"(recompute uses H10 {L.FROZEN_EPS[10]}, H50 {L.FROZEN_EPS[50]} for both)")

    # gate 0b: recomputed H10 at the stored eps vs stored SE
    res["match_h10"] = {}
    for run in ("classic", "pro"):
        m, n = match_rate([e for e in E if e["run"] == run])
        res["match_h10"][run] = {"share_identical": m, "readings": n}
        print(f"[gate 0b] {run}: H10 recomputed at stored eps identical on "
              f"{'n/a' if m is None else f'{100 * m:.2f}%'} of {n} readings")
    mp = res["match_h10"]["pro"]["share_identical"]
    if mp is None or mp < GATE_MATCH:
        res["verdict"] = "GATE 0b FAILED"
        L.write_json(os.path.join(a.out, "triple_results.json"), res)
        print("GATE 0b FAILED (PRO reproduction below 98%) - stop here")
        sys.exit(1)

    # per-task successes and lengths
    res["per_task_success"] = {run: L.split_from_meta([e for e in E if e["run"] == run])[2] for run in ("classic", "pro")}
    res["success_lengths"] = {run: {int(t): [int(e["steps"]) for e in E if e["run"] == run and e["task"] == t and e["success"]]
                                    for t in range(10)} for run in ("classic", "pro")}

    # K
    K = min(e["steps"] for e in E if e["success"])
    res["K"] = int(K)
    print(f"[K] shortest success over both runs, all tasks: {K}")
    if K < K_MIN:
        res["verdict"] = "K too small"
        L.write_json(os.path.join(a.out, "triple_results.json"), res)
        print(f"K too small (< {K_MIN}) - stop for review")
        sys.exit(1)

    # features
    for e in E:
        e["feat"] = {H: float(e["se"][H][:K].mean()) for H in L.HORIZONS}
    with open(os.path.join(a.out, "triple_features.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "task", "ep", "success", "length", "meanSE_H10_K", "meanSE_H50_K"])
        for e in E:
            w.writerow([e["run"], e["task"], e["ep"], int(e["success"]), e["steps"], f"{e['feat'][10]:.6f}", f"{e['feat'][50]:.6f}"])

    ok, fail = (lambda e: e["success"]), (lambda e: not e["success"])
    res["legs"] = {}
    for H in L.HORIZONS:
        key = lambda e, H=H: e["feat"][H]
        A = leg(cells(E, LEG_A, ok, ok, key))
        d, p, _ = perm_test(cells(E, LEG_A, ok, ok, key))
        A["perm_p_two_sided"] = p
        A["n_perm"] = N_PERM
        A["positive_tasks"] = int(sum(1 for v in A["per_task"].values() if v["d"] is not None and v["d"] > 0))
        A_s = leg(cells(E, LEG_A_SENS, ok, ok, key))
        _, p_s, _ = perm_test(cells(E, LEG_A_SENS, ok, ok, key))
        A_s["perm_p_two_sided"] = p_s
        B = leg(cells(E, LEG_B, ok, fail, key))
        B["gap_B_minus_gap_A"] = B["pooled_d_pro_minus_classic"] - A["pooled_d_pro_minus_classic"]
        three = {}
        for t in THREE_CELL:
            cS = np.array([key(e) for e in E if e["task"] == t and e["run"] == "classic" and e["success"]])
            pS = np.array([key(e) for e in E if e["task"] == t and e["run"] == "pro" and e["success"]])
            pF = np.array([key(e) for e in E if e["task"] == t and e["run"] == "pro" and not e["success"]])
            m = lambda v: float(v.mean()) if len(v) else None
            three[int(t)] = {"n": [len(cS), len(pS), len(pF)], "mean_classic_S": m(cS), "mean_pro_S": m(pS), "mean_pro_F": m(pF),
                             "proS_minus_classic": (m(pS) - m(cS)) if len(pS) and len(cS) else None,
                             "proF_minus_proS": (m(pF) - m(pS)) if len(pF) and len(pS) else None,
                             "proF_minus_classic": (m(pF) - m(cS)) if len(pF) and len(cS) else None}
        res["legs"][f"H{H}"] = {"legA": A, "legA_sensitivity_0_2_8": A_s, "legB_descriptive": B,
                                "three_cell_t1_t4_descriptive_NOT_independent_of_40_pair_test": three}

    # verdict (locked): primary = H10 leg A
    A = res["legs"]["H10"]["legA"]
    sd = A["within_group_sd"]
    if A["perm_p_two_sided"] < 0.05 and A["positive_tasks"] >= 3:
        verdict = "SHIFT"
    elif sd and A["ci95"][1] < SD_BOUND * sd:
        verdict = "NO DETECTABLE SHIFT, BOUNDED"
    else:
        verdict = "INCONCLUSIVE"
    res["verdict"] = verdict
    res["rule"] = ("SHIFT if two-sided stratified permutation p < 0.05 AND d_t > 0 in >= 3 of 4 leg-A tasks (H10); "
                   "NO DETECTABLE SHIFT, BOUNDED if the CI95 of the pooled d lies below +0.5 pooled within-group SD; "
                   "else INCONCLUSIVE. Fixed 2026-11-10 before any run on real data.")
    L.write_json(os.path.join(a.out, "triple_results.json"), res)
    for H in L.HORIZONS:
        plot(E, K, H, a.out)

    print(f"\n=== VERDICT (H10, leg A, locked rule): {verdict} ===")
    for H in L.HORIZONS:
        A, B = res["legs"][f"H{H}"]["legA"], res["legs"][f"H{H}"]["legB_descriptive"]
        print(f"\n--- H{H} (eps {L.FROZEN_EPS[H]}), window K={K} ---")
        print(f"leg A (tasks {LEG_A}, classic S vs PRO S): pooled d = {A['pooled_d_pro_minus_classic']:+.4f} "
              f"CI95 [{A['ci95'][0]:+.4f}, {A['ci95'][1]:+.4f}], p = {A['perm_p_two_sided']:.4f}, "
              f"effect {A['effect_size_sd']:+.2f} SD (within-group SD {A['within_group_sd']:.4f}), "
              f"positive in {A['positive_tasks']}/4 tasks")
        for t, v in A["per_task"].items():
            print(f"   task {t}: classic {v['mean_classic']:.4f} (n={v['n_classic']})  PRO {v['mean_pro']:.4f} (n={v['n_pro']})  "
                  f"d = {v['d']:+.4f} CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
        S = res["legs"][f"H{H}"]["legA_sensitivity_0_2_8"]
        print(f"sensitivity tasks {LEG_A_SENS}: d = {S['pooled_d_pro_minus_classic']:+.4f} CI95 [{S['ci95'][0]:+.4f}, {S['ci95'][1]:+.4f}] p = {S['perm_p_two_sided']:.4f}")
        print(f"leg B (tasks {LEG_B}, classic S vs PRO F, descriptive): pooled d = {B['pooled_d_pro_minus_classic']:+.4f} "
              f"CI95 [{B['ci95'][0]:+.4f}, {B['ci95'][1]:+.4f}]; gap_B - gap_A = {B['gap_B_minus_gap_A']:+.4f}")
        for t, v in B["per_task"].items():
            print(f"   task {t}: classic S {v['mean_classic']:.4f} (n={v['n_classic']})  PRO F {v['mean_pro']:.4f} (n={v['n_pro']})  d = {v['d']:+.4f}")
        th = res["legs"][f"H{H}"]["three_cell_t1_t4_descriptive_NOT_independent_of_40_pair_test"]
        for t, v in th.items():
            print(f"three cells task {t} (n classicS/proS/proF = {v['n']}; NOT independent of the 40-pair test): "
                  f"proS-classic {v['proS_minus_classic']}, proF-proS {v['proF_minus_proS']}, proF-classic {v['proF_minus_classic']}")
    print("\nsuccess lengths per task (classic | pro):")
    for t in range(10):
        lc, lp = res["success_lengths"]["classic"][t], res["success_lengths"]["pro"][t]
        f = lambda v: f"n={len(v)} median {int(np.median(v))} min {min(v)} max {max(v)}" if v else "none"
        print(f"   task {t}: {f(lc)} | {f(lp)}")
    print("TRIPLE DONE")


if __name__ == "__main__":
    main()
