#!/usr/bin/env python3
"""Step 5 (plots only, no test, changes nothing) - SE over the episode for every variant, one figure per horizon.
Needs: s3_series.npz from step 3.  Reads only. Writes PNG (300 dpi) + PDF next to it.

Layout of each figure: rows = task 1, task 4 ; columns = the six variants (V6 is on its own scale: it is a
dispersion, not nats). Thin lines = single episodes, thick line = mean of the episodes still running at that step
(success green, failure red), band = +-1 standard error. The dashed vertical line is K = the equal-length window
of the test (first K readings); episodes are longer than K only for failures (all failures hit 280 steps), so
right of that line a failure curve has no success to compare with - descriptive only, not evidence.

Usage: python s5_plot.py --out OUT_DIR       (OUT_DIR = .../se_variants)
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANTS = ["V1_endpoint_g0", "V2_endpoint_g1", "V3_rawmax_g1", "V4_normmax_g1", "V5_normlate_g1", "V6_dispersion"]
SHORT = {"V1_endpoint_g0": "V1 endpoint, old gripper", "V2_endpoint_g1": "V2 endpoint, new gripper",
         "V3_rawmax_g1": "V3 raw max", "V4_normmax_g1": "V4 normalized max (primary H10)",
         "V5_normlate_g1": "V5 normalized late third", "V6_dispersion": "V6 dispersion (no eps)"}
GREEN, RED = "#1b7f4c", "#c0392b"
CEIL = np.log(10) + 9 / 20


def split(z, name):
    steps = z["steps"]
    cuts = np.cumsum(steps)[:-1]
    return np.split(z[name], cuts)


def mean_sem(curves, L):
    M = np.full((len(curves), L), np.nan)
    for i, c in enumerate(curves):
        M[i, :len(c)] = c
    n = np.sum(~np.isnan(M), axis=0)
    with np.errstate(all="ignore"):
        mu = np.nanmean(M, axis=0)
        sd = np.nanstd(M, axis=0, ddof=1) if len(curves) > 1 else np.zeros(L)
        sem = sd / np.sqrt(np.maximum(n, 1))
    mu[n < 1] = np.nan
    return mu, sem, n


def make(z, H, K, out):
    task, succ = z["task"], z["success"].astype(bool)
    fig, axes = plt.subplots(2, len(VARIANTS), figsize=(30, 9), dpi=100)
    for r, t in enumerate((1, 4)):
        for c, v in enumerate(VARIANTS):
            ax = axes[r, c]
            ser = split(z, f"{v}_H{H}")
            for ok, col, lab in ((True, GREEN, "success"), (False, RED, "failure")):
                idx = [i for i in range(len(ser)) if task[i] == t and succ[i] == ok]
                if not idx:
                    continue
                cur = [ser[i] for i in idx]
                for cc in cur:
                    ax.plot(np.arange(len(cc)), cc, color=col, lw=0.5, alpha=0.25)
                L = max(len(cc) for cc in cur)
                mu, sem, n = mean_sem(cur, L)
                x = np.arange(L)
                ax.fill_between(x, mu - sem, mu + sem, color=col, alpha=0.18, lw=0)
                ax.plot(x, mu, color=col, lw=2.4, label=f"{lab} (n={len(idx)})")
            ax.axvline(K, color="#555", ls="--", lw=1)
            if c < 5:
                ax.axhline(CEIL, color="#999", ls=":", lw=1)
            ax.set_title(f"task {t} | {SHORT[v]}" if True else "", fontsize=10)
            ax.set_xlabel("reading (episode step)")
            if c == 0:
                ax.set_ylabel("SE (nats)")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"Semantic entropy per reading, all variants, H={H}  (dashed line = test window K={K}; dotted = SE ceiling {CEIL:.3f})",
                 fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"s5_se_H{H}.{ext}"), dpi=300)
    plt.close(fig)


def make_overview(z, H, K, out):
    """One panel per variant, all tasks of the test pair side by side: mean over the first K readings per episode."""
    task, succ = z["task"], z["success"].astype(bool)
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(24, 5), dpi=100)
    rng = np.random.default_rng(0)
    for c, v in enumerate(VARIANTS):
        ax = axes[c]
        ser = split(z, f"{v}_H{H}")
        for k, t in enumerate((1, 4)):
            for ok, col, off in ((True, GREEN, -0.15), (False, RED, 0.15)):
                vals = [np.nanmean(ser[i][:K]) for i in range(len(ser)) if task[i] == t and succ[i] == ok]
                ax.scatter(k + off + rng.uniform(-0.07, 0.07, len(vals)), vals, color=col, s=36, alpha=0.8)
                if vals:
                    ax.hlines(np.mean(vals), k + off - 0.12, k + off + 0.12, color=col, lw=3)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["task 1", "task 4"])
        ax.set_title(SHORT[v], fontsize=10)
        ax.grid(alpha=0.25)
        if c == 0:
            ax.set_ylabel(f"mean SE over first {K} readings")
    fig.suptitle(f"Per-episode feature of the test (mean SE over the first K={K} readings), H={H}; green = success, red = failure",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"s5_feature_H{H}.{ext}"), dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(os.path.join(a.out, "s3_series.npz"))
    K = int(z["K"])
    for H in (10, 50):
        make(z, H, K, a.out)
        make_overview(z, H, K, a.out)
        print(f"[s5] wrote s5_se_H{H}.png/.pdf and s5_feature_H{H}.png/.pdf", flush=True)
    print("STEP 5 DONE (plots only; nothing was tested or changed)")


if __name__ == "__main__":
    main()
