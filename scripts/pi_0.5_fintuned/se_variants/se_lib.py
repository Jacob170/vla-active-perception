"""Shared maths for the SE-variant study (steps s0-s4). CPU only, numpy only.

Reads the per-episode files written by a13b_own_loop.py (v2): <run_dir>/se_logs/taskTT_epEE.npz with
plans (T, N, 50, 7) float16 in env action units, eef_pos (T, 3), eef_mat (T, 3, 3), actions (T, 7),
pos_scale, rot_scale, eps, se_h10, se_h50, success, task_id, episode_in_task.

Notation: a READING is one episode step t. At each reading there are N = 10 sampled plans. A plan is
integrated from the current pose; "step s" (1..50) means the pose after the first s actions of a plan.
"""
import glob, itertools, json, math, os
import numpy as np

# ---- constants copied from a13b_own_loop.py, do not change ------------------------------------------
LAMBDA_R, LAMBDA_G = 0.1, 10.0            # a13b_own_loop.py:76-77 (metres per radian; gripper = hard split)
SHUFFLE_SEED = 0                          # a13b_own_loop.py:78 CLUSTER_SHUFFLE_SEED
CHUNK = 50
HORIZONS = (10, 50)
SE_CEIL = math.log(10) + 9 / 20           # N = 10: ln(10) + (10 - 1) / (2 * 10) = 2.7526

# ---- pre-registered choices (2026-11-04). Changing any of these after seeing s4 is a new study. ---------
FROZEN_EPS = {10: 0.01, 50: 0.03}         # the frozen eps of the endpoint metric (decision 2026-10-29)
CALIB_TASKS = (0, 2, 5, 8)                # all-success tasks: the success norm is built ONLY here
TEST_TASKS = (1, 4)                       # the only mixed tasks: every test pair comes from here
LAMBDA_T_RATIO = 0.4                      # switch-time weight = 0.4 * eps: 1-2 steps apart merge, >= 3 split
VARIANTS = ["V1_endpoint_g0", "V2_endpoint_g1", "V3_rawmax_g1", "V4_normmax_g1", "V5_normlate_g1", "V6_dispersion"]
CLUSTERED = VARIANTS[:5]                  # V6 has no clustering and no eps
PRIMARY = ("V4_normmax_g1", 10)           # Jacob's proposal at the executed horizon
NORM_VARIANTS = ("V4_normmax_g1", "V5_normlate_g1", "V6_dispersion")

PAIRS = np.triu_indices(10, 1)            # 45 unordered pairs, i < j


# ---- loading -------------------------------------------------------------------------------------------
def episode_files(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "se_logs", "task*_ep*.npz")))
    if not files:
        raise SystemExit(f"no npz files under {run_dir}/se_logs")
    return files


def load(f):
    z = np.load(f)
    return {"file": os.path.basename(f), "task": int(z["task_id"]), "ep": int(z["episode_in_task"]),
            "success": bool(z["success"]), "plans": z["plans"], "eef_pos": z["eef_pos"], "eef_mat": z["eef_mat"],
            "actions": z["actions"], "ps": float(z["pos_scale"]), "rs": float(z["rot_scale"]),
            "eps_stored": float(z["eps"]), "steps": int(z["plans"].shape[0]),
            "se_stored": {H: np.asarray(z[f"se_h{H}"], float) for H in HORIZONS if f"se_h{H}" in z.files}}


def split_from_meta(meta):
    """Tasks where every episode succeeded (calibration) and tasks with both outcomes (test)."""
    by = {}
    for m in meta:
        by.setdefault(m["task"], []).append(m["success"])
    calib = tuple(sorted(t for t, v in by.items() if all(v)))
    test = tuple(sorted(t for t, v in by.items() if any(v) and not all(v)))
    return calib, test, {t: [int(sum(v)), len(v)] for t, v in sorted(by.items())}


# ---- geometry ------------------------------------------------------------------------------------------
def rodrigues_batch(v):
    """(N,3) axis-angle -> (N,3,3). Same as a2_analysis.py."""
    th = np.linalg.norm(v, axis=1)
    small = th < 1e-12
    k = v / np.where(small, 1.0, th)[:, None]
    K = np.zeros((len(v), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s = np.where(small, 0.0, np.sin(th))[:, None, None]
    c = np.where(small, 0.0, 1.0 - np.cos(th))[:, None, None]
    return np.eye(3)[None] + s * K + c * (K @ K)


def step_poses(plans_t, p0, R0, ps, rs, hmax=CHUNK):
    """plans_t (N,50,7) -> P (hmax,N,3), R (hmax,N,3,3), G (hmax,N): pose after s = 1..hmax actions, raw gripper.
    Position: running sum; rotation: composed on SO(3) (left-multiplied, as in the run); actions clipped to [-1, 1]."""
    A = np.clip(plans_t.astype(np.float64), -1.0, 1.0)[:, :hmax]
    N = A.shape[0]
    P = p0[None, None, :] + np.cumsum(A[:, :, :3] * ps, axis=1)
    R = np.empty((hmax, N, 3, 3))
    cur = np.broadcast_to(R0, (N, 3, 3)).copy()
    for h in range(hmax):
        cur = rodrigues_batch(A[:, h, 3:6] * rs) @ cur
        R[h] = cur
    return P.transpose(1, 0, 2), R, A[:, :, 6].T


def pose_gaps(P, R):
    """(h,N,3), (h,N,3,3) -> (h,N,N) pose distance per step: ||dp|| + LAMBDA_R * geodesic angle. NO gripper."""
    dp = np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=3)
    Rf = R.reshape(R.shape[0], R.shape[1], 9)
    cos = np.clip((np.einsum("hik,hjk->hij", Rf, Rf) - 1.0) / 2.0, -1.0, 1.0)   # trace(Ri Rj^T)
    return dp + LAMBDA_R * np.arccos(cos)


def condensed(M):
    """(..., N, N) -> (..., 45) upper triangle."""
    return M[..., PAIRS[0], PAIRS[1]]


def square(v, n=10):
    M = np.zeros((n, n))
    M[PAIRS] = v
    return M + M.T


# ---- gripper -------------------------------------------------------------------------------------------
def g0_term(G, H):
    """The run's gripper term: LAMBDA_G * |sign(last gripper value in H)_i - sign(...)_j| (np.sign, as in the run)."""
    g = np.sign(G[H - 1])
    return LAMBDA_G * np.abs(g[:, None] - g[None, :])


def gripper_ref(actions, t):
    """Gripper command in effect when reading t is taken: the executed action of the previous step (t = 0: step 0)."""
    return 1 if actions[max(t - 1, 0), 6] >= 0 else -1


def switch_times(G, g_ref, H):
    """(N,) first step s in 1..H at which a plan's gripper sign differs from g_ref; 0 = no switch inside H."""
    b = np.where(G[:H] >= 0, 1, -1)
    diff = b != g_ref
    return np.where(diff.any(axis=0), np.argmax(diff, axis=0) + 1, 0)


def g1_term(sw, eps):
    """New gripper term (Jacob, 2026-11-04). Exactly one plan of the pair switches inside H -> hard split (inf).
    Both switch -> LAMBDA_T_RATIO * eps * |switch time difference|. Neither -> 0."""
    a, b = sw[:, None], sw[None, :]
    out = np.where((a > 0) & (b > 0), LAMBDA_T_RATIO * eps * np.abs(a - b), 0.0)
    out[(a > 0) != (b > 0)] = np.inf
    return out


# ---- clustering ----------------------------------------------------------------------------------------
ORDER = np.random.default_rng(SHUFFLE_SEED).permutation(10)


def se_from_dist(D, eps, order=ORDER):
    """Greedy eps-clustering in a fixed order, then Miller-Madow SE (nats). Same as the run."""
    n = D.shape[0]
    reps, labels = [], np.full(n, -1)
    for i in order:
        for c, r in enumerate(reps):
            if D[i, r] <= eps:
                labels[i] = c
                break
        else:
            labels[i] = len(reps)
            reps.append(i)
    p = np.bincount(labels) / n
    C = len(reps)
    return float(-(p * np.log(p)).sum() + (C - 1) / (2 * n)), C


def late_steps(H):
    """Last third of the horizon (H10: steps 8-10, H50: steps 35-50), as 0-based indices."""
    return np.arange(H - H // 3, H)


def pose_parts(gaps_c, H, m):
    """gaps_c (50,45) condensed pose gaps of one reading, m (50,) normalizer -> pose part per clustered variant."""
    g = gaps_c[:H]
    gn = g / m[:H, None]
    late = late_steps(H)
    return {"V1_endpoint_g0": g[H - 1], "V2_endpoint_g1": g[H - 1], "V3_rawmax_g1": g.max(axis=0),
            "V4_normmax_g1": gn.max(axis=0), "V5_normlate_g1": gn[late].mean(axis=0)}


def dispersion(gaps_c, H, m):
    """V6: per step, the mean normalized pose gap over the 45 pairs; then the max over steps 1..H. No eps."""
    return float((gaps_c[:H] / m[:H, None]).mean(axis=1).max())


def variant_se(variant, pose_c, G, g_ref, H, eps):
    D = square(pose_c)
    if variant == "V1_endpoint_g0":
        D = D + g0_term(G, H)
    else:
        D = D + g1_term(switch_times(G, g_ref, H), eps)
    return se_from_dist(D, eps)


# ---- cache written by s1 --------------------------------------------------------------------------------
def load_cache(out_dir):
    z = np.load(os.path.join(out_dir, "s1_cache.npz"))
    meta = json.load(open(os.path.join(out_dir, "s1_episodes.json")))
    return z, meta


def normalizer(gaps, rows):
    """Median pose gap per step (50,) over the given readings and all 45 pairs."""
    m = np.median(gaps[rows].astype(np.float64), axis=(0, 2))
    return np.maximum(m, 1e-12)


# ---- statistics (copied from a2_analysis.py) --------------------------------------------------------------
def auroc(fail_scores, succ_scores):
    f, s = np.asarray(fail_scores, float), np.asarray(succ_scores, float)
    if len(f) == 0 or len(s) == 0:
        return None
    gt = (f[:, None] > s[None, :]).sum() + 0.5 * (f[:, None] == s[None, :]).sum()
    return float(gt / (len(f) * len(s)))


def within_task_auroc(E, key):
    num = den = 0.0
    for t in sorted({e["task"] for e in E}):
        f = [key(e) for e in E if e["task"] == t and not e["success"]]
        s = [key(e) for e in E if e["task"] == t and e["success"]]
        if f and s:
            num += auroc(f, s) * len(f) * len(s)
            den += len(f) * len(s)
    return None if den == 0 else num / den


def bootstrap_ci(E, key, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    by = {}
    for e in E:
        by.setdefault(e["task"], []).append(e)
    vals = []
    for _ in range(n_boot):
        samp = []
        for es in by.values():
            samp += [es[i] for i in rng.integers(0, len(es), len(es))]
        a = within_task_auroc(samp, key)
        if a is not None:
            vals.append(a)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (None, None)


class WithinTaskPerm:
    """Exact permutation distribution of the pooled within-task AUROC (labels shuffled inside each mixed task).
    For t1 (2 of 10 succeed) and t4 (6 of 10): 45 * 210 = 9450 arrangements. Copied from a2_analysis.py."""

    def __init__(self, E, max_exact=200_000, n_mc=20_000, seed=0):
        self.blocks, idx = [], []
        for t in sorted({e["task"] for e in E}):
            ix = [i for i, e in enumerate(E) if e["task"] == t]
            k = sum(E[i]["success"] for i in ix)
            if 0 < k < len(ix):
                self.blocks.append((len(idx), len(idx) + len(ix), k))
                idx += ix
        self.idx = np.array(idx, int)
        self.mixed_tasks = [E[idx[a]]["task"] for a, _, _ in self.blocks]
        self.n_pairs = sum(k * (b - a - k) for a, b, k in self.blocks)
        self.obs = np.array([1.0 if E[i]["success"] else 0.0 for i in idx])
        sizes = [math.comb(b - a, k) for a, b, k in self.blocks]
        total = int(np.prod([float(s) for s in sizes])) if sizes else 0
        self.exact = 0 < total <= max_exact
        if self.exact:
            combos = [list(itertools.combinations(range(b - a), k)) for a, b, k in self.blocks]
            Y = np.zeros((total, len(idx)))
            for row, choice in enumerate(itertools.product(*combos)):
                for (a, b, k), comb in zip(self.blocks, choice):
                    Y[row, a + np.array(comb)] = 1.0
        else:
            rng = np.random.default_rng(seed)
            Y = np.zeros((n_mc, len(idx)))
            for r in range(n_mc):
                for a, b, k in self.blocks:
                    Y[r, a:b] = rng.permutation(self.obs[a:b])
        self.Y = Y

    def auroc_of(self, x_all, Y):
        x = np.asarray(x_all, float)[self.idx]
        num = np.zeros(Y.shape[0])
        for a, b, k in self.blocks:
            xb = x[a:b]
            Gm = (xb[:, None] > xb[None, :]).astype(float) + 0.5 * (xb[:, None] == xb[None, :])
            Yb = Y[:, a:b]
            num += np.einsum("pi,ij,pj->p", 1.0 - Yb, Gm, Yb)
        return num / self.n_pairs

    def observed(self, x_all):
        return float(self.auroc_of(x_all, self.obs[None, :])[0])

    def perm_values(self, x_all):
        return self.auroc_of(x_all, self.Y)

    def pvalue(self, d_perm, d_obs):
        hit = int((d_perm >= d_obs - 1e-12).sum())
        return hit / len(d_perm) if self.exact else (1 + hit) / (1 + len(d_perm))


def write_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o)))
