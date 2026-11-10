#!/usr/bin/env python3
"""Self-test for se_triple.py on FAKE runs in the real npz format (generator shared with selftest_fake_run.py).

Two cases, each a pair of fake runs (classic: all successes, shorter episodes, stored eps 0.02;
PRO: the real A.2 success pattern, stored eps 0.01):
  null  - same plan spread in both runs       -> verdict must NOT be SHIFT
  shift - PRO plans much more spread out      -> verdict must be SHIFT
Checks: gates pass, K equals the shortest success over both runs, outputs written, "TRIPLE DONE" printed.
    python selftest_triple.py /tmp/fake_triple
The numbers mean nothing; never point se_triple.py at a fake run and report it.
"""
import json, os, subprocess, sys
import numpy as np
import selftest_fake_run as F

HERE = os.path.dirname(os.path.abspath(__file__))
N_EP = 5                                           # episodes per task (speed); real runs have 10
OK_PRO = {0: 5, 1: 1, 2: 5, 3: 0, 4: 3, 5: 5, 6: 0, 7: 0, 8: 5, 9: 0}
OK_CLASSIC = {t: 5 for t in range(10)}


def make_episode(rng, success, T, spread):
    """selftest_fake_run.make_episode with the plan spread as a parameter (success 0.5 there, failure 1.0)."""
    t_grasp = int(T * rng.uniform(0.3, 0.5))
    conf = spread if success else max(spread, 1.0)
    plans = np.zeros((T, 10, 50, 7), np.float32)
    pos, mat = np.zeros((T, 3)), np.zeros((T, 3, 3))
    p, R = np.array([0.0, 0.0, 0.9]), np.eye(3)
    direction = rng.normal(0, 0.3, 3)
    for t in range(T):
        pos[t], mat[t] = p, R
        base = np.zeros((50, 7))
        base[:, :3] = direction + 0.05 * np.sin(np.arange(50)[:, None] / 7.0)
        base[:, 3:6] = rng.normal(0, 0.02, 3)
        for i in range(10):
            pl = base.copy()
            pl[:, :3] += rng.normal(0, 0.04 * conf, (50, 3)) + rng.normal(0, 0.03 * conf, 3)
            pl[:, 3:6] += rng.normal(0, 0.01 * conf, (50, 3))
            s_switch = t_grasp - t + int(np.round(rng.normal(0, 1.0 + 2.0 * conf)))
            g = np.where(np.arange(1, 51) >= s_switch, 0.9, -0.9) + rng.normal(0, 0.05, 50)
            pl[:, 6] = g
            plans[t, i] = np.clip(pl, -1.2, 1.2)
        a0 = np.clip(plans[t, 0, 0].astype(np.float64), -1, 1)
        p = p + a0[:3] * F.POS_SCALE
        R = F.axisangle_to_mat(a0[3:6] * F.ROT_SCALE) @ R
    return plans, pos, mat


def write_run(out, pattern, eps_stored, spread, T_ok, T_fail, seed):
    os.makedirs(os.path.join(out, "se_logs"), exist_ok=True)
    rng = np.random.default_rng(seed)
    shortest = 10 ** 9
    for task, n_ok in pattern.items():
        for ep in range(N_EP):
            success = ep < n_ok
            T = int(rng.integers(*T_ok)) if success else T_fail
            if success:
                shortest = min(shortest, T)
            plans, pos, mat = make_episode(rng, success, T, spread)
            se = {10: np.zeros(T), 50: np.zeros(T)}
            ncl = {10: np.zeros(T, int), 50: np.zeros(T, int)}
            for t in range(T):
                for H in (10, 50):
                    poses = [F.integrate_chunk(c, pos[t], mat[t], H) for c in plans[t]]
                    F.EPS = eps_stored                      # semantic_entropy reads the module constant
                    se[H][t], ncl[H][t] = F.semantic_entropy(poses)
            actions = np.array([np.clip(plans[t, 0, 0], -1, 1) for t in range(T)], np.float32)
            np.savez_compressed(os.path.join(out, "se_logs", f"task{task:02d}_ep{ep:02d}.npz"),
                                success=np.array(success), task_id=np.array(task), episode_in_task=np.array(ep),
                                plans=plans.astype(np.float16), eef_pos=pos, eef_mat=mat,
                                refilled=(np.arange(T) % 10 == 0), actions=actions, infer_s=np.full(T, 0.7),
                                pos_scale=np.array(F.POS_SCALE), rot_scale=np.array(F.ROT_SCALE),
                                eps=np.array(eps_stored), se_h10=se[10], ncl_h10=ncl[10], se_h50=se[50], ncl_h50=ncl[50])
    return shortest


def run_case(root, name, pro_spread):
    cdir, pdir, odir = (os.path.join(root, name, x) for x in ("classic", "pro", "out"))
    k_c = write_run(cdir, OK_CLASSIC, 0.02, 0.5, (32, 42), 60, seed=11)
    k_p = write_run(pdir, OK_PRO, 0.01, pro_spread, (40, 52), 60, seed=23)
    K_expected = min(k_c, k_p)
    r = subprocess.run([sys.executable, os.path.join(HERE, "se_triple.py"), "--classic", cdir, "--pro", pdir, "--out", odir],
                       capture_output=True, text=True)
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise SystemExit(f"[{name}] se_triple.py failed")
    assert "TRIPLE DONE" in r.stdout
    res = json.load(open(os.path.join(odir, "triple_results.json")))
    assert res["K"] == K_expected, (res["K"], K_expected)
    assert res["match_h10"]["pro"]["share_identical"] >= 0.98, res["match_h10"]
    assert res["eps_stored"] == {"classic": [0.02], "pro": [0.01]}, res["eps_stored"]
    for f in ("triple_features.csv", "triple_curves.png", "triple_curves_H50.png"):
        assert os.path.exists(os.path.join(odir, f)), f
    return res["verdict"], res["legs"]["H10"]["legA"]


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fake_triple"
    v0, a0 = run_case(root, "null", pro_spread=0.5)
    v1, a1 = run_case(root, "shift", pro_spread=1.0)
    print(f"\nnull case:  verdict {v0}, pooled d {a0['pooled_d_pro_minus_classic']:+.3f}, p {a0['perm_p_two_sided']:.3f}")
    print(f"shift case: verdict {v1}, pooled d {a1['pooled_d_pro_minus_classic']:+.3f}, p {a1['perm_p_two_sided']:.3f}")
    assert v0 in ("NO DETECTABLE SHIFT, BOUNDED", "INCONCLUSIVE"), v0
    assert v1 == "SHIFT", v1
    print("SELFTEST TRIPLE PASSED (fake data; the numbers mean nothing)")


if __name__ == "__main__":
    main()
