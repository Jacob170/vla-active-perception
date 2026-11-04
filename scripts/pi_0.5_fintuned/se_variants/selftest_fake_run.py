#!/usr/bin/env python3
"""Self-test: writes a FAKE run in the exact npz format of a13b_own_loop.py, with the real A.2 success pattern
(t0 10/10, t1 2/10, t2 10/10, t3 0/10, t4 6/10, t5 10/10, t6-t7 0/10, t8 10/10, t9 0/10), short episodes, and
the stored SE computed the way the run did it (float32 plans, then stored as float16). Then:

    python selftest_fake_run.py /tmp/fake_run
    PY=python3 bash run_all.sh /tmp/fake_run /tmp/fake_run/se_variants

checks the whole chain end to end in about a minute. The numbers it prints mean nothing; only that every step
runs and step 0 reproduces. Never point the chain at a fake run and report it.
"""
import os, sys
import numpy as np

POS_SCALE, ROT_SCALE, EPS, LAMBDA_R, LAMBDA_G = 0.05, 0.5, 0.01, 0.1, 10.0
OK_PATTERN = {0: 10, 1: 2, 2: 10, 3: 0, 4: 6, 5: 10, 6: 0, 7: 0, 8: 10, 9: 0}


# ---- the run's own functions (a13b_own_loop.py:85-125), copied unchanged in substance ----
def axisangle_to_mat(v):
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return np.eye(3)
    k = v / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def integrate_chunk(chunk, p0, R0, H):
    p, R = p0.copy(), R0.copy()
    for a in np.clip(chunk[:H], -1.0, 1.0):
        p = p + a[:3] * POS_SCALE
        R = axisangle_to_mat(a[3:6] * ROT_SCALE) @ R
    return p, R, float(np.sign(chunk[H - 1, 6]))


def pose_distance(A, B):
    pa, Ra, ga = A
    pb, Rb, gb = B
    cos = np.clip((np.trace(Ra @ Rb.T) - 1.0) / 2.0, -1.0, 1.0)
    return np.linalg.norm(pa - pb) + LAMBDA_R * np.arccos(cos) + LAMBDA_G * abs(ga - gb)


def semantic_entropy(poses):
    n = len(poses)
    order = np.random.default_rng(0).permutation(n)
    reps, labels = [], np.full(n, -1)
    for i in order:
        for c, r in enumerate(reps):
            if pose_distance(poses[i], poses[r]) <= EPS:
                labels[i] = c
                break
        else:
            labels[i] = len(reps)
            reps.append(i)
    p = np.bincount(labels) / n
    C = len(reps)
    return float(-(p * np.log(p)).sum() + (C - 1) / (2 * n)), C


def make_episode(rng, success, T):
    t_grasp = int(T * rng.uniform(0.3, 0.5))
    conf = 0.5 if success else 1.0                      # failures: wider plan spread and sloppier gripper timing
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
        p = p + a0[:3] * POS_SCALE
        R = axisangle_to_mat(a0[3:6] * ROT_SCALE) @ R
    return plans, pos, mat


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fake_run"
    os.makedirs(os.path.join(out, "se_logs"), exist_ok=True)
    rng = np.random.default_rng(7)
    for task, n_ok in OK_PATTERN.items():
        for ep in range(10):
            success = ep < n_ok
            T = int(rng.integers(40, 70)) if success else 90
            plans, pos, mat = make_episode(rng, success, T)
            se = {10: np.zeros(T), 50: np.zeros(T)}
            ncl = {10: np.zeros(T, int), 50: np.zeros(T, int)}
            for t in range(T):
                for H in (10, 50):
                    poses = [integrate_chunk(c, pos[t], mat[t], H) for c in plans[t]]
                    se[H][t], ncl[H][t] = semantic_entropy(poses)
            actions = np.array([np.clip(plans[t, 0, 0], -1, 1) for t in range(T)], np.float32)
            np.savez_compressed(os.path.join(out, "se_logs", f"task{task:02d}_ep{ep:02d}.npz"),
                                success=np.array(success), task_id=np.array(task), episode_in_task=np.array(ep),
                                plans=plans.astype(np.float16), eef_pos=pos, eef_mat=mat,
                                refilled=(np.arange(T) % 10 == 0), actions=actions, infer_s=np.full(T, 0.7),
                                pos_scale=np.array(POS_SCALE), rot_scale=np.array(ROT_SCALE), eps=np.array(EPS),
                                se_h10=se[10], ncl_h10=ncl[10], se_h50=se[50], ncl_h50=ncl[50])
        print(f"fake task {task}: {n_ok}/10 ok", flush=True)
    print(f"fake run written to {out}")


if __name__ == "__main__":
    main()
