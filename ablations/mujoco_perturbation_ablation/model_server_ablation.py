"""
pi0.5 model server for the visual-nuisance ablation (LeRobot PI05Policy).

Replaces the OpenVLA server. Same calling convention as before:
8-byte big-endian length prefix + pickle, over stdin/stdout.

MUST run in a SEPARATE process from MuJoCo/LIBERO (transformers + OSMesa in one
process segfaulted on the cluster). Run with USE_TF=0.

Preprocessing is LeRobot's own lerobot-eval chain, in the same order:
    preprocess_observation -> LiberoProcessorStep -> policy preprocessor
    -> PI05 sampling -> policy postprocessor
LiberoProcessorStep already rotates both images 180 deg and builds the 8-dim
state (eef pos, eef axis-angle, gripper qpos). The orchestrator therefore sends
RAW renders: flipping there too would show the model upside-down images.
No OpenVLA preprocessing here: no 0.9 centre crop, no prompt template, no
gripper normalise/invert.

Per request the server:
  1. draws N_SAMPLES action chunks (50 x 7) from N independent noise seeds,
     in one batched pass;
  2. computes discrete semantic entropy (SE) over the N chunks with
     epsilon-pose clustering, for every H in HORIZONS;
  3. returns the next action to execute from an action queue that is refilled
     every N_ACTION_STEPS steps with sample 0's chunk (same behaviour as
     select_action with n_action_steps=10: sample 0 is an ordinary draw).

Requests:
  {"cmd": "reset", "seed": int}            -> clears the action queue, reseeds noise
  {"agentview": uint8 HxWx3, "wrist": uint8 HxWx3,
   "eef_pos": (3,), "eef_quat": (4,) xyzw, "gripper_qpos": (2,),
   "instruction": str}                      -> response dict (see handle_step)
  {"cmd": "stop"}                           -> exit
"""
import sys
import os
import signal
import struct
import pickle
import time
import traceback
import numpy as np

MODEL_ID = "lerobot/pi05_libero_finetuned"  # _v044 is byte-identical; never pi05_libero_base (lerobot #2533)
DEVICE = "cuda"
N_ACTION_STEPS = 10   # actions executed per chunk; checkpoint config defaults to 50 (open loop)
N_SAMPLES = 10        # N for SE. A.3 sweeps {10, 20, 50, 100}
SE_EVERY_STEP = True  # Stage A rule: SE at every env step. False = only when the queue refills
HORIZONS = (10, 50)   # H is undecided: log both from the same samples

# epsilon-pose clustering. [Guessing] first-pass values; A.3 sweeps EPS over two
# orders of magnitude. Chunks are saved raw, so any EPS/lambda/H can be recomputed offline.
EPS = 0.02            # cluster radius in the units of d below
LAMBDA_R = 0.1        # metres per radian of geodesic rotation distance
LAMBDA_G = 10.0       # must dominate: gripper disagreement always splits a cluster
CLUSTER_SHUFFLE_SEED = 0

# [Likely] robosuite OSC_POSE defaults used by LIBERO: input clipped to [-1, 1],
# output_max 0.05 m (position) and 0.5 rad (rotation). Verify against the
# controller config on the cluster before trusting EPS in metres.
POS_SCALE = 0.05
ROT_SCALE = 0.5


# ---------------- process plumbing (unchanged from the OpenVLA server) ----------------

def install_parent_death_watchdog():
    """SIGKILL this process if the orchestrator dies, so no orphaned model holds GPU memory."""
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
        if os.getppid() == 1:
            sys.exit(0)
    except Exception:
        pass


def read_msg():
    hdr = sys.stdin.buffer.read(8)
    if len(hdr) < 8:
        return None
    n = struct.unpack(">Q", hdr)[0]
    return pickle.loads(sys.stdin.buffer.read(n))


def write_msg(obj):
    payload = pickle.dumps(obj)
    sys.stdout.buffer.write(struct.pack(">Q", len(payload)))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


# ---------------- pose integration and semantic entropy (numpy only) ----------------

def quat_xyzw_to_mat(q):
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def axisangle_to_mat(v):
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return np.eye(3)
    k = v / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def integrate_chunk(chunk, p0, R0, H):
    """Terminal pose after the first H actions of one chunk (H x 7, env action units).
    Position: running sum. Orientation: composed on SO(3), never summed.
    Gripper: sign of the last value (open/close), so small continuous noise in the
    gripper output cannot split clusters on its own; only a real open/close disagreement does."""
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


def semantic_entropy(poses, eps=EPS, seed=CLUSTER_SHUFFLE_SEED):
    """Greedy clustering against one representative per cluster, then
    SE = -sum p_c ln p_c + (C - 1) / (2N) (Miller-Madow). Returns (SE, C, labels)."""
    N = len(poses)
    order = np.random.default_rng(seed).permutation(N)
    reps, labels = [], np.full(N, -1)
    for i in order:
        for c, r in enumerate(reps):
            if pose_distance(poses[i], poses[r]) <= eps:
                labels[i] = c
                break
        else:
            labels[i] = len(reps)
            reps.append(i)
    p = np.bincount(labels) / N
    C = len(reps)
    return float(-(p * np.log(p)).sum() + (C - 1) / (2 * N)), C, labels


def se_for_horizons(chunks, eef_pos, eef_quat):
    p0, R0 = np.asarray(eef_pos, dtype=np.float64), quat_xyzw_to_mat(eef_quat)
    out = {}
    for H in HORIZONS:
        poses = [integrate_chunk(c, p0, R0, H) for c in chunks]
        se, C, _ = semantic_entropy(poses)
        out[f"se_h{H}"] = se
        out[f"n_clusters_h{H}"] = C
    return out


# ---------------- model ----------------

class Pi05Sampler:
    def __init__(self):
        import torch
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.processor import LiberoProcessorStep, PolicyProcessorPipeline
        from lerobot.envs.utils import preprocess_observation

        self.torch = torch
        self.preprocess_observation = preprocess_observation
        self.policy = PI05Policy.from_pretrained(MODEL_ID).to(DEVICE).eval()
        self.policy.config.n_action_steps = N_ACTION_STEPS
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, MODEL_ID,
            preprocessor_overrides={"device_processor": {"device": DEVICE}},
        )
        # the env-side step lerobot-eval applies for LIBERO (180 deg flip + 8-dim state)
        self.env_pre = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
        self.queue = []
        self.gen = torch.Generator(device=DEVICE)

    def reset(self, seed):
        self.queue = []
        self.gen.manual_seed(int(seed))

    def sample_chunks(self, msg, n):
        """n chunks in env action units, shape (n, chunk_size, 7). One batched pass."""
        torch = self.torch
        rep = lambda x: np.repeat(np.asarray(x)[None], n, axis=0)
        raw = {
            "pixels": {"image": rep(msg["agentview"]), "image2": rep(msg["wrist"])},
            "robot_state": {
                "eef": {"pos": rep(msg["eef_pos"]).astype(np.float64),
                        "quat": rep(msg["eef_quat"]).astype(np.float64)},
                "gripper": {"qpos": rep(msg["gripper_qpos"]).astype(np.float64)},
            },
        }
        obs = self.preprocess_observation(raw)
        obs["task"] = [msg["instruction"]] * n
        obs = self.env_pre(obs)
        batch = self.pre(obs)
        cfg = self.policy.config
        noise = torch.randn(
            (n, cfg.chunk_size, cfg.max_action_dim), generator=self.gen,
            device=DEVICE, dtype=torch.float32,
        )
        with torch.no_grad():
            actions = self.policy.predict_action_chunk(batch, noise=noise)  # (n, 50, 7), normalised
        flat = actions.reshape(-1, actions.shape[-1])
        flat = self.post(flat)  # unnormalise to env action units
        return flat.reshape(actions.shape).float().cpu().numpy()

    def handle_step(self, msg):
        torch = self.torch
        need_refill = len(self.queue) == 0
        n = N_SAMPLES if (SE_EVERY_STEP or need_refill) else 1
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        chunks = self.sample_chunks(msg, n)
        torch.cuda.synchronize()
        infer_time = time.perf_counter() - t0

        if need_refill:
            self.queue = list(chunks[0, :N_ACTION_STEPS])
        action = self.queue.pop(0)

        response = {"action": np.asarray(action, dtype=np.float64), "infer_time": infer_time,
                    "refilled": need_refill}
        if n > 1:
            response.update(se_for_horizons(chunks, msg["eef_pos"], msg["eef_quat"]))
            response["chunks"] = chunks.astype(np.float32)  # raw samples for offline re-clustering
        return response


def main():
    install_parent_death_watchdog()  # before the model load
    sampler = Pi05Sampler()
    while True:
        msg = read_msg()
        if msg is None or msg.get("cmd") == "stop":
            break
        try:
            if msg.get("cmd") == "reset":
                sampler.reset(msg.get("seed", 0))
                response = {"ok": True}
            else:
                response = sampler.handle_step(msg)
        except Exception:
            sys.stderr.write(f"SERVER ERROR:\n{traceback.format_exc()}\n")
            sys.stderr.flush()
            response = {"error": traceback.format_exc()}
        write_msg(response)


if __name__ == "__main__":
    main()
