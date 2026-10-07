"""
Orchestrator -- MuJoCo/LIBERO process, pi0.5 version. NEVER import LeRobot,
transformers or model code here (TF auto-import + MuJoCo/OSMesa in one process
-> segfault). Talks to model_server_ablation.py via subprocess pipe using the
8-byte length-prefixed pickle protocol.

Changes from the OpenVLA version:
  - sends BOTH cameras (agentview + wrist) and the raw robot state; the server
    builds observation.state and does the 180 deg flip (LeRobot LiberoProcessorStep)
  - images are sent RAW: no [::-1, ::-1] here
  - no gripper normalise/invert: pi0.5 outputs env-ready actions after unnormalising
  - prompt is the raw task.language (no OpenVLA "In: ... Out:" template)
  - adds a "nominal" condition: there are no pi0.5 Stage-4 logs to reuse
  - MAX_STEPS 280 = LeRobot's libero_spatial cap, so nominal can be checked against lerobot-eval
  - logs SE for H=10 and H=50 plus the raw sampled chunks per step
  - task_id saved as int, np.savez_compressed
  - success = env.check_success(), the same test lerobot-eval uses
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("NVIDIA_DRIVER_CAPABILITIES", "compute,utility")

import sys
import struct
import pickle
import subprocess
import hashlib
import numpy as np
import mujoco

# PyTorch >=2.6: weights_only=True default breaks LIBERO's init-state loading.
import torch
_orig_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)
torch.load = _patched_load

import libero
# hf-libero (pip, site-packages) is the one LeRobot was validated with. A LIBERO
# clone on PYTHONPATH silently shadows it -- refuse to run if that happened.
assert "site-packages" in libero.__file__, (
    f"wrong LIBERO imported: {libero.__file__} -- run under `env -u PYTHONPATH`"
)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from perturbations import (
    CONDITIONS, N_EPISODES_PER_CONDITION, TASK_IDS,
    perturb_brightness, perturb_color_texture,
    apply_camera_jitter, restore_camera,
)

SUITE = "libero_spatial"      # TODO for A.2: LIBERO-PRO Spatial-Pos once the Pos mechanism is decided
CAMERA_NAME = "agentview"     # camera jitter moves this camera only (the wrist camera rides on the arm)
AGENT_KEY = "agentview_image"
WRIST_KEY = "robot0_eye_in_hand_image"
PERTURB_WRIST = True          # brightness/texture hit both views (scene-level change). False = agentview only
OUT_DIR = "outputs/ablation_pi05"
MAX_STEPS = 280               # LeRobot TASK_SUITE_MAX_STEPS["libero_spatial"]
NUM_SETTLE_STEPS = 10         # LeRobot num_steps_wait default
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]
MODEL_PYTHON = os.environ.get("MODEL_PYTHON", sys.executable)  # interpreter of the LeRobot env

CONDITIONS_PI05 = {"nominal": {"type": "none"}, **CONDITIONS}


def _send(proc, obj):
    payload = pickle.dumps(obj)
    proc.stdin.write(struct.pack(">Q", len(payload)))
    proc.stdin.write(payload)
    proc.stdin.flush()


def send_recv(proc, obj):
    _send(proc, obj)
    hdr = proc.stdout.read(8)
    if len(hdr) < 8:
        raise RuntimeError(
            "model server subprocess died mid-request (short/empty read on response "
            "header) -- its stderr traceback should be printed directly above this error"
        )
    n = struct.unpack(">Q", hdr)[0]
    return pickle.loads(proc.stdout.read(n))


# Lazy singleton -- the model is only loaded on the first request, so crashes in
# env/orchestration code exit without paying the load cost.
_model_proc_holder = {}


def get_model_proc():
    if "proc" not in _model_proc_holder:
        _model_proc_holder["proc"] = subprocess.Popen(
            [MODEL_PYTHON, "model_server_ablation.py"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        )
    return _model_proc_holder["proc"]


def shutdown_model_proc():
    """Always call this on exit (normal or crash) -- see main()'s try/finally."""
    proc = _model_proc_holder.get("proc")
    if proc is None or proc.poll() is not None:
        return
    try:
        _send(proc, {"cmd": "stop"})
        proc.wait(timeout=10)
    except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
        pass
    if proc.poll() is None:
        proc.kill()
        proc.wait()


def make_env(task_id):
    task_suite = benchmark.get_benchmark_dict()[SUITE]()
    task = task_suite.get_task(task_id)
    bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file, camera_heights=256, camera_widths=256
    )
    init_states = task_suite.get_task_init_states(task_id)  # canonical 50 trials/task
    return env, task, init_states


def perturb_pixels(rgb, cond_cfg, rng):
    if cond_cfg["type"] == "brightness":
        return perturb_brightness(rgb, cond_cfg["scale"])
    if cond_cfg["type"] == "texture":
        return perturb_color_texture(
            rgb, cond_cfg["hue_deg"], cond_cfg["sat_range"], cond_cfg["noise_std"], rng,
        )
    return rgb  # nominal and camera conditions: no pixel change


def run_episode(env, task, task_id, init_state, prompt, condition_name, cond_cfg, seed):
    rng = np.random.default_rng(seed)
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(NUM_SETTLE_STEPS):
        obs, _, _, _ = env.step(DUMMY_ACTION)

    raw_model = env.env.sim.model._model  # mujoco._structs.MjModel, required by mj_name2id
    cam_id = mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    base_pos = raw_model.cam_pos[cam_id].copy()
    base_quat = raw_model.cam_quat[cam_id].copy()

    if cond_cfg["type"] == "camera":
        apply_camera_jitter(
            raw_model, cam_id, base_pos, base_quat,
            cond_cfg["pos_m"], cond_cfg["angle_deg"], rng,
        )
    # self-verification: nonzero for camera conditions, exactly zero otherwise
    applied_pos_delta = raw_model.cam_pos[cam_id].copy() - base_pos
    applied_quat_delta = raw_model.cam_quat[cam_id].copy() - base_quat

    proc = get_model_proc()
    reply = send_recv(proc, {"cmd": "reset", "seed": seed})
    if "error" in reply:
        raise RuntimeError(reply["error"])

    log = {k: [] for k in ("se_h10", "se_h50", "n_clusters_h10", "n_clusters_h50",
                           "action", "infer_time", "eef_pos", "eef_quat", "chunks")}
    success = False

    try:
        for t in range(MAX_STEPS):
            agent = perturb_pixels(obs[AGENT_KEY].copy(), cond_cfg, rng)
            wrist = obs[WRIST_KEY].copy()
            if PERTURB_WRIST:
                wrist = perturb_pixels(wrist, cond_cfg, rng)

            result = send_recv(proc, {
                "agentview": agent, "wrist": wrist,
                "eef_pos": obs["robot0_eef_pos"].copy(),
                "eef_quat": obs["robot0_eef_quat"].copy(),
                "gripper_qpos": obs["robot0_gripper_qpos"].copy(),
                "instruction": prompt,
            })
            if "error" in result:
                print(f"!! server error [{condition_name}] task={task_id} "
                      f"step={t}: {result['error'][:200]}")
                break

            for k in ("se_h10", "se_h50", "n_clusters_h10", "n_clusters_h50", "chunks"):
                if k in result:
                    log[k].append(result[k])
            log["infer_time"].append(result["infer_time"])
            log["eef_pos"].append(obs["robot0_eef_pos"].copy())
            log["eef_quat"].append(obs["robot0_eef_quat"].copy())

            action = np.asarray(result["action"], dtype=np.float64)
            log["action"].append(action)

            obs, reward, done, info = env.step(action.tolist())
            if env.check_success():
                success = True
                break
            if done:
                break
    finally:
        if cond_cfg["type"] == "camera":
            restore_camera(raw_model, cam_id, base_pos, base_quat)

    out = {f"{k}_per_step": np.array(v) for k, v in log.items() if k != "chunks"}
    if log["chunks"]:
        out["chunks_per_step"] = np.stack(log["chunks"])  # (T, N, 50, 7) float32
    out.update({
        "success": success,
        "num_steps": len(log["action"]),
        "condition": condition_name,
        "camera_pos_delta": applied_pos_delta,
        "camera_quat_delta": applied_quat_delta,
        "task_id": int(task_id),
        "task_name": getattr(task, "name", str(task)),
        "suite": SUITE,
        "seed": seed,
    })
    return out


def stable_seed(*parts):
    """Deterministic across processes (built-in hash() is randomised per run)."""
    s = "_".join(str(p) for p in parts)
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**32)


def main():
    print(f"libero from: {libero.__file__}")
    try:
        for task_id in TASK_IDS:
            env, task, init_states = make_env(task_id)
            prompt = task.language  # raw instruction; the pi0.5 preprocessor adds the newline
            try:
                for cond_name, cond_cfg in CONDITIONS_PI05.items():
                    cond_dir = os.path.join(OUT_DIR, cond_name)
                    os.makedirs(cond_dir, exist_ok=True)
                    for ep in range(N_EPISODES_PER_CONDITION):
                        out_path = os.path.join(cond_dir, f"task{task_id}_ep{ep}.npz")
                        if os.path.exists(out_path):
                            print(f"[{cond_name}] task={task_id} ep={ep} already exists, skipping")
                            continue
                        seed = stable_seed(task_id, cond_name, ep)
                        result = run_episode(
                            env, task, task_id, init_states[ep], prompt, cond_name, cond_cfg, seed
                        )
                        np.savez_compressed(out_path, **result)
                        print(f"[{cond_name}] task={task_id} ep={ep} success={result['success']} "
                              f"steps={result['num_steps']}")
            finally:
                env.close()
    finally:
        shutdown_model_proc()


if __name__ == "__main__":
    main()
