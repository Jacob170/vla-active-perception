"""Stage 0 smoke test: hf-libero + LeRobot env pipeline, no model.

Run (LeRobot env, clone off the path):
    env -u PYTHONPATH USE_TF=0 MUJOCO_GL=osmesa ~/env/vla/bin/python scripts/pi_0.5_fintuned/smoke_libero_lerobot.py

Checks:
  A. hf-libero alone: libero_spatial task 0, OffScreenRenderEnv, fixed init state, dummy steps.
  B. LeRobot side: make_env(LiberoEnv) -> preprocess_observation -> env preprocessor
     (LiberoProcessorStep), i.e. the lerobot-eval observation chain minus the policy.
"""

import os
import sys
import time

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoke_out")
N_STEPS = 10

import libero  # noqa: E402

print("libero.__file__:", libero.__file__)
assert "site-packages" in libero.__file__, "a LIBERO clone is shadowing hf-libero; run under env -u PYTHONPATH"

# ---------- A. hf-libero alone ----------
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

print("bddl_files:", get_libero_path("bddl_files"))
suite = benchmark.get_benchmark_dict()["libero_spatial"]()
print("libero_spatial n_tasks:", suite.n_tasks)
task = suite.get_task(0)
print("task 0:", task.language)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
t0 = time.time()
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(0)
env.reset()
init_states = suite.get_task_init_states(0)
print("init_states:", init_states.shape)
obs = env.set_init_state(init_states[0])
for _ in range(N_STEPS):
    obs, reward, done, info = env.step([0.0] * 6 + [-1.0])
print(f"A: {N_STEPS} steps ok in {time.time() - t0:.1f}s | agentview {obs['agentview_image'].shape} "
      f"wrist {obs['robot0_eye_in_hand_image'].shape} eef_pos {np.round(obs['robot0_eef_pos'], 3)} "
      f"reward {reward} done {done}")
os.makedirs(OUT, exist_ok=True)
from PIL import Image  # noqa: E402

Image.fromarray(obs["agentview_image"]).save(os.path.join(OUT, "A_agentview_raw.png"))
env.close()

# ---------- B. LeRobot env + env processor ----------
import torch  # noqa: E402

import lerobot  # noqa: E402
from lerobot.envs.configs import LiberoEnv  # noqa: E402
from lerobot.envs.factory import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.envs.utils import preprocess_observation  # noqa: E402

print("lerobot.__file__:", lerobot.__file__)
cfg = LiberoEnv(task="libero_spatial", task_ids=[0], observation_height=256, observation_width=256)
t0 = time.time()
envs = make_env(cfg, n_envs=1)
vec = envs["libero_spatial"][0]
obs, info = vec.reset(seed=[0])
env_pre, _ = make_env_pre_post_processors(cfg, policy_cfg=None)
for _ in range(N_STEPS):
    obs, reward, terminated, truncated, info = vec.step(np.array([[0.0] * 6 + [-1.0]], dtype=np.float32))
batch = env_pre(preprocess_observation(obs))
print(f"B: {N_STEPS} steps ok in {time.time() - t0:.1f}s")
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"   {k}: {tuple(v.shape)} {v.dtype} [{v.min().item():.3f}, {v.max().item():.3f}]")
assert batch["observation.state"].shape[-1] == 8, "expected state = eef pos 3 + axis-angle 3 + gripper 2"
img = batch["observation.images.image"][0].permute(1, 2, 0).cpu().numpy()
Image.fromarray((img * 255).clip(0, 255).astype(np.uint8)).save(os.path.join(OUT, "B_agentview_after_processor.png"))
vec.close()
print("SMOKE OK; images in", OUT)
sys.exit(0)
