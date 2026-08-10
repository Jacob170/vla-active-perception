"""
STAGE 4 — Tier-1 entropy signal collection (rollout + logging half).
Belongs to Stage 4 of the thesis plan: entropy-driven active perception.

Wraps the LIBERO-Spatial eval loop and, for EVERY episode, saves to disk:
  - per-step entropy H (shape [T, 7])
  - per-step perplexity PPL (shape [T, 7])
  - per-step action taken (shape [T, 7])
  - per-step wall-clock inference time (shape [T]) for throughput/Hz analysis
  - whether the episode succeeded (bool)
  - task id, episode index, task language string

One .npz file per episode, under ~/runs/stage4_entropy/.
No cameras move here -- this is pure data collection for OFFLINE analysis.
Runs the model in a separate process (step_4_model.py) to avoid the
TF/MuJoCo/CUDA segfault that forced the split architecture.
"""
import os; os.environ['MUJOCO_GL'] = 'osmesa'
import torch
# PyTorch >=2.6 defaults weights_only=True, which breaks LIBERO's init-state loading. Restore old behavior.
_orig_torch_load = torch.load
torch.load = lambda *a, **k: _orig_torch_load(*a, **{**k, 'weights_only': False})

import subprocess, pickle, sys, numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# ---------------- configuration ----------------
SUITE_NAME = 'libero_spatial'
NUM_EPISODES_PER_TASK = 50          # full protocol; set to 2 for a smoke test
MAX_STEPS_PER_EPISODE = 220         # official libero_spatial cap
NUM_SETTLE_STEPS = 10               # dummy steps after reset so objects stop falling
MODEL_SCRIPT = os.path.expanduser('~/thesis/vla-active-perception/step_4_model.py')
DATA_DIR = os.path.expanduser('~/runs/stage4_entropy')       # one .npz per episode
PROGRESS_LOG = os.path.join(DATA_DIR, 'progress.txt')        # human-readable success log
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------- model subprocess ----------------
server_stderr = open(os.path.join(DATA_DIR, 'server_stderr.log'), 'w')
model_proc = subprocess.Popen(
    [sys.executable, MODEL_SCRIPT],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=server_stderr)

def infer(image, prompt):
    """Send one frame to the model subprocess, get back {action, H, P, infer_time} or {error}."""
    request = pickle.dumps({'image': image, 'prompt': prompt})
    model_proc.stdin.write(len(request).to_bytes(8, 'big'))
    model_proc.stdin.write(request); model_proc.stdin.flush()
    header = model_proc.stdout.read(8)
    response_len = int.from_bytes(header, 'big')
    return pickle.loads(model_proc.stdout.read(response_len))

# ---------------- gripper action fixes (verbatim from official robot_utils.py) ----------------
def normalize_gripper_action(action, binarize=True):
    action[..., -1] = 2 * (action[..., -1] - 0.0) / (1.0 - 0.0) - 1
    if binarize: action[..., -1] = np.sign(action[..., -1])
    return action
def invert_gripper_action(action):
    action[..., -1] = action[..., -1] * -1.0
    return action

def log_progress(msg):
    print(msg, flush=True)
    with open(PROGRESS_LOG, 'a') as f: f.write(msg + '\n')

# ---------------- resume support: skip episodes whose .npz already exists ----------------
def episode_filename(task_id, episode_index):
    return os.path.join(DATA_DIR, f'task{task_id:02d}_ep{episode_index:02d}.npz')

# ---------------- main collection loop ----------------
suite = benchmark.get_benchmark_dict()[SUITE_NAME]()
log_progress(f"=== STAGE 4 entropy collection: {SUITE_NAME}, "
             f"{NUM_EPISODES_PER_TASK} eps/task, {suite.n_tasks} tasks ===")

for task_id in range(suite.n_tasks):
    task = suite.get_task(task_id)
    bddl_path = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256)
    official_initial_states = suite.get_task_init_states(task_id)
    prompt = f'In: What action should the robot take to {task.language.lower()}?\nOut:'

    for episode_index in range(NUM_EPISODES_PER_TASK):
        out_path = episode_filename(task_id, episode_index)
        if os.path.exists(out_path):            # resume: already collected
            continue

        env.reset()
        obs = env.set_init_state(official_initial_states[episode_index])
        for _ in range(NUM_SETTLE_STEPS):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

        # per-episode accumulators
        entropy_history = []      # each element: np.array shape (7,) -- H at that step
        action_history = []       # each element: np.array shape (7,) -- action taken
        perplexity_history = []
        infer_time_history = []   # each element: float -- inference seconds at that step
        episode_succeeded = False

        for step_index in range(MAX_STEPS_PER_EPISODE):
            frame = obs['agentview_image'][::-1, ::-1].copy()   # full 180-degree rotation
            result = infer(frame, prompt)
            if 'error' in result:               # model reported a per-step failure
                log_progress(f"!! server error task {task_id} ep {episode_index} "
                             f"step {step_index}: {result['error'][:120]}")
                break

            # SAVE the entropy for this step (this is the whole point of Stage 4)
            entropy_history.append(result['H'])
            perplexity_history.append(result['PPL'])
            infer_time_history.append(result['infer_time'])

            action = np.array(result['action'], dtype=np.float64).copy()
            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)
            action_history.append(action)

            obs, reward, done, info = env.step(action.tolist())
            if done:
                episode_succeeded = True
                break

        # write one .npz per episode
        np.savez_compressed(
            out_path,
            entropy_per_step=np.array(entropy_history),      # shape (T, 7)
            perplexity_per_step=np.array(perplexity_history), # shape (T, 7)
            action_per_step=np.array(action_history),        # shape (T, 7)
            infer_time_per_step=np.array(infer_time_history),# shape (T,)
            num_steps=len(entropy_history),                  # int T
            success=episode_succeeded,                       # bool
            task_id=task_id,
            episode_index=episode_index,
            task_language=task.language,
        )
        log_progress(f"task {task_id} ep {episode_index:2d}  "
                     f"success={episode_succeeded}  steps={len(entropy_history)}  saved")
    env.close()

model_proc.terminate()
log_progress("=== STAGE 4 collection complete ===")