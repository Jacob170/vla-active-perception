"""Full LIBERO-Spatial baseline via the split pipeline (no TF/torch in this process).
Reproduces official run_libero_eval protocol: official initial states, 50 trials/task."""
import os; os.environ['MUJOCO_GL'] = 'osmesa'
import torch
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, 'weights_only': False})
import subprocess, pickle, sys, numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

SUITE = 'libero_spatial'
N_TRIALS = 50
MAX_STEPS = 220
NUM_WAIT = 10
LOG = os.path.expanduser(f'~/runs/baseline/{SUITE}_baseline.txt')
os.makedirs(os.path.dirname(LOG), exist_ok=True)

server_log = open(os.path.expanduser('~/runs/baseline/server_stderr.log'), 'w')
srv = subprocess.Popen(
    [sys.executable, os.path.expanduser('~/thesis/vla-active-perception/model_server.py')],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=server_log)

def infer(image, prompt):
    req = pickle.dumps({'image': image, 'prompt': prompt})
    srv.stdin.write(len(req).to_bytes(8,'big')); srv.stdin.write(req); srv.stdin.flush()
    n = int.from_bytes(srv.stdout.read(8),'big')
    return pickle.loads(srv.stdout.read(n))

def normalize_gripper_action(action, binarize=True):
    action[..., -1] = 2 * (action[..., -1] - 0.0) / (1.0 - 0.0) - 1
    if binarize: action[..., -1] = np.sign(action[..., -1])
    return action
def invert_gripper_action(action):
    action[..., -1] = action[..., -1] * -1.0
    return action

def log(msg):
    print(msg, flush=True)
    with open(LOG, 'a') as f: f.write(msg + '\n')

suite = benchmark.get_benchmark_dict()[SUITE]()
done_set = set()
if os.path.exists(LOG):
    for line in open(LOG):
        if line.startswith('task ') and 'success=' in line:
            p = line.split()
            done_set.add((int(p[1]), int(p[3])))   # (task_id, ep)
if done_set:
    log(f"=== RESUMING: {len(done_set)} episodes already done, skipping them ===")
log(f"=== Baseline {SUITE}, {N_TRIALS} trials/task, {suite.n_tasks} tasks ===")

total_ep, total_succ = 0, 0
for task_id in range(suite.n_tasks):
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    init_states = suite.get_task_init_states(task_id)          # OFFICIAL initial states
    prompt = f'In: What action should the robot take to {task.language.lower()}?\nOut:'

    task_ep, task_succ = 0, 0
    for ep in range(N_TRIALS):
        if (task_id, ep) in done_set:            # SKIP already-done
            continue
        env.reset()
        obs = env.set_init_state(init_states[ep]) # OFFICIAL start config
        for _ in range(NUM_WAIT):
            obs, _, _, _ = env.step([0,0,0,0,0,0,-1])

        done = False
        for t in range(MAX_STEPS):
            frame = obs['agentview_image'][::-1, ::-1].copy()
            r = infer(frame, prompt)
            if 'error' in r:                     # SERVER reported a failure this step
                log(f"!! server error task {task_id} ep {ep} step {t}: {r['error'][:120]}")
                break                            # abandon this episode, done stays False
            action = np.array(r['action'], dtype=np.float64).copy()
            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)
            obs, reward, done, info = env.step(action.tolist())
            if done:
                task_succ += 1
                break
        task_ep += 1
        log(f"task {task_id} ep {ep:2d}  success={done}")
    env.close()

srv.terminate()
# recompute totals from the full log (handles resumed runs correctly)
succ = tot = 0
for line in open(LOG):
    if line.startswith('task ') and 'success=' in line:
        tot += 1
        if 'success=True' in line: succ += 1
log(f"\n=== FINAL (from log): {succ}/{tot} = {100*succ/tot:.1f}% ===")