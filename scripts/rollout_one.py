"""One full LIBERO episode through the split pipeline. Logs H per step + saves video."""
import os; os.environ['MUJOCO_GL'] = 'osmesa'
import subprocess, pickle, sys, numpy as np, imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

srv = subprocess.Popen([sys.executable, os.path.expanduser('~/thesis/vla-active-perception/model_server.py')],
                       stdin=subprocess.PIPE, stdout=subprocess.PIPE)

def infer(image, prompt):
    req = pickle.dumps({'image': image, 'prompt': prompt})
    srv.stdin.write(len(req).to_bytes(8,'big')); srv.stdin.write(req); srv.stdin.flush()
    n = int.from_bytes(srv.stdout.read(8),'big')
    return pickle.loads(srv.stdout.read(n))

# --- verbatim from official robot_utils.py ---
def normalize_gripper_action(action, binarize=True):
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action

def invert_gripper_action(action):
    action[..., -1] = action[..., -1] * -1.0
    return action
# ---------------------------------------------

suite = benchmark.get_benchmark_dict()['libero_spatial']()
task = suite.get_task(0)
bddl = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(0); obs = env.reset()
print(f"task: {task.language}")

DUMMY = [0, 0, 0, 0, 0, 0, -1]
for _ in range(10):                                   # settle steps before first inference
    obs, _, _, _ = env.step(DUMMY)

prompt = f'In: What action should the robot take to {task.language.lower()}?\nOut:'
H_log, act_log, frames = [], [], []
done = False
print("waiting for model...", flush=True)

for t in range(220):                                  # official max_steps for libero_spatial
    frame = obs['agentview_image'][::-1, ::-1].copy() # full 180-degree rotation
    r = infer(frame, prompt)
    action = np.array(r['action'], dtype=np.float64).copy()
    action = normalize_gripper_action(action, binarize=True)
    action = invert_gripper_action(action)
    obs, reward, done, info = env.step(action.tolist())
    H_log.append(r['H']); act_log.append(action)
    frames.append(obs['agentview_image'][::-1, ::-1].copy())
    if t % 20 == 0:
        print(f"step {t:3d}  H_sum={r['H'].sum():.2f}  reward={reward}", flush=True)
    if done: break

H_log = np.array(H_log); act_log = np.array(act_log)
run_dir = os.path.expanduser('~/runs/first_episode')
os.makedirs(run_dir, exist_ok=True)
np.savez_compressed(f'{run_dir}/ep_0000_fixed.npz',
                    H=H_log, actions=act_log, success=bool(done), T=len(H_log),
                    task_name=task.name, task_language=task.language)

vid_dir = os.path.expanduser('~/videos')
os.makedirs(vid_dir, exist_ok=True)
imageio.mimsave(f'{vid_dir}/first_episode_fixed.mp4', frames, fps=30)

print(f"\nDONE: {'SUCCESS' if done else 'FAIL'} in {len(H_log)} steps")
print(f"H_sum: mean={H_log.sum(1).mean():.2f} min={H_log.sum(1).min():.2f} max={H_log.sum(1).max():.2f}")
env.close(); srv.terminate()