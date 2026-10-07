import os; os.environ['MUJOCO_GL'] = 'osmesa'
import subprocess, pickle, sys, cv2, numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

srv = subprocess.Popen([sys.executable, os.path.expanduser('~/thesis/vla-active-perception/model_server.py')],
                       stdin=subprocess.PIPE, stdout=subprocess.PIPE)
def infer(image, prompt):
    req = pickle.dumps({'image': image, 'prompt': prompt})
    srv.stdin.write(len(req).to_bytes(8,'big')); srv.stdin.write(req); srv.stdin.flush()
    n = int.from_bytes(srv.stdout.read(8),'big')
    return pickle.loads(srv.stdout.read(n))

suite = benchmark.get_benchmark_dict()['libero_spatial']()
task = suite.get_task(0)
bddl = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(0); obs = env.reset()
prompt = f'In: What action should the robot take to {task.language.lower()}?\nOut:'

frame = obs['agentview_image'][::-1, ::-1].copy()          # note: full 180 rotation
results = {}
for sigma in [0, 3, 10, 20]:
    img = frame if sigma == 0 else cv2.GaussianBlur(frame, (31, 31), sigma)
    H = infer(img, prompt)['H']
    results[sigma] = H
    print(f"sigma={sig
    
H_sharp = infer(frame, prompt)['H']
H_blur  = infer(blurred, prompt)['H']
print(f"sharp: H_sum={H_sharp.sum():.2f}   per-DoF={np.round(H_sharp,2)}")
print(f"blur:  H_sum={H_blur.sum():.2f}   per-DoF={np.round(H_blur,2)}")
env.close(); srv.terminate()