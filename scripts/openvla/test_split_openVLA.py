"""Main process: sim only, NEVER imports torch. Talks to model_server subprocess."""
import os; os.environ['MUJOCO_GL'] = 'osmesa'
import subprocess, pickle, sys, numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# launch model in its own process
srv = subprocess.Popen([sys.executable, os.path.expanduser('~/thesis/vla-active-perception/model_server.py')],
                       stdin=subprocess.PIPE, stdout=subprocess.PIPE)
# wait for MODEL_READY on stderr (inherited → prints to our terminal)

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

print("waiting for model...", flush=True)
r = infer(obs['agentview_image'][::-1].copy(),
          f'In: What action should the robot take to {task.language.lower()}?\nOut:')

print('TASK  :', task.language)
print('ACTION:', np.round(r['action'],4))
print('H     :', np.round(r['H'],3))
env.close(); srv.terminate()
