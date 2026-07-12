import os; os.environ['MUJOCO_GL'] = 'osmesa'
import numpy as np, imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
 
suite = benchmark.get_benchmark_dict()['libero_spatial']()
task  = suite.get_task(0)
print('TASK:', task.language)          # e.g. 'pick up the black bowl...'
 
bddl = os.path.join(get_libero_path('bddl_files'),
                    task.problem_folder, task.bddl_file)
 
env = OffScreenRenderEnv(bddl_file_name=bddl,
                         camera_heights=256, camera_widths=256)
env.seed(0)
obs = env.reset()
print('OBS KEYS:', list(obs.keys()))   # <-- LOOK AT THIS. memorize it.
 
frames = []
for t in range(200):
    action = np.random.uniform(-0.3, 0.3, size=7)   # random!
    obs, reward, done, info = env.step(action)
    frames.append(obs['agentview_image'][::-1])     # [::-1] = vertical flip!
    if done: break
 
os.makedirs(f"{os.environ['THESIS']}/videos", exist_ok=True)
imageio.mimsave(f"{os.environ['THESIS']}/videos/smoke.mp4", frames, fps=30)
env.close()
print('wrote', len(frames), 'frames')
