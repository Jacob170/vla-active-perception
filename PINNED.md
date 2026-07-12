# Verified working stack — this exact combo runs

## Simulator
LIBERO      8f1084e
mujoco      3.10.0
robosuite   1.4.0          # pip install --no-deps (egl_probe won't build)
cv2         opencv-python-headless (NOT opencv-python — needs libGL)
numpy       1.26.4

## Model
transformers 4.40.1        # NOT 5.x — OpenVLA remote code breaks
tokenizers   0.19.1
timm         0.9.10
peft         0.11.1
torch        2.13.0+cu130  # newer than OpenVLA's pin (2.2.0) — fine, do NOT downgrade

## Rendering
MUJOCO_GL=osmesa           # container is compute,utility only. No libEGL_nvidia.
LD_LIBRARY_PATH=$CONDA_PREFIX/lib

## Deliberately NOT installed (training-only)
tensorflow, tensorflow_datasets, tensorflow_graphics, dlimp, torchaudio

## Gotchas that cost hours
- NEVER pip install in (base) → fills container overlay → container dies
- git needs LD_LIBRARY_PATH= prefix (conda OpenSSL breaks ssh)
- cmake must be <4 (egl_probe uses cmake_minimum_required <3.5)

## G1 PASSED 2026-07-13
mujoco 2.3.7   (NOT 3.x — robosuite 1.4.0's mj_fullM signature)
libero via plain .pth (editable install hooks broken)
smoke test: 200 frames, agentview_image (256,256,3)

## obs keys that matter
agentview_image           → OpenVLA input
robot0_eye_in_hand_image  → wrist cam
robot0_eef_pos/_quat      → EE pose, free FK for semantic entropy
