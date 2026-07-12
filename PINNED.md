# Verified working stack — 2026-07-12

LIBERO      8f1084e
mujoco      3.10.0
robosuite   1.4.0        (pip install --no-deps — egl_probe unbuildable)
torch       2.13.0+cu130
python      3.10
numpy       <2.0

## Rendering
MUJOCO_GL=osmesa          # NOT egl
  Container runs NVIDIA_DRIVER_CAPABILITIES=compute,utility.
  No graphics → no libEGL_nvidia → EGL impossible.
  Admin request sent to add `graphics`. Until then: CPU rendering.
  Cost: ~+25% wall-clock on rollouts. Inference is the bottleneck, not rendering.

## Deliberately NOT pinned
transformers  — LIBERO pins 4.21.1 for its own (unused) policies.
                OpenVLA needs >=4.40. Installed with OpenVLA.
