#!/bin/bash
# pi0.5 visual-nuisance ablation. Run inside the LeRobot + hf-libero conda env
# (NOT the OpenVLA env). Set MODEL_PYTHON if the model server needs another interpreter.
set -e

# Refuse to start if a previous instance is still alive (concurrent-model-load OOM fix).
if pgrep -f "run_perturbation_ablation.py|model_server_ablation.py" > /dev/null; then
    echo "REFUSING TO START: a previous instance is still running:"
    pgrep -af "run_perturbation_ablation.py|model_server_ablation.py"
    echo "kill it, confirm the same pgrep command comes back empty, then rerun this script."
    exit 1
fi

export MUJOCO_GL=osmesa                          # EGL is unavailable on the cluster
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export USE_TF=0                                  # stop transformers importing TensorFlow (segfault with OSMesa)

LOG_FILE="run_pi05_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to ${LOG_FILE}"
# env -u PYTHONPATH: a LIBERO clone on PYTHONPATH shadows hf-libero; the orchestrator asserts this too.
env -u PYTHONPATH python run_perturbation_ablation.py 2>&1 | tee "${LOG_FILE}"
