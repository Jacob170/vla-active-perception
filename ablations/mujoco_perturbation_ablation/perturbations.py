"""
Perturbation functions for MuJoCo/LIBERO visual-nuisance ablation.
Task-state at t=0 is held fixed across ALL conditions via LIBERO's canonical
init_states (see run_perturbation_ablation.py) -- only rendering/pixels or
camera pose differ; downstream state naturally diverges as actions diverge.
"""
import numpy as np
import cv2
import mujoco


# ---- Image-space perturbations (brightness, color/texture) ----
# These touch pixels only, post-render. Simulation state is untouched.

def perturb_brightness(rgb: np.ndarray, scale: float) -> np.ndarray:
    """Scale HSV Value channel only (isolates brightness from hue/saturation). scale=1.0 is no-op."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 2] = np.clip(hsv[..., 2] * scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def perturb_color_texture(rgb: np.ndarray, hue_shift_deg: float,
                           sat_scale_range: tuple, noise_std: float,
                           rng: np.random.Generator) -> np.ndarray:
    """Hue shift + saturation jitter + additive Gaussian noise (proxy for material/texture appearance shift)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue_delta = rng.uniform(-hue_shift_deg, hue_shift_deg)
    hsv[..., 0] = (hsv[..., 0] + hue_delta) % 180  # OpenCV H range is [0,180)
    sat_scale = rng.uniform(*sat_scale_range)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    noise = rng.normal(0, noise_std, size=out.shape)
    return np.clip(out + noise, 0, 255).astype(np.uint8)


# ---- Camera pose perturbation (MuJoCo model-level, NOT pixel-space) ----
# Applied once per episode (persistent vantage shift), not per-frame --
# this mirrors how your active-perception controller will actually behave
# (reposition once, then observe), not frame-by-frame jitter.

def apply_camera_jitter(model: mujoco.MjModel, cam_id: int,
                         base_pos: np.ndarray, base_quat: np.ndarray,
                         pos_range_m: float, angle_range_deg: float,
                         rng: np.random.Generator) -> None:
    """Writes perturbed pose into model.cam_pos/cam_quat for cam_id. MUST call restore_camera after episode."""
    delta_pos = rng.uniform(-pos_range_m, pos_range_m, size=3)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle_rad = np.deg2rad(rng.uniform(-angle_range_deg, angle_range_deg))
    delta_quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(delta_quat, axis, angle_rad)
    new_quat = np.zeros(4)
    mujoco.mju_mulQuat(new_quat, delta_quat, base_quat)
    model.cam_pos[cam_id] = base_pos + delta_pos
    model.cam_quat[cam_id] = new_quat


def restore_camera(model: mujoco.MjModel, cam_id: int,
                    base_pos: np.ndarray, base_quat: np.ndarray) -> None:
    model.cam_pos[cam_id] = base_pos
    model.cam_quat[cam_id] = base_quat


# ---- Condition matrix ----
# [Guessing] magnitudes are first-pass estimates, not calibrated against real
# sensor data. Inspect a few rendered frames per condition before trusting
# results -- re-scale if "mild" looks aggressive or vice versa.

CONDITIONS = {
    "brightness_dim":      {"type": "brightness", "scale": 0.5},
    "brightness_bright":   {"type": "brightness", "scale": 1.6},
    "texture_mild":        {"type": "texture", "hue_deg": 5,  "sat_range": (0.85, 1.15), "noise_std": 5},
    "texture_aggressive":  {"type": "texture", "hue_deg": 20, "sat_range": (0.6, 1.4),   "noise_std": 15},
    "camera_jitter_small": {"type": "camera", "pos_m": 0.01, "angle_deg": 2},
    "camera_jitter_large": {"type": "camera", "pos_m": 0.04, "angle_deg": 8},
}

N_EPISODES_PER_CONDITION = 10  # 10  original 
TASK_IDS = [4, 7]   # [4, 7] original 

# nominal (unperturbed) comparison point re-uses your existing Stage 4 logs --
# not re-run here, since it's already validated against published numbers.
