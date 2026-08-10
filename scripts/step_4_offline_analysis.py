"""
STAGE 4 — Offline analysis of collected entropy data.
Belongs to Stage 4 of the thesis plan: entropy-driven active perception.

Reads the per-episode .npz files produced by step_4_entropy_logger.py and:
  1. Builds the composite uncertainty signal S_t for each episode.
  2. Plots S_t curves for SUCCESSFUL vs FAILED episodes (the Stage-4 deliverable).
  3. Runs the z-score trigger on each episode to see when it WOULD fire.

Nothing here touches the robot or the model. Pure post-hoc analysis.

>>> PLACEHOLDERS YOU MUST FILL (see notes at bottom):
    DOF_WEIGHTS_BETA, WINDOW_SIZE,  (WINDOW_SIZE ISNT 50 ITS selected by a fucntion find_best_window_size)
    COLD_START_MEAN, COLD_START_STD are decided using a chosedn window - dynamic 
    , MIN_HISTORY_BEFORE_TRUSTING_BUFFER
"""
import os, glob
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

DATA_DIR = os.path.expanduser('~/runs/stage4_entropy')

# ==================== PLACEHOLDERS — see notes at bottom of file ====================
# β: per-DoF weight vector. Start with uniform (all 1s) just to see curves; later
#    replace with Tang's published per-DoF weights. Length must be 7.
DOF_WEIGHTS_BETA = np.array([1.0, 2.6, 10.0, 1.0, 1.0, 1.0, 10.0])  # Tang Table A1, LIBERO-Spatial

# Window length for the trailing max-pool. Plan says 50–90. Pick one to start.
WINDOW_SIZE = 50                                   # PLACEHOLDER — try 50, 70, 90

# Cold-start stats: used before the buffer has MIN_HISTORY samples. Compute these
# from a held-out calibration set once you have data; 0/1 is a temporary stand-in.
COLD_START_MEAN = 0.0                              # PLACEHOLDER
COLD_START_STD = 1.0                               # PLACEHOLDER
MIN_HISTORY_BEFORE_TRUSTING_BUFFER = 30            # PLACEHOLDER — plan uses ~30
# ====================================================================================

def compute_auroc(episodes, window_size):
    """
    AUROC for failure prediction: can peak S_t distinguish failed from successful
    episodes? Directly comparable to Tang's reported 0.936 on LIBERO-Spatial.
    Label: 1 = failure (the 'positive' class we want to detect), 0 = success.
    Score: peak S_t over the episode.
    """
    labels, scores = [], []
    for ep in episodes:
        if ep['num_steps'] == 0:
            continue
        u_t = collapse_entropy_to_scalar(ep['entropy_per_step'])
        S_t = compute_signal_St(u_t, window_size)
        labels.append(0 if ep['success'] else 1)
        scores.append(S_t.max())
    auroc = roc_auc_score(labels, scores)
    print(f"AUROC (failure prediction, window={window_size}): {auroc:.3f}   "
          f"[Tang Spatial reference: 0.936]")
    return auroc

def collapse_entropy_to_scalar(entropy_per_step):
    """
    7 entropies per step -> 1 number per step (u_t), via weighted sum with β.
    entropy_per_step: np.array shape (T, 7)
    Returns: np.array shape (T,) -- the scalar uncertainty u_t at each step.
    NOTE: the plan also gates entropy by ATR (action-toggle rate) before this sum
          (H_tilde = (1 + lambda*ATR) * H). That gating is a PLACEHOLDER too --
          left out here so you first see the plain weighted-entropy signal. Add
          ATR once the basic version is validated.
    """
    return entropy_per_step @ DOF_WEIGHTS_BETA        # (T,7) @ (7,) -> (T,)

def compute_perplexity_scalar(perplexity_per_step):
    """
    7 per-DoF perplexities per step -> 1 number per step.
    Following Karli et al.: mean perplexity across the sequence (here, across the
    7 DoF "tokens" in one action). Unlike entropy, no beta weighting is applied --
    the paper uses a plain mean, so we replicate that to keep our perplexity signal
    a faithful comparison against their reported entropy-vs-perplexity timing result.
    perplexity_per_step: np.array shape (T, 7)
    Returns: np.array shape (T,) -- scalar perplexity at each timestep.
    """
    return perplexity_per_step.mean(axis=1)     # (T,7) -> (T,)

def compute_signal_St(scalar_uncertainty_per_step, window_size):
    """
    Max-pool u_t over a trailing window -> S_t (one value per step).
    S_t = the peak combined uncertainty seen in the last `window_size` steps.
    Returns: np.array shape (T,) -- S_t at each step.
    """
    T = len(scalar_uncertainty_per_step)
    signal_St = np.zeros(T)
    for step in range(T):
        window_start = max(0, step - window_size + 1)
        signal_St[step] = scalar_uncertainty_per_step[window_start: step + 1].max()
    return signal_St

def zscore_trigger_over_episode(signal_St, cold_start_mean, cold_start_std, trigger_threshold):

    """
    Walk S_t through the z-score trigger, mimicking how it would fire LIVE.
    cold_start_mean/std: fallback 'typical uncertainty' stats, computed from a
    calibration set (see compute_calibration_stats), used before this episode's
    own buffer has enough history to trust its own mean/std.
    """
    recent_signals = deque(maxlen=100)   # rolling memory of recent S_t values
    trigger_fired = np.zeros(len(signal_St), dtype=bool)
    for step, current_St in enumerate(signal_St):
        recent_signals.append(current_St)
        if len(recent_signals) < MIN_HISTORY_BEFORE_TRUSTING_BUFFER:
            hist_mean, hist_std = cold_start_mean, cold_start_std   # cold start
        else:
            hist_mean = np.mean(recent_signals)
            hist_std = np.std(recent_signals) + 1e-8
        zscore = (current_St - hist_mean) / hist_std
        trigger_fired[step] = zscore > trigger_threshold
    return trigger_fired

def load_all_episodes():
    """Load every episode .npz into a list of dicts."""
    episodes = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, 'task*_ep*.npz'))):
        d = np.load(path, allow_pickle=True)
        episodes.append({
            'entropy_per_step': d['entropy_per_step'],
            'perplexity_per_step': d['perplexity_per_step'], # per Karli paper  
            'success': bool(d['success']),
            'task_id': int(d['task_id']),
            'episode_index': int(d['episode_index']),
            'num_steps': int(d['num_steps']),
        })
    return episodes

H_MAX_PER_DOF = np.log(256)   # ln(256) ≈ 5.545 nats -- max possible entropy over 256 bins (uniform distribution)

def normalize_entropy(entropy_per_step):
    """
    Rescale raw entropy (nats, range [0, ln(256)]) to [0,1].
    Useful for comparing entropy across different bin-counts or metrics on the
    same scale -- NOT needed for the raw H you already collected, just a transform.
    entropy_per_step: np.array shape (T, 7)
    Returns: np.array shape (T, 7), each value in [0, 1]
    """
    return entropy_per_step / H_MAX_PER_DOF

def plot_normalized_vs_raw(episodes, save_dir, window_size):
    """
    Two-panel comparison: raw entropy S_t curves vs normalized entropy S_t curves,
    for the same episodes. Purpose: check whether normalizing (dividing by ln(256))
    changes which episodes look uncertain relative to each other, or whether it's
    just a rescaling that preserves the same success/failure separation.
    """
    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    for ep in episodes:
        if ep['num_steps'] == 0:
            continue
        color = 'green' if ep['success'] else 'red'

        # raw entropy path (nats, unbounded up to ln(256) per DoF)
        u_t_raw = collapse_entropy_to_scalar(ep['entropy_per_step'])
        S_t_raw = compute_signal_St(u_t_raw, window_size)
        ax_raw.plot(S_t_raw, color=color, alpha=0.3, linewidth=0.8)

        # normalized entropy path (each DoF's H rescaled to [0,1] before weighting)
        H_normalized = normalize_entropy(ep['entropy_per_step'])
        u_t_norm = collapse_entropy_to_scalar(H_normalized)
        S_t_norm = compute_signal_St(u_t_norm, window_size)
        ax_norm.plot(S_t_norm, color=color, alpha=0.3, linewidth=0.8)

    ax_raw.set_title('raw entropy S_t  (nats)')
    ax_norm.set_title('normalized entropy S_t  (0 to 1 scale)')
    for ax in (ax_raw, ax_norm):
        ax.set_xlabel('timestep'); ax.set_ylabel('S_t')
    plt.tight_layout()
    out_path = os.path.join(save_dir, 'St_raw_vs_normalized.png')
    plt.savefig(out_path, dpi=150)
    print(f"saved comparison plot to {out_path}")

def first_trigger_step(fired_mask):
    """Index of the first True in a boolean trigger array, or None if it never fires."""
    idx = np.argmax(fired_mask) if fired_mask.any() else None
    return idx

def compare_entropy_vs_perplexity_timing(episodes, window_size, cold_start_mean, cold_start_std, trigger_threshold):
    """
    Reproduces Karli et al.'s Fig. 4 analysis on YOUR data: for each FAILED episode,
    find the first entropy-trigger step and the first perplexity-trigger step, and
    check whether each fires BEFORE the episode ends (proactive) or never fires
    before the failure (reactive/never). Prints counts so you can see, quantified,
    whether perplexity anticipates failure earlier than entropy on LIBERO-Spatial --
    the exact question Karli et al. left as "not rigorously quantified."
    """
    entropy_before, entropy_never = 0, 0
    ppl_before, ppl_never = 0, 0
    for ep in episodes:
        if ep['success'] or ep['num_steps'] == 0:
            continue    # this comparison only makes sense for episodes that FAILED

        u_t_entropy = collapse_entropy_to_scalar(ep['entropy_per_step'])
        S_t_entropy = compute_signal_St(u_t_entropy, window_size)
        fired_entropy = zscore_trigger_over_episode(S_t_entropy, cold_start_mean, cold_start_std, trigger_threshold)
        step_entropy = first_trigger_step(fired_entropy)

        u_t_ppl = compute_perplexity_scalar(ep['perplexity_per_step'])
        S_t_ppl = compute_signal_St(u_t_ppl, window_size)
        fired_ppl = zscore_trigger_over_episode(S_t_ppl, cold_start_mean, cold_start_std, trigger_threshold)
        step_ppl = first_trigger_step(fired_ppl)

        # "before failure" = triggered at all during the episode (episode ended in failure,
        # so any trigger necessarily precedes that failure)
        if step_entropy is not None: entropy_before += 1
        else: entropy_never += 1
        if step_ppl is not None: ppl_before += 1
        else: ppl_never += 1

    print(f"failed episodes -- entropy triggered: {entropy_before}, never: {entropy_never}")
    print(f"failed episodes -- perplexity triggered: {ppl_before}, never: {ppl_never}")
    # If perplexity triggers on MORE failed episodes than entropy, that's your
    # quantified version of Karli's "perplexity anticipates failure" observation.


def find_best_window_size(episodes, window_candidates=(50, 70, 90, 93, 100)):
    """
    Try each candidate window size, measure how well peak S_t separates
    successful from failed episodes (mean peak S_t, failure minus success --
    bigger gap = better separation). Prints results AND returns the window
    with the largest gap, so the rest of the pipeline can use it.
    """
    best_window, best_gap = None, -np.inf
    for window_size in window_candidates:
        peaks_success, peaks_failure = [], []
        for ep in episodes:
            if ep['num_steps'] == 0:
                continue
            u_t = collapse_entropy_to_scalar(ep['entropy_per_step'])
            S_t = compute_signal_St(u_t, window_size)
            (peaks_success if ep['success'] else peaks_failure).append(S_t.max())
        gap = np.mean(peaks_failure) - np.mean(peaks_success)
        print(f"window={window_size:3d}  success_peak={np.mean(peaks_success):.3f}  "
              f"failure_peak={np.mean(peaks_failure):.3f}  gap={gap:.3f}")
        if gap > best_gap:
            best_gap, best_window = gap, window_size
    print(f">>> best window: {best_window} (gap={best_gap:.3f})")
    return best_window

def compute_calibration_stats(episodes, calibration_task_ids, window_size):
    """
    Compute COLD_START_MEAN / COLD_START_STD from a held-out set of confident,
    successful episodes (e.g. task 0, which ran 50/50 with no ambiguity).
    These become the fallback 'what normal uncertainty looks like' stats used
    before an episode's own buffer has enough history to trust.
    calibration_task_ids: list of task_ids to use for calibration, e.g. [0]
    """
    calibration_S_t_values = []
    for ep in episodes:
        if ep['task_id'] not in calibration_task_ids or not ep['success']:
            continue
        u_t = collapse_entropy_to_scalar(ep['entropy_per_step'])
        S_t = compute_signal_St(u_t, window_size)
        calibration_S_t_values.extend(S_t.tolist())
    mean = np.mean(calibration_S_t_values)
    std = np.std(calibration_S_t_values)
    print(f"calibration from tasks {calibration_task_ids}: "
          f"mean={mean:.3f}  std={std:.3f}  (from {len(calibration_S_t_values)} steps)")
    return mean, std

def precision_recall_at_k(episodes, window_size, cold_mean, cold_std, trigger_threshold):
    true_pos = false_pos = false_neg = 0
    for ep in episodes:
        if ep['num_steps'] == 0:
            continue
        u_t = collapse_entropy_to_scalar(ep['entropy_per_step'])
        S_t = compute_signal_St(u_t, window_size)
        fired = zscore_trigger_over_episode(S_t, cold_mean, cold_std, trigger_threshold).any()
        failed = not ep['success']
        if fired and failed:       true_pos  += 1
        elif fired and not failed: false_pos += 1
        elif not fired and failed: false_neg += 1
    precision = true_pos / (true_pos + false_pos + 1e-9)
    recall    = true_pos / (true_pos + false_neg + 1e-9)
    print(f"threshold={trigger_threshold:.1f}  precision={precision:.3f}  recall={recall:.3f}")
    return precision, recall

def main():
    episodes = load_all_episodes()
    print(f"loaded {len(episodes)} episodes")
    
    chosen_window = find_best_window_size(episodes) 
    COLD_START_MEAN, COLD_START_STD = compute_calibration_stats(
        episodes, calibration_task_ids=[0], window_size=chosen_window)
    print("=== Stage 1: SW only (uniform beta, no ATR) ===")
    compute_auroc(episodes, chosen_window)   # with DOF_WEIGHTS_BETA = ones, no ATR
    compare_entropy_vs_perplexity_timing(episodes, chosen_window, COLD_START_MEAN, COLD_START_STD, 2.0) # threshold = 2.0 just for this part,  for each FAILED episode,find the first entropy-trigger step and the first perplexity-trigger step
   
    for TRIGGER_Z_THRESHOLD in [1.5, 2.0, 2.5, 3.0]:
        precision_recall_at_k(episodes, chosen_window, COLD_START_MEAN, COLD_START_STD, TRIGGER_Z_THRESHOLD)
    # --- plot S_t for successful vs failed episodes (the Stage-4 deliverable) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for ep in episodes:
        if ep['num_steps'] == 0:
            continue
        u_t = collapse_entropy_to_scalar(ep['entropy_per_step'])
        S_t = compute_signal_St(u_t, chosen_window)
        color = 'green' if ep['success'] else 'red'
        ax.plot(S_t, color=color, alpha=0.3, linewidth=0.8)
    ax.set_xlabel('timestep')
    ax.set_ylabel('S_t (composite uncertainty)')
    ax.set_title('S_t over time: green = success, red = failure')
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'St_success_vs_failure.png'), dpi=150)
    print(f"saved plot to {DATA_DIR}/St_success_vs_failure.png")

    # --- quick numeric check: mean peak S_t, success vs failure ---
    peak_success = [compute_signal_St(collapse_entropy_to_scalar(e['entropy_per_step']),
                                      chosen_window).max()
                    for e in episodes if e['success'] and e['num_steps'] > 0]
    peak_failure = [compute_signal_St(collapse_entropy_to_scalar(e['entropy_per_step']),
                                      chosen_window).max()
                    for e in episodes if not e['success'] and e['num_steps'] > 0]
    print(f"mean peak S_t  success: {np.mean(peak_success):.3f}   "
          f"failure: {np.mean(peak_failure):.3f}")
    # If failure peak >> success peak, entropy predicts failure -> thesis premise holds.
    plot_normalized_vs_raw(episodes, DATA_DIR, chosen_window)

if __name__ == '__main__':
    main()