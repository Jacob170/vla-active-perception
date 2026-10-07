"""
STAGE 4 — Offline analysis of collected entropy data.
Belongs to Stage 4 of the thesis plan: entropy-driven active perception.

Reads the per-episode .npz files produced by step_4_entropy_logger.py and:
  1. Builds the composite uncertainty signal S_t (entropy) and S_t_ppl (perplexity)
     for each episode.
  2. Compares entropy vs. perplexity vs. a normalized COMBINATION of both as
     failure predictors (AUROC).
  3. Plots S_t curves for SUCCESSFUL vs FAILED episodes.
  4. Runs the z-score trigger on each episode / threshold to get precision/recall.

Nothing here touches the robot or the model. Pure post-hoc analysis.

>>> STILL OPEN / NOT YET IMPLEMENTED:
    ATR (action-toggle reweighting) -- ablation stage 2 of the thesis plan.
    MIN_HISTORY_BEFORE_TRUSTING_BUFFER is a fixed guess (30), not calibrated.
"""
import os, glob
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# ============================== CONFIG ==============================
DATA_DIR = os.path.expanduser('~/runs/stage4_entropy')

# β: per-DoF weight vector (Tang et al., Table A1, LIBERO-Spatial).
# Order: [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
DOF_WEIGHTS_BETA = np.array([1.0, 2.6, 10.0, 1.0, 1.0, 1.0, 10.0])

H_MAX_PER_DOF = np.log(256)   # ln(256) ~= 5.545 nats -- max entropy over 256 bins

# How many recent S_t samples before the z-score trigger trusts its own buffer
# stats instead of falling back to the cold-start calibration values.
MIN_HISTORY_BEFORE_TRUSTING_BUFFER = 30


# ======================== PER-STEP SCALARIZATION ========================
def collapse_entropy_to_scalar(entropy_per_step):
    """
    7 entropies per step -> 1 number per step (u_t), via beta-weighted sum.
    entropy_per_step: np.array shape (T, 7)  ->  returns np.array shape (T,)
    NOTE: ATR gating (H_tilde = (1 + lambda*ATR) * H) is NOT applied here yet --
          this is the plain SW+beta signal. Add ATR as the next ablation stage.
    """
    return entropy_per_step @ DOF_WEIGHTS_BETA


def compute_perplexity_scalar(perplexity_per_step):
    """
    7 per-DoF perplexities per step -> 1 number per step, via plain mean
    (Karli et al.'s convention -- no beta weighting for perplexity).
    perplexity_per_step: np.array shape (T, 7)  ->  returns np.array shape (T,)
    """
    return perplexity_per_step.mean(axis=1)


def normalize_entropy(entropy_per_step):
    """Rescale raw entropy (nats, [0, ln(256)]) to [0,1]. Pure transform, no new info."""
    return entropy_per_step / H_MAX_PER_DOF


# ======================== SIGNAL: S_t AND TRIGGER ========================
def compute_signal_St(scalar_uncertainty_per_step, window_size):
    """
    Max-pool u_t over a trailing window -> S_t (one value per step).
    S_t = the peak combined uncertainty seen in the last `window_size` steps.
    Returns: np.array shape (T,)
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
    Before MIN_HISTORY_BEFORE_TRUSTING_BUFFER samples exist, falls back to
    cold_start_mean/std (from a held-out calibration set); after that, uses
    this episode's own recent history.
    Returns: boolean np.array shape (T,) -- True where a trigger WOULD fire.
    """
    recent_signals = deque(maxlen=100)
    trigger_fired = np.zeros(len(signal_St), dtype=bool)
    for step, current_St in enumerate(signal_St):
        recent_signals.append(current_St)
        if len(recent_signals) < MIN_HISTORY_BEFORE_TRUSTING_BUFFER:
            hist_mean, hist_std = cold_start_mean, cold_start_std
        else:
            hist_mean = np.mean(recent_signals)
            hist_std = np.std(recent_signals) + 1e-8
        zscore = (current_St - hist_mean) / hist_std
        trigger_fired[step] = zscore > trigger_threshold
    return trigger_fired


def first_trigger_step(fired_mask):
    """Index of the first True in a boolean trigger array, or None if it never fires."""
    return np.argmax(fired_mask) if fired_mask.any() else None


# ======================== CALIBRATION ========================
def _calibration_St_values(episodes, calibration_task_ids, window_size, scalar_fn, per_step_key):
    """Shared helper: collect S_t values from held-out confident/successful episodes."""
    values = []
    for ep in episodes:
        if ep['task_id'] not in calibration_task_ids or not ep['success']:
            continue
        u_t = scalar_fn(ep[per_step_key])
        S_t = compute_signal_St(u_t, window_size)
        values.extend(S_t.tolist())
    return values


def compute_calibration_stats(episodes, calibration_task_ids, window_size):
    """COLD_START_MEAN/STD for ENTROPY, from held-out confident/successful episodes."""
    values = _calibration_St_values(episodes, calibration_task_ids, window_size,
                                     collapse_entropy_to_scalar, 'entropy_per_step')
    mean, std = np.mean(values), np.std(values)
    print(f"calibration (entropy) from tasks {calibration_task_ids}: "
          f"mean={mean:.3f}  std={std:.3f}  (from {len(values)} steps)")
    return mean, std


def compute_calibration_stats_perplexity(episodes, calibration_task_ids, window_size):
    """COLD_START_MEAN/STD for PERPLEXITY, from held-out confident/successful episodes."""
    values = _calibration_St_values(episodes, calibration_task_ids, window_size,
                                     compute_perplexity_scalar, 'perplexity_per_step')
    mean, std = np.mean(values), np.std(values)
    print(f"[PPL] calibration from tasks {calibration_task_ids}: "
          f"mean={mean:.3f}  std={std:.3f}  (from {len(values)} steps)")
    return mean, std


# ======================== METRICS: AUROC ========================
def _episode_scores(episodes, window_size, scalar_fn, per_step_key, reduce_fn=lambda S_t: S_t.mean()):
    """Shared helper: (labels, scores) across all episodes for a given metric."""
    labels, scores = [], []
    for ep in episodes:
        if ep['num_steps'] == 0:
            continue
        u_t = scalar_fn(ep[per_step_key])
        S_t = compute_signal_St(u_t, window_size)
        labels.append(0 if ep['success'] else 1)   # 1 = failure (positive class)
        scores.append(reduce_fn(S_t))
    return labels, scores


def compute_auroc(episodes, window_size):
    """AUROC for failure prediction using ENTROPY. Comparable to Tang's 0.936 (Spatial)."""
    labels, scores = _episode_scores(episodes, window_size,
                                      collapse_entropy_to_scalar, 'entropy_per_step')
    auroc = roc_auc_score(labels, scores)
    print(f"AUROC (entropy, window={window_size}): {auroc:.3f}   [Tang Spatial ref: 0.936]")
    return auroc


def compute_auroc_perplexity(episodes, window_size):
    """AUROC for failure prediction using PERPLEXITY."""
    labels, scores = _episode_scores(episodes, window_size,
                                      compute_perplexity_scalar, 'perplexity_per_step')
    auroc = roc_auc_score(labels, scores)
    print(f"AUROC (perplexity, window={window_size}): {auroc:.3f}")
    return auroc


def compute_auroc_combined(episodes, window_size,
                            entropy_mean, entropy_std, ppl_mean, ppl_std):
    """
    AUROC using a NORMALIZED COMBINATION of entropy + perplexity.
    Each metric is z-scored against its own calibration stats (so the two,
    which live on very different raw scales, become comparable) and summed:
        combined = z(entropy_S_t) + z(perplexity_S_t)
    Answers: does combining beat entropy alone (0.801) or perplexity alone?
    """
    labels, scores = [], []
    for ep in episodes:
        if ep['num_steps'] == 0:
            continue
        S_t_e = compute_signal_St(collapse_entropy_to_scalar(ep['entropy_per_step']), window_size)
        S_t_p = compute_signal_St(compute_perplexity_scalar(ep['perplexity_per_step']), window_size)
        z_e = (S_t_e.mean() - entropy_mean) / (entropy_std + 1e-8)
        z_p = (S_t_p.mean() - ppl_mean) / (ppl_std + 1e-8)
        labels.append(0 if ep['success'] else 1)
        scores.append(z_e + z_p)
    auroc = roc_auc_score(labels, scores)
    print(f"AUROC (normalized entropy+perplexity COMBINED, window={window_size}): {auroc:.3f}")
    return auroc


def find_best_window_size(episodes, window_candidates=(30, 50, 70, 90, 100)):
    """
    Pick the window that best separates success from failure, measured by AUROC
    (entropy signal). Uses S_t.mean(), not .max() -- the global max is
    window-independent, only the mean of the windowed signal changes with window.
    """
    best_window, best_auroc = None, -1.0
    for window_size in window_candidates:
        labels, scores = _episode_scores(episodes, window_size,
                                          collapse_entropy_to_scalar, 'entropy_per_step')
        auroc = roc_auc_score(labels, scores)
        print(f"window={window_size:3d}  AUROC={auroc:.3f}")
        if auroc > best_auroc:
            best_auroc, best_window = auroc, window_size
    print(f">>> best window: {best_window} (AUROC={best_auroc:.3f})")
    return best_window


# ======================== METRICS: PRECISION / RECALL ========================
def _precision_recall(episodes, window_size, scalar_fn, per_step_key,
                       cold_mean, cold_std, trigger_threshold):
    """Shared helper for precision/recall at a given z-score threshold."""
    true_pos = false_pos = false_neg = 0
    for ep in episodes:
        if ep['num_steps'] == 0:
            continue
        u_t = scalar_fn(ep[per_step_key])
        S_t = compute_signal_St(u_t, window_size)
        fired = zscore_trigger_over_episode(S_t, cold_mean, cold_std, trigger_threshold).any()
        failed = not ep['success']
        if fired and failed:       true_pos  += 1
        elif fired and not failed: false_pos += 1
        elif not fired and failed: false_neg += 1
    precision = true_pos / (true_pos + false_pos + 1e-9)
    recall    = true_pos / (true_pos + false_neg + 1e-9)
    return precision, recall


def precision_recall_at_k(episodes, window_size, cold_mean, cold_std, trigger_threshold):
    precision, recall = _precision_recall(episodes, window_size,
                                           collapse_entropy_to_scalar, 'entropy_per_step',
                                           cold_mean, cold_std, trigger_threshold)
    print(f"[H]   threshold={trigger_threshold:.1f}  precision={precision:.3f}  recall={recall:.3f}")
    return precision, recall


def precision_recall_at_k_perplexity(episodes, window_size, cold_mean, cold_std, trigger_threshold):
    precision, recall = _precision_recall(episodes, window_size,
                                           compute_perplexity_scalar, 'perplexity_per_step',
                                           cold_mean, cold_std, trigger_threshold)
    print(f"[PPL] threshold={trigger_threshold:.1f}  precision={precision:.3f}  recall={recall:.3f}")
    return precision, recall


# ======================== DIAGNOSTICS ========================
def show_trigger_values(episodes, window_size, cold_mean, cold_std, trigger_threshold,
                         metric='entropy', n_examples=10):
    """
    For failed episodes where the trigger fired, print the actual S_t value and
    z-score at the moment it fired -- not just whether/when, but WHAT VALUE.
    """
    scalar_fn = collapse_entropy_to_scalar if metric == 'entropy' else compute_perplexity_scalar
    per_step_key = 'entropy_per_step' if metric == 'entropy' else 'perplexity_per_step'
    shown = 0
    for ep in episodes:
        if ep['success'] or ep['num_steps'] == 0:
            continue
        S_t = compute_signal_St(scalar_fn(ep[per_step_key]), window_size)
        fired = zscore_trigger_over_episode(S_t, cold_mean, cold_std, trigger_threshold)
        step = first_trigger_step(fired)
        if step is not None:
            zscore_at_trigger = (S_t[step] - cold_mean) / (cold_std + 1e-8)
            print(f"[{metric}] task={ep['task_id']} ep={ep['episode_index']}  "
                  f"fired at step {step}/{ep['num_steps']}  S_t={S_t[step]:.3f}  "
                  f"z-score={zscore_at_trigger:.2f}")
            shown += 1
            if shown >= n_examples:
                break


def compare_entropy_vs_perplexity_timing(episodes, window_size,
                                          cold_start_mean_h, cold_start_std_h,
                                          cold_start_mean_p, cold_start_std_p,
                                          trigger_threshold):
    entropy_before = ppl_before = 0
    total_failed = 0
    for ep in episodes:
        if ep['success'] or ep['num_steps'] == 0:
            continue
        total_failed += 1
        S_t_h = compute_signal_St(collapse_entropy_to_scalar(ep['entropy_per_step']), window_size)
        if first_trigger_step(zscore_trigger_over_episode(
                S_t_h, cold_start_mean_h, cold_start_std_h, trigger_threshold)) is not None:
            entropy_before += 1
        S_t_p = compute_signal_St(compute_perplexity_scalar(ep['perplexity_per_step']), window_size)
        if first_trigger_step(zscore_trigger_over_episode(
                S_t_p, cold_start_mean_p, cold_start_std_p, trigger_threshold)) is not None:
            ppl_before += 1
    print(f"failed episodes -- entropy triggered: {entropy_before}, never: {total_failed - entropy_before}")
    print(f"failed episodes -- perplexity triggered: {ppl_before}, never: {total_failed - ppl_before}")

# ======================== PLOTTING ========================
def _plot_St_by_outcome(episodes, window_size, scalar_fn, per_step_key, title, ylabel, out_path):
    """Shared plotting helper: S_t curves colored green=success, red=failure."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for ep in episodes:
        if ep['num_steps'] == 0:
            continue
        S_t = compute_signal_St(scalar_fn(ep[per_step_key]), window_size)
        color = 'green' if ep['success'] else 'red'
        ax.plot(S_t, color=color, alpha=0.3, linewidth=0.8)
    ax.set_xlabel('timestep'); ax.set_ylabel(ylabel); ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"saved plot to {out_path}")


def plot_St_entropy(episodes, save_dir, window_size):
    _plot_St_by_outcome(episodes, window_size, collapse_entropy_to_scalar, 'entropy_per_step',
                         'S_t over time: green = success, red = failure',
                         'S_t (composite uncertainty)',
                         os.path.join(save_dir, 'St_success_vs_failure.png'))


def plot_St_perplexity(episodes, save_dir, window_size):
    _plot_St_by_outcome(episodes, window_size, compute_perplexity_scalar, 'perplexity_per_step',
                         'Perplexity S_t over time: green = success, red = failure',
                         'S_t (perplexity-based)',
                         os.path.join(save_dir, 'St_perplexity_success_vs_failure.png'))


def plot_normalized_vs_raw(episodes, save_dir, window_size):
    """
    Two-panel comparison: raw entropy S_t vs normalized (÷ ln(256)) entropy S_t.
    Purpose: check whether normalizing changes relative ordering between episodes,
    or is just a rescale that preserves the same success/failure separation.
    """
    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    for ep in episodes:
        if ep['num_steps'] == 0:
            continue
        color = 'green' if ep['success'] else 'red'
        S_t_raw = compute_signal_St(collapse_entropy_to_scalar(ep['entropy_per_step']), window_size)
        ax_raw.plot(S_t_raw, color=color, alpha=0.3, linewidth=0.8)
        H_normalized = normalize_entropy(ep['entropy_per_step'])
        S_t_norm = compute_signal_St(collapse_entropy_to_scalar(H_normalized), window_size)
        ax_norm.plot(S_t_norm, color=color, alpha=0.3, linewidth=0.8)
    ax_raw.set_title('raw entropy S_t  (nats)')
    ax_norm.set_title('normalized entropy S_t  (0 to 1 scale)')
    for ax in (ax_raw, ax_norm):
        ax.set_xlabel('timestep'); ax.set_ylabel('S_t')
    plt.tight_layout()
    out_path = os.path.join(save_dir, 'St_raw_vs_normalized.png')
    plt.savefig(out_path, dpi=150)
    print(f"saved comparison plot to {out_path}")


# ======================== DATA LOADING ========================
def load_all_episodes():
    """Load every episode .npz into a list of dicts."""
    episodes = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, 'task*_ep*.npz'))):
        d = np.load(path, allow_pickle=True)
        episodes.append({
            'entropy_per_step': d['entropy_per_step'],
            'perplexity_per_step': d['perplexity_per_step'],
            'success': bool(d['success']),
            'task_id': int(d['task_id']),
            'episode_index': int(d['episode_index']),
            'num_steps': int(d['num_steps']),
        })
    return episodes

# ======================== LEARN LAMBDA  ========================


def sweep_combined_lambda(episodes, window_size,
                           entropy_mean, entropy_std, ppl_mean, ppl_std,
                           lambdas=(0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)):
    """
    Sweep lambda in combined = z_entropy + lambda * z_perplexity.
    lambda=0 reduces to entropy alone; find whether ANY lambda beats entropy's
    own AUROC (0.801), rather than assuming equal weight (lambda=1) is fair.
    """
    labels = []
    z_e_all, z_p_all = [], []
    for ep in episodes:
        if ep['num_steps'] == 0:
            continue
        S_t_h = compute_signal_St(collapse_entropy_to_scalar(ep['entropy_per_step']), window_size)
        S_t_p = compute_signal_St(compute_perplexity_scalar(ep['perplexity_per_step']), window_size)
        z_e_all.append((S_t_h.mean() - entropy_mean) / (entropy_std + 1e-8))
        z_p_all.append((S_t_p.mean() - ppl_mean) / (ppl_std + 1e-8))
        labels.append(0 if ep['success'] else 1)

    best_lambda, best_auroc = None, -1.0
    for lam in lambdas:
        scores = [ze + lam * zp for ze, zp in zip(z_e_all, z_p_all)]
        auroc = roc_auc_score(labels, scores)
        print(f"lambda={lam:.2f}  combined AUROC={auroc:.3f}")
        if auroc > best_auroc:
            best_auroc, best_lambda = auroc, lam
    print(f">>> best lambda={best_lambda}  AUROC={best_auroc:.3f}  "
          f"(entropy-alone reference: use lambda=0 row above)")
    return best_lambda, best_auroc

# ================================ MAIN ================================
def main():
    episodes = load_all_episodes()
    print(f"loaded {len(episodes)} episodes")

    chosen_window = find_best_window_size(episodes)
    print(f" ===== chosen window: {chosen_window} =====")

    COLD_START_MEAN, COLD_START_STD = compute_calibration_stats(
        episodes, calibration_task_ids=[0], window_size=chosen_window)
    COLD_START_MEAN_PPL, COLD_START_STD_PPL = compute_calibration_stats_perplexity(
        episodes, calibration_task_ids=[0], window_size=chosen_window)

    print("=== Stage 1: SW only (Tang beta, no ATR) ===")
    entropy_auroc = compute_auroc(episodes, chosen_window)
    perplexity_auroc = compute_auroc_perplexity(episodes, chosen_window)
    combined_auroc = compute_auroc_combined(
        episodes, chosen_window, COLD_START_MEAN, COLD_START_STD, COLD_START_MEAN_PPL, COLD_START_STD_PPL)
    print(f">>> entropy AUROC={entropy_auroc:.3f}  perplexity AUROC={perplexity_auroc:.3f}  "
          f"combined AUROC={combined_auroc:.3f}")
    print(f">>> combined {'BEATS' if combined_auroc > entropy_auroc else 'does NOT beat'} entropy alone")

    sweep_combined_lambda(episodes, chosen_window,
                       COLD_START_MEAN, COLD_START_STD, COLD_START_MEAN_PPL, COLD_START_STD_PPL)
                       
    compare_entropy_vs_perplexity_timing(episodes, chosen_window,
                                      COLD_START_MEAN, COLD_START_STD,
                                      COLD_START_MEAN_PPL, COLD_START_STD_PPL, 2.0)

    plot_St_entropy(episodes, DATA_DIR, chosen_window)
    plot_St_perplexity(episodes, DATA_DIR, chosen_window)
    plot_normalized_vs_raw(episodes, DATA_DIR, chosen_window)

    print("--- entropy trigger values (k=2.0) ---")
    show_trigger_values(episodes, chosen_window, COLD_START_MEAN, COLD_START_STD, 2.0, metric='entropy')
    print("--- perplexity trigger values (k=2.0) ---")
    show_trigger_values(episodes, chosen_window, COLD_START_MEAN_PPL, COLD_START_STD_PPL, 2.0, metric='perplexity')

    for trigger_threshold in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        precision_recall_at_k(episodes, chosen_window, COLD_START_MEAN, COLD_START_STD, trigger_threshold)
        precision_recall_at_k_perplexity(episodes, chosen_window, COLD_START_MEAN_PPL, COLD_START_STD_PPL,
                                          trigger_threshold)

    peak_success = [compute_signal_St(collapse_entropy_to_scalar(e['entropy_per_step']), chosen_window).max()
                    for e in episodes if e['success'] and e['num_steps'] > 0]
    peak_failure = [compute_signal_St(collapse_entropy_to_scalar(e['entropy_per_step']), chosen_window).max()
                    for e in episodes if not e['success'] and e['num_steps'] > 0]
    print(f"mean peak S_t  success: {np.mean(peak_success):.3f}   failure: {np.mean(peak_failure):.3f}")


if __name__ == '__main__':
    main()