"""
STAGE 4 — Tier-1 entropy signal collection (inference server half).
Belongs to Stage 4 of the thesis plan: entropy-driven active perception.

Runs in its OWN process (torch + CUDA only; NEVER imports mujoco/libero).
Reads {image, prompt} from stdin, returns {action, H, P, infer_time} via stdout.

DIFFERENCE from the baseline model_server.py: also returns per-call inference
time so the offline analysis can report the policy's throughput (Hz). Entropy
computation is unchanged from the validated baseline.
"""
import sys, pickle, os, math, time, traceback
import torch, numpy as np
import torch.nn.functional as F
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

CKPT = 'openvla/openvla-7b-finetuned-libero-spatial'
proc = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
vla  = AutoModelForVision2Seq.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    trust_remote_code=True).to('cuda:0').eval()

VOCAB_SIZE = vla.vocab_size            # 32000 real tokens (logits are padded wider to 32064)
NUM_ACTION_BINS = 256                  # OpenVLA action tokens = last 256 real vocab IDs
NUM_DOF = 7                            # 7 action dimensions: [dx dy dz droll dpitch dyaw gripper]

def entropy_per_dof(scores):
    """
    Shannon entropy (natural log -> nats) of each DoF's action-token distribution.
    scores: tuple of NUM_DOF tensors, each shape [1, logits_width].
    Returns:
        entropy_per_dim: np.array shape (7,)   -- H for each DoF
        perplexity_per_dim: np.array shape (7,)   -- NOT (7,256): perplexity is one
                             number per DoF (of the chosen token), unlike H or P
        prob_per_dim:    np.array shape (7,256) -- full 256-bin distribution per DoF
    """
    perplexity_per_dim = np.zeros(NUM_DOF)
    entropy_per_dim = np.zeros(NUM_DOF)
    prob_per_dim = np.zeros((NUM_DOF, NUM_ACTION_BINS))
    for dof_index, dof_scores in enumerate(scores):
        # slice the 256 action bins: real vocab IDs [VOCAB_SIZE-256 : VOCAB_SIZE]
        action_bin_logits = dof_scores[0, VOCAB_SIZE - NUM_ACTION_BINS: VOCAB_SIZE].float()
        probs = F.softmax(action_bin_logits, dim=-1)
        probs = torch.flip(probs, [0])          # OpenVLA bin order is reversed
        prob_per_dim[dof_index] = probs.cpu().numpy()
        entropy_per_dim[dof_index] = -(probs * (probs + 1e-12).log()).sum().item()
        top_prob = probs.max().item()                    # probability the model assigned to its OWN chosen bin
        perplexity_per_dim[dof_index] = np.exp(-np.log(top_prob + 1e-12))           # perplexity of the selected token, per DoF
    return entropy_per_dim, perplexity_per_dim, prob_per_dim



def center_crop_image(img, crop_scale=0.9):
    """Eval-time center crop, PIL-only (geometry matches official). Returns 224x224 RGB."""
    w, h = img.size
    side = math.sqrt(crop_scale)
    cw, ch = int(round(w * side)), int(round(h * side))
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize((224, 224), Image.BILINEAR)

sys.stderr.write("MODEL_READY\n"); sys.stderr.flush()
sys.stderr.write(f"VOCAB_SIZE={VOCAB_SIZE}  tokenizer={proc.tokenizer.vocab_size}\n"); sys.stderr.flush()

stdin_bytes = sys.stdin.buffer
stdout_bytes = sys.stdout.buffer
step_count = 0
while True:
    header = stdin_bytes.read(8)
    if not header: break
    payload_len = int.from_bytes(header, 'big')
    request = pickle.loads(stdin_bytes.read(payload_len))
    try:
        infer_start = time.time()
        img = Image.fromarray(request['image']).convert('RGB')
        img = center_crop_image(img, 0.9)
        model_input = proc(request['prompt'], img).to('cuda:0', dtype=torch.bfloat16)
        action = vla.predict_action(**model_input, unnorm_key='libero_spatial', do_sample=False)
        generated = vla.generate(**model_input, max_new_tokens=NUM_DOF, do_sample=False,
                                 output_scores=True, return_dict_in_generate=True)
        entropy_per_dim, perplexity_per_dim, prob_per_dim = entropy_per_dof(generated.scores)
        infer_time = time.time() - infer_start        # seconds for this inference (for Hz)
        response = {'action': action, 'H': entropy_per_dim, 'PPL': perplexity_per_dim, 'P': prob_per_dim,
                    'infer_time': infer_time}
        step_count += 1
        if step_count % 25 == 0:
            torch.cuda.empty_cache()
    except Exception:
        sys.stderr.write(f"SERVER ERROR:\n{traceback.format_exc()}\n"); sys.stderr.flush()
        response = {'error': traceback.format_exc()}
    out = pickle.dumps(response)
    stdout_bytes.write(len(out).to_bytes(8, 'big')); stdout_bytes.write(out); stdout_bytes.flush()