"""Per-DoF entropy from OpenVLA action logits. VERIFIED 2026-07-13
against openvla-7b-finetuned-libero-spatial:
  vocab_size   = 32000
  logits width = 32064   (64 pad tokens -- do NOT slice [-256:])
  action range = [31744, 32000)
  bin = vocab_size - token_id - 1   (REVERSED)
  argmax_bin == expected for all 7 DoFs.
"""
import numpy as np
import torch
import torch.nn.functional as F

N_BINS, N_DOF = 256, 7

def entropy_per_dof(scores, vla, eps=1e-12):
    """scores: tuple of 7 tensors [1, vocab_width] from generate(output_scores=True).
       returns H (7,) in nats, P (7,256) in bin order 0..255."""
    V = vla.vocab_size
    H = np.zeros(N_DOF)
    P = np.zeros((N_DOF, N_BINS))
    for i, s in enumerate(scores):
        p = F.softmax(s[0, V-N_BINS:V].float(), dim=-1)
        p = torch.flip(p, [0])                 # token order -> bin order
        P[i] = p.cpu().numpy()
        H[i] = -(p * (p + eps).log()).sum().item()
    assert (H >= 0).all() and (H <= np.log(N_BINS) + 1e-6).all()
    return H, P
