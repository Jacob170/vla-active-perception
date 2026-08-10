"""Runs in its OWN process. Imports torch, NEVER mujoco/libero.
Reads {image, prompt} from stdin, writes {action, H} to stdout. Pickle protocol."""
import sys, pickle, os
import torch, numpy as np
import torch.nn.functional as F
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import sys

CKPT = 'openvla/openvla-7b-finetuned-libero-spatial'
proc = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
vla  = AutoModelForVision2Seq.from_pretrained(CKPT, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True).to('cuda:0').eval()
V = vla.vocab_size
def entropy_per_dof(scores):
    H = np.zeros(7); P = np.zeros((7,256))
    for i,s in enumerate(scores):
        p = F.softmax(s[0, V-256:V].float(), -1); p = torch.flip(p,[0])
        P[i] = p.cpu().numpy(); H[i] = -(p*(p+1e-12).log()).sum().item()
    return H, P

sys.stderr.write("MODEL_READY\n"); sys.stderr.flush()

rd = sys.stdin.buffer
wr = sys.stdout.buffer
sys.stderr.write(f"V={V}  tok={proc.tokenizer.vocab_size}\n")


import math


def center_crop_image(img, crop_scale=0.9):
    """Eval-time center crop, PIL-only (geometry matches official). Returns 224x224 PIL RGB."""
    w, h = img.size
    side = math.sqrt(crop_scale)
    cw, ch = int(round(w*side)), int(round(h*side))
    left, top = (w-cw)//2, (h-ch)//2
    return img.crop((left, top, left+cw, top+ch)).resize((224, 224), Image.BILINEAR)



import traceback
step_count = 0
while True:
    n = rd.read(8)
    if not n: break
    length = int.from_bytes(n, 'big')
    req = pickle.loads(rd.read(length))
    try:
        img = Image.fromarray(req['image']).convert('RGB')
        img = center_crop_image(img, 0.9)
        inp = proc(req['prompt'], img).to('cuda:0', dtype=torch.bfloat16)
        action = vla.predict_action(**inp, unnorm_key='libero_spatial', do_sample=False)
        g = vla.generate(**inp, max_new_tokens=7, do_sample=False,
                         output_scores=True, return_dict_in_generate=True)
        H, P = entropy_per_dof(g.scores)
        out = pickle.dumps({'action': action, 'H': H, 'P': P})
        step_count += 1
        if step_count % 25 == 0:
            torch.cuda.empty_cache()
    except Exception:
        sys.stderr.write(f"SERVER ERROR:\n{traceback.format_exc()}\n"); sys.stderr.flush()
        out = pickle.dumps({'error': traceback.format_exc()})
    wr.write(len(out).to_bytes(8,'big')); wr.write(out); wr.flush()