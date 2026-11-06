#!/usr/bin/env bash
# One stock-LeRobot episode per classic LIBERO task (no perturbation), WITH video, to see what a success looks like.
# Order: first the 8 LIBERO-Pro2.0 evaluation tasks (Pro2.0 Table 17: Spatial 0,8; Object 1,8; Goal 3,6; Long/libero_10 5,8),
# then the other 32 tasks of the same four suites. Stop the script whenever you have enough.
# Needs the 24 GB node (pi0.5 does not fit the small pods). One eval at a time. ~5 min per attempt incl. model load, so 40 tasks is about 3.5 h; the first 8 take about 40 min.
# For each task: try seeds 0,1,2 until the episode succeeds; keep every attempt's folder (videos inside).
#   bash run.sh [OUT_DIR]
# Written 2026-11-06, NOT yet run. Flags copied from README step 0.6; only --seed and --output_dir are new.
set -uo pipefail
OUT=${1:-$HOME/runs/pi05_classic_success_videos}
export MUJOCO_GL=osmesa USE_TF=0
export HF_HOME=${HF_HOME:-/gpfs0/bgu-ashapiro/users/yaacubov/data/hf_cache}
LEROBOT_EVAL=${LEROBOT_EVAL:-$HOME/env/vla/bin/lerobot-eval}
mkdir -p "$OUT"
PRIORITY="libero_spatial:0 libero_spatial:8 libero_object:1 libero_object:8 libero_goal:3 libero_goal:6 libero_10:5 libero_10:8"
REST=""
for SUITE in libero_spatial libero_object libero_goal libero_10; do
  for I in 0 1 2 3 4 5 6 7 8 9; do
    case " $PRIORITY " in *" $SUITE:$I "*) ;; *) REST="$REST $SUITE:$I";; esac
  done
done
for ITEM in $PRIORITY $REST; do
  SUITE=${ITEM%%:*}; T=${ITEM##*:}
  for S in 0 1 2; do
    D="$OUT/${SUITE}_task${T}_seed${S}"
    echo "=== $SUITE task $T seed $S -> $D ($(date -u +%H:%M:%S)) ==="
    env -u PYTHONPATH "$LEROBOT_EVAL" --output_dir="$D" \
      --env.type=libero --env.task="$SUITE" --env.task_ids="[$T]" \
      --eval.batch_size=1 --eval.n_episodes=1 --env.max_parallel_tasks=1 --seed=$S \
      --policy.path=lerobot/pi05_libero_finetuned_v044 --policy.n_action_steps=10 \
      > "$D.log" 2>&1
    # success = pc_success (percent) found anywhere in eval_info.json; 100.0 means the one episode succeeded
    OK=$(env -u PYTHONPATH "$HOME/env/vla/bin/python" -I - "$D/eval_info.json" <<'PY'
import json, sys
def find(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "pc_success" and isinstance(v, (int, float)):
                yield v
            yield from find(v)
    elif isinstance(o, list):
        for v in o:
            yield from find(v)
try:
    print(any(v >= 99.9 for v in find(json.load(open(sys.argv[1])))))
except Exception as e:
    print("unreadable:", e)
PY
)
    echo "$SUITE task $T seed $S success: $OK"
    [ "$OK" = "True" ] && break
  done
done
echo "done. videos: find $OUT -name '*.mp4'"
