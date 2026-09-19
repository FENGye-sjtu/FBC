#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
cd "$EFA_ROOT/code"
# Preserve the in-flight comparison and start the next SFT only after its
# completed summary exists. Never restart a failed evaluation or model run.
for attempt in $(seq 1 240); do
    if [ -f "$EFA_ROOT/runs/gqa_reader_objectives_eval_v1/summary.json" ]; then break; fi
    if ! pgrep -f '^python -m efa.evaluate_episodes .*--output /data_cinema/zqc/VBC/runs/gqa_reader_objectives_eval_v1 ' > /dev/null; then
        echo 'READER_EVALUATION_STOPPED_WITHOUT_SUMMARY'; exit 4
    fi
    sleep 30
done
test -f "$EFA_ROOT/runs/gqa_reader_objectives_eval_v1/summary.json"
bash "$EFA_ROOT/code/scripts/package_reader_objectives.sh"
bash "$EFA_ROOT/code/scripts/train_mixed_programs_2b.sh"
