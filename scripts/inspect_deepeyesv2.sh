#!/usr/bin/env bash
set -eu
source /data_cinema/zqc/VBC/code/scripts/server_env.sh
python - <<'PY'
import json
from pathlib import Path
p=Path('/data_cinema/zqc/VBC/data/raw/deepeyesv2_sft/json/sft_part_0.json')
if not p.exists():
    print('METADATA_PENDING')
else:
    rows=json.loads(p.read_text())
    print('type',type(rows).__name__,'count',len(rows))
    def search_used(row):
        return any(m['role']=='assistant' and '<tool_call>' in m['content'] for m in row['messages'])
    print('assistant_tool_call_rows',sum(search_used(r) for r in rows))
    for row in rows[:3]:
        print('metadata',json.dumps({k:v for k,v in row.items() if k!='messages'})[:1000])
        print('user',next(m['content'] for m in row['messages'] if m['role']=='user')[:500])
        print('final',next(m['content'] for m in reversed(row['messages']) if m['role']=='assistant')[-500:])
PY
