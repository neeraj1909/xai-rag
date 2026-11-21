"""One externally coordinated ES pause; append markers to the soak interval log."""
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

base=Path('/home/neeraj/Code/personalized_interview_materials/projects/xai-rag/evaluation')
summary=base/'soak-summary.json'
intervals=base/'soak-intervals.jsonl'
name='xai-soak-es-19310'
def now(): return datetime.now(timezone.utc)
def docker(*args): return subprocess.run(['docker',*args],capture_output=True,text=True,timeout=30)
def append(action,**extra):
    initial=datetime.fromisoformat(json.loads(summary.read_text())['measurement_started_utc'])
    t=now(); row={'utc':t.isoformat(),'elapsed_seconds':round((t-initial).total_seconds(),3),'kind':'fault_amendment','action':action,**extra}
    with intervals.open('a') as f:f.write(json.dumps(row,separators=(',',':'))+'\n')
    print(json.dumps(row),flush=True)
initial=json.loads(summary.read_text())
assert initial['state']=='running'
start=datetime.fromisoformat(initial['measurement_started_utc'])
p=docker('inspect','-f','{{.Id}}',name)
assert p.returncode==0
container_id=p.stdout.strip()
append('es_fault_scheduled',scheduled_elapsed_seconds=2700,duration_seconds=20)
while (now()-start).total_seconds()<2700:
    if json.loads(summary.read_text())['state']!='running':
        append('es_fault_omitted',reason='soak no longer running'); raise SystemExit(0)
    time.sleep(min(30,max(0,2700-(now()-start).total_seconds())))
p=docker('inspect','-f','{{.Id}} {{.State.Running}}',name)
if p.returncode!=0 or p.stdout.strip()!=container_id+' true' or psutil.virtual_memory().available<1024**3 or shutil.disk_usage('/').free<3*1024**3 or json.loads(summary.read_text())['state']!='running':
    append('es_fault_omitted',reason='container, run state, or resource gate changed'); raise SystemExit(0)
paused=False
try:
    p=docker('pause',name)
    if p.returncode!=0: raise RuntimeError('docker pause failed')
    paused=True; append('pause_elasticsearch')
    time.sleep(20)
finally:
    if paused:
        p=docker('unpause',name)
        append('unpause_elasticsearch',docker_exit=p.returncode)
        if p.returncode!=0: raise RuntimeError('docker unpause failed')
