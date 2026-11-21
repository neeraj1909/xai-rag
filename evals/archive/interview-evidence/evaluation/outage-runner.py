"""Isolated, synthetic XAI-RAG stability run; no real model API calls."""
import asyncio
import hashlib
import json
import os
import pathlib
import re
import secrets
import shutil
import socket
import subprocess
import tarfile
import tempfile
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import psutil

SOURCE = pathlib.Path('/home/neeraj/Documents/xai-rag')
OUT = pathlib.Path('/home/neeraj/Code/personalized_interview_materials/projects/xai-rag/evaluation')
CHROMA = 'xai-outage-chroma-18189'
ES = 'xai-outage-es-19330'
DURATION = 75 * 60
ROOT = pathlib.Path(tempfile.mkdtemp(prefix='xai-es-outage-'))
SNAPSHOT = ROOT / 'snapshot'
SNAPSHOT.mkdir()
shutil.copytree(SOURCE / 'src', SNAPSHOT / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
manifest = {str(p.relative_to(SNAPSHOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((SNAPSHOT / 'src').rglob('*.py'))}
manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
archive = OUT / 'outage-source-snapshot.tar.gz'
with tarfile.open(archive, 'w:gz') as tar:
    for p in sorted((SNAPSHOT / 'src').rglob('*')):
        if p.is_file():
            tar.add(p, arcname=p.relative_to(SNAPSHOT))
archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
fixture = ROOT / 'fixtures'
fixture.mkdir()
(fixture / 'synthetic-policy.txt').write_text('Synthetic Cedar policy: refund requests must be filed within 30 days of purchase. Initial support response is due within four business hours.\n')
api_key = secrets.token_hex(32)
containers = []
process = None
log = None
mock = None
paused = False
start = None
records = OUT / 'outage-intervals.jsonl'
summary_path = OUT / 'outage-summary.json'
records.write_text('')
summary = {'state': 'starting', 'started_utc': datetime.now(timezone.utc).isoformat(), 'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip(), 'snapshot_manifest_sha256': manifest_hash, 'source_manifest': manifest, 'snapshot_archive_sha256': archive_hash, 'generation': 'loopback deterministic stub; no paid calls', 'ports': [18189, 19330, 18089, 18090], 'cleanup': []}
def timestamp(): return datetime.now(timezone.utc).isoformat()
def event(kind, **data):
    row = {'utc': timestamp(), 'elapsed_seconds': round(time.monotonic() - start, 3) if start else None, 'kind': kind, **data}
    with records.open('a') as f: f.write(json.dumps(row, separators=(',', ':')) + '\n')
    if kind in {'startup', 'fault', 'failure', 'stop', 'progress'}: print(json.dumps(row), flush=True)
def shell(args):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=45)
    if p.returncode: raise RuntimeError(f'{args[0]} returned {p.returncode}: {p.stderr[:300]}')
    return p.stdout.strip()
def port_free(port):
    with socket.socket() as s:
        try: s.bind(('127.0.0.1', port)); return True
        except OSError: return False
MOCK_CALLS = 0
class MockHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global MOCK_CALLS
        MOCK_CALLS += 1
        try:
            if self.path != '/v1/chat/completions': self.send_error(404); return
            request = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
            prompt = request['messages'][0]['content']
            m = re.search(r'<retrieved_chunk id="([^"]+)">', prompt)
            if not m: self.send_error(400); return
            cid = m.group(1)
            content = json.dumps({'answer': f'Refund requests must be filed within 30 days [{cid}].', 'claims': [{'text': 'Refund requests must be filed within 30 days.', 'source_chunk_ids': [cid]}], 'sources': [cid]})
            body = json.dumps({'id': 'soak-local-stub', 'object': 'chat.completion', 'created': 0, 'model': 'local-controlled-stub', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}}).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
        except Exception: self.send_error(500)
    def log_message(self, *_): pass
async def wait_http(client, url, seconds=180):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None: raise RuntimeError('API exited during startup')
        try:
            r = await client.get(url, timeout=3)
            if r.status_code == 200: return
        except httpx.HTTPError: pass
        await asyncio.sleep(2)
    raise TimeoutError(f'startup timeout: {url}')
async def main():
    global process, log, mock, start, paused
    for port in summary['ports']:
        if not port_free(port): raise RuntimeError(f'port {port} in use')
    mem=psutil.virtual_memory(); disk=shutil.disk_usage('/')
    if mem.available < 2*1024**3 or disk.free < 5*1024**3: raise RuntimeError('insufficient free memory or disk')
    summary['initial_resources']={'memory_available_bytes':mem.available,'disk_free_bytes':disk.free,'swap_free_bytes':psutil.swap_memory().free}
    ci='chromadb/chroma:1.5.9@sha256:1e0b73a187a28757c572acba508c46f48c9e8b0acaf5c20e6d95cdedce1acdf6'
    ei='docker.elastic.co/elasticsearch/elasticsearch:8.17.0@sha256:2f602552550869fb29b6fd5848c5118d3ef3a2e1d5d45802e3ab9088cb2de8e2'
    shell(['docker','run','-d','--rm','--name',CHROMA,'--memory','700m','--cpus','1','-p','127.0.0.1:18189:8000','-e','ANONYMIZED_TELEMETRY=false',ci]);containers.append(CHROMA)
    shell(['docker','run','-d','--rm','--name',ES,'--memory','1200m','--cpus','2','-p','127.0.0.1:19330:9200','-e','discovery.type=single-node','-e','xpack.security.enabled=false','-e','cluster.routing.allocation.disk.threshold_enabled=false','-e','ES_JAVA_OPTS=-Xms384m -Xmx384m',ei]);containers.append(ES)
    mock=ThreadingHTTPServer(('127.0.0.1',18090),MockHandler);threading.Thread(target=mock.serve_forever,daemon=True).start()
    env={k:v for k,v in os.environ.items() if not (k.startswith('XAI_RAG_') or k.startswith('OPENAI_') or k.lower() in ('http_proxy','https_proxy','all_proxy'))}
    env.update({'PYTHONPATH':str(SNAPSHOT/'src'),'XAI_RAG_API_KEY':api_key,'XAI_RAG_OPENAI_API_KEY':'local-synthetic-test-key','XAI_RAG_LLM_BASE_URL':'http://127.0.0.1:18090/v1','XAI_RAG_CHROMA_HOST':'127.0.0.1','XAI_RAG_CHROMA_PORT':'18189','XAI_RAG_ELASTICSEARCH_URL':'http://127.0.0.1:19330','XAI_RAG_INGEST_ROOT':str(fixture),'XAI_RAG_OTEL_ENDPOINT':'','XAI_RAG_RATE_LIMIT_PER_MINUTE':'1000','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','ANONYMIZED_TELEMETRY':'false','DO_NOT_TRACK':'1','OTEL_SDK_DISABLED':'true','NO_PROXY':'127.0.0.1,localhost'})
    async with httpx.AsyncClient(trust_env=False,timeout=120) as client:
        await wait_http(client,'http://127.0.0.1:18189/api/v2/heartbeat');await wait_http(client,'http://127.0.0.1:19330/')
        log=open(ROOT/'api.log','w');process=subprocess.Popen([str(SOURCE/'.venv/bin/python'),'-m','uvicorn','xai_rag.api.app:app','--host','127.0.0.1','--port','18089'],cwd=SNAPSHOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        await wait_http(client,'http://127.0.0.1:18089/ready');start=time.monotonic();hdr={'X-API-Key':api_key}
        r=await client.post('http://127.0.0.1:18089/ingest',headers=hdr,params={'source_path':str(fixture),'strategy':'fixed'});b=r.json();event('ingestion',status=r.status_code,body=b)
        if r.status_code!=200 or b.get('parity_consistent') is not True: raise RuntimeError('initial ingestion failed')
        payload={'query':'What is the refund deadline?','top_k':1}
        r=await client.post('http://127.0.0.1:18089/query',headers=hdr,json=payload,timeout=90);b=r.json();event('before_fault_query',status=r.status_code,answer=b.get('answer'),sources=b.get('sources'))
        if r.status_code!=200:raise RuntimeError('pre-fault query failed')
        before=MOCK_CALLS;shell(['docker','pause',ES]);paused=True;event('fault',action='pause_elasticsearch')
        try:
            t=time.monotonic();r=await client.post('http://127.0.0.1:18089/query',headers=hdr,json=payload,timeout=90); elapsed=time.monotonic()-t
            summary['outage_query']={'status':r.status_code,'body':r.json(),'duration_seconds':round(elapsed,3),'stub_calls_before':before,'stub_calls_after':MOCK_CALLS,'sanitized_503':r.status_code==503 and r.json()=={'detail':'Search dependency unavailable'},'no_generation_during_outage':MOCK_CALLS==before}
            event('outage_query',**summary['outage_query'])
        finally:
            shell(['docker','unpause',ES]);paused=False;event('fault',action='unpause_elasticsearch')
        await wait_http(client,'http://127.0.0.1:18089/ready',seconds=60)
        t=time.monotonic();r=await client.post('http://127.0.0.1:18089/query',headers=hdr,json=payload,timeout=90);b=r.json()
        summary['recovery_query']={'status':r.status_code,'duration_seconds':round(time.monotonic()-t,3),'answer':b.get('answer'),'sources':b.get('sources'),'passed':r.status_code==200 and '30 days' in b.get('answer','') and bool(b.get('sources'))};event('recovery_query',**summary['recovery_query'])
        summary['state']='completed' if all((summary['outage_query']['sanitized_503'],summary['outage_query']['no_generation_during_outage'],summary['recovery_query']['passed'])) else 'assertion_failed'
        event('stop',reason=summary['state'])
try:
    asyncio.run(main())
except BaseException as exc:
    summary['state']='failed'; summary['error']={'type':type(exc).__name__,'message':str(exc)}; event('failure',error_type=type(exc).__name__,message=str(exc))
finally:
    if paused:
        p=subprocess.run(['docker','unpause',ES],capture_output=True,text=True,timeout=30); summary['cleanup'].append({'action':'unpause','exit':p.returncode})
    if process:
        process.terminate()
        try: process.wait(timeout=15)
        except subprocess.TimeoutExpired: process.kill(); process.wait(); summary['cleanup'].append({'action':'api_kill'})
    if log: log.close()
    if mock: mock.shutdown(); mock.server_close()
    for name in reversed(containers):
        p=subprocess.run(['docker','stop','-t','5',name],capture_output=True,text=True,timeout=30); summary['cleanup'].append({'container':name,'stop_exit':p.returncode})
    summary['finished_utc']=timestamp(); summary['actual_elapsed_seconds']=round(time.monotonic()-start,3) if start else 0
    if (ROOT/'api.log').exists(): summary['api_log_tail']=(ROOT/'api.log').read_text(errors='replace')[-6000:]
    summary_path.write_text(json.dumps(summary,indent=2)+'\n')
    shutil.rmtree(ROOT,ignore_errors=True)
    print('OUTAGE_RESULT',summary['state'],flush=True)
