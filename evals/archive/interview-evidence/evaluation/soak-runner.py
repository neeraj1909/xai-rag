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
CHROMA = 'xai-soak-chroma-18169'
ES = 'xai-soak-es-19310'
DURATION = 75 * 60
ROOT = pathlib.Path(tempfile.mkdtemp(prefix='xai-isolated-soak-'))
SNAPSHOT = ROOT / 'snapshot'
SNAPSHOT.mkdir()
shutil.copytree(SOURCE / 'src', SNAPSHOT / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
manifest = {str(p.relative_to(SNAPSHOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((SNAPSHOT / 'src').rglob('*.py'))}
manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
archive = OUT / 'soak-source-snapshot.tar.gz'
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
records = OUT / 'soak-intervals.jsonl'
summary_path = OUT / 'soak-summary.json'
records.write_text('')
summary = {'state': 'starting', 'started_utc': datetime.now(timezone.utc).isoformat(), 'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip(), 'duration_target_seconds': DURATION, 'snapshot_manifest_sha256': manifest_hash, 'source_manifest': manifest, 'snapshot_archive_sha256': archive_hash, 'generation': 'loopback deterministic stub; no paid calls', 'ports': [18169, 19310, 18069, 18070], 'cleanup': []}
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
class MockHandler(BaseHTTPRequestHandler):
    def do_POST(self):
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
        if not port_free(port): raise RuntimeError(f'port {port} is in use')
    mem = psutil.virtual_memory(); disk = shutil.disk_usage('/')
    if mem.available < 2 * 1024**3 or disk.free < 5 * 1024**3: raise RuntimeError('insufficient free memory or disk for isolated run')
    summary['initial_resources'] = {'memory_available_bytes': mem.available, 'disk_free_bytes': disk.free, 'swap_free_bytes': psutil.swap_memory().free}
    chroma_image='chromadb/chroma:1.5.9@sha256:1e0b73a187a28757c572acba508c46f48c9e8b0acaf5c20e6d95cdedce1acdf6'
    es_image='docker.elastic.co/elasticsearch/elasticsearch:8.17.0@sha256:2f602552550869fb29b6fd5848c5118d3ef3a2e1d5d45802e3ab9088cb2de8e2'
    shell(['docker','run','-d','--rm','--name',CHROMA,'--memory','700m','--cpus','1','-p','127.0.0.1:18169:8000','-e','ANONYMIZED_TELEMETRY=false',chroma_image]); containers.append(CHROMA)
    shell(['docker','run','-d','--rm','--name',ES,'--memory','1200m','--cpus','2','-p','127.0.0.1:19310:9200','-e','discovery.type=single-node','-e','xpack.security.enabled=false','-e','cluster.routing.allocation.disk.threshold_enabled=false','-e','ES_JAVA_OPTS=-Xms384m -Xmx384m',es_image]); containers.append(ES)
    mock = ThreadingHTTPServer(('127.0.0.1',18070), MockHandler)
    threading.Thread(target=mock.serve_forever, daemon=True).start()
    env = {k:v for k,v in os.environ.items() if not (k.startswith('XAI_RAG_') or k.startswith('OPENAI_') or k.lower() in ('http_proxy','https_proxy','all_proxy'))}
    env.update({'PYTHONPATH':str(SNAPSHOT/'src'),'XAI_RAG_API_KEY':api_key,'XAI_RAG_OPENAI_API_KEY':'local-synthetic-test-key','XAI_RAG_LLM_BASE_URL':'http://127.0.0.1:18070/v1','XAI_RAG_CHROMA_HOST':'127.0.0.1','XAI_RAG_CHROMA_PORT':'18169','XAI_RAG_ELASTICSEARCH_URL':'http://127.0.0.1:19310','XAI_RAG_INGEST_ROOT':str(fixture),'XAI_RAG_OTEL_ENDPOINT':'','XAI_RAG_RATE_LIMIT_PER_MINUTE':'1000','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','ANONYMIZED_TELEMETRY':'false','DO_NOT_TRACK':'1','OTEL_SDK_DISABLED':'true','NO_PROXY':'127.0.0.1,localhost'})
    async with httpx.AsyncClient(trust_env=False, timeout=120) as client:
        await wait_http(client,'http://127.0.0.1:18169/api/v2/heartbeat'); await wait_http(client,'http://127.0.0.1:19310/')
        log=open(ROOT/'api.log','w')
        process=subprocess.Popen([str(SOURCE/'.venv/bin/python'),'-m','uvicorn','xai_rag.api.app:app','--host','127.0.0.1','--port','18069'],cwd=SNAPSHOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        await wait_http(client,'http://127.0.0.1:18069/ready')
        hdr={'X-API-Key':api_key}
        r=await client.post('http://127.0.0.1:18069/ingest',headers=hdr,params={'source_path':str(fixture),'strategy':'fixed'})
        if r.status_code != 200 or r.json().get('parity_consistent') is not True: raise RuntimeError(f'initial ingestion/parity failed: HTTP {r.status_code}')
        event('startup',ingestion=r.json(), note='real models and stores; generation is a local mock')
        start=time.monotonic(); summary['measurement_started_utc']=timestamp(); summary['state']='running'; summary_path.write_text(json.dumps(summary,indent=2)+'\n')
        async def health_loop():
            tick=0
            while time.monotonic()-start < DURATION:
                target=start+tick*5
                if target>time.monotonic(): await asyncio.sleep(target-time.monotonic())
                try:
                    before=time.monotonic(); r=await client.get('http://127.0.0.1:18069/ready',timeout=6)
                    event('readiness',status=r.status_code,latency_ms=round((time.monotonic()-before)*1000,2),body=r.json(),fault_window=paused)
                except Exception as e: event('readiness',error_type=type(e).__name__,fault_window=paused)
                tick+=1
        async def query_loop():
            tick=0
            while start+tick*300 < start+DURATION:
                target=start+tick*300
                if target>time.monotonic(): await asyncio.sleep(target-time.monotonic())
                try:
                    before=time.monotonic(); r=await client.post('http://127.0.0.1:18069/query',headers=hdr,json={'query':'What is the refund deadline?','top_k':1},timeout=90)
                    b=r.json(); ok=r.status_code==200 and '30 days' in b.get('answer','') and bool(b.get('sources')) and bool(b.get('retrieval_explanations'))
                    event('query',status=r.status_code,passed=ok,latency_ms=round((time.monotonic()-before)*1000,2),stages=b.get('stage_latency_ms'),answer=b.get('answer'),sources=b.get('sources'))
                except Exception as e: event('query',passed=False,error_type=type(e).__name__)
                tick+=1
        async def parity_loop():
            tick=1
            while tick*900 < DURATION:
                await asyncio.sleep(max(0,start+tick*900-time.monotonic()))
                try:
                    before=time.monotonic(); r=await client.post('http://127.0.0.1:18069/ingest',headers=hdr,params={'source_path':str(fixture),'strategy':'fixed'},timeout=90); b=r.json()
                    event('parity',status=r.status_code,passed=r.status_code==200 and b.get('parity_consistent') is True and b.get('total_chunks')==1,latency_ms=round((time.monotonic()-before)*1000,2),body=b)
                except Exception as e: event('parity',passed=False,error_type=type(e).__name__)
                tick+=1
        async def resources_loop():
            tick=0; low=0
            while tick*30 < DURATION:
                await asyncio.sleep(max(0,start+tick*30-time.monotonic()))
                vm=psutil.virtual_memory(); sw=psutil.swap_memory(); disk=shutil.disk_usage('/'); rss=psutil.Process(process.pid).memory_info().rss if process.poll() is None else None
                event('resource',memory_available_bytes=vm.available,swap_free_bytes=sw.free,disk_free_bytes=disk.free,api_rss_bytes=rss)
                low=low+1 if vm.available<1024**3 or disk.free<3*1024**3 else 0
                if process.poll() is not None or low>=3: raise RuntimeError('resource or API process stop gate triggered')
                if tick and tick%20==0: event('progress',minutes=tick/2,memory_available_bytes=vm.available,api_rss_bytes=rss)
                tick+=1
        async def fault():
            global paused
            await asyncio.sleep(max(0,start+1200-time.monotonic()))
            shell(['docker','pause',CHROMA]); paused=True; event('fault',action='pause_chroma')
            try: await asyncio.sleep(20)
            finally:
                shell(['docker','unpause',CHROMA]); paused=False; event('fault',action='unpause_chroma')
        await asyncio.gather(health_loop(),query_loop(),parity_loop(),resources_loop(),fault())
        summary['state']='completed'; event('stop',reason='target duration reached')
try:
    asyncio.run(main())
except BaseException as exc:
    summary['state']='failed'; summary['error']={'type':type(exc).__name__,'message':str(exc)}; event('failure',error_type=type(exc).__name__,message=str(exc))
finally:
    if paused:
        p=subprocess.run(['docker','unpause',CHROMA],capture_output=True,text=True,timeout=30); summary['cleanup'].append({'action':'unpause','exit':p.returncode})
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
    print('SOAK_RESULT',summary['state'],flush=True)
