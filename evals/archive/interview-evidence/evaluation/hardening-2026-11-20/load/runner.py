"""Bounded c375903 local load diagnostic; no paid provider calls."""
from __future__ import annotations

import asyncio
import hashlib
import html
import json
import math
import os
import pathlib
import re
import secrets
import shutil
import socket
import subprocess
import tarfile
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import psutil

BASE = pathlib.Path('/tmp/xai-perf-baseline')
SOURCE = pathlib.Path('/home/neeraj/Documents/xai-rag')
ROOT = BASE / 'run'
SNAPSHOT = ROOT / 'snapshot'
DOCS = ROOT / 'documents'
EVENTS = BASE / 'events.jsonl'
SUMMARY = BASE / 'summary.json'
PORTS = {'chroma': 18261, 'es': 19461, 'api': 18081, 'stub': 18082}
TOKEN = secrets.token_hex(32)
SUFFIX = secrets.token_hex(4)
CHROMA = f'xai-perf-chroma-{SUFFIX}'
ES = f'xai-perf-es-{SUFFIX}'
containers = []
api = None
stub = None
logfile = None
start = time.monotonic()
stop = asyncio.Event()
stop_reason = None
consecutive_failures = 0
summary = {'state': 'starting', 'git_commit': 'c375903fb46327744d076438cb90f7b02711a94f', 'ports': PORTS, 'results': {}, 'cleanup': [], 'provider': 'synthetic loopback stub, no paid calls', 'concurrency': [1, 2, 4, 8], 'requests_per_level': 30, 'corpus_documents': 100, 'rate_limit_per_minute': 1000, 'max_concurrent_queries': 4, 'start_utc': datetime.now(timezone.utc).isoformat()}

def event(kind, **kwargs):
    row = {'kind': kind, 'utc': datetime.now(timezone.utc).isoformat(), 'elapsed_s': round(time.monotonic() - start, 3), **kwargs}
    with EVENTS.open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n')

def command(args, timeout=60):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'{args[0]} exited {result.returncode}: {result.stderr[:400]}')
    return result.stdout.strip()

def resources():
    return {'memory_available': psutil.virtual_memory().available, 'swap_free': psutil.swap_memory().free, 'disk_free': shutil.disk_usage('/').free, 'api_rss': psutil.Process(api.pid).memory_info().rss if api and api.poll() is None else None}

def check_resources(r):
    if r['memory_available'] < 3*1024**3: return 'memory_available_below_3gib'
    if r['swap_free'] < 1024**3: return 'swap_free_below_1gib'
    if r['disk_free'] < 10*1024**3: return 'disk_free_below_10gib'
    if r['api_rss'] is not None and r['api_rss'] > 4*1024**3: return 'api_rss_above_4gib'
    if api and api.poll() is not None: return 'api_exited'
    return None

class Stub(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            if self.path != '/v1/chat/completions':
                self.send_error(404); return
            req = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            prompt = req['messages'][0]['content']
            m = re.search(r'<retrieved_chunk id="([^"]+)">\n(.*?)\n</retrieved_chunk>', prompt, re.S)
            if not m: self.send_error(400); return
            cid, content = m.group(1), html.unescape(m.group(2)).strip()
            claim = content[:180].replace('[', '(').replace(']', ')')
            payload = json.dumps({'answer': f'{claim} [{cid}]', 'claims': [{'text': claim, 'source_chunk_ids': [cid]}], 'sources': [cid]})
            body = json.dumps({'id':'local-stub', 'object':'chat.completion', 'created':0, 'model':'local-stub', 'choices':[{'index':0,'message':{'role':'assistant','content':payload},'finish_reason':'stop'}], 'usage':{'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}}).encode()
            self.send_response(200); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
        except Exception:
            self.send_error(500)
    def log_message(self, *_): pass

async def watchdog():
    global stop_reason
    while not stop.is_set():
        r = resources(); event('resource', **r)
        reason = check_resources(r)
        if reason:
            stop_reason = reason; event('stop', reason=reason); stop.set(); return
        try: await asyncio.wait_for(stop.wait(), 5)
        except TimeoutError: pass

async def wait_http(client, url, timeout=180):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        if stop.is_set(): raise RuntimeError(f'watchdog: {stop_reason}')
        if api and api.poll() is not None: raise RuntimeError('api exited during startup')
        try:
            r=await client.get(url,timeout=3)
            if r.status_code == 200: return
        except httpx.HTTPError: pass
        await asyncio.sleep(1)
    raise TimeoutError('dependency startup timeout')

def percentile(values, p):
    return sorted(values)[max(0, math.ceil(p*len(values))-1)] if values else None

async def run():
    global api, stub, logfile, stop_reason
    if ROOT.exists(): raise RuntimeError('run directory already exists; inspect it before rerun')
    for port in PORTS.values():
        with socket.socket() as sock:
            try: sock.bind(('127.0.0.1', port))
            except OSError as exc: raise RuntimeError(f'test port {port} unavailable') from exc
    reason=check_resources(resources())
    if reason: raise RuntimeError(reason)
    SNAPSHOT.mkdir(parents=True); DOCS.mkdir()
    archive = BASE/'source-c375903.tar'
    with archive.open('wb') as f:
        subprocess.run(['git','archive','c375903'],cwd=SOURCE,stdout=f,check=True)
    with tarfile.open(archive) as tar: tar.extractall(SNAPSHOT,filter='data')
    summary['source_archive_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
    for i in range(100):
        (DOCS/f'policy-{i:03}.txt').write_text(f'Policy item P{i:03} belongs to category C{i//10:02}. For identifier P{i:03}, the authorized response deadline is {10+i} days. Contact the fictional support desk for policy P{i:03}.\n')
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DOCS.iterdir())}
    summary['fixture_sha256']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    event('start', resources=resources())
    chroma='chromadb/chroma:1.5.9@sha256:1e0b73a187a28757c572acba508c46f48c9e8b0acaf5c20e6d95cdedce1acdf6'
    es='docker.elastic.co/elasticsearch/elasticsearch:8.17.0@sha256:2f602552550869fb29b6fd5848c5118d3ef3a2e1d5d45802e3ab9088cb2de8e2'
    command(['docker','run','--rm','-d','--name',CHROMA,'--memory','700m','--cpus','1','-p',f"127.0.0.1:{PORTS['chroma']}:8000",'-e','ANONYMIZED_TELEMETRY=false',chroma]);containers.append(CHROMA)
    command(['docker','run','--rm','-d','--name',ES,'--memory','1200m','--cpus','2','-p',f"127.0.0.1:{PORTS['es']}:9200",'-e','discovery.type=single-node','-e','xpack.security.enabled=false','-e','cluster.routing.allocation.disk.threshold_enabled=false','-e','ES_JAVA_OPTS=-Xms384m -Xmx384m',es]);containers.append(ES)
    stub=ThreadingHTTPServer(('127.0.0.1',PORTS['stub']),Stub);threading.Thread(target=stub.serve_forever,daemon=True).start()
    env={k:v for k,v in os.environ.items() if not(k.startswith('XAI_RAG_') or k.startswith('OPENAI_') or k.lower() in ('http_proxy','https_proxy','all_proxy'))}
    env.update({'PYTHONPATH':str(SNAPSHOT/'src'),'XAI_RAG_API_KEY':TOKEN,'XAI_RAG_OPENAI_API_KEY':'local-synthetic-key','XAI_RAG_LLM_BASE_URL':f"http://127.0.0.1:{PORTS['stub']}/v1",'XAI_RAG_CHROMA_HOST':'127.0.0.1','XAI_RAG_CHROMA_PORT':str(PORTS['chroma']),'XAI_RAG_ELASTICSEARCH_URL':f"http://127.0.0.1:{PORTS['es']}",'XAI_RAG_INGEST_ROOT':str(DOCS),'XAI_RAG_OTEL_ENDPOINT':'','XAI_RAG_RATE_LIMIT_PER_MINUTE':'1000','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','ANONYMIZED_TELEMETRY':'false','DO_NOT_TRACK':'1','OTEL_SDK_DISABLED':'true','NO_PROXY':'127.0.0.1,localhost'})
    watchdog_task=asyncio.create_task(watchdog())
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=180, limits=httpx.Limits(max_connections=20, max_keepalive_connections=8)) as client:
            await wait_http(client,f"http://127.0.0.1:{PORTS['chroma']}/api/v2/heartbeat")
            await wait_http(client,f"http://127.0.0.1:{PORTS['es']}/")
            logfile=(BASE/'api.log').open('w')
            api=subprocess.Popen([str(SOURCE/'.venv/bin/python'),'-m','uvicorn','xai_rag.api.app:app','--host','127.0.0.1','--port',str(PORTS['api']),'--no-access-log'],cwd=SNAPSHOT,env=env,stdout=logfile,stderr=subprocess.STDOUT)
            await wait_http(client,f"http://127.0.0.1:{PORTS['api']}/ready")
            headers={'X-API-Key':TOKEN}; root=f"http://127.0.0.1:{PORTS['api']}"
            t=time.monotonic(); resp=await client.post(root+'/ingest',headers=headers,params={'source_path':str(DOCS),'strategy':'fixed'},timeout=600)
            data=resp.json(); event('ingest',status=resp.status_code,elapsed_s=round(time.monotonic()-t,3),body=data)
            if resp.status_code!=200 or data.get('ingested_files')!=100 or data.get('parity_consistent') is not True: raise RuntimeError('fixture ingestion or parity failed')
            async def query(i, phase, concurrency):
                global consecutive_failures, stop_reason
                if stop.is_set(): return {'error':'cancelled_by_watchdog'}
                qid=(i*37)%100; before=time.monotonic()
                try:
                    r=await client.post(root+'/query',headers=headers,json={'query':f'For policy P{qid:03}, what is the response deadline?', 'top_k':5, 'include_explanations':True,'include_faithfulness':False},timeout=150)
                    payload=r.json(); previews=[x.get('content_preview','') for x in payload.get('retrieval_explanations',[])]
                    row={'phase':phase,'concurrency':concurrency,'case':qid,'status':r.status_code,'latency_s':round(time.monotonic()-before,3),'hit_at_5':any(f'P{qid:03}' in p for p in previews),'sources':len(payload.get('sources',[])),'stage_latency_ms':payload.get('stage_latency_ms')}
                except Exception as exc:
                    row={'phase':phase,'concurrency':concurrency,'case':qid,'error':type(exc).__name__,'latency_s':round(time.monotonic()-before,3)}
                if row.get('status', 0) >= 500 or 'error' in row or row.get('status') == 429:
                    consecutive_failures += 1
                else:
                    consecutive_failures = 0
                if consecutive_failures >= 3:
                    stop_reason='three_consecutive_request_failures'; stop.set()
                event('query',**row); return row
            for i in range(5):
                row=await query(i,'warmup',1)
                if row.get('status')!=200: raise RuntimeError('warmup failed')
            for concurrency in (1,2,4,8):
                if stop.is_set(): break
                sem=asyncio.Semaphore(concurrency)
                async def limited(i):
                    async with sem: return await query(i,'measured',concurrency)
                t=time.monotonic(); rows=await asyncio.gather(*(limited(i) for i in range(30))); duration=time.monotonic()-t
                ok=[x for x in rows if x.get('status')==200]; latencies=[x['latency_s'] for x in ok]
                result={'attempted':len(rows),'success':len(ok),'errors':len(rows)-len(ok),'throughput_success_per_s':round(len(ok)/duration,3),'elapsed_s':round(duration,3),'p50_success_s':percentile(latencies,.5),'p95_success_s':percentile(latencies,.95),'max_success_s':max(latencies,default=None),'hit_at_5':sum(x.get('hit_at_5',False) for x in ok),'rows':rows}
                summary['results'][str(concurrency)]=result; SUMMARY.write_text(json.dumps(summary,indent=2)+'\n'); event('level_done',**{k:v for k,v in result.items() if k!='rows'},concurrency=concurrency)
                print(f'level {concurrency}: {len(ok)}/30 successful',flush=True)
                if len(ok)<27: stop_reason='level_failure_threshold';stop.set();break
            if not stop.is_set(): summary['state']='completed'
            else: summary['state']='stopped';summary['stop_reason']=stop_reason
    finally:
        stop.set();await watchdog_task

if __name__=='__main__':
    EVENTS.write_text('')
    try:
        asyncio.run(asyncio.wait_for(run(),timeout=1800))
    except BaseException as exc:
        summary['state']='failed';summary['error']={'type':type(exc).__name__,'message':str(exc)};event('failure',**summary['error'])
    finally:
        if api:
            api.terminate()
            try: api.wait(timeout=15)
            except subprocess.TimeoutExpired: api.kill();api.wait()
        if logfile: logfile.close()
        if stub: stub.shutdown();stub.server_close()
        for name in reversed(containers):
            try:
                p=subprocess.run(['docker','stop','-t','5',name],capture_output=True,text=True,timeout=30)
                summary['cleanup'].append({'container':name,'exit':p.returncode})
            except Exception as exc: summary['cleanup'].append({'container':name,'error':type(exc).__name__})
        summary['finished_utc']=datetime.now(timezone.utc).isoformat();summary['elapsed_s']=round(time.monotonic()-start,3)
        SUMMARY.write_text(json.dumps(summary,indent=2)+'\n')
        print('FINAL',summary['state'],flush=True)
