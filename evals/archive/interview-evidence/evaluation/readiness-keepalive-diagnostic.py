import asyncio
import json
import platform
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import fastapi
import httpx
import uvicorn

OUT = Path('/tmp/xai-ready-diagnostic-rerun/events.jsonl')
OUT.parent.mkdir(parents=True, exist_ok=True)
DURATION = 20 * 60
INTERVAL = 5.0
app = fastapi.FastAPI()

@app.get('/ready')
async def ready():
    return {'status': 'healthy'}

async def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('127.0.0.1', 0))
    sock.listen(128)
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level='error', access_log=False, timeout_keep_alive=5, lifespan='off'))
    server_task = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        if server_task.done():
            raise RuntimeError('local server failed to start')
        await asyncio.sleep(0.01)
    meta = {'event': 'start', 'ts': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(), 'httpx': httpx.__version__, 'uvicorn': uvicorn.__version__, 'fastapi': fastapi.__version__, 'port': port, 'interval_s': INTERVAL, 'duration_s': DURATION, 'uvicorn_keepalive_s': 5, 'proxy': False}
    with OUT.open('w', encoding='utf8', buffering=1) as f:
        def log(row):
            f.write(json.dumps(row) + '\n')
        log(meta)
        print(json.dumps(meta), flush=True)
        async def arm(name, headers):
            limits = httpx.Limits(max_connections=1, max_keepalive_connections=1, keepalive_expiry=60)
            start = time.monotonic()
            async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{port}', timeout=3, limits=limits, headers=headers, trust_env=False) as client:
                seq = 0
                while True:
                    delay = start + seq * INTERVAL - time.monotonic()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    now = time.monotonic()
                    if now - start > DURATION + 0.5:
                        break
                    row = {'event': 'request', 'arm': name, 'seq': seq, 'ts': datetime.now(timezone.utc).isoformat(), 'elapsed_s': round(now-start, 3)}
                    try:
                        response = await client.get('/ready')
                        row['status'] = response.status_code
                        row['ok'] = response.status_code == 200 and response.json().get('status') == 'healthy'
                    except Exception as exc:
                        row['ok'] = False
                        row['exception'] = type(exc).__name__
                    row['latency_ms'] = round((time.monotonic()-now)*1000, 2)
                    log(row)
                    if not row['ok']:
                        print(json.dumps(row), flush=True)
                    seq += 1
            return seq
        try:
            counts = await asyncio.gather(arm('pooled', {}), arm('connection_close', {'Connection':'close'}))
            log({'event':'finished','ts':datetime.now(timezone.utc).isoformat(),'counts':counts})
            print(json.dumps({'event':'finished','counts':counts}), flush=True)
        finally:
            server.should_exit = True
            await server_task
            sock.close()

asyncio.run(main())
