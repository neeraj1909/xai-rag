"""Exercise readiness on an already-started, disposable local Compose project."""
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
BASE = 'http://127.0.0.1:18143'
PROJECT = 'xai-ready-20261122'
results = []

def probe(path):
    try:
        with urllib.request.urlopen(BASE + path, timeout=8) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)
    except (OSError, TimeoutError) as exc:
        return None, {'error': type(exc).__name__}

def wait_for(name, predicate, timeout=240):
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        status, body = probe('/ready')
        if predicate(status, body):
            live_code, live = probe('/live')
            result = {'stage': name, 'elapsed_seconds': round(time.monotonic() - started, 3), 'ready_status': status, 'ready': body, 'live_status': live_code, 'live': live}
            results.append(result)
            (ROOT / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
            print(json.dumps(result), flush=True)
            assert live_code == 200 and live.get('status') == 'alive'
            return
        time.sleep(3)
    raise RuntimeError(f'{name} timed out: {status}, {body}')

def docker(*args):
    subprocess.run(['docker', *args], check=True, timeout=60)

def healthy(code, body):
    return code == 200 and body.get('status') == 'healthy' and body.get('checks') == {'chromadb': 'ok', 'elasticsearch': 'ok'}

def degraded(name):
    return lambda code, body: code == 503 and body.get('status') == 'degraded' and body.get('checks', {}).get(name) in {'error', 'unavailable'} and body.get('checks', {}).get('elasticsearch' if name == 'chromadb' else 'chromadb') == 'ok'

wait_for('initial_healthy', healthy)
for name in ('chromadb', 'elasticsearch'):
    container = f'{PROJECT}-{name}-1'
    docker('stop', container)
    wait_for(f'{name}_stopped', degraded(name))
    docker('start', container)
    wait_for(f'{name}_recovered', healthy)
print('PASS', flush=True)
