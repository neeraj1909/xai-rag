"""Exercise the Docker healthcheck against a real loopback HTTP endpoint."""

import json
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

DOCKERFILE = Path(__file__).resolve().parents[2] / "Dockerfile"


def _healthcheck():
    dockerfile = DOCKERFILE.read_text()
    outer_timeout = int(re.search(r"HEALTHCHECK .*?--timeout=(\d+)s", dockerfile).group(1))
    command = json.loads(
        next(
            line.split("CMD ", 1)[1]
            for line in dockerfile.splitlines()
            if 'CMD ["python", "-c"' in line
        )
    )
    inner_timeout = int(re.search(r"timeout=(\d+)\)", command[2]).group(1))
    return outer_timeout, inner_timeout, command


@pytest.fixture
def readiness_server():
    class Handler(BaseHTTPRequestHandler):
        delay = 0.0
        status = 200

        def do_GET(self):
            time.sleep(self.delay)
            try:
                self.send_response(self.status)
                self.end_headers()
                self.wfile.write(b"{}")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, Handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _run_check(server, command, timeout_override=None):
    code = command[2].replace("127.0.0.1:8000", f"127.0.0.1:{server.server_port}")
    if timeout_override is not None:
        code = re.sub(r"timeout=\d+\)", f"timeout={timeout_override})", code)
    assert command[0] == "python"
    return subprocess.run([sys.executable, command[1], code], capture_output=True, timeout=10)


def test_docker_healthcheck_allows_sequential_ready_budget(readiness_server):
    outer_timeout, inner_timeout, command = _healthcheck()
    # Readiness has two sequential 2-second budgets. The client and Docker
    # budgets must both allow that maximum plus some scheduling margin.
    server, handler = readiness_server
    handler.delay = 3.2
    assert _run_check(server, command).returncode == 0
    assert inner_timeout >= 6
    assert outer_timeout > inner_timeout


def test_docker_healthcheck_fails_for_unhealthy_and_hung_response(readiness_server):
    _, _, command = _healthcheck()
    server, handler = readiness_server
    handler.status = 503
    assert _run_check(server, command).returncode != 0
    handler.status = 200
    handler.delay = 0.2
    assert _run_check(server, command, timeout_override=0.05).returncode != 0


def test_docker_runtime_uses_extended_keepalive():
    command = json.loads(
        next(
            line.removeprefix("CMD ")
            for line in DOCKERFILE.read_text().splitlines()
            if line.startswith('CMD ["uvicorn"')
        )
    )
    assert command[command.index("--timeout-keep-alive") + 1] == "15"
