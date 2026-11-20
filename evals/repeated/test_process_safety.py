"""Local-only tests for listener ownership and watchdog child termination."""

import socket
import subprocess
import sys
import time
from pathlib import Path

with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
server = subprocess.Popen(
    [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
check = Path(__file__).with_name("owns_port.py")
try:
    for _ in range(100):
        if (
            subprocess.run(
                [sys.executable, str(check), str(server.pid), str(port)], stdout=subprocess.DEVNULL
            ).returncode
            == 0
        ):
            break
        time.sleep(0.05)
    else:
        raise AssertionError("server did not own expected port")
    assert (
        subprocess.run(
            [sys.executable, str(check), str(server.pid), str(port + 1)], stdout=subprocess.DEVNULL
        ).returncode
        != 0
    )
    assert (
        subprocess.run(
            [sys.executable, str(check), str(__import__("os").getpid()), str(port)],
            stdout=subprocess.DEVNULL,
        ).returncode
        != 0
    )
finally:
    server.terminate()
    server.wait(timeout=5)

# Emulate the watchdog's direct-child termination while the shell waits in a foreground command.
script = """main_pid=$$
( sleep 0.5; for child in $(pgrep -P "$main_pid" || true); do
  if [ "$child" != "$BASHPID" ]; then kill -TERM "$child" 2>/dev/null || true; fi
done; kill -TERM "$main_pid" 2>/dev/null || true ) &
sleep 30
"""
started = time.monotonic()
subprocess.run(
    ["bash", "-c", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5
)
assert time.monotonic() - started < 5
print("listener identity and watchdog foreground termination passed")
