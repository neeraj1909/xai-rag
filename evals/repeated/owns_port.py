"""Confirm the launched API process owns the expected loopback listener."""

import sys

import psutil

try:
    pid, port = map(int, sys.argv[1:])
    sockets = psutil.Process(pid).net_connections(kind="inet")
    ok = any(
        s.status == psutil.CONN_LISTEN and s.laddr.ip == "127.0.0.1" and s.laddr.port == port
        for s in sockets
    )
except (psutil.Error, ValueError):
    ok = False
raise SystemExit(0 if ok else 1)
