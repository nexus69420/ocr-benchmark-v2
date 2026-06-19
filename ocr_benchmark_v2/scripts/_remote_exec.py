#!/usr/bin/env python3
"""Run bash on H100 via vm5 jump (avoids PowerShell quoting)."""
from __future__ import annotations

import subprocess
import sys

REMOTE = "ssh -o BatchMode=yes amul-vm5-ai-backend ssh -o BatchMode=yes aicloud@10.185.25.197 bash -s"
script = sys.stdin.read() if not sys.argv[1:] else "\n".join(sys.argv[1:])
p = subprocess.run(REMOTE.split(), input=script.encode(), capture_output=True)
sys.stdout.buffer.write(p.stdout)
sys.stderr.buffer.write(p.stderr)
raise SystemExit(p.returncode)
