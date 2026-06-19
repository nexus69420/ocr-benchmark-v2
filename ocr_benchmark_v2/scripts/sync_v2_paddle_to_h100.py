#!/usr/bin/env python3
"""Stream-sync ocr_benchmark_v2 tarball to H100 (single SSH pipe)."""
from __future__ import annotations

import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V2 = ROOT / "evals" / "ocr_benchmark_v2"
REMOTE_TGZ = "/tmp/v2_paddle_sync.tgz"

with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as tmp:
    tgz = Path(tmp.name)

with tarfile.open(tgz, "w:gz") as tar:
    init = ROOT / "evals" / "__init__.py"
    tar.add(init, arcname="evals/__init__.py")
    for path in V2.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            tar.add(path, arcname=str(path.relative_to(ROOT)).replace("\\", "/"))

size = tgz.stat().st_size
print(f"tarball {size} bytes")

remote = ["ssh", "-o", "BatchMode=yes", "amul-vm5-ai-backend",
          "ssh", "-o", "BatchMode=yes", "aicloud@10.185.25.197",
          f"cat > {REMOTE_TGZ}"]

with tgz.open("rb") as f:
    p = subprocess.run(remote, stdin=f, capture_output=True)
if p.returncode != 0:
    print(p.stderr.decode())
    raise SystemExit(p.returncode)

extract = "\n".join([
    "set -e",
    "cd ~/amul-oan-api",
    f"ls -l {REMOTE_TGZ}",
    f"tar xzf {REMOTE_TGZ}",
    "for f in ocr_benchmark_v2/scripts/*.sh; do sed -i 's/\\r$//' \"$f\"; done",
    "grep paddleocr-vl ocr_benchmark_v2/config/models.yaml",
    "ls ocr_benchmark_v2/src/runners/paddle_runner.py",
    "echo SYNC_OK",
])
p2 = subprocess.run(
    ["ssh", "-o", "BatchMode=yes", "amul-vm5-ai-backend",
     "ssh", "-o", "BatchMode=yes", "aicloud@10.185.25.197", "bash", "-s"],
    input=extract.encode(),
    capture_output=True,
)
print(p2.stdout.decode())
print(p2.stderr.decode())
tgz.unlink(missing_ok=True)
raise SystemExit(p2.returncode)
