#!/usr/bin/env python3
"""SCP paddle integration files to H100 via vm5."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FILES = [
    "ocr_benchmark_v2/cli.py",
    "ocr_benchmark_v2/src/paddle_env.py",
    "ocr_benchmark_v2/src/runners/paddle_runner.py",
    "ocr_benchmark_v2/src/runners/factory.py",
    "ocr_benchmark_v2/config/models.yaml",
    "ocr_benchmark_v2/config/rate_limits.yaml",
    "ocr_benchmark_v2/scripts/install_h100_paddle.sh",
    "ocr_benchmark_v2/scripts/run_h100_paddle.sh",
    "ocr_benchmark_v2/scripts/h100_env.sh",
]
REMOTE = "amul-vm5-ai-backend"
H100 = "aicloud@10.185.25.197"

with tempfile.TemporaryDirectory() as td:
    tdir = Path(td)
    # bundle
    import tarfile

    tgz = tdir / "paddle_files.tgz"
    with tarfile.open(tgz, "w:gz") as tar:
        for rel in FILES:
            p = ROOT / rel
            tar.add(p, arcname=rel)

    vm_tgz = "/tmp/paddle_files.tgz"
    subprocess.check_call(["scp", "-o", "BatchMode=yes", str(tgz), f"{REMOTE}:{vm_tgz}"])
    subprocess.check_call(["ssh", "-o", "BatchMode=yes", REMOTE,
                           "scp", "-o", "BatchMode=yes", vm_tgz, f"{H100}:{vm_tgz}"])

cmds = "\n".join([
    "set -e",
    "cd ~/amul-oan-api",
    "tar xzf /tmp/paddle_files.tgz",
    "for f in ocr_benchmark_v2/scripts/*.sh; do sed -i 's/\\r$//' \"$f\"; done",
    "test -f ocr_benchmark_v2/src/runners/paddle_runner.py",
    "grep paddleocr-vl ocr_benchmark_v2/config/models.yaml",
    "echo FILES_OK",
])
p = subprocess.run(
    ["ssh", "-o", "BatchMode=yes", REMOTE, "ssh", "-o", "BatchMode=yes", H100, "bash", "-s"],
    input=cmds.encode(),
    capture_output=True,
)
print(p.stdout.decode())
if p.returncode:
    print(p.stderr.decode())
    raise SystemExit(p.returncode)
