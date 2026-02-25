"""
RunPod Serverless handler for PersonaPlex Coffee.

Wraps the existing Moshi server (start.sh) and exposes its connection
details via progress_update so clients can connect via WebSocket.
"""

import os
import subprocess
import time

import requests
import runpod

MOSHI_PORT = 8998
HEALTH_URL = f"http://localhost:{MOSHI_PORT}/"
STARTUP_TIMEOUT_SECS = 300  # 5 minutes max for model loading
SESSION_TIMEOUT_SECS = 900  # 15 minutes max session length


def _wait_for_moshi(timeout: int) -> bool:
    """Poll localhost until Moshi HTTP endpoint responds."""
    for _ in range(timeout):
        try:
            r = requests.get(HEALTH_URL, timeout=2)
            if r.status_code < 500:
                return True
        except requests.ConnectionError:
            pass
        except requests.Timeout:
            pass
        time.sleep(1)
    return False


def handler(job):
    """Start Moshi, share connection details, block until session ends."""

    # 1. Launch Moshi server via existing start.sh
    proc = subprocess.Popen(
        ["bash", "/app/start.sh"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # 2. Wait for Moshi to be ready
    if not _wait_for_moshi(STARTUP_TIMEOUT_SECS):
        proc.terminate()
        return {"error": "Moshi server failed to start within 5 minutes"}

    # 3. Read exposed TCP port details from RunPod environment
    public_ip = os.environ.get("RUNPOD_PUBLIC_IP", "")
    tcp_port = os.environ.get(f"RUNPOD_TCP_PORT_{MOSHI_PORT}", "")

    if not public_ip or not tcp_port:
        proc.terminate()
        return {
            "error": (
                "Exposed TCP port not configured. "
                "Enable 'Expose TCP Ports' and add port 8998 in endpoint settings."
            )
        }

    # 4. Share connection details via progress_update
    runpod.serverless.progress_update(job, {
        "status": "ready",
        "ip": public_ip,
        "port": tcp_port,
        "wsUrl": f"{public_ip}:{tcp_port}",
    })

    # 5. Block until Moshi process exits or session times out
    try:
        proc.wait(timeout=SESSION_TIMEOUT_SECS)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait(timeout=10)
        return {"status": "timeout", "message": "Session exceeded 15 minute limit"}

    return {"status": "completed"}


runpod.serverless.start({"handler": handler})