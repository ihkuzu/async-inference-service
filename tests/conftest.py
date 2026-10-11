import shutil
import socket
import subprocess
import time

import pytest
import pytest_asyncio

from inferq.backends import make_backend


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def redis_url():
    if not shutil.which("redis-server"):
        pytest.skip("redis-server not installed")
    port = _free_port()
    proc = subprocess.Popen(
        ["redis-server", "--port", str(port), "--save", "", "--appendonly", "no"],
        stdout=subprocess.DEVNULL,
    )
    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
            break
        except OSError:
            time.sleep(0.1)
    yield f"redis://127.0.0.1:{port}/0"
    proc.terminate()
    proc.wait(timeout=5)


@pytest_asyncio.fixture(params=["memory", "redis"])
async def backend(request):
    if request.param == "memory":
        b = make_backend("memory")
    else:
        url = request.getfixturevalue("redis_url")
        b = make_backend("redis", url, worker_id="test")
        await b.r.flushdb()
    yield b
    await b.close()
