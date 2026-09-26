"""Real-server smoke test: start `streamlit run app/streamlit_app.py` exactly as Docker / Vercel do, open a real WebSocket session and run
EVERY page, failing on any Python exception. Streamlit's AppTest cannot catch import-path problems (it adds the project root itself), which is how
`ModuleNotFoundError: No module named 'app'` reached production once."""
import asyncio
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from tests._paths import ROOT

pytestmark = pytest.mark.integration


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_pages(port: int, timeout: float = 60.0) -> dict:
    """Connect to the live server, run the main page and every other page; return {page_name: [exception messages]}."""
    from streamlit.proto import BackMsg_pb2, ForwardMsg_pb2
    url = f"ws://127.0.0.1:{port}/_stcore/stream"

    def send_rerun(page_hash: str = "") -> bytes:
        m = BackMsg_pb2.BackMsg()
        m.rerun_script.query_string = ""
        if page_hash:
            m.rerun_script.page_script_hash = page_hash
        return m.SerializeToString()

    def decode(raw) -> ForwardMsg_pb2.ForwardMsg:
        f = ForwardMsg_pb2.ForwardMsg()
        f.ParseFromString(raw)
        return f

    def exception_text(f) -> str:
        if f.WhichOneof("type") == "delta" and f.delta.WhichOneof("type") == "new_element":
            el = f.delta.new_element
            if el.WhichOneof("type") == "exception":
                return f"{el.exception.type}: {el.exception.message}"
        return ""

    results: dict = {}

    async def with_tornado():
        import tornado.websocket
        conn = await tornado.websocket.websocket_connect(url, subprotocols=["streamlit"])
        async def next_msg():
            raw = await asyncio.wait_for(conn.read_message(), timeout)
            return None if raw is None else decode(raw)
        await drive(conn.write_message, next_msg)

    async def with_websockets():
        from websockets.asyncio.client import connect
        async with connect(url, subprotocols=["streamlit"]) as conn:
            async def next_msg():
                return decode(await asyncio.wait_for(conn.recv(), timeout))
            async def write(data, binary=True):
                await conn.send(data)
            await drive(write, next_msg)

    async def drive(write, next_msg):
        pages, queue = [], [""]
        first = True
        while queue:
            h = queue.pop(0)
            await write(send_rerun(h), binary=True)
            name, errors = "?", []
            while True:
                f = await next_msg()
                if f is None:
                    raise RuntimeError("server closed the WebSocket")
                kind = f.WhichOneof("type")
                if kind == "navigation" and first:                      # the page list arrives in the navigation message
                    first = False
                    pages = [(p.page_script_hash, p.page_name or p.url_pathname or "home") for p in f.navigation.app_pages]
                    queue = [ph for ph, _ in pages[1:]]
                if kind == "navigation":
                    name = next((n for ph, n in pages if ph == (h or pages[0][0])), name)
                if exception_text(f):
                    errors.append(exception_text(f))
                if kind == "script_finished":
                    break
            results[name or h] = errors

    try:
        import websockets  # noqa: F401
        asyncio.run(with_websockets())
    except ImportError:
        asyncio.run(with_tornado())
    return results


def test_every_page_runs_on_a_real_streamlit_server(tmp_path):
    port = _free_port()
    env = {**os.environ, "STREAMLIT_SERVER_HEADLESS": "true", "STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION": "false",
           "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false", "PHARMAGUARD_OBS_DB": str(tmp_path / "o.db"),
           "PHARMAGUARD_DB_PATH": str(tmp_path / "r.db"), "PHARMAGUARD_LOG_DIR": str(tmp_path / "logs")}
    log = open(tmp_path / "server.log", "w")
    # Use the `streamlit` console script (as the container's CMD does): `python -m streamlit` would add the cwd to sys.path and hide import bugs.
    exe = Path(sys.executable).parent / "streamlit"
    srv = subprocess.Popen([str(exe), "run", "app/streamlit_app.py", "--server.port", str(port), "--server.address", "127.0.0.1"],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(60):
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2).status == 200:
                    break
            except OSError:
                time.sleep(0.5)
        else:
            pytest.fail("server did not become healthy")
        try:
            results = _run_pages(port)
        except ImportError:
            pytest.skip("neither websockets nor tornado is available")
    finally:
        srv.terminate()
        srv.wait(timeout=10)
        log.close()
    server_log = (tmp_path / "server.log").read_text()
    assert "Uncaught app execution" not in server_log and "ModuleNotFoundError" not in server_log, server_log[-2000:]
    assert len(results) >= 9, f"expected the home page and 8 pages, ran: {list(results)}"
    assert not {k: v for k, v in results.items() if v}, results
