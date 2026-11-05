"""🗜 (DECISION_LOG 475) the dashboard + API responses go out gzip-compressed; the live terminal stream never does.
Hermetic: the middleware class is lifted from main.py's source and mounted on a tiny Starlette app (main.py itself is not imported —
it boots the trading engine)."""
import os

from starlette.applications import Starlette
from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import HTMLResponse, StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()


def _middleware_cls():
    src = MAIN[MAIN.index("class _GZipExceptStreams("):MAIN.index("app.add_middleware(_GZipExceptStreams")]
    ns = {"GZipMiddleware": GZipMiddleware}
    exec(src, ns)
    return ns["_GZipExceptStreams"]


def _app():
    big = "<html>" + "dashboard row " * 5000 + "</html>"

    async def page(request):
        return HTMLResponse(big)

    async def tiny(request):
        return HTMLResponse("ok")

    async def stream(request):
        async def gen():
            for i in range(3):
                yield f"data: {'x' * 600} {i}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    async def other_stream(request):          # a stream on another path, labelled as plain text: name-based skip is path-only
        async def gen():
            yield "a" * 2000
        return StreamingResponse(gen(), media_type="text/csv")

    app = Starlette(routes=[Route("/page", page), Route("/tiny", tiny), Route("/api/terminal/stream", stream), Route("/csv", other_stream)])
    app.add_middleware(_middleware_cls(), minimum_size=1000, compresslevel=6)
    return app


def test_page_is_compressed_and_much_smaller():
    c = TestClient(_app())
    r = c.get("/page", headers={"Accept-Encoding": "gzip"})
    assert r.headers.get("content-encoding") == "gzip"
    assert int(r.headers["content-length"]) < len(r.text) / 5
    assert r.text.startswith("<html>dashboard row")          # the client unzips to the same page


def test_small_bodies_and_clients_without_gzip_are_untouched():
    c = TestClient(_app())
    assert "content-encoding" not in c.get("/tiny", headers={"Accept-Encoding": "gzip"}).headers
    assert "content-encoding" not in c.get("/page", headers={"Accept-Encoding": "identity"}).headers


def test_live_terminal_stream_is_never_compressed():
    c = TestClient(_app())
    r = c.get("/api/terminal/stream", headers={"Accept-Encoding": "gzip"})
    assert "content-encoding" not in r.headers and r.text.count("data: ") == 3
    assert c.get("/csv", headers={"Accept-Encoding": "gzip"}).headers.get("content-encoding") == "gzip"   # downloads still shrink


def test_wiring_in_main():
    assert "from starlette.middleware.gzip import GZipMiddleware" in MAIN
    assert '@app.get("/api/terminal/stream")' in MAIN
    assert '"/api/terminal/stream"' in MAIN[MAIN.index("class _GZipExceptStreams("):]
    assert MAIN.count('media_type="text/event-stream"') == 1           # a new live stream must be added to STREAM_PATHS
    # outermost: added after the session + auth layers
    assert MAIN.index("app.add_middleware(_GZipExceptStreams") > MAIN.index("app.add_middleware(\n    SessionMiddleware")


def test_dashboard_is_precompressed_once_and_served_as_is(tmp_path, monkeypatch):
    import gzip
    src = MAIN[MAIN.index("_PAGE_GZ = {"):MAIN.index('@app.get("/", response_class=HTMLResponse)')]
    route = MAIN[MAIN.index('@app.get("/", response_class=HTMLResponse)'):]
    route = route[:route.index("\n\n\n")].split("\n", 1)[1]                        # drop the decorator
    from starlette.requests import Request
    ns = {"os": os, "HTMLResponse": HTMLResponse, "Request": Request}
    exec(src + "\n" + route, ns)
    (tmp_path / "templates").mkdir()
    page = ("<html>" + "row " * 20000 + "</html>").encode()
    (tmp_path / "templates" / "index.html").write_bytes(page)
    monkeypatch.chdir(tmp_path)
    calls = []
    real = gzip.compress
    monkeypatch.setattr(gzip, "compress", lambda *a, **k: calls.append(1) or real(*a, **k))
    app = Starlette(routes=[Route("/", ns["root"]), Route("/x.png", lambda r: HTMLResponse("p" * 5000))])
    app.add_middleware(_middleware_cls(), minimum_size=1000, compresslevel=6)
    c = TestClient(app)
    for _ in range(3):
        r = c.get("/", headers={"Accept-Encoding": "gzip"})
        assert r.headers["content-encoding"] == "gzip" and r.content == page            # one encoding layer, unzips to the page
    assert len(calls) == 1                                                             # compressed once, not per reload
    assert "content-encoding" not in c.get("/", headers={"Accept-Encoding": "identity"}).headers
    assert "content-encoding" not in c.get("/x.png", headers={"Accept-Encoding": "gzip"}).headers   # already-compressed types skipped
