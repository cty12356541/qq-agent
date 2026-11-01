"""Wave W5-01 unit tests: low-latency parallel search race, HTTP connection pool reuse, async thread DNS SSRF guard, concurrent/threaded vision image normalization, and compact sticker/tool schemas."""

from __future__ import annotations

import asyncio
import base64
import io
import json
from pathlib import Path
import socket
import threading
import time
import tomllib
from typing import Any
import urllib.parse

import httpx
from PIL import Image
import pytest

from qq_agent.channel_bridge import DirectMemoryBus
import qq_agent.provider as provider_mod
from qq_agent.provider import fetch_vision_images, normalize_image_bytes_for_vision
from qq_agent.stickers import StickerRegistry
from qq_agent.tools.chat_tools import ChatTurnState, build_send_message_tool
from qq_agent.tools.web_search import (
    SsrfBlockedError,
    WebPageFetcher,
    WebSearchClient,
    validate_public_http_url,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STICKERS_MANIFEST = REPO_ROOT / "config" / "stickers.toml"


def _make_bing_html(hits: list[tuple[str, str, str]]) -> str:
    items = "\n".join(
        f'<li class="b_algo"><h2><a href="{url}">{title}</a></h2>'
        f'<div class="b_caption"><p>{snippet}</p></div></li>'
        for title, url, snippet in hits
    )
    return f'<html><body><ol id="b_results">{items}</ol></body></html>'


def _make_ddg_html(hits: list[tuple[str, str, str]]) -> str:
    blocks = []
    for title, url, snippet in hits:
        uddg = urllib.parse.quote(url, safe="")
        blocks.append(
            f'<div class="result">'
            f'<h2><a class="result__a" href="//duckduckgo.com/l/?uddg={uddg}">{title}</a></h2>'
            f'<div class="result__snippet">{snippet}</div>'
            f"</div>"
        )
    return f"<html><body>{''.join(blocks)}</body></html>"


def _make_large_rgb_png(size: tuple[int, int], color: tuple[int, int, int]) -> bytes:
    im = Image.new("RGB", size, color)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _make_large_animated_gif(size: tuple[int, int]) -> bytes:
    f1 = Image.new("RGB", size, (240, 60, 60))
    f2 = Image.new("RGB", size, (60, 200, 120))
    buf = io.BytesIO()
    f1.save(buf, format="GIF", save_all=True, append_images=[f2])
    return buf.getvalue()


def test_w5_parallel_search_race_cancels_slow_engine_and_merges_when_fast_insufficient() -> None:
    # Case 1: Slow Bing (2.0s) vs Fast DDG (0.02s with 2 hits) -> returns < 0.3s and cancels Bing cleanly
    bing_cancelled = False
    seen_bing_params: dict[str, str] = {}

    ddg_two_hits_html = _make_ddg_html(
        [
            ("Fast DDG Hit 1", "https://fast.example.org/hit-1", "First rapid result"),
            ("Fast DDG Hit 2", "https://fast.example.org/hit-2", "Second rapid result"),
        ]
    )
    bing_two_hits_html = _make_bing_html(
        [
            ("Fast Bing Hit 1", "https://bing-fast.example.org/1", "Bing primary hit 1"),
            ("Fast Bing Hit 2", "https://bing-fast.example.org/2", "Bing primary hit 2"),
        ]
    )

    async def _slow_bing_fast_ddg(request: httpx.Request) -> httpx.Response:
        nonlocal bing_cancelled, seen_bing_params
        if request.url.host == "www.bing.com":
            seen_bing_params = dict(request.url.params)
            try:
                await asyncio.sleep(2.0)
            except asyncio.CancelledError:
                bing_cancelled = True
                raise
            return httpx.Response(200, text=bing_two_hits_html)
        if request.url.host == "html.duckduckgo.com":
            await asyncio.sleep(0.02)
            return httpx.Response(200, text=ddg_two_hits_html)
        raise AssertionError(f"Unexpected host: {request.url}")

    # Case 2: Fast Bing (0.02s with 2 hits) vs Slow DDG (2.0s) -> returns < 0.3s and cancels DDG cleanly
    ddg_cancelled = False

    async def _fast_bing_slow_ddg(request: httpx.Request) -> httpx.Response:
        nonlocal ddg_cancelled
        if request.url.host == "www.bing.com":
            await asyncio.sleep(0.02)
            return httpx.Response(200, text=bing_two_hits_html)
        if request.url.host == "html.duckduckgo.com":
            try:
                await asyncio.sleep(2.0)
            except asyncio.CancelledError:
                ddg_cancelled = True
                raise
            return httpx.Response(200, text=ddg_two_hits_html)
        raise AssertionError(f"Unexpected host: {request.url}")

    # Case 3: Fast engine returns 0 results (0.01s) -> smoothly awaits slower engine (0.06s) and merges
    async def _fast_empty_slow_valid(request: httpx.Request) -> httpx.Response:
        if request.url.host == "www.bing.com":
            await asyncio.sleep(0.01)
            return httpx.Response(200, text="<html><body>No results</body></html>")
        if request.url.host == "html.duckduckgo.com":
            await asyncio.sleep(0.06)
            return httpx.Response(200, text=ddg_two_hits_html)
        raise AssertionError(f"Unexpected host: {request.url}")

    async def _run() -> None:
        assert WebSearchClient.BING_CN_URL == "https://www.bing.com/search"

        async with WebSearchClient(transport=httpx.MockTransport(_slow_bing_fast_ddg)) as c1:
            t0 = time.perf_counter()
            hits1 = await c1.search_items("asyncio 并发竞速", max_results=5)
            elapsed1 = time.perf_counter() - t0

        assert elapsed1 < 0.3, f"Expected <0.3s fast-path race, took {elapsed1:.3f}s"
        assert bing_cancelled is True
        assert seen_bing_params == {"q": "asyncio 并发竞速", "ensearch": "0", "mkt": "zh-CN"}
        assert [h.url for h in hits1] == [
            "https://fast.example.org/hit-1",
            "https://fast.example.org/hit-2",
        ]

        async with WebSearchClient(transport=httpx.MockTransport(_fast_bing_slow_ddg)) as c2:
            t0 = time.perf_counter()
            hits2 = await c2.search_items("必应优先竞速", max_results=5)
            elapsed2 = time.perf_counter() - t0

        assert elapsed2 < 0.3, f"Expected <0.3s fast-path race, took {elapsed2:.3f}s"
        assert ddg_cancelled is True
        assert [h.url for h in hits2] == [
            "https://bing-fast.example.org/1",
            "https://bing-fast.example.org/2",
        ]

        async with WebSearchClient(transport=httpx.MockTransport(_fast_empty_slow_valid)) as c3:
            hits3 = await c3.search_items("零结果兜底合并", max_results=5)

        assert [h.url for h in hits3] == [
            "https://fast.example.org/hit-1",
            "https://fast.example.org/hit-2",
        ]

    asyncio.run(_run())


def test_w5_persistent_http_pool_reuse_and_aclose_for_search_and_fetcher() -> None:
    bing_html = _make_bing_html(
        [
            ("Pool Hit 1", "https://pool.example.org/1", "Snippet 1"),
            ("Pool Hit 2", "https://pool.example.org/2", "Snippet 2"),
        ]
    )

    def _search_handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "www.bing.com":
            return httpx.Response(200, text=bing_html)
        return httpx.Response(200, text="<html></html>")

    def _page_handler(request: httpx.Request) -> httpx.Response:
        body = "<p>" + ("常驻连接池复用网页正文测试内容。" * 15) + "</p>"
        return httpx.Response(
            200,
            text=f"<html><head><title>Pool Page</title></head><body>{body}</body></html>",
            headers={"Content-Type": "text/html; charset=utf-8"},
        )

    async def _run() -> None:
        search_client = WebSearchClient(transport=httpx.MockTransport(_search_handler))
        assert search_client._owned_client is None

        res1 = await search_client.search_items("第一次查询")
        owned_first = search_client._owned_client
        assert len(res1) == 2
        assert owned_first is not None
        assert not owned_first.is_closed
        assert owned_first.follow_redirects is True

        res2 = await search_client.search_items("第二次查询")
        assert len(res2) == 2
        assert search_client._owned_client is owned_first

        await search_client.aclose()
        assert owned_first.is_closed is True
        assert search_client._owned_client is None

        # Injected external client must NOT be closed by WebSearchClient.aclose()
        ext_client = httpx.AsyncClient(transport=httpx.MockTransport(_search_handler))
        try:
            injected_search = WebSearchClient(client=ext_client)
            await injected_search.search_items("外部注入连接池")
            assert injected_search._owned_client is None
            await injected_search.aclose()
            assert ext_client.is_closed is False
        finally:
            await ext_client.aclose()

        # WebPageFetcher persistent pool reuse (keeping follow_redirects=False) + aclose()
        fetcher = WebPageFetcher(
            transport=httpx.MockTransport(_page_handler),
            resolver=lambda host, port: ["93.184.216.34"],
            enable_jina_fallback=False,
        )
        assert fetcher._owned_client is None
        out1 = await fetcher.fetch("https://docs.example.org/p1")
        f_pool = fetcher._owned_client
        assert "Pool Page" in out1
        assert f_pool is not None and not f_pool.is_closed
        assert f_pool.follow_redirects is False

        out2 = await fetcher.fetch("https://docs.example.org/p2")
        assert "Pool Page" in out2
        assert fetcher._owned_client is f_pool

        await fetcher.aclose()
        assert f_pool.is_closed is True
        assert fetcher._owned_client is None

    asyncio.run(_run())


def test_w5_validate_public_http_url_async_thread_dns_and_ssrf_interception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_tid = threading.get_ident()
    dns_calls: list[tuple[int, str, int, int, int]] = []

    def _fake_blocking_getaddrinfo(
        host: str,
        port: int,
        family: int = 0,
        socktype: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[tuple[Any, ...]]:
        tid = threading.get_ident()
        dns_calls.append((tid, host, port, family, socktype))
        time.sleep(0.06)
        if host == "rebind-loopback.example.com":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]
        if host == "rebind-private.example.com":
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.10.5", port)),
            ]
        if host == "nxdomain.example.com":
            raise socket.gaierror("Name or service not known")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

    monkeypatch.setattr(socket, "getaddrinfo", _fake_blocking_getaddrinfo)

    async def _run() -> None:
        loop_ticks = 0

        async def _heartbeat() -> None:
            nonlocal loop_ticks
            for _ in range(5):
                await asyncio.sleep(0.01)
                loop_ticks += 1

        hb_task = asyncio.create_task(_heartbeat())
        normalized = await validate_public_http_url("https://Public-Host.Example.COM./api/v1?x=1#top")
        await hb_task

        assert normalized == "https://Public-Host.Example.COM./api/v1?x=1"
        # Event loop kept ticking while blocking DNS ran in worker thread
        assert loop_ticks >= 3
        assert len(dns_calls) == 1
        called_tid, called_host, called_port, called_family, called_socktype = dns_calls[0]
        assert called_tid != main_tid
        assert called_host == "public-host.example.com"
        assert called_port == 443
        assert called_family == 0
        assert called_socktype == socket.SOCK_STREAM

        # SSRF blocks via default threaded socket.getaddrinfo path
        with pytest.raises(SsrfBlockedError, match="127.0.0.1"):
            await validate_public_http_url("https://rebind-loopback.example.com/secret")

        with pytest.raises(SsrfBlockedError, match="192.168.10.5"):
            await validate_public_http_url("http://rebind-private.example.com:8080/internal")

        with pytest.raises(SsrfBlockedError, match="DNS resolution failed"):
            await validate_public_http_url("https://nxdomain.example.com/")

    asyncio.run(_run())


def test_w5_fetch_vision_images_concurrent_download_client_reuse_and_threaded_512_resize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    png_bytes = _make_large_rgb_png((1600, 1200), (30, 140, 230))
    gif_bytes = _make_large_animated_gif((1200, 1500))

    main_tid = threading.get_ident()
    real_normalize = normalize_image_bytes_for_vision
    decode_records: list[tuple[int, int]] = []

    def _tracked_normalize(raw: bytes, *, max_side: int = 768) -> tuple[str, str] | None:
        decode_records.append((threading.get_ident(), max_side))
        return real_normalize(raw, max_side=max_side)

    monkeypatch.setattr(provider_mod, "normalize_image_bytes_for_vision", _tracked_normalize)

    # Direct call backward compatibility check: default max_side remains 768
    direct_res = real_normalize(png_bytes)
    assert direct_res is not None
    with Image.open(io.BytesIO(base64.b64decode(direct_res[1]))) as direct_im:
        assert max(direct_im.width, direct_im.height) == 768

    async def _img_handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.10)
        if request.url.path == "/first.png":
            return httpx.Response(200, content=png_bytes, headers={"Content-Type": "image/png"})
        if request.url.path == "/second.gif":
            return httpx.Response(200, content=gif_bytes, headers={"Content-Type": "image/gif"})
        return httpx.Response(404, content=b"missing")

    async def _run() -> None:
        shared_client = httpx.AsyncClient(transport=httpx.MockTransport(_img_handler))
        try:
            t0 = time.perf_counter()
            images = await fetch_vision_images(
                [
                    "https://cdn.example.org/first.png",
                    "https://cdn.example.org/second.gif",
                ],
                max_images=2,
                client=shared_client,
            )
            elapsed = time.perf_counter() - t0
            assert shared_client.is_closed is False
        finally:
            await shared_client.aclose()

        # Two 0.10s downloads executed via asyncio.gather complete well under sequential 0.20s+ overhead
        assert elapsed < 0.19, f"Expected concurrent download, took {elapsed:.3f}s"
        assert len(images) == 2
        assert len(decode_records) == 2

        for rec_tid, rec_max_side in decode_records:
            assert rec_tid != main_tid
            assert rec_max_side == 512

        # Input order preserved: first is landscape (512x384), second is portrait (410x512)
        with Image.open(io.BytesIO(base64.b64decode(images[0][1]))) as im0:
            assert im0.width == 512 and im0.height < 512
        with Image.open(io.BytesIO(base64.b64decode(images[1][1]))) as im1:
            assert im1.height == 512 and im1.width < 512

    asyncio.run(_run())


def test_w5_compact_sticker_prompt_guide_and_compact_send_message_schema(tmp_path: Path) -> None:
    sticker_dir = tmp_path / "stickers"
    sticker_dir.mkdir()
    tiny_png = _make_large_rgb_png((16, 16), (255, 210, 50))

    manifest_data = tomllib.loads(DEFAULT_STICKERS_MANIFEST.read_bytes().decode("utf-8-sig"))
    raw_stickers = manifest_data["stickers"]
    assert len(raw_stickers) >= 16
    for item in raw_stickers:
        (sticker_dir / item["file"]).write_bytes(tiny_png)

    reg = StickerRegistry(sticker_dir, manifest_path=DEFAULT_STICKERS_MANIFEST)
    entries = reg.available_entries()
    assert len(entries) == len(raw_stickers)

    default_guide = reg.render_prompt_guide()
    full_guide = reg.render_prompt_guide(compact=False)
    compact_guide = reg.render_prompt_guide(compact=True)

    assert default_guide == full_guide
    assert "【表情包（奶龙/奶蛙系列）使用指南】" in full_guide

    # Compact guide covers 100% of sticker IDs + short labels while saving >= 55% characters
    for entry in entries:
        assert f"{entry.id}(" in compact_guide
    assert "send_message" in compact_guide
    assert "sticker_id" in compact_guide
    assert "括号" in compact_guide
    assert len(compact_guide) <= int(len(full_guide) * 0.45), (
        f"Expected >=55% char reduction, got {len(compact_guide)} vs {len(full_guide)}"
    )

    async def _noop_sleep(_: float) -> None:
        return None

    async def _run() -> None:
        bus = DirectMemoryBus()
        await bus.start()

        full_tool = build_send_message_tool(
            bus,
            ChatTurnState(),
            chat_type="group",
            target_id=700001,
            stickers=reg,
            compact_schema=False,
            sleep_fn=_noop_sleep,
        )
        compact_state = ChatTurnState()
        compact_tool = build_send_message_tool(
            bus,
            compact_state,
            chat_type="group",
            target_id=700001,
            stickers=reg,
            compact_schema=True,
            sleep_fn=_noop_sleep,
        )

        # Schema structure, types, and required fields remain strictly identical
        full_schema = full_tool.definition.input_schema
        comp_schema = compact_tool.definition.input_schema
        assert compact_tool.definition.name == "send_message"
        assert compact_tool.definition.is_chat_terminal is True
        assert compact_tool.definition.requires_followup is False
        assert comp_schema["type"] == full_schema["type"] == "object"
        assert comp_schema["required"] == full_schema["required"] == ["bubbles"]
        assert comp_schema["additionalProperties"] is False
        assert set(comp_schema["properties"].keys()) == set(full_schema["properties"].keys())
        for key in ("bubbles", "sticker_id", "reply_to_message_id"):
            assert comp_schema["properties"][key]["type"] == full_schema["properties"][key]["type"]
        assert comp_schema["properties"]["bubbles"]["minItems"] == 1
        assert comp_schema["properties"]["bubbles"]["maxItems"] == 3

        # Compact schema omits repeating 16 sticker IDs in sticker_id.description
        assert entries[0].id in full_schema["properties"]["sticker_id"]["description"]
        for entry in entries:
            assert entry.id not in comp_schema["properties"]["sticker_id"]["description"]

        full_json_len = len(
            json.dumps(
                {"d": full_tool.definition.description, "s": full_schema},
                ensure_ascii=False,
            )
        )
        comp_json_len = len(
            json.dumps(
                {"d": compact_tool.definition.description, "s": comp_schema},
                ensure_ascii=False,
            )
        )
        assert comp_json_len < full_json_len * 0.55

        # Runtime execution & sticker whitelist validation remain fully intact
        await compact_tool.execute(
            {
                "bubbles": ["先看第一条", "再补第二条"],
                "sticker_id": "naiwa_stare",
                "reply_to_message_id": 9001,
            }
        )
        assert compact_state.sent_bubbles == ["先看第一条", "再补第二条"]
        assert compact_state.last_sticker_id == "naiwa_stare"

    asyncio.run(_run())
