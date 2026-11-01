"""Wave W6-04 governance tests: web_search modular split line budgets, symbol identity, and circular-import safety."""

from __future__ import annotations

import asyncio
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any

import pytest

import qq_agent.tools as tools_pkg
import qq_agent.tools.search_engines as search_engines_mod
import qq_agent.tools.web_search as web_search_mod
import qq_agent.tools.web_ssrf as web_ssrf_mod
import qq_agent.tools.webpage_fetcher as webpage_fetcher_mod

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "qq_agent" / "tools"
MAX_MODULE_LINES = 680
MAX_FACADE_LINES = 95


def test_w6_web_search_split_file_line_budgets() -> None:
    expected_files = (
        "web_ssrf.py",
        "webpage_fetcher.py",
        "search_engines.py",
        "web_search.py",
    )
    for filename in expected_files:
        path = TOOLS_DIR / filename
        assert path.is_file(), f"Expected module {path} to exist"
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        assert line_count <= MAX_MODULE_LINES, (
            f"{filename} has {line_count} lines, exceeding {MAX_MODULE_LINES}L ceiling"
        )

    facade_lines = len((TOOLS_DIR / "web_search.py").read_text(encoding="utf-8").splitlines())
    assert facade_lines <= MAX_FACADE_LINES, (
        f"web_search.py facade has {facade_lines} lines, exceeding {MAX_FACADE_LINES}L ceiling"
    )


def test_w6_web_search_facade_and_tools_package_symbol_identity() -> None:
    ssrf_symbols = (
        "DEFAULT_BROWSER_USER_AGENT",
        "ResponseTooLargeError",
        "SsrfBlockedError",
        "WebFetchError",
        "_assert_public_ip",
        "_collapse_inline_ws",
        "_escape_xml_attr",
        "_extract_ips_from_resolver_output",
        "_invoke_resolver",
        "_normalize_multiline_text",
        "sanitize_untrusted_web_text",
        "validate_public_http_url",
    )
    for sym in ssrf_symbols:
        assert getattr(web_search_mod, sym) is getattr(web_ssrf_mod, sym), f"Mismatch on web_ssrf.{sym}"

    fetcher_symbols = (
        "DEFAULT_FETCH_MAX_CHARS",
        "DEFAULT_MAX_FETCH_BYTES",
        "DEFAULT_MAX_REDIRECTS",
        "HARD_FETCH_MAX_CHARS",
        "JINA_FALLBACK_MIN_CHARS",
        "WebPageFetcher",
        "_ReadableHtmlParser",
        "_decode_http_bytes",
        "_is_allowed_text_mime",
        "_wrap_untrusted_webpage",
        "build_fetch_webpage_tool",
        "extract_readable_html",
    )
    for sym in fetcher_symbols:
        assert getattr(web_search_mod, sym) is getattr(webpage_fetcher_mod, sym), (
            f"Mismatch on webpage_fetcher.{sym}"
        )

    search_symbols = (
        "SearchResultItem",
        "WebSearchClient",
        "_BingCnHtmlParser",
        "_DuckDuckGoHtmlParser",
        "_unwrap_bing_href",
        "build_web_search_tool",
        "format_untrusted_web_results",
        "normalize_url_for_dedup",
        "parse_bing_cn_html",
        "parse_duckduckgo_html",
        "unwrap_ddg_href",
    )
    for sym in search_symbols:
        assert getattr(web_search_mod, sym) is getattr(search_engines_mod, sym), (
            f"Mismatch on search_engines.{sym}"
        )

    tools_pkg_reexports = {
        "ResponseTooLargeError": web_ssrf_mod.ResponseTooLargeError,
        "SearchResultItem": search_engines_mod.SearchResultItem,
        "SsrfBlockedError": web_ssrf_mod.SsrfBlockedError,
        "WebFetchError": web_ssrf_mod.WebFetchError,
        "WebPageFetcher": webpage_fetcher_mod.WebPageFetcher,
        "WebSearchClient": search_engines_mod.WebSearchClient,
        "build_fetch_webpage_tool": webpage_fetcher_mod.build_fetch_webpage_tool,
        "build_web_search_tool": search_engines_mod.build_web_search_tool,
        "validate_public_http_url": web_ssrf_mod.validate_public_http_url,
    }
    for sym, canonical_obj in tools_pkg_reexports.items():
        assert getattr(tools_pkg, sym) is canonical_obj, f"qq_agent.tools.{sym} does not match canonical object"
        assert getattr(web_search_mod, sym) is canonical_obj


def test_w6_web_search_no_circular_imports_across_all_entry_orders() -> None:
    probe_script = (
        "import importlib, sys\n"
        "targets = [\n"
        "    'qq_agent.tools.web_ssrf',\n"
        "    'qq_agent.tools.webpage_fetcher',\n"
        "    'qq_agent.tools.search_engines',\n"
        "    'qq_agent.tools.web_search',\n"
        "    'qq_agent.tools',\n"
        "]\n"
        "for mod_name in targets:\n"
        "    for k in [k for k in list(sys.modules) if k == 'qq_agent' or k.startswith('qq_agent.')]:\n"
        "        del sys.modules[k]\n"
        "    mod = importlib.import_module(mod_name)\n"
        "    assert mod is not None\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe_script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Circular or failed import detected:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"


def test_w6_web_ssrf_dynamic_socket_getaddrinfo_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_hosts: list[str] = []

    def _fake_getaddrinfo(
        host: str,
        port: int,
        family: int = 0,
        socktype: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[tuple[Any, ...]]:
        seen_hosts.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo)
    normalized = asyncio.run(web_ssrf_mod.validate_public_http_url("https://example.org/docs#anchor"))
    assert normalized == "https://example.org/docs"
    assert seen_hosts == ["example.org"]
