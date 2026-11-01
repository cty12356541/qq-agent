"""Zero-trust streaming webpage text extractor with hop-by-hop redirect SSRF validation and `r.jina.ai` SPA fallback."""

from __future__ import annotations

from html.parser import HTMLParser
import logging
import re
from typing import Any
import urllib.parse

import httpx

from qq_agent.tools import ToolArgumentError, ToolBinding
from qq_agent.types import ToolDef

from .web_ssrf import (
    DEFAULT_BROWSER_USER_AGENT,
    ResolverFn,
    ResponseTooLargeError,
    SsrfBlockedError,
    WebFetchError,
    _collapse_inline_ws,
    _escape_xml_attr,
    _normalize_multiline_text,
    sanitize_untrusted_web_text,
    validate_public_http_url,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_FETCH_BYTES = 2 * 1024 * 1024  # 2 MiB hard streaming cap
DEFAULT_MAX_REDIRECTS = 4
DEFAULT_FETCH_MAX_CHARS = 4000
HARD_FETCH_MAX_CHARS = 8000
JINA_FALLBACK_MIN_CHARS = 160

_REDIRECT_STATUS_CODES: frozenset[int] = frozenset({301, 302, 303, 307, 308})
_ALLOWED_EXACT_MIMES: frozenset[str] = frozenset(
    {
        "application/json",
        "application/xhtml+xml",
        "application/xml",
    }
)
_STRIPPED_HTML_TAGS: frozenset[str] = frozenset(
    {
        "script",
        "style",
        "noscript",
        "svg",
        "nav",
        "footer",
        "header",
    }
)
_BLOCK_HTML_TAGS: frozenset[str] = frozenset(
    {
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "hr",
        "li",
        "main",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
)
_META_CHARSET_RE = re.compile(
    rb"""<meta[^>]+charset\s*=\s*['"]?\s*([A-Za-z0-9._-]+)""",
    flags=re.IGNORECASE,
)


class _ReadableHtmlParser(HTMLParser):
    """Standard-library HTML cleaner stripping script/style/svg/nav/header/footer subtrees."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignore_stack: list[str] = []
        self._in_title: bool = False
        self._title_parts: list[str] = []
        self._body_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in _STRIPPED_HTML_TAGS:
            self._ignore_stack.append(t)
            return
        if self._ignore_stack:
            return
        if t == "title":
            self._in_title = True
        elif t in _BLOCK_HTML_TAGS:
            self._body_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in _STRIPPED_HTML_TAGS:
            if t in self._ignore_stack:
                while self._ignore_stack:
                    popped = self._ignore_stack.pop()
                    if popped == t:
                        break
            return
        if self._ignore_stack:
            return
        if t == "title":
            self._in_title = False
        elif t in _BLOCK_HTML_TAGS:
            self._body_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignore_stack:
            return
        if self._in_title:
            self._title_parts.append(data)
        else:
            self._body_parts.append(data)

    def extract(self) -> tuple[str, str]:
        title = _collapse_inline_ws("".join(self._title_parts))
        body = _normalize_multiline_text("".join(self._body_parts))
        return title, body


def extract_readable_html(raw_html: str) -> tuple[str, str]:
    """Extract `(title, body_text)` from HTML while completely discarding noisy/executable tags."""
    parser = _ReadableHtmlParser()
    parser.feed(raw_html)
    parser.close()
    return parser.extract()


def _is_allowed_text_mime(content_type_header: str) -> tuple[bool, str]:
    raw = content_type_header.strip()
    if not raw:
        # Missing Content-Type treated tentatively as text/html; binary null-byte guard still applies
        return True, "text/html"
    mime = raw.split(";", 1)[0].strip().lower()
    if mime.startswith("text/") or mime in _ALLOWED_EXACT_MIMES:
        return True, mime
    return False, mime


def _decode_http_bytes(raw_bytes: bytes, encoding_hint: str | None) -> str:
    candidates: list[str] = []
    if encoding_hint:
        candidates.append(encoding_hint)
    meta_match = _META_CHARSET_RE.search(raw_bytes[:4096])
    if meta_match:
        candidates.append(meta_match.group(1).decode("ascii", errors="ignore"))
    candidates.extend(["utf-8", "gb18030"])

    for enc in candidates:
        try:
            return raw_bytes.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw_bytes.decode("utf-8", errors="replace")


def _wrap_untrusted_webpage(url: str, body: str, *, max_chars: int) -> str:
    limit = min(HARD_FETCH_MAX_CHARS, max(1, int(max_chars)))
    sanitized = sanitize_untrusted_web_text(body.strip())
    if len(sanitized) > limit:
        sanitized = sanitized[:limit].rstrip()
        # Re-sanitize in case slicing left a partial tag prefix at the tail
        sanitized = sanitize_untrusted_web_text(sanitized)
    safe_url = _escape_xml_attr(url)
    return f'<untrusted_webpage url="{safe_url}">\n{sanitized}\n</untrusted_webpage>'


class WebPageFetcher:
    """Zero-trust streaming webpage text extractor with hop-by-hop redirect SSRF validation and `r.jina.ai` SPA fallback."""

    def __init__(
        self,
        *,
        timeout_s: float = 10.0,
        max_bytes: int = DEFAULT_MAX_FETCH_BYTES,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        enable_jina_fallback: bool = True,
        jina_min_chars: int = JINA_FALLBACK_MIN_CHARS,
        resolver: ResolverFn | None = None,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        user_agent: str = DEFAULT_BROWSER_USER_AGENT,
    ) -> None:
        self._timeout_s = max(0.5, float(timeout_s))
        self._max_bytes = max(1024, int(max_bytes))
        self._max_redirects = max(0, int(max_redirects))
        self._enable_jina_fallback = bool(enable_jina_fallback)
        self._jina_min_chars = max(0, int(jina_min_chars))
        self._resolver = resolver
        self._client = client
        self._owned_client: httpx.AsyncClient | None = None
        self._transport = transport
        self._user_agent = user_agent

    def _get_or_create_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        if self._owned_client is None or self._owned_client.is_closed:
            self._owned_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout_s),
                follow_redirects=False,
                transport=self._transport,
            )
        return self._owned_client

    async def __aenter__(self) -> WebPageFetcher:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the lazily-allocated internal HTTP connection pool (never closes an injected client)."""
        owned = self._owned_client
        self._owned_client = None
        if owned is not None and not owned.is_closed:
            await owned.aclose()

    async def _stream_validated_url(
        self,
        http_client: httpx.AsyncClient,
        initial_url: str,
        *,
        accept_header: str,
    ) -> tuple[str, str, bytes, str | None]:
        """Follow up to `max_redirects` hops manually, re-running `validate_public_http_url` on every hop."""
        current_url = await validate_public_http_url(initial_url, resolver=self._resolver)
        headers = {
            "User-Agent": self._user_agent,
            "Accept": accept_header,
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

        for hop in range(self._max_redirects + 1):
            async with http_client.stream(
                "GET",
                current_url,
                headers=headers,
                follow_redirects=False,
            ) as resp:
                if resp.status_code in _REDIRECT_STATUS_CODES:
                    location = resp.headers.get("Location", "").strip()
                    if not location:
                        raise WebFetchError(
                            f"HTTP {resp.status_code} redirect from {current_url} missing Location header"
                        )
                    if hop >= self._max_redirects:
                        raise WebFetchError(
                            f"Exceeded max redirects ({self._max_redirects}) fetching {initial_url}"
                        )
                    resolved_target = urllib.parse.urljoin(current_url, location)
                    # Critical: re-validate every redirect Location before issuing the next request
                    current_url = await validate_public_http_url(
                        resolved_target,
                        resolver=self._resolver,
                    )
                    continue

                if resp.status_code >= 400:
                    raise WebFetchError(f"HTTP {resp.status_code} when fetching {current_url}")

                ct_raw = resp.headers.get("Content-Type", "")
                allowed_mime, mime = _is_allowed_text_mime(ct_raw)
                if not allowed_mime:
                    raise WebFetchError(
                        f"Rejected non-text Content-Type {mime!r} from {current_url}"
                    )

                cl_raw = resp.headers.get("Content-Length", "").strip()
                if cl_raw.isdigit() and int(cl_raw) > self._max_bytes:
                    raise ResponseTooLargeError(
                        f"Content-Length {cl_raw} exceeds max_bytes limit ({self._max_bytes})"
                    )

                chunks: list[bytes] = []
                total_bytes = 0
                async for chunk in resp.aiter_bytes():
                    if not chunk:
                        continue
                    total_bytes += len(chunk)
                    if total_bytes > self._max_bytes:
                        raise ResponseTooLargeError(
                            f"Streamed response exceeded max_bytes limit ({self._max_bytes} bytes)"
                        )
                    chunks.append(chunk)

                body_bytes = b"".join(chunks)
                if not ct_raw.strip() and b"\x00" in body_bytes[:2048]:
                    raise WebFetchError(f"Rejected binary payload without Content-Type from {current_url}")

                return current_url, mime, body_bytes, resp.encoding

        raise WebFetchError(f"Redirect loop terminated abnormally for {initial_url}")

    async def _fetch_with_client(
        self,
        http_client: httpx.AsyncClient,
        url: str,
        *,
        max_chars: int,
    ) -> str:
        final_url, mime, raw_bytes, enc_hint = await self._stream_validated_url(
            http_client,
            url,
            accept_header="text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.5",
        )
        decoded = _decode_http_bytes(raw_bytes, enc_hint)

        is_html = mime in {"text/html", "application/xhtml+xml"}
        if is_html:
            title, body_text = extract_readable_html(decoded)
            if title and body_text:
                primary_text = f"Title: {title}\n\n{body_text}"
            elif title:
                primary_text = f"Title: {title}"
            else:
                primary_text = body_text
            measured_len = len(body_text)
        else:
            primary_text = _normalize_multiline_text(decoded)
            measured_len = len(primary_text)

        if self._enable_jina_fallback and measured_len < self._jina_min_chars:
            jina_url = f"https://r.jina.ai/{final_url}"
            try:
                _, jina_mime, jina_bytes, jina_enc = await self._stream_validated_url(
                    http_client,
                    jina_url,
                    accept_header="text/markdown,text/plain;q=0.9,text/html;q=0.7",
                )
                jina_decoded = _decode_http_bytes(jina_bytes, jina_enc)
                if jina_mime in {"text/html", "application/xhtml+xml"}:
                    j_title, j_body = extract_readable_html(jina_decoded)
                    jina_text = f"Title: {j_title}\n\n{j_body}".strip() if j_title else j_body
                else:
                    jina_text = _normalize_multiline_text(jina_decoded)
                if len(jina_text) > len(primary_text):
                    primary_text = jina_text
            except SsrfBlockedError:
                raise
            except Exception as exc:
                logger.debug("r.jina.ai fallback skipped/failed for %s: %s", final_url, exc)

        return _wrap_untrusted_webpage(final_url, primary_text, max_chars=max_chars)

    async def fetch(
        self,
        url: str,
        *,
        max_chars: int = DEFAULT_FETCH_MAX_CHARS,
    ) -> str:
        """Validate, stream, clean, optionally Jina-render, truncate, and wrap `url` in `<untrusted_webpage>`."""
        effective_max_chars = min(HARD_FETCH_MAX_CHARS, max(1, int(max_chars)))
        http_client = self._get_or_create_client()
        return await self._fetch_with_client(http_client, url, max_chars=effective_max_chars)


def build_fetch_webpage_tool(
    fetcher: WebPageFetcher | None = None,
    *,
    default_max_chars: int = DEFAULT_FETCH_MAX_CHARS,
    max_chars_limit: int = HARD_FETCH_MAX_CHARS,
) -> ToolBinding:
    """Build the `fetch_webpage` `ToolBinding` for slow-brain `depth=1` background research sub-agents."""
    page_fetcher = fetcher if fetcher is not None else WebPageFetcher()
    eff_limit = min(HARD_FETCH_MAX_CHARS, max(200, int(max_chars_limit)))
    eff_default = min(eff_limit, max(200, int(default_max_chars)))

    definition = ToolDef(
        name="fetch_webpage",
        description=(
            "安全抓取公网 HTTP/HTTPS 网页并提取清洗后的正文（仅供后台深度检索子任务使用）。"
            "返回包裹在 <untrusted_webpage> 标签内的不可信外部网页文本。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "要抓取正文的完整 http:// 或 https:// 公网 URL。",
                },
                "max_chars": {
                    "type": "integer",
                    "minimum": 200,
                    "maximum": eff_limit,
                    "default": eff_default,
                    "description": f"提取正文最大字符数（默认 {eff_default}，上限 {eff_limit}）。",
                },
            },
            "required": ["url"],
            "additionalProperties": False,
        },
        is_chat_terminal=False,
        requires_followup=True,
    )

    async def _execute(args: dict[str, Any]) -> str:
        raw_url = args.get("url")
        if not isinstance(raw_url, str) or not raw_url.strip():
            raise ToolArgumentError("fetch_webpage requires a non-empty string 'url'")

        raw_max = args.get("max_chars", eff_default)
        if raw_max is None:
            req_max = eff_default
        elif isinstance(raw_max, bool):
            raise ToolArgumentError("'max_chars' must be an integer, not boolean")
        else:
            try:
                req_max = int(raw_max)
            except (TypeError, ValueError) as exc:
                raise ToolArgumentError(f"'max_chars' must be an integer, got {raw_max!r}") from exc

        clamped_max = min(eff_limit, max(200, req_max))
        return await page_fetcher.fetch(raw_url.strip(), max_chars=clamped_max)

    return ToolBinding(definition=definition, execute=_execute)


__all__ = [
    "DEFAULT_FETCH_MAX_CHARS",
    "DEFAULT_MAX_FETCH_BYTES",
    "DEFAULT_MAX_REDIRECTS",
    "HARD_FETCH_MAX_CHARS",
    "JINA_FALLBACK_MIN_CHARS",
    "WebPageFetcher",
    "build_fetch_webpage_tool",
    "extract_readable_html",
]
