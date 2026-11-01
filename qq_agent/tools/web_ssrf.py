"""Zero-trust SSRF public-IP/DNS validator, fetch exceptions, and untrusted web text sanitizers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
import html
import inspect
import ipaddress
import re
import socket
from typing import Any
import urllib.parse

DEFAULT_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

_CGNAT_IPV4_NET = ipaddress.IPv4Network("100.64.0.0/10")
_THIS_IPV4_NET = ipaddress.IPv4Network("0.0.0.0/8")

_UNTRUSTED_TAG_CLOSED_RE = re.compile(
    r"<\s*(/?)\s*(untrusted_web_results|untrusted_webpage|untrusted_chat_message|untrusted_group_context|system|developer|tool_result)\b([^>]*)>",
    flags=re.IGNORECASE,
)
_UNTRUSTED_TAG_OPEN_FRAGMENT_RE = re.compile(
    r"<\s*(/?)\s*(untrusted_web_results|untrusted_webpage|untrusted_chat_message|untrusted_group_context|system|developer|tool_result)",
    flags=re.IGNORECASE,
)
_HORIZ_WS_RE = re.compile(r"[ \t\r\f\v]+")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")

ResolverResult = Sequence[Any]
ResolverFn = Callable[..., ResolverResult | Awaitable[ResolverResult]]


class WebFetchError(RuntimeError):
    """Raised when fetching or decoding an untrusted webpage fails safety or protocol checks."""


class SsrfBlockedError(WebFetchError, ValueError):
    """Raised fail-closed when a URL or any of its resolved IPs / redirects violates public-network SSRF rules."""


class ResponseTooLargeError(WebFetchError):
    """Raised when an HTTP response exceeds the 2 MiB streaming byte ceiling."""


def _escape_xml_attr(value: str) -> str:
    cleaned = value.replace("\r", " ").replace("\n", " ").strip()
    return html.escape(cleaned, quote=True)


def sanitize_untrusted_web_text(text: str) -> str:
    """Neutralize prompt-injection boundary breakout tags inside untrusted web titles, snippets, URLs, and pages."""
    escaped = _UNTRUSTED_TAG_CLOSED_RE.sub(r"&lt;\1\2\3&gt;", text)
    escaped = _UNTRUSTED_TAG_OPEN_FRAGMENT_RE.sub(r"&lt;\1\2", escaped)
    return escaped


def _collapse_inline_ws(text: str) -> str:
    return " ".join(text.split())


def _normalize_multiline_text(raw: str) -> str:
    lines: list[str] = []
    for raw_line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        cleaned = _HORIZ_WS_RE.sub(" ", raw_line).strip()
        lines.append(cleaned)
    joined = "\n".join(lines).strip()
    return _MULTI_BLANK_RE.sub("\n\n", joined)


def _assert_public_ip(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address, *, context: str) -> None:
    if (
        not ip_obj.is_global
        or ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_multicast
        or ip_obj.is_reserved
        or ip_obj.is_unspecified
    ):
        raise SsrfBlockedError(f"SSRF blocked non-public IP {ip_obj} for {context}")

    if isinstance(ip_obj, ipaddress.IPv4Address):
        if ip_obj in _CGNAT_IPV4_NET or ip_obj in _THIS_IPV4_NET:
            raise SsrfBlockedError(f"SSRF blocked restricted IPv4 range {ip_obj} for {context}")
        return

    if ip_obj.ipv4_mapped is not None:
        _assert_public_ip(ip_obj.ipv4_mapped, context=context)
    if ip_obj.sixtofour is not None:
        _assert_public_ip(ip_obj.sixtofour, context=context)
    if ip_obj.teredo is not None:
        _assert_public_ip(ip_obj.teredo[0], context=context)
        _assert_public_ip(ip_obj.teredo[1], context=context)


def _extract_ips_from_resolver_output(raw_items: ResolverResult, *, host: str) -> list[str]:
    ips: list[str] = []
    for item in raw_items:
        if isinstance(item, str):
            ips.append(item)
        elif isinstance(item, (ipaddress.IPv4Address, ipaddress.IPv6Address)):
            ips.append(str(item))
        elif isinstance(item, (tuple, list)):
            # Standard socket.getaddrinfo 5-tuple: (family, type, proto, canonname, sockaddr)
            if len(item) >= 5 and isinstance(item[4], (tuple, list)) and len(item[4]) >= 1:
                ips.append(str(item[4][0]))
            elif len(item) >= 1 and isinstance(item[0], str):
                ips.append(item[0])
            else:
                raise SsrfBlockedError(f"Unrecognized resolver tuple format for host {host!r}")
        else:
            raise SsrfBlockedError(f"Unrecognized resolver entry type {type(item).__name__} for host {host!r}")
    return ips


async def _invoke_resolver(resolver: ResolverFn, host: str, port: int) -> ResolverResult:
    try:
        sig = inspect.signature(resolver)
        positional = [
            p
            for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
        if not has_varargs and len(positional) == 1:
            maybe = resolver(host)
        else:
            maybe = resolver(host, port)
    except (TypeError, ValueError):
        maybe = resolver(host, port)

    if inspect.isawaitable(maybe):
        return await maybe
    return maybe


async def validate_public_http_url(
    url: str,
    *,
    resolver: ResolverFn | None = None,
) -> str:
    """Zero-trust SSRF validator: enforce public-only `http(s)` URL with fail-closed DNS inspection."""
    if not isinstance(url, str):
        raise SsrfBlockedError("URL must be a string")
    cleaned = url.strip()
    if not cleaned:
        raise SsrfBlockedError("URL must not be empty")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in cleaned):
        raise SsrfBlockedError("URL contains forbidden control characters")

    try:
        parsed = urllib.parse.urlsplit(cleaned)
    except ValueError as exc:
        raise SsrfBlockedError(f"Malformed URL: {exc}") from exc

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise SsrfBlockedError(f"Blocked URL scheme {scheme!r}: only http and https are allowed")

    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        raise SsrfBlockedError("Blocked URL containing embedded credentials")

    raw_host = parsed.hostname
    if not raw_host:
        raise SsrfBlockedError("Blocked URL with empty hostname")
    if "%" in raw_host:
        raise SsrfBlockedError("Blocked URL hostname containing IPv6 zone identifier")

    host_normalized = raw_host.rstrip(".").lower()
    if not host_normalized:
        raise SsrfBlockedError("Blocked URL with root-dot empty hostname")

    if (
        host_normalized == "localhost"
        or host_normalized.endswith(".localhost")
        or host_normalized == "local"
        or host_normalized.endswith(".local")
        or host_normalized == "internal"
        or host_normalized.endswith(".internal")
    ):
        raise SsrfBlockedError(f"Blocked internal/local hostname {raw_host!r}")

    try:
        explicit_port = parsed.port
    except ValueError as exc:
        raise SsrfBlockedError(f"Invalid port in URL: {exc}") from exc

    port = explicit_port if explicit_port is not None else (443 if scheme == "https" else 80)
    if not (1 <= port <= 65535):
        raise SsrfBlockedError(f"Invalid TCP port {port}")

    # Direct IP literal fast-fail check (including decimal/hex/octal/short IPv4 obfuscations via inet_aton)
    literal_ip: ipaddress.IPv4Address | ipaddress.IPv6Address | None = None
    try:
        literal_ip = ipaddress.ip_address(host_normalized)
    except ValueError:
        if ":" not in host_normalized:
            try:
                packed_v4 = socket.inet_aton(host_normalized)
                literal_ip = ipaddress.IPv4Address(packed_v4)
            except OSError:
                literal_ip = None

    if literal_ip is not None:
        _assert_public_ip(literal_ip, context=raw_host)

    # Resolve all A / AAAA records and fail-closed if 0 records or ANY record is non-public
    try:
        if resolver is not None:
            raw_resolved = await _invoke_resolver(resolver, host_normalized, port)
        elif literal_ip is not None:
            raw_resolved = [str(literal_ip)]
        else:
            raw_resolved = await asyncio.to_thread(
                socket.getaddrinfo,
                host_normalized,
                port,
                0,
                socket.SOCK_STREAM,
            )
    except SsrfBlockedError:
        raise
    except Exception as exc:
        raise SsrfBlockedError(f"DNS resolution failed for {raw_host!r}: {exc}") from exc

    ip_strings = _extract_ips_from_resolver_output(raw_resolved, host=raw_host)
    if not ip_strings:
        raise SsrfBlockedError(f"DNS resolution returned no IP addresses for {raw_host!r}")

    for raw_ip_str in ip_strings:
        ip_token = raw_ip_str.strip()
        if "%" in ip_token:
            raise SsrfBlockedError(f"Blocked scoped IPv6 address {raw_ip_str!r} for {raw_host!r}")
        try:
            ip_obj = ipaddress.ip_address(ip_token)
        except ValueError as exc:
            raise SsrfBlockedError(
                f"Resolver returned invalid IP address {raw_ip_str!r} for {raw_host!r}"
            ) from exc
        _assert_public_ip(ip_obj, context=raw_host)

    normalized_path = parsed.path if parsed.path else "/"
    return urllib.parse.urlunsplit((scheme, parsed.netloc, normalized_path, parsed.query, ""))


__all__ = [
    "DEFAULT_BROWSER_USER_AGENT",
    "ResolverFn",
    "ResolverResult",
    "ResponseTooLargeError",
    "SsrfBlockedError",
    "WebFetchError",
    "sanitize_untrusted_web_text",
    "validate_public_http_url",
]
