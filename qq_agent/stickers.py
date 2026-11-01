"""Hot-reloadable sticker/meme registry (奶龙 / 奶蛙 series) and LLM reply tag extraction."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import logging
from pathlib import Path
import random
import re
import tomllib
from typing import Any

logger = logging.getLogger(__name__)

_SUPPORTED_IMAGE_SUFFIXES = {".gif", ".png", ".jpg", ".jpeg", ".webp"}

# Matches `[[sticker:foo]]`, `[sticker:foo]`, `[[表情:foo]]`, `[[表情包:foo]]`, `[表情包:foo]`
_STICKER_TAG_RE = re.compile(
    r"\[\[?\s*(?:sticker|表情包|表情)\s*[:：]\s*([^\[\]\r\n]+?)\s*\]?\]",
    flags=re.IGNORECASE,
)


@dataclass(slots=True, frozen=True)
class StickerEntry:
    """Metadata and filename of a single local sticker/meme asset."""

    id: str
    series: str
    label: str
    file: str
    tags: tuple[str, ...] = ()


def _normalize_tags(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        val = raw.strip()
        return (val,) if val else ()
    if isinstance(raw, list) and all(isinstance(x, str) for x in raw):
        return tuple(x.strip() for x in raw if x.strip())
    return ()


def _parse_manifest(path: Path) -> list[StickerEntry]:
    if not path.is_file():
        return []
    raw_data = tomllib.loads(path.read_bytes().decode("utf-8-sig"))
    raw_items = raw_data.get("stickers", [])
    if not isinstance(raw_items, list):
        raise ValueError("stickers.toml field 'stickers' must be an array of tables")

    entries: list[StickerEntry] = []
    seen_ids: set[str] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("id") or "").strip()
        fname = str(item.get("file") or "").strip()
        if not sid or not fname or sid in seen_ids:
            continue
        series = str(item.get("series") or "表情").strip() or "表情"
        label = str(item.get("label") or sid).strip() or sid
        tags = _normalize_tags(item.get("tags"))
        seen_ids.add(sid)
        entries.append(
            StickerEntry(
                id=sid,
                series=series,
                label=label,
                file=fname,
                tags=tags,
            )
        )
    return entries


class StickerRegistry:
    """Manages local sticker pack metadata, auto-discovery, prompt guide, and base64 loading."""

    def __init__(
        self,
        sticker_dir: Path | str,
        *,
        manifest_path: Path | str | None = None,
    ) -> None:
        self._dir = Path(sticker_dir)
        self._manifest_path = Path(manifest_path) if manifest_path is not None else None
        self._entries: dict[str, StickerEntry] = {}
        self._last_signature: tuple[Any, ...] | None = None
        self.reload()

    @property
    def sticker_dir(self) -> Path:
        return self._dir

    def _compute_signature(self) -> tuple[Any, ...]:
        m_sig: tuple[int, int] | None = None
        if self._manifest_path is not None and self._manifest_path.is_file():
            st = self._manifest_path.stat()
            m_sig = (st.st_mtime_ns, st.st_size)

        d_files: list[tuple[str, int]] = []
        if self._dir.is_dir():
            for child in sorted(self._dir.iterdir(), key=lambda p: p.name):
                if child.is_file() and child.suffix.lower() in _SUPPORTED_IMAGE_SUFFIXES:
                    d_files.append((child.name, child.stat().st_size))
        return (m_sig, tuple(d_files))

    def reload(self) -> list[StickerEntry]:
        """Reload manifest + auto-discovered image files present inside `sticker_dir`."""
        manifest_entries: list[StickerEntry] = []
        if self._manifest_path is not None and self._manifest_path.is_file():
            manifest_entries = _parse_manifest(self._manifest_path)

        active: dict[str, StickerEntry] = {}
        claimed_files: set[str] = set()

        if self._dir.is_dir():
            root_resolved = self._dir.resolve()
            for entry in manifest_entries:
                candidate = (self._dir / entry.file).resolve()
                if (
                    candidate.parent == root_resolved
                    and candidate.is_file()
                    and candidate.suffix.lower() in _SUPPORTED_IMAGE_SUFFIXES
                ):
                    active[entry.id] = entry
                    claimed_files.add(candidate.name)

            # Auto-discover any extra sticker files dropped into `data/stickers/`
            for child in sorted(self._dir.iterdir(), key=lambda p: p.name):
                if (
                    not child.is_file()
                    or child.suffix.lower() not in _SUPPORTED_IMAGE_SUFFIXES
                    or child.name in claimed_files
                ):
                    continue
                stem = child.stem.strip()
                if not stem or stem in active:
                    continue
                lower_stem = stem.lower()
                if "naiwa" in lower_stem or "奶蛙" in stem:
                    series = "奶蛙"
                elif "nailong" in lower_stem or "奶龙" in stem:
                    series = "奶龙"
                else:
                    series = "表情"
                active[stem] = StickerEntry(
                    id=stem,
                    series=series,
                    label=stem,
                    file=child.name,
                    tags=(stem,),
                )

        self._entries = active
        self._last_signature = self._compute_signature()
        return list(self._entries.values())

    def available_entries(self) -> list[StickerEntry]:
        """Return currently available on-disk stickers, hot-reloading if directory or TOML changed."""
        try:
            sig = self._compute_signature()
            if sig != self._last_signature:
                return self.reload()
        except (tomllib.TOMLDecodeError, ValueError, OSError) as exc:
            logger.warning("Failed to hot-reload stickers (%s); keeping cached registry", exc)
        return list(self._entries.values())

    @staticmethod
    def _compact_label(entry: StickerEntry) -> str:
        raw = entry.label.strip()
        if not raw or raw == entry.id:
            return entry.series
        head = re.split(r"[（(《]", raw, maxsplit=1)[0].strip()
        if head.endswith("动图") and len(head) > 2:
            head = head[:-2]
        short = head or raw
        return short[:6]

    def render_prompt_guide(self, *, compact: bool = False) -> str:
        """Build system-prompt instructions listing available 奶龙/奶蛙 sticker IDs."""
        entries = self.available_entries()
        if not entries:
            return ""

        if compact:
            items = " ".join(f"{e.id}({self._compact_label(e)})" for e in entries)
            return (
                f"【表情包】{items}\n"
                "适度通过 send_message 的 sticker_id（或 [[sticker:ID]]）配1张图；禁用（狗头）等括号文字表情。"
            )

        lines = [
            "【表情包（奶龙/奶蛙系列）使用指南】",
            "你手里存了一套奶龙、奶蛙抽象表情包。需要发表情包时，在回复末尾附上 `[[sticker:表情ID]]`（每次最多 1 张）：",
        ]
        for entry in entries:
            lines.append(f"- `{entry.id}`：[{entry.series}] {entry.label}")
        lines.extend(
            [
                "使用原则：",
                "1. 群友发来 `[动画表情]` 斗图、抛梗调侃、让你发奶龙/奶蛙，或者你觉得适合嘲笑/无语凝视/吃瓜/疑惑/红温时，自然配上一张对应情绪的 `[[sticker:表情ID]]`；斗图或无语时甚至可以不打字、只单独输出一个 `[[sticker:表情ID]]`。",
                "2. 日常闲聊适度带图（约三到五成回合即可），正经聊具体技术细节时不必强行塞图；严禁用 `（狗头）`、`（捂脸）` 等文字括号代替表情包。",
            ]
        )
        return "\n".join(lines)

    def resolve(
        self,
        token: str,
        *,
        rng: random.Random | None = None,
    ) -> StickerEntry | None:
        """Resolve an exact sticker ID, fuzzy emotion tag, or series keyword (`奶蛙`/`奶龙`)."""
        entries = self.available_entries()
        if not entries:
            return None
        key = token.strip()
        if not key:
            return None
        key_lower = key.lower()
        picker = rng if rng is not None else random

        # 1) Exact ID match (case-insensitive)
        for entry in entries:
            if entry.id.lower() == key_lower:
                return entry

        # 2) Exact or substring match against tags / label
        tag_matches = [
            entry
            for entry in entries
            if any(key_lower == t.lower() or key_lower in t.lower() for t in entry.tags)
            or key_lower in entry.label.lower()
        ]
        if tag_matches:
            return picker.choice(tag_matches)

        # 3) Series fallback ("奶蛙" / "naiwa" / "奶龙" / "nailong" / "random")
        if "奶蛙" in key or "naiwa" in key_lower or "蛙" in key:
            pool = [e for e in entries if e.series == "奶蛙"]
            return picker.choice(pool or entries)
        if "奶龙" in key or "nailong" in key_lower or "龙" in key:
            pool = [e for e in entries if e.series == "奶龙"]
            return picker.choice(pool or entries)
        if key_lower in {"random", "任意", "随机", "表情", "表情包"}:
            return picker.choice(entries)

        return None

    def extract_from_reply(
        self,
        raw_text: str,
        *,
        rng: random.Random | None = None,
    ) -> tuple[str, str | None]:
        """Strip `[[sticker:...]]` tags from LLM reply text and return `(clean_text, sticker_id)`."""
        resolved_id: str | None = None

        for match in _STICKER_TAG_RE.finditer(raw_text):
            if resolved_id is None:
                entry = self.resolve(match.group(1), rng=rng)
                if entry is not None:
                    resolved_id = entry.id

        cleaned = _STICKER_TAG_RE.sub("", raw_text).strip()
        return cleaned, resolved_id

    def read_base64_uri(self, sticker_id: str) -> str | None:
        """Read sticker bytes safely inside `sticker_dir` and encode as `base64://...` for OB11."""
        entries = {e.id: e for e in self.available_entries()}
        entry = entries.get(sticker_id)
        if entry is None:
            return None
        if not self._dir.is_dir():
            return None
        root_resolved = self._dir.resolve()
        target = (self._dir / entry.file).resolve()
        if target.parent != root_resolved or not target.is_file():
            return None
        encoded = base64.b64encode(target.read_bytes()).decode("ascii")
        return f"base64://{encoded}"


__all__ = [
    "StickerEntry",
    "StickerRegistry",
]
