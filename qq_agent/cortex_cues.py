"""Public-interest cue dictionary, memory sanitization, query relevance features, and domain types."""

from __future__ import annotations

from dataclasses import dataclass
import re

from qq_agent.anti_injection import detect_injection_signals

_MEMORY_TAG_CLOSED_RE = re.compile(
    r"<\s*(/?)\s*(untrusted_chat_message|untrusted_group_context|untrusted_web_results|untrusted_webpage|subagent_research_brief|tool_result|system|developer)\b([^>]*)>",
    flags=re.IGNORECASE,
)
_MEMORY_TAG_OPEN_FRAGMENT_RE = re.compile(
    r"<\s*(/?)\s*(untrusted_chat_message|untrusted_group_context|untrusted_web_results|untrusted_webpage|subagent_research_brief|tool_result|system|developer)",
    flags=re.IGNORECASE,
)

_SHARED_FACT_MAX_BRIEF_CHARS = 360
_SELF_STATE_MAX_CHARS = 42

_SHARED_FACT_ERROR_MARKERS: tuple[str, ...] = (
    "未能查证",
    "查证超时",
    "查证失败",
    "后台查证中止",
    "配额已耗尽",
    "no matching web results",
)

# Conversational filler words stripped before CJK 2-gram slicing to prevent false boundary bigrams.
_CJK_FILLER_PHRASES: tuple[str, ...] = (
    "请问一下",
    "帮我查查",
    "查一下",
    "搜一下",
    "问一下",
    "怎么样",
    "为什么",
    "是什么",
    "怎么看",
    "有没有",
    "是不是",
    "能不能",
    "最新",
    "什么",
    "怎么",
    "一下",
    "帮我",
    "查查",
    "搜搜",
    "现在",
    "今天",
    "请问",
    "知道",
    "告诉",
    "看看",
    "多少",
    "哪些",
    "哪个",
    "哪里",
    "最近",
    "目前",
    "刚才",
    "到底",
    "究竟",
    "那个",
    "这个",
    "问下",
)
_CJK_FILLER_RE = re.compile("|".join(re.escape(p) for p in _CJK_FILLER_PHRASES))
_CJK_PARTICLE_RE = re.compile(r"[的了吗呢吧啊呀哦哈嘛呐啥]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_MIXED_ALPHA_CJK_RE = re.compile(
    r"(?<![a-z0-9])[a-z]{1,2}[\u4e00-\u9fff]",
    flags=re.IGNORECASE,
)
_ALNUM_COMPOUND_RE = re.compile(r"[a-z0-9]+(?:[._+-][a-z0-9]+)*", flags=re.IGNORECASE)
_ALNUM_PART_RE = re.compile(r"[a-z]{2,}|\d+\.\d+|\d{2,}", flags=re.IGNORECASE)
_DOTTED_VERSION_RE = re.compile(r"^\d+(?:\.\d+)+$")
_PURE_DIGITS_RE = re.compile(r"^\d+$")
_TOPIC_KEY_CLEAN_RE = re.compile(r"[^\w\u4e00-\u9fff.+#-]+")

# Generic predicate bigrams that may contribute to score when paired with an entity,
# but must NEVER trigger a cross-topic match on their own (e.g. "杭州天气预报" vs "北京天气预报").
_GENERIC_PREDICATE_BIGRAMS: frozenset[str] = frozenset(
    {
        "天气",
        "气预",
        "预报",
        "版本",
        "本更",
        "更新",
        "新内",
        "内容",
        "价格",
        "多少",
        "少钱",
        "时间",
        "官方",
        "公告",
        "新闻",
        "消息",
        "攻略",
        "教程",
        "推荐",
        "排行",
        "排名",
        "发布",
        "上线",
        "维护",
        "情况",
    }
)

_EN_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "what",
        "how",
        "why",
        "who",
        "when",
        "where",
        "are",
        "was",
        "were",
        "have",
        "has",
        "had",
        "not",
        "but",
        "can",
        "will",
        "just",
        "new",
        "latest",
        "today",
        "now",
        "news",
        "info",
        "url",
        "snippet",
        "http",
        "https",
        "www",
        "com",
        "org",
        "net",
        "cn",
        "html",
        "htm",
    }
)

_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_CQ_CODE_RE = re.compile(r"\[CQ:[^\]]*\]", flags=re.IGNORECASE)
_XML_TAG_RE = re.compile(r"<[^>]+>")
_ANGLE_BRACKET_RE = re.compile(r"[<>]")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_AT_MENTION_RE = re.compile(r"@[^\s,，。!！?？:：;；、()（）【】\[\]\"'“”‘’<>]+")
_PHONE_LIKE_RE = re.compile(r"(?<!\d)1[3-9]\d[-\s]?\d{4}[-\s]?\d{4}(?!\d)")
_LONG_DIGITS_RE = re.compile(r"\d{5,}")
_EFFECTIVE_CHAR_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")
_SELF_STATE_PREFIX_RE = re.compile(r"^(?:刚聊过|刚查过|最近在聊)\s*[:：]\s*")


def _compile_public_cue_rules() -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Compile zero-latency safe public-interest dictionary patterns (fixed labels only, zero PII extraction)."""
    raw_rules: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("聊过战舰世界", ("战舰世界", "窝窝屎", "打船", "开船", "莱服", "wows")),
        ("玩帝国时代", ("帝国时代", "aoe4", "aoe2")),
        ("游戏开黑党", ("steam", "开黑", "联机")),
        ("备考学习中", ("专升本", "考研", "刷题", "备考", "四六级", "考公")),
        ("常熬夜", ("睡不着", "熬夜", "凌晨", "通宵", "失眠")),
        (
            "关注编程/AI",
            (
                "写代码",
                "敲代码",
                "编程",
                "大模型",
                "python",
                "rust",
                "golang",
                "typescript",
                "llm",
                "ai",
            ),
        ),
        ("二次元手游党", ("原神", "崩铁", "星穹铁道", "绝区零", "鸣潮", "二游", "抽卡")),
        ("FPS玩家", ("cs2", "csgo", "瓦罗兰特", "无畏契约", "打瓦", "apex")),
        ("关注动漫番剧", ("番剧", "新番", "追番", "动漫")),
        ("关注数码硬件", ("显卡", "装机", "配电脑", "机械键盘", "路由器")),
        ("美食干饭人", ("火锅", "烧烤", "夜宵", "奶茶", "干饭")),
        ("喜欢猫狗萌宠", ("撸猫", "猫猫", "养猫", "吸猫", "修狗", "养狗")),
    )
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for label, keywords in raw_rules:
        parts: list[str] = []
        for kw in keywords:
            if kw.isascii():
                parts.append(rf"(?<![A-Za-z0-9_]){re.escape(kw)}(?![A-Za-z0-9_])")
            else:
                parts.append(re.escape(kw))
        compiled.append((label, re.compile("|".join(parts), flags=re.IGNORECASE)))
    return tuple(compiled)


_PUBLIC_CUE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = _compile_public_cue_rules()


def _sanitize_memory_text(text: str) -> str:
    """Escape XML-like untrusted/system boundary tags to prevent prompt-injection tag breakout."""
    escaped = _MEMORY_TAG_CLOSED_RE.sub(r"&lt;\1\2\3&gt;", text)
    escaped = _MEMORY_TAG_OPEN_FRAGMENT_RE.sub(r"&lt;\1\2", escaped)
    return escaped.strip()


def extract_public_interest_cues(text: str) -> list[str]:
    """Zero-latency dictionary extractor returning fixed safe public-interest tags (never extracts digits/PII)."""
    if not text or not text.strip():
        return []
    scrubbed = _URL_RE.sub(" ", text)
    scrubbed = _CQ_CODE_RE.sub(" ", scrubbed)
    scrubbed = _EMAIL_RE.sub(" ", scrubbed)
    scrubbed = _AT_MENTION_RE.sub(" ", scrubbed)
    if not scrubbed.strip():
        return []

    cues: list[str] = []
    for label, pattern in _PUBLIC_CUE_RULES:
        if pattern.search(scrubbed):
            cues.append(label)
    return cues


def _normalize_topic_key(query: str) -> str:
    """Produce a canonical lowercase topic key for `shared_facts` deduplication."""
    collapsed = _TOPIC_KEY_CLEAN_RE.sub(" ", query.lower()).strip()
    collapsed = " ".join(collapsed.split())
    return collapsed[:160]


def _is_high_distinctiveness_token(token: str, *, from_fact_query: bool = True) -> bool:
    """Return True if an ASCII/numeric token (`len >= 3`) is specific enough to anchor a topic match."""
    if len(token) < 3 or token in _EN_STOPWORDS:
        return False
    if _DOTTED_VERSION_RE.match(token):
        return True
    if any("a" <= ch <= "z" for ch in token):
        return True
    if not from_fact_query:
        return False
    if _PURE_DIGITS_RE.match(token):
        val = int(token)
        if 1900 <= val <= 2099:
            return False
        if token in {"100", "1000", "10000", "100000"}:
            return False
        return True
    return False


@dataclass(slots=True, frozen=True)
class _QueryFeatures:
    alnum_tokens: frozenset[str]
    high_alnum_tokens: frozenset[str]
    cjk_bigrams: frozenset[str]
    specific_cjk_bigrams: frozenset[str]
    norm_core: str


def _is_ignored_alnum_token(tok: str) -> bool:
    if len(tok) < 2 or tok in _EN_STOPWORDS:
        return True
    if _PURE_DIGITS_RE.match(tok) and 1900 <= int(tok) <= 2099:
        return True
    return False


def _extract_query_features(text: str, *, is_fact_query: bool = True) -> _QueryFeatures:
    """Extract normalized alphanumeric tokens, CJK 2-grams (with filler words removed), and core string."""
    lowered = _URL_RE.sub(" ", text.lower())
    lowered = _XML_TAG_RE.sub(" ", lowered)

    alnum: set[str] = set()
    for match in _ALNUM_COMPOUND_RE.finditer(lowered):
        tok = match.group(0).strip("._+-")
        if not _is_ignored_alnum_token(tok):
            alnum.add(tok)
            compact = tok.replace("-", "").replace("_", "")
            if not _is_ignored_alnum_token(compact):
                alnum.add(compact)
        for sub in _ALNUM_PART_RE.findall(tok):
            if not _is_ignored_alnum_token(sub):
                alnum.add(sub)

    mixed_tokens = _MIXED_ALPHA_CJK_RE.findall(lowered)
    for mixed in mixed_tokens:
        alnum.add(mixed)

    high_alnum = frozenset(
        tok
        for tok in alnum
        if _is_high_distinctiveness_token(tok, from_fact_query=is_fact_query)
    )

    stripped_cjk = _CJK_FILLER_RE.sub(" ", lowered)
    stripped_cjk = _CJK_PARTICLE_RE.sub(" ", stripped_cjk)

    bigrams: set[str] = set(mixed_tokens)
    for run in _CJK_RUN_RE.findall(stripped_cjk):
        for idx in range(len(run) - 1):
            bigrams.add(run[idx : idx + 2])

    specific_bigrams = frozenset(bg for bg in bigrams if bg not in _GENERIC_PREDICATE_BIGRAMS)

    ordered_core_tokens: list[str] = []
    for raw_part in re.findall(
        r"(?<![a-z0-9])[a-z]{1,2}[\u4e00-\u9fff]+|[a-z0-9]+(?:[._+-][a-z0-9]+)*|[\u4e00-\u9fff]+",
        stripped_cjk,
    ):
        if raw_part.isascii():
            clean_tok = raw_part.strip("._+-")
            if not _is_ignored_alnum_token(clean_tok):
                ordered_core_tokens.append(clean_tok)
        elif len(raw_part) >= 2:
            ordered_core_tokens.append(raw_part)
    norm_core = "".join(ordered_core_tokens)

    return _QueryFeatures(
        alnum_tokens=frozenset(alnum),
        high_alnum_tokens=high_alnum,
        cjk_bigrams=frozenset(bigrams),
        specific_cjk_bigrams=specific_bigrams,
        norm_core=norm_core,
    )


def _is_safe_core_containment(core_a: str, core_b: str) -> bool:
    """Check normalized core containment without letting short ASCII tokens match inside unrelated words."""
    if len(core_a) < 2 or len(core_b) < 2:
        return False
    if core_a == core_b:
        return True
    shorter, longer = (core_a, core_b) if len(core_a) <= len(core_b) else (core_b, core_a)
    if shorter.isascii():
        return False
    prefix = r"(?<![a-z0-9])" if shorter[0].isascii() else ""
    suffix = r"(?![a-z0-9])" if shorter[-1].isascii() else ""
    return re.search(f"{prefix}{re.escape(shorter)}{suffix}", longer) is not None


def _sanitize_public_self_topic(topic_hint: str) -> str:
    """Scrub `@mentions`, `>=5` digit IDs/phones, URLs, XML tags, injection attempts, and newlines for public `bot_self_state`."""
    if not topic_hint or not topic_hint.strip():
        return ""
    cleaned = _URL_RE.sub(" ", topic_hint)
    cleaned = _CQ_CODE_RE.sub(" ", cleaned)
    cleaned = _XML_TAG_RE.sub(" ", cleaned)
    cleaned = _ANGLE_BRACKET_RE.sub(" ", cleaned)
    if detect_injection_signals(cleaned):
        return ""
    cleaned = _EMAIL_RE.sub(" ", cleaned)
    cleaned = _AT_MENTION_RE.sub(" ", cleaned)
    cleaned = _PHONE_LIKE_RE.sub(" ", cleaned)
    cleaned = _LONG_DIGITS_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = _SELF_STATE_PREFIX_RE.sub("", cleaned).strip()
    cleaned = cleaned.strip(" ,，。!！?？:：;；、-—~")

    if len(_EFFECTIVE_CHAR_RE.findall(cleaned)) < 2:
        return ""

    formatted = _sanitize_memory_text(f"刚聊过：{cleaned}")
    if len(formatted) > _SELF_STATE_MAX_CHARS:
        formatted = formatted[:_SELF_STATE_MAX_CHARS].rstrip()
    return formatted


@dataclass(slots=True, frozen=True)
class MemoryFact:
    """Single L3 memory fact isolated by visibility `scope`."""

    id: int
    scope: str
    content: str
    created_at: float


@dataclass(slots=True, frozen=True)
class UserProfile:
    """Cross-session L3 acquaintance profile keyed by QQ `user_id`."""

    user_id: int
    display_name: str
    traits_text: str
    updated_at: float


@dataclass(slots=True, frozen=True)
class SharedFact:
    """TTL-cached objective research fact shared across sessions without group/user metadata."""

    topic_key: str
    query_text: str
    brief_text: str
    updated_at: float
    expires_at: float


__all__ = [
    "MemoryFact",
    "SharedFact",
    "UserProfile",
    "extract_public_interest_cues",
]
