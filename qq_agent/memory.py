"""SQLite-backed L2 rolling session summaries, L3 privacy-isolated memories, and W5 Shared Cortex.

Backward-compatible facade re-exporting symbols from `cortex_cues`, `shared_cortex`, and `memory_store`.
"""

from __future__ import annotations

from qq_agent.cortex_cues import (
    MemoryFact,
    SharedFact,
    UserProfile,
    _ALNUM_COMPOUND_RE,
    _ALNUM_PART_RE,
    _ANGLE_BRACKET_RE,
    _AT_MENTION_RE,
    _CJK_FILLER_PHRASES,
    _CJK_FILLER_RE,
    _CJK_PARTICLE_RE,
    _CJK_RUN_RE,
    _CQ_CODE_RE,
    _DOTTED_VERSION_RE,
    _EFFECTIVE_CHAR_RE,
    _EMAIL_RE,
    _EN_STOPWORDS,
    _GENERIC_PREDICATE_BIGRAMS,
    _LONG_DIGITS_RE,
    _MEMORY_TAG_CLOSED_RE,
    _MEMORY_TAG_OPEN_FRAGMENT_RE,
    _MIXED_ALPHA_CJK_RE,
    _PHONE_LIKE_RE,
    _PUBLIC_CUE_RULES,
    _PURE_DIGITS_RE,
    _QueryFeatures,
    _SELF_STATE_MAX_CHARS,
    _SELF_STATE_PREFIX_RE,
    _SHARED_FACT_ERROR_MARKERS,
    _SHARED_FACT_MAX_BRIEF_CHARS,
    _TOPIC_KEY_CLEAN_RE,
    _URL_RE,
    _XML_TAG_RE,
    _compile_public_cue_rules,
    _extract_query_features,
    _is_high_distinctiveness_token,
    _is_ignored_alnum_token,
    _is_safe_core_containment,
    _normalize_topic_key,
    _sanitize_memory_text,
    _sanitize_public_self_topic,
    extract_public_interest_cues,
)
from qq_agent.memory_store import MemoryStore, _VALID_ROLES, logger
from qq_agent.shared_cortex import SharedCortexMixin

__all__ = [
    "MemoryFact",
    "MemoryStore",
    "SharedCortexMixin",
    "SharedFact",
    "UserProfile",
    "extract_public_interest_cues",
]
