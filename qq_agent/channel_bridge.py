# qq_agent/channel_bridge.py
"""双队列消息总线（DirectMemoryBus / LlmosChannelBus）与 NapCat OB11 正向 WS 适配器兼容门面。"""

from __future__ import annotations

from .bus_base import (
    AckCoordinator,
    AdaptivePoller,
    BusQueueMetrics,
    DEFAULT_CONSUMER_FIBER_ID,
    DirectMemoryBus,
    MessageBus,
    _MonotonicAckQueue,
    _validate_nonzero_id16,
)
from .llmos_bus import (
    CONSUMER_FIBER_GENERATION,
    LlmosChannelBus,
    MAX_REGISTERED_SEQ_CACHE,
    ResilientLlmosBus,
    _accepts_kwarg,
    _extract_acked_seq,
    _extract_backlog_bytes,
    _extract_trim_seq,
    _load_channel_client_cls,
    _random_nonzero_id16,
    _try_parse_int,
)
from .napcat_ws import (
    MSG_CACHE_CAPACITY,
    MsgCacheEntry,
    NapCatWsAdapter,
    OUTBOUND_ECHO_CACHE_CAPACITY,
)

__all__ = [
    "AckCoordinator",
    "AdaptivePoller",
    "BusQueueMetrics",
    "CONSUMER_FIBER_GENERATION",
    "DEFAULT_CONSUMER_FIBER_ID",
    "DirectMemoryBus",
    "LlmosChannelBus",
    "MAX_REGISTERED_SEQ_CACHE",
    "MSG_CACHE_CAPACITY",
    "MessageBus",
    "MsgCacheEntry",
    "NapCatWsAdapter",
    "OUTBOUND_ECHO_CACHE_CAPACITY",
    "ResilientLlmosBus",
    "_load_channel_client_cls",
]
