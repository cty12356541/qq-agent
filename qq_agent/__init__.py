"""qq-agent: Async Pi-Agent QQ companion running on llmos ChannelControl SABI v1.0."""

from __future__ import annotations

from .agent import AgentKernel, ManagedLlmosDaemon
from .channel_bridge import (
    AckCoordinator,
    AdaptivePoller,
    BusQueueMetrics,
    DirectMemoryBus,
    LlmosChannelBus,
    MessageBus,
    NapCatWsAdapter,
    ResilientLlmosBus,
)
from .commands import AdminCommandRouter
from .config import AgentConfig, BusMode, RuntimeRole, load_dotenv_file
from .guardrails import GuardrailManager, QuotaSnapshot
from .memory import MemoryFact, MemoryStore, UserProfile
from .ob11 import format_forward_nodes
from .persona import PersonaConfig, PersonaManager
from .pi_core import (
    BudgetBlockedError,
    CancelToken,
    RegisteredTool,
    SessionActor,
    SubAgentRunner,
    agent_loop,
)
from .provider import LLMProvider, OpenAICompatProvider, ProviderError
from .session import SessionStore
from .stickers import StickerEntry, StickerRegistry
from .tools import ChatTurnState, ToolBinding, WebPageFetcher, WebSearchClient
from .trigger import DEFAULT_TOPIC_KEYWORDS, RuleTrigger
from .types import (
    ChatMessage,
    ChatResult,
    CommandResult,
    GuardDecision,
    InboundMessage,
    OutboundReply,
    SubAgentRequest,
    SubAgentResult,
    ToolCall,
    ToolDef,
    ToolOutcome,
    TriggerDecision,
)

__version__ = "0.3.0"

__all__ = [
    "AckCoordinator",
    "AdaptivePoller",
    "AdminCommandRouter",
    "AgentConfig",
    "AgentKernel",
    "BudgetBlockedError",
    "BusMode",
    "BusQueueMetrics",
    "CancelToken",
    "ChatMessage",
    "ChatResult",
    "ChatTurnState",
    "CommandResult",
    "DEFAULT_TOPIC_KEYWORDS",
    "DirectMemoryBus",
    "GuardDecision",
    "GuardrailManager",
    "InboundMessage",
    "LLMProvider",
    "LlmosChannelBus",
    "ManagedLlmosDaemon",
    "MemoryFact",
    "MemoryStore",
    "MessageBus",
    "NapCatWsAdapter",
    "OpenAICompatProvider",
    "OutboundReply",
    "PersonaConfig",
    "PersonaManager",
    "ProviderError",
    "QuotaSnapshot",
    "RegisteredTool",
    "ResilientLlmosBus",
    "RuleTrigger",
    "RuntimeRole",
    "SessionActor",
    "SessionStore",
    "StickerEntry",
    "StickerRegistry",
    "SubAgentRequest",
    "SubAgentResult",
    "SubAgentRunner",
    "ToolBinding",
    "ToolCall",
    "ToolDef",
    "ToolOutcome",
    "TriggerDecision",
    "UserProfile",
    "WebPageFetcher",
    "WebSearchClient",
    "__version__",
    "agent_loop",
    "format_forward_nodes",
    "load_dotenv_file",
]
