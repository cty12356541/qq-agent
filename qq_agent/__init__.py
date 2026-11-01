"""qq-agent: QQ group/private AI companion running on llmos ChannelControl SABI v1.0."""

from __future__ import annotations

from .agent import AgentKernel, ManagedLlmosDaemon
from .channel_bridge import DirectMemoryBus, LlmosChannelBus, MessageBus, NapCatWsAdapter
from .commands import AdminCommandRouter
from .config import AgentConfig, BusMode, load_dotenv_file
from .guardrails import GuardrailManager, QuotaSnapshot
from .persona import PersonaConfig, PersonaManager
from .provider import LLMProvider, OpenAICompatProvider, ProviderError
from .session import SessionStore
from .trigger import RuleTrigger
from .types import (
    ChatMessage,
    ChatResult,
    CommandResult,
    GuardDecision,
    InboundMessage,
    OutboundReply,
    TriggerDecision,
)

__version__ = "0.1.0"

__all__ = [
    "AdminCommandRouter",
    "AgentConfig",
    "AgentKernel",
    "BusMode",
    "ChatMessage",
    "ChatResult",
    "CommandResult",
    "DirectMemoryBus",
    "GuardDecision",
    "GuardrailManager",
    "InboundMessage",
    "LLMProvider",
    "LlmosChannelBus",
    "ManagedLlmosDaemon",
    "MessageBus",
    "NapCatWsAdapter",
    "OpenAICompatProvider",
    "OutboundReply",
    "PersonaConfig",
    "PersonaManager",
    "ProviderError",
    "QuotaSnapshot",
    "RuleTrigger",
    "SessionStore",
    "TriggerDecision",
    "__version__",
    "load_dotenv_file",
]
