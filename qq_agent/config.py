"""Environment and `.env` configuration loader for qq-agent (zero third-party dependencies)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Literal

from .types import RuntimeRole

BusMode = Literal["llmos", "direct-ws"]
_VALID_ROLES: frozenset[str] = frozenset({"supervisor", "adapter", "brain", "all-in-one"})


def _default_llmos_root() -> Path:
    preferred = Path(r"D:\llmos")
    if preferred.exists():
        return preferred
    return Path(r"D:\AI\llmos")


def _strip_value(raw_val: str) -> str:
    val = raw_val.strip()
    if len(val) >= 2 and (
        (val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")
    ):
        return val[1:-1]
    if val.startswith("#"):
        return ""
    for marker in (" #", "\t#"):
        idx = val.find(marker)
        if idx != -1:
            val = val[:idx].rstrip()
    return val


def load_dotenv_file(
    path: Path | str,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Parse a `.env` file into a dictionary without overwriting explicit process env vars."""
    env_path = Path(path)
    if not env_path.is_file():
        return {}

    active_env: Mapping[str, str] = os.environ if environ is None else environ
    parsed: dict[str, str] = {}

    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export ") and "=" in line[7:]:
            line = line[7:].strip()
        if "=" not in line:
            continue
        key_raw, val_raw = line.split("=", 1)
        key = key_raw.strip()
        if not key:
            continue
        value = _strip_value(val_raw)
        if key in active_env:
            parsed[key] = active_env[key]
        else:
            parsed[key] = value

    return parsed


def _parse_optional_int(raw: str | None, field_name: str) -> int | None:
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {field_name}: {raw!r}") from exc


def _parse_int_with_default(raw: str | None, default: int, field_name: str) -> int:
    parsed = _parse_optional_int(raw, field_name)
    return default if parsed is None else parsed


def _parse_float_with_default(raw: str | None, default: float, field_name: str) -> float:
    if raw is None:
        return default
    stripped = raw.strip()
    if not stripped:
        return default
    try:
        return float(stripped)
    except ValueError as exc:
        raise ValueError(f"Invalid float for {field_name}: {raw!r}") from exc


def _parse_bool_with_default(raw: str | None, default: bool, field_name: str) -> bool:
    if raw is None:
        return default
    token = raw.strip().lower()
    if not token:
        return default
    if token in {"1", "true", "yes", "on", "enabled"}:
        return True
    if token in {"0", "false", "no", "off", "disabled"}:
        return False
    raise ValueError(f"Invalid boolean for {field_name}: {raw!r}")


def _parse_int_set(raw: str | None, field_name: str) -> set[int]:
    if raw is None:
        return set()
    items: set[int] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            items.add(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid integer item {token!r} in {field_name}") from exc
    return items


def _parse_runtime_role(raw: str | None, default: RuntimeRole = "supervisor") -> RuntimeRole:
    if raw is None or not raw.strip():
        return default
    token = raw.strip().lower()
    if token not in _VALID_ROLES:
        raise ValueError(
            f"Invalid QQ_RUNTIME_ROLE: {raw!r} (expected one of {sorted(_VALID_ROLES)})"
        )
    return token  # type: ignore[return-value]


@dataclass(slots=True)
class AgentConfig:
    """Normalized runtime configuration aligned 1-to-1 with `.env.example`."""

    bot_qq: int | None = None
    admin_user_ids: set[int] = field(default_factory=set)
    allowed_group_ids: set[int] | None = None

    napcat_ws_url: str = "ws://127.0.0.1:3001"
    napcat_access_token: str | None = field(default=None, repr=False)

    llm_api_key: str = field(default="", repr=False)
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    llm_cheap_model: str | None = None
    llm_enable_thinking: bool = False
    enable_vision: bool = True

    llmos_root: Path = field(default_factory=_default_llmos_root)
    llmos_plain_endpoint: str | None = None
    runtime_role: RuntimeRole = "supervisor"

    data_dir: Path = field(default_factory=lambda: Path("data"))
    persona_path: Path = field(default_factory=lambda: Path("config/persona.default.toml"))
    persona_local_path: Path | None = None
    personas_dir: Path | None = None
    stickers_manifest_path: Path = field(default_factory=lambda: Path("config/stickers.toml"))

    private_debounce_s: float = 1.8
    group_debounce_s: float = 1.2
    group_followup_window_s: float = 60.0
    group_proactive_prob: float = 0.22
    group_topic_prob: float = 0.65
    group_proactive_every_n: int = 6
    group_proactive_cooldown_s: float = 12.0

    agent_max_steps: int = 3
    agent_turn_timeout_s: float = 35.0
    bubble_typing_delay_s: float = 0.45

    subagent_enabled: bool = True
    subagent_model: str | None = None
    subagent_enable_thinking: bool = True
    subagent_max_steps: int = 4
    subagent_timeout_s: float = 30.0
    subagent_max_concurrent: int = 2

    web_search_enabled: bool = True
    web_search_timeout_s: float = 8.0
    search_api_key: str | None = field(default=None, repr=False)
    search_api_base_url: str | None = None

    daily_max_calls: int = 200
    daily_max_tokens: int = 200_000
    daily_soft_cost_micros_usd: int = 1_000_000
    daily_hard_cost_micros_usd: int = 2_000_000

    mode: BusMode = "llmos"

    @property
    def sessions_db_path(self) -> Path:
        return self.data_dir / "sessions.sqlite3"

    @property
    def memory_db_path(self) -> Path:
        return self.data_dir / "memory.sqlite3"

    @property
    def guardrails_db_path(self) -> Path:
        return self.data_dir / "guardrails.sqlite3"

    @property
    def stickers_dir(self) -> Path:
        return self.data_dir / "stickers"

    @property
    def effective_persona_local_path(self) -> Path:
        if self.persona_local_path is not None:
            return self.persona_local_path
        return self.data_dir / "persona.local.toml"

    @property
    def effective_personas_dir(self) -> Path:
        if self.personas_dir is not None:
            return self.personas_dir
        return self.data_dir / "personas"

    @property
    def llmos_daemon_root(self) -> Path:
        return (self.data_dir / "llmos_daemon").resolve()

    @property
    def llmos_runtime_state_path(self) -> Path:
        return (self.data_dir / "llmos_runtime.json").resolve()

    @property
    def effective_subagent_model(self) -> str:
        return self.subagent_model if self.subagent_model else self.llm_model

    @classmethod
    def from_env(
        cls,
        env_file: Path | str | None = ".env",
        *,
        direct_ws: bool = False,
        environ: Mapping[str, str] | None = None,
    ) -> AgentConfig:
        """Load AgentConfig from process environment + optional `.env` file fallback."""
        active_env: Mapping[str, str] = os.environ if environ is None else environ
        dotenv_vars = (
            load_dotenv_file(env_file, environ=active_env)
            if env_file is not None
            else {}
        )

        def get_raw(key: str) -> str | None:
            if key in active_env:
                return active_env[key]
            return dotenv_vars.get(key)

        def get_nonempty(key: str, default: str) -> str:
            raw = get_raw(key)
            if raw is None:
                return default
            stripped = raw.strip()
            return stripped if stripped else default

        def get_optional_str(key: str) -> str | None:
            raw = get_raw(key)
            if raw is None:
                return None
            stripped = raw.strip()
            return stripped if stripped else None

        bot_qq = _parse_optional_int(get_raw("QQ_BOT_ID"), "QQ_BOT_ID")
        admin_ids = _parse_int_set(get_raw("QQ_ADMIN_IDS"), "QQ_ADMIN_IDS")
        allowed_groups_set = _parse_int_set(get_raw("QQ_ALLOWED_GROUPS"), "QQ_ALLOWED_GROUPS")
        allowed_groups: set[int] | None = allowed_groups_set if allowed_groups_set else None

        llmos_root_raw = get_optional_str("LLMOS_ROOT")
        llmos_root = Path(llmos_root_raw) if llmos_root_raw is not None else _default_llmos_root()

        persona_local_raw = get_optional_str("QQ_PERSONA_LOCAL_PATH")
        persona_local_path = Path(persona_local_raw) if persona_local_raw is not None else None

        personas_dir_raw = get_optional_str("QQ_PERSONAS_DIR")
        personas_dir = Path(personas_dir_raw) if personas_dir_raw is not None else None

        api_key_raw = get_raw("LLM_API_KEY")
        llm_api_key = api_key_raw.strip() if api_key_raw is not None else ""

        mode: BusMode = "direct-ws" if direct_ws else "llmos"
        default_role: RuntimeRole = "all-in-one" if direct_ws else "supervisor"
        runtime_role = _parse_runtime_role(get_raw("QQ_RUNTIME_ROLE"), default=default_role)

        return cls(
            bot_qq=bot_qq,
            admin_user_ids=admin_ids,
            allowed_group_ids=allowed_groups,
            napcat_ws_url=get_nonempty("NAPCAT_WS_URL", "ws://127.0.0.1:3001"),
            napcat_access_token=get_optional_str("NAPCAT_ACCESS_TOKEN"),
            llm_api_key=llm_api_key,
            llm_base_url=get_nonempty("LLM_BASE_URL", "https://api.deepseek.com/v1"),
            llm_model=get_nonempty("LLM_MODEL", "deepseek-chat"),
            llm_cheap_model=get_optional_str("LLM_CHEAP_MODEL"),
            llm_enable_thinking=_parse_bool_with_default(
                get_raw("LLM_ENABLE_THINKING"), False, "LLM_ENABLE_THINKING"
            ),
            enable_vision=_parse_bool_with_default(
                get_raw("QQ_ENABLE_VISION"), True, "QQ_ENABLE_VISION"
            ),
            llmos_root=llmos_root,
            llmos_plain_endpoint=get_optional_str("LLMOS_PLAIN_ENDPOINT"),
            runtime_role=runtime_role,
            data_dir=Path(get_nonempty("QQ_DATA_DIR", "data")),
            persona_path=Path(get_nonempty("QQ_PERSONA_PATH", "config/persona.default.toml")),
            persona_local_path=persona_local_path,
            personas_dir=personas_dir,
            stickers_manifest_path=Path(
                get_nonempty("QQ_STICKERS_MANIFEST", "config/stickers.toml")
            ),
            private_debounce_s=_parse_float_with_default(
                get_raw("QQ_PRIVATE_DEBOUNCE_S"), 1.8, "QQ_PRIVATE_DEBOUNCE_S"
            ),
            group_debounce_s=_parse_float_with_default(
                get_raw("QQ_GROUP_DEBOUNCE_S"), 1.2, "QQ_GROUP_DEBOUNCE_S"
            ),
            group_followup_window_s=_parse_float_with_default(
                get_raw("QQ_GROUP_FOLLOWUP_WINDOW_S"), 60.0, "QQ_GROUP_FOLLOWUP_WINDOW_S"
            ),
            group_proactive_prob=_parse_float_with_default(
                get_raw("QQ_GROUP_PROACTIVE_PROB"), 0.22, "QQ_GROUP_PROACTIVE_PROB"
            ),
            group_topic_prob=_parse_float_with_default(
                get_raw("QQ_GROUP_TOPIC_PROB"), 0.65, "QQ_GROUP_TOPIC_PROB"
            ),
            group_proactive_every_n=_parse_int_with_default(
                get_raw("QQ_GROUP_PROACTIVE_EVERY_N"), 6, "QQ_GROUP_PROACTIVE_EVERY_N"
            ),
            group_proactive_cooldown_s=_parse_float_with_default(
                get_raw("QQ_GROUP_PROACTIVE_COOLDOWN_S"), 12.0, "QQ_GROUP_PROACTIVE_COOLDOWN_S"
            ),
            agent_max_steps=_parse_int_with_default(
                get_raw("QQ_AGENT_MAX_STEPS"), 3, "QQ_AGENT_MAX_STEPS"
            ),
            agent_turn_timeout_s=_parse_float_with_default(
                get_raw("QQ_AGENT_TURN_TIMEOUT_S"), 35.0, "QQ_AGENT_TURN_TIMEOUT_S"
            ),
            bubble_typing_delay_s=_parse_float_with_default(
                get_raw("QQ_BUBBLE_TYPING_DELAY_S"), 0.45, "QQ_BUBBLE_TYPING_DELAY_S"
            ),
            subagent_enabled=_parse_bool_with_default(
                get_raw("SUBAGENT_ENABLED"), True, "SUBAGENT_ENABLED"
            ),
            subagent_model=get_optional_str("SUBAGENT_MODEL"),
            subagent_enable_thinking=_parse_bool_with_default(
                get_raw("SUBAGENT_ENABLE_THINKING"), True, "SUBAGENT_ENABLE_THINKING"
            ),
            subagent_max_steps=_parse_int_with_default(
                get_raw("SUBAGENT_MAX_STEPS"), 4, "SUBAGENT_MAX_STEPS"
            ),
            subagent_timeout_s=_parse_float_with_default(
                get_raw("SUBAGENT_TIMEOUT_S"), 30.0, "SUBAGENT_TIMEOUT_S"
            ),
            subagent_max_concurrent=_parse_int_with_default(
                get_raw("SUBAGENT_MAX_CONCURRENT"), 2, "SUBAGENT_MAX_CONCURRENT"
            ),
            web_search_enabled=_parse_bool_with_default(
                get_raw("WEB_SEARCH_ENABLED"), True, "WEB_SEARCH_ENABLED"
            ),
            web_search_timeout_s=_parse_float_with_default(
                get_raw("WEB_SEARCH_TIMEOUT_S"), 8.0, "WEB_SEARCH_TIMEOUT_S"
            ),
            search_api_key=get_optional_str("SEARCH_API_KEY"),
            search_api_base_url=get_optional_str("SEARCH_API_BASE_URL"),
            daily_max_calls=_parse_int_with_default(
                get_raw("QQ_DAILY_MAX_CALLS"), 200, "QQ_DAILY_MAX_CALLS"
            ),
            daily_max_tokens=_parse_int_with_default(
                get_raw("QQ_DAILY_MAX_TOKENS"), 200_000, "QQ_DAILY_MAX_TOKENS"
            ),
            daily_soft_cost_micros_usd=_parse_int_with_default(
                get_raw("QQ_DAILY_SOFT_COST_MICROS_USD"),
                1_000_000,
                "QQ_DAILY_SOFT_COST_MICROS_USD",
            ),
            daily_hard_cost_micros_usd=_parse_int_with_default(
                get_raw("QQ_DAILY_HARD_COST_MICROS_USD"),
                2_000_000,
                "QQ_DAILY_HARD_COST_MICROS_USD",
            ),
            mode=mode,
        )


__all__ = [
    "AgentConfig",
    "BusMode",
    "RuntimeRole",
    "load_dotenv_file",
]
