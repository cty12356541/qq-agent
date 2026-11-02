"""HTTP JSON API and background server for the qq-agent Web Admin Dashboard."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .config import AgentConfig
from .dashboard_ui import DASHBOARD_HTML
from .guardrails import GuardrailManager
from .kernel_maintenance import request_brain_reload_sentinel
from .memory import MemoryStore
from .persona import PersonaManager
from .session import SessionStore

logger = logging.getLogger(__name__)

_UNTRUSTED_MSG_RE = re.compile(
    r'<untrusted_chat_message[^>]*sender_id="([^"]*)"[^>]*sender_name="([^"]*)"[^>]*>\s*(.*?)\s*</untrusted_chat_message>',
    flags=re.DOTALL,
)


def _is_pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return int(exit_code.value) == STILL_ACTIVE
            return True
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _parse_l1_message_for_display(role: str, content: str) -> tuple[str, str]:
    """Extract readable `(speaker_label, display_text)` from raw L1 `<untrusted_chat_message>` content."""
    if role == "assistant":
        return "机器人 (Bot)", content
    match = _UNTRUSTED_MSG_RE.search(content)
    if match:
        sender_id, sender_name, body = match.group(1), match.group(2), match.group(3).strip()
        label = f"{sender_name} ({sender_id})" if sender_name else f"用户 ({sender_id})"
        return label, body
    return role, content


class DashboardService:
    """Thread-safe service layer backing the Web Admin Dashboard REST endpoints."""

    def __init__(self, cfg: AgentConfig) -> None:
        self.cfg = cfg

    def _open_persona(self) -> PersonaManager:
        return PersonaManager(
            self.cfg.persona_path,
            local_overlay_path=self.cfg.effective_persona_local_path,
            personas_dir=self.cfg.effective_personas_dir,
        )

    def _open_memory(self) -> MemoryStore:
        self.cfg.data_dir.mkdir(parents=True, exist_ok=True)
        return MemoryStore(self.cfg.memory_db_path)

    def _open_sessions(self) -> SessionStore:
        self.cfg.data_dir.mkdir(parents=True, exist_ok=True)
        return SessionStore(self.cfg.sessions_db_path, max_window=24)

    def _open_guardrails(self) -> GuardrailManager:
        self.cfg.data_dir.mkdir(parents=True, exist_ok=True)
        return GuardrailManager(
            self.cfg.guardrails_db_path,
            user_cooldown_s=1.0,
            max_daily_calls=self.cfg.daily_max_calls,
            max_daily_tokens=self.cfg.daily_max_tokens,
            soft_daily_budget_micros=self.cfg.daily_soft_cost_micros_usd,
            hard_daily_budget_micros=self.cfg.daily_hard_cost_micros_usd,
            cheap_model=self.cfg.llm_cheap_model,
        )

    def get_overview(self) -> dict[str, Any]:
        if self.cfg.data_dir == Path("data") and Path(".env").is_file():
            try:
                env_cfg = AgentConfig.from_env(".env")
                self.cfg.allowed_group_ids = env_cfg.allowed_group_ids
                self.cfg.group_trigger_tiers = env_cfg.group_trigger_tiers
            except Exception:
                pass
        rt_data: dict[str, Any] = {"status": "stopped"}
        if self.cfg.llmos_runtime_state_path.is_file():
            try:
                rt_data = json.loads(
                    self.cfg.llmos_runtime_state_path.read_text(encoding="utf-8")
                )
            except Exception:
                pass
        rt_data["supervisor_alive"] = _is_pid_alive(rt_data.get("supervisor_pid"))
        rt_data["daemon_alive"] = _is_pid_alive(rt_data.get("daemon_pid"))
        rt_data["adapter_alive"] = _is_pid_alive(rt_data.get("adapter_pid"))
        rt_data["brain_alive"] = _is_pid_alive(rt_data.get("brain_pid"))

        persona = self._open_persona()
        p_cfg = persona.current()
        profiles = persona.list_profiles()

        with self._open_guardrails() as guard:
            snap = guard.snapshot()

        self_state_obj: dict[str, Any] = {"state_text": "", "updated_at": 0.0}
        shared_facts_list: list[dict[str, Any]] = []
        with self._open_memory() as mem:
            for sf in mem.list_shared_facts():
                shared_facts_list.append(
                    {
                        "topic_key": sf.topic_key,
                        "query_text": sf.query_text,
                        "brief_text": sf.brief_text,
                        "updated_at": sf.updated_at,
                        "expires_at": sf.expires_at,
                    }
                )
        if self.cfg.memory_db_path.is_file():
            with sqlite3.connect(self.cfg.memory_db_path) as conn:
                row = conn.execute(
                    "SELECT state_text, updated_at FROM bot_self_state WHERE id = 1"
                ).fetchone()
                if row:
                    self_state_obj = {
                        "state_text": str(row[0] or ""),
                        "updated_at": float(row[1] or 0.0),
                    }

        return {
            "runtime": rt_data,
            "persona": {
                "name": p_cfg.name,
                "aliases": list(p_cfg.aliases),
                "active_profile": p_cfg.active_profile or "default",
                "available_profiles": ["default", *[p for p in profiles if p != "default"]],
            },
            "guardrails": {
                "day_key": getattr(snap, "date_str", ""),
                "calls_used": snap.calls_used,
                "max_daily_calls": snap.max_daily_calls,
                "tokens_used": snap.tokens_used,
                "max_daily_tokens": snap.max_daily_tokens,
                "cost_micros_used": snap.cost_micros_used,
                "soft_budget_micros": snap.soft_budget_micros,
                "hard_budget_micros": snap.hard_budget_micros,
            },
            "trigger": {
                "followup_window_s": self.cfg.group_followup_window_s,
                "followup_max_turns": self.cfg.group_followup_max_turns,
                "topic_prob": self.cfg.group_topic_prob,
                "proactive_prob": self.cfg.group_proactive_prob,
                "proactive_every_n": self.cfg.group_proactive_every_n,
                "proactive_cooldown_s": self.cfg.group_proactive_cooldown_s,
                "allowed_group_ids": (
                    sorted(self.cfg.allowed_group_ids) if self.cfg.allowed_group_ids else []
                ),
                "group_tiers": {str(k): v for k, v in sorted(self.cfg.group_trigger_tiers.items())},
            },
            "self_state": self_state_obj,
            "shared_facts": shared_facts_list,
        }

    # ------------------------------------------------------------------
    # User Profiles CRUD
    # ------------------------------------------------------------------

    def list_profiles(self) -> list[dict[str, Any]]:
        with self._open_memory():
            pass
        with sqlite3.connect(self.cfg.memory_db_path) as conn:
            rows = conn.execute(
                """
                SELECT user_id, display_name, traits_text, updated_at
                FROM user_profiles
                ORDER BY CASE WHEN traits_text != '' THEN 0 ELSE 1 END, updated_at DESC
                """
            ).fetchall()
        return [
            {
                "user_id": int(r[0]),
                "display_name": str(r[1] or ""),
                "traits_text": str(r[2] or ""),
                "updated_at": float(r[3] or 0.0),
            }
            for r in rows
        ]

    def upsert_profile(
        self,
        *,
        user_id: int,
        display_name: str,
        traits_text: str,
    ) -> dict[str, Any]:
        uid = int(user_id)
        now = time.time()
        with self._open_memory():
            pass
        with sqlite3.connect(self.cfg.memory_db_path) as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO user_profiles (user_id, display_name, traits_text, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        display_name = excluded.display_name,
                        traits_text = excluded.traits_text,
                        updated_at = excluded.updated_at
                    """,
                    (uid, display_name.strip(), traits_text.strip(), now),
                )
        return {
            "user_id": uid,
            "display_name": display_name.strip(),
            "traits_text": traits_text.strip(),
            "updated_at": now,
        }

    def delete_profile(self, user_id: int) -> bool:
        with self._open_memory():
            pass
        with sqlite3.connect(self.cfg.memory_db_path) as conn:
            with conn:
                cur = conn.execute(
                    "DELETE FROM user_profiles WHERE user_id = ?",
                    (int(user_id),),
                )
                return int(cur.rowcount) > 0

    # ------------------------------------------------------------------
    # L3 Circle Memories CRUD
    # ------------------------------------------------------------------

    def list_memories(self) -> list[dict[str, Any]]:
        with self._open_memory() as mem:
            items = mem.list_memories(include_admin_private=True)
        return [
            {
                "id": m.id,
                "scope": m.scope,
                "content": m.content,
                "created_at": m.created_at,
            }
            for m in reversed(items)
        ]

    def save_memory(
        self,
        *,
        scope: str,
        content: str,
        fact_id: int | None = None,
    ) -> dict[str, Any]:
        cleaned_scope = scope.strip()
        cleaned_content = content.strip()
        if not cleaned_scope or not cleaned_content:
            raise ValueError("scope 与 content 不能为空")

        if fact_id is None:
            with self._open_memory() as mem:
                created = mem.add_memory(cleaned_content, scope=cleaned_scope)
            return {
                "id": created.id,
                "scope": created.scope,
                "content": created.content,
                "created_at": created.created_at,
            }

        now = time.time()
        with self._open_memory():
            pass
        with sqlite3.connect(self.cfg.memory_db_path) as conn:
            with conn:
                conn.execute(
                    "UPDATE circle_memories SET scope = ?, content = ? WHERE id = ?",
                    (cleaned_scope, cleaned_content, int(fact_id)),
                )
        return {
            "id": int(fact_id),
            "scope": cleaned_scope,
            "content": cleaned_content,
            "created_at": now,
        }

    def delete_memory(self, fact_id: int) -> bool:
        with self._open_memory() as mem:
            return mem.delete_memory(int(fact_id))

    # ------------------------------------------------------------------
    # L2 Rolling Summaries & Shared Cortex CRUD
    # ------------------------------------------------------------------

    def list_summaries(self) -> list[dict[str, Any]]:
        with self._open_memory():
            pass
        with sqlite3.connect(self.cfg.memory_db_path) as conn:
            rows = conn.execute(
                """
                SELECT session_key, summary_text, pending_json, updated_at
                FROM session_summaries
                ORDER BY updated_at DESC
                """
            ).fetchall()
        result: list[dict[str, Any]] = []
        for r in rows:
            try:
                pending = json.loads(str(r[2] or "[]"))
                pending_count = len(pending) if isinstance(pending, list) else 0
            except Exception:
                pending_count = 0
            result.append(
                {
                    "session_key": str(r[0]),
                    "summary_text": str(r[1] or ""),
                    "pending_count": pending_count,
                    "updated_at": float(r[3] or 0.0),
                }
            )
        return result

    def set_summary(self, session_key: str, summary_text: str) -> None:
        key = session_key.strip()
        if not key:
            raise ValueError("session_key 不能为空")
        with self._open_memory() as mem:
            mem.set_summary(key, summary_text.strip())

    def set_self_state(self, state_text: str) -> None:
        text = state_text.strip()
        now = time.time()
        with self._open_memory():
            pass
        with sqlite3.connect(self.cfg.memory_db_path) as conn:
            with conn:
                if not text:
                    conn.execute("DELETE FROM bot_self_state WHERE id = 1")
                else:
                    conn.execute(
                        """
                        INSERT INTO bot_self_state (id, state_text, updated_at)
                        VALUES (1, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            state_text = excluded.state_text,
                            updated_at = excluded.updated_at
                        """,
                        (text[:42], now),
                    )

    def delete_shared_facts(self, topic_key: str | None = None) -> int:
        with self._open_memory():
            pass
        with sqlite3.connect(self.cfg.memory_db_path) as conn:
            with conn:
                if topic_key:
                    cur = conn.execute(
                        "DELETE FROM shared_facts WHERE topic_key = ?",
                        (topic_key,),
                    )
                else:
                    cur = conn.execute("DELETE FROM shared_facts")
                return int(cur.rowcount)

    # ------------------------------------------------------------------
    # L1 Sessions Inspector
    # ------------------------------------------------------------------

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._open_sessions():
            pass
        with sqlite3.connect(self.cfg.sessions_db_path) as conn:
            rows = conn.execute(
                """
                SELECT session_key, COUNT(*) AS cnt, MAX(created_at) AS last_ts
                FROM session_messages
                GROUP BY session_key
                ORDER BY last_ts DESC
                """
            ).fetchall()
        return [
            {
                "session_key": str(r[0]),
                "count": int(r[1]),
                "last_ts": float(r[2] or 0.0),
            }
            for r in rows
        ]

    def get_session_messages(self, session_key: str) -> list[dict[str, Any]]:
        with self._open_sessions() as sess:
            msgs = sess.get_window(session_key.strip())
        out: list[dict[str, Any]] = []
        for m in msgs:
            speaker, display_text = _parse_l1_message_for_display(m.role, m.content)
            out.append(
                {
                    "role": m.role,
                    "speaker": speaker,
                    "display_text": display_text,
                    "content": m.content,
                    "created_at": m.created_at,
                }
            )
        return out

    def clear_session(self, session_key: str) -> None:
        with self._open_sessions() as sess:
            sess.clear(session_key.strip())

    # ------------------------------------------------------------------
    # Actions (Persona Switch, Brain Reload, Quota Reset)
    # ------------------------------------------------------------------

    def switch_persona(self, profile: str) -> dict[str, Any]:
        persona = self._open_persona()
        target = profile.strip()
        if target == "default":
            updated = persona.reset()
        else:
            updated = persona.use_profile(target)
        request_brain_reload_sentinel(self.cfg.brain_reload_sentinel_path)
        return {
            "name": updated.name,
            "active_profile": updated.active_profile or "default",
            "aliases": list(updated.aliases),
        }

    def reload_brain(self) -> None:
        request_brain_reload_sentinel(self.cfg.brain_reload_sentinel_path)

    def reset_quota(self) -> None:
        with self._open_guardrails():
            pass
        with sqlite3.connect(self.cfg.guardrails_db_path) as conn:
            with conn:
                conn.execute("DELETE FROM daily_usage")


class _DashboardRequestHandler(BaseHTTPRequestHandler):
    service: DashboardService

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, data: Any, *, status: int = 200) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_html(self, html: str) -> None:
        payload = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        return json.loads(raw) if raw.strip() else {}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                self._send_html(DASHBOARD_HTML)
                return
            if path == "/api/overview":
                self._send_json(self.service.get_overview())
                return
            if path == "/api/profiles":
                self._send_json({"items": self.service.list_profiles()})
                return
            if path == "/api/memories":
                self._send_json({"items": self.service.list_memories()})
                return
            if path == "/api/summaries":
                self._send_json({"items": self.service.list_summaries()})
                return
            if path == "/api/sessions":
                session_key = (qs.get("session_key") or [""])[0]
                if session_key:
                    self._send_json(
                        {
                            "session_key": session_key,
                            "messages": self.service.get_session_messages(session_key),
                        }
                    )
                else:
                    self._send_json({"sessions": self.service.list_sessions()})
                return
            self._send_json({"error": f"Not found: {path}"}, status=404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            body = self._read_json_body()
            if path == "/api/profiles":
                item = self.service.upsert_profile(
                    user_id=int(body["user_id"]),
                    display_name=str(body.get("display_name", "")),
                    traits_text=str(body.get("traits_text", "")),
                )
                self._send_json({"ok": True, "item": item})
                return
            if path == "/api/memories":
                fact_id = body.get("id")
                item = self.service.save_memory(
                    scope=str(body.get("scope", "public")),
                    content=str(body.get("content", "")),
                    fact_id=int(fact_id) if fact_id is not None else None,
                )
                self._send_json({"ok": True, "item": item})
                return
            if path == "/api/summaries":
                self.service.set_summary(
                    str(body.get("session_key", "")),
                    str(body.get("summary_text", "")),
                )
                self._send_json({"ok": True})
                return
            if path == "/api/self-state":
                self.service.set_self_state(str(body.get("state_text", "")))
                self._send_json({"ok": True})
                return
            if path == "/api/actions/switch-persona":
                res = self.service.switch_persona(str(body.get("profile", "default")))
                self._send_json({"ok": True, "persona": res})
                return
            if path == "/api/actions/reload-brain":
                self.service.reload_brain()
                self._send_json({"ok": True})
                return
            if path == "/api/actions/reset-quota":
                self.service.reset_quota()
                self._send_json({"ok": True})
                return
            self._send_json({"error": f"Not found: {path}"}, status=404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if path.startswith("/api/profiles/"):
                uid = int(unquote(path.split("/api/profiles/", 1)[1]))
                self.service.delete_profile(uid)
                self._send_json({"ok": True})
                return
            if path.startswith("/api/memories/"):
                mid = int(unquote(path.split("/api/memories/", 1)[1]))
                self.service.delete_memory(mid)
                self._send_json({"ok": True})
                return
            if path == "/api/shared-facts":
                topic_key = (qs.get("topic_key") or [None])[0]
                self.service.delete_shared_facts(topic_key)
                self._send_json({"ok": True})
                return
            if path == "/api/sessions":
                session_key = (qs.get("session_key") or [""])[0]
                if session_key:
                    self.service.clear_session(session_key)
                self._send_json({"ok": True})
                return
            self._send_json({"error": f"Not found: {path}"}, status=404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)


class DashboardHttpServer:
    """Background-thread HTTP server hosting the qq-agent Web Admin Dashboard."""

    def __init__(
        self,
        cfg: AgentConfig,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self.cfg = cfg
        self.host = host
        self.port = int(port)
        self.service = DashboardService(cfg)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        actual_port = self._httpd.server_port if self._httpd is not None else self.port
        return f"http://{self.host}:{actual_port}"

    def start(self) -> str:
        if self._httpd is not None:
            return self.url

        handler_cls = type(
            "BoundDashboardHandler",
            (_DashboardRequestHandler,),
            {"service": self.service},
        )
        ThreadingHTTPServer.allow_reuse_address = True
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler_cls)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="qq-agent-dashboard",
            daemon=True,
        )
        self._thread.start()
        logger.info("qq-agent Web Admin Dashboard listening on %s", self.url)
        return self.url

    def close(self) -> None:
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


def main() -> int:
    parser = argparse.ArgumentParser(description="qq-agent Web Admin Dashboard server")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--host", default=os.environ.get("QQ_DASHBOARD_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("QQ_DASHBOARD_PORT", "8765")),
    )
    args = parser.parse_args()
    cfg = AgentConfig.from_env(args.env_file)
    srv = DashboardHttpServer(cfg, host=args.host, port=args.port)
    url = srv.start()
    print(f"[qq-agent] 后台仪表盘已启动：{url}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        srv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = ["DashboardHttpServer", "DashboardService", "main"]
