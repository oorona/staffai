import asyncio
import json
import logging
import os
import random
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import redis
from openai import AsyncOpenAI

from utils.litellm_client import LiteLLMClient

logger = logging.getLogger(__name__)


class UserMemoryManager:
    """Persistent user memory manager (Redis + async file backup)."""

    PIPELINE_MODE_DISABLED = "disabled"
    PIPELINE_MODE_TINY_EXTRACT = "tiny_extract"
    PIPELINE_MODE_FRONTIER = "frontier_pipeline"
    PIPELINE_MODE_TINY_GATE_FRONTIER_CORE = "tiny_gate_frontier_core"
    _VALID_PIPELINE_MODES = {
        PIPELINE_MODE_DISABLED,
        PIPELINE_MODE_TINY_EXTRACT,
        PIPELINE_MODE_FRONTIER,
        PIPELINE_MODE_TINY_GATE_FRONTIER_CORE,
    }
    _MAX_MEMORY_PIECE_CHARS = 180
    _PROFILE_REASSESS_EVERY_WORTHWHILE = 8
    _DEFAULT_STYLE_TRAITS_LINE = "be calm, courteous, uplifting, practical."
    _DEFAULT_EXPERTISE_LEVEL = "intermediate"
    _VALID_EXPERTISE_LEVELS = {"beginner", "intermediate", "advanced"}
    _INJECTION_CONFIDENCE_LEVELS = {"low", "medium", "high"}
    _INJECTION_BLOCK_CONFIDENCE = "high"
    _MEMORY_FILENAME = "memory.json"
    _USER_LABEL_FILENAME = "user.txt"
    _STYLE_FILENAME = "style.json"
    _EXPERTISE_FILENAME = "expertise.json"

    def __init__(
        self,
        prompts_root_path: str,
        memory_root_path: str,
        redis_client: Optional[redis.Redis],
        litellm_client: LiteLLMClient,
        enabled: bool,
        update_chance: float,
        min_message_chars: int,
        min_message_words: int,
        max_memory_chars: int,
        pipeline_mode: str,
        ollama_base_url: str,
        ollama_api_key: str,
        ollama_timeout_seconds: float,
        tiny_model: str,
        tiny_model_extract: Optional[str],
        tiny_model_classifier: Optional[str],
        tiny_accumulate_max_tokens: int,
        memory_audit_max_entries: int,
        debug_classification_logs: bool,
    ):
        self.prompts_root_path = prompts_root_path
        self.memory_root_path = memory_root_path
        self.user_profiles_dir = self.memory_root_path
        self.redis_client = redis_client
        self.litellm_client = litellm_client

        self.enabled = enabled
        self.update_chance = update_chance
        self.min_message_chars = min_message_chars
        self.min_message_words = min_message_words
        self.max_memory_chars = max_memory_chars
        mode = (pipeline_mode or "").strip().lower()
        if mode in self._VALID_PIPELINE_MODES:
            self.pipeline_mode = mode
        else:
            self.pipeline_mode = self.PIPELINE_MODE_TINY_EXTRACT
            if mode:
                logger.warning(
                    "Invalid user-memory pipeline mode '%s'. Falling back to '%s'.",
                    mode,
                    self.PIPELINE_MODE_TINY_EXTRACT
                )

        self.ollama_timeout_seconds = ollama_timeout_seconds
        self.tiny_model_extract = (tiny_model_extract or tiny_model).strip()
        self.tiny_model_classifier = (tiny_model_classifier or tiny_model).strip()
        self.tiny_accumulate_max_tokens = tiny_accumulate_max_tokens
        self.max_storage_chars = max(self.max_memory_chars, self.tiny_accumulate_max_tokens * 6)
        self.memory_audit_max_entries = memory_audit_max_entries
        self.debug_classification_logs = debug_classification_logs

        proxy_url = (ollama_base_url or "").strip().rstrip("/")
        if proxy_url and not proxy_url.endswith("/v1"):
            proxy_url = f"{proxy_url}/v1"
        self.ollama_base_url = proxy_url
        self.ollama_client: Optional[AsyncOpenAI] = None
        if self.ollama_base_url:
            try:
                self.ollama_client = AsyncOpenAI(
                    base_url=self.ollama_base_url,
                    api_key=ollama_api_key or "ollama"
                )
            except Exception as e:
                logger.error("Failed to initialize Ollama client (%s): %s", self.ollama_base_url, e)

        # Prompt packs (single folder convention: <purpose>/{system_prompt.txt,user_prompt.txt,schema.json})
        memory_update_pack = os.path.join(self.prompts_root_path, "user_memory_frontier_update")
        tiny_worthwhile_pack = os.path.join(self.prompts_root_path, "user_memory_tiny_worthwhile")
        tiny_extract_pack = os.path.join(self.prompts_root_path, "user_memory_tiny_extract")
        tiny_compact_pack = os.path.join(self.prompts_root_path, "user_memory_tiny_compact")
        frontier_core_extract_pack = os.path.join(self.prompts_root_path, "user_memory_frontier_core_extract")
        style_extract_pack = os.path.join(self.prompts_root_path, "user_style_extract")
        expertise_extract_pack = os.path.join(self.prompts_root_path, "user_expertise_extract")
        injection_guard_pack = os.path.join(self.prompts_root_path, "user_memory_injection_guard")

        # Frontier memory update prompts/schemas
        self.memory_update_system_prompt_path = os.path.join(memory_update_pack, "system_prompt.txt")
        self.memory_update_user_prompt_path = os.path.join(memory_update_pack, "user_prompt.txt")
        self.memory_update_schema_path = os.path.join(memory_update_pack, "schema.json")

        # Tiny worthwhile classifier prompts/schemas
        self.tiny_worthwhile_system_prompt_path = os.path.join(tiny_worthwhile_pack, "system_prompt.txt")
        self.tiny_worthwhile_user_prompt_path = os.path.join(tiny_worthwhile_pack, "user_prompt.txt")
        self.tiny_worthwhile_schema_path = os.path.join(tiny_worthwhile_pack, "schema.json")

        # Tiny direct extraction prompts/schemas
        self.tiny_extract_system_prompt_path = os.path.join(tiny_extract_pack, "system_prompt.txt")
        self.tiny_extract_user_prompt_path = os.path.join(tiny_extract_pack, "user_prompt.txt")
        self.tiny_extract_schema_path = os.path.join(tiny_extract_pack, "schema.json")
        self.tiny_compact_system_prompt_path = os.path.join(tiny_compact_pack, "system_prompt.txt")
        self.tiny_compact_user_prompt_path = os.path.join(tiny_compact_pack, "user_prompt.txt")
        self.tiny_compact_schema_path = os.path.join(tiny_compact_pack, "schema.json")
        self.frontier_core_extract_system_prompt_path = os.path.join(frontier_core_extract_pack, "system_prompt.txt")
        self.frontier_core_extract_user_prompt_path = os.path.join(frontier_core_extract_pack, "user_prompt.txt")
        self.frontier_core_extract_schema_path = os.path.join(frontier_core_extract_pack, "schema.json")
        self.style_extract_system_prompt_path = os.path.join(style_extract_pack, "system_prompt.txt")
        self.style_extract_user_prompt_path = os.path.join(style_extract_pack, "user_prompt.txt")
        self.style_extract_schema_path = os.path.join(style_extract_pack, "schema.json")
        self.expertise_extract_system_prompt_path = os.path.join(expertise_extract_pack, "system_prompt.txt")
        self.expertise_extract_user_prompt_path = os.path.join(expertise_extract_pack, "user_prompt.txt")
        self.expertise_extract_schema_path = os.path.join(expertise_extract_pack, "schema.json")
        self.injection_guard_system_prompt_path = os.path.join(injection_guard_pack, "system_prompt.txt")
        self.injection_guard_user_prompt_path = os.path.join(injection_guard_pack, "user_prompt.txt")
        self.injection_guard_schema_path = os.path.join(injection_guard_pack, "schema.json")

        self._user_locks: Dict[int, asyncio.Lock] = {}
        self._pending_file_payloads: Dict[int, Dict[str, Any]] = {}
        self._file_writer_tasks: Dict[int, asyncio.Task] = {}
        self._style_reassess_counter_local: Dict[int, int] = {}
        self._expertise_reassess_counter_local: Dict[int, int] = {}
        self._history_profile_bootstrap_attempted_local: Dict[int, bool] = {}

        os.makedirs(self.user_profiles_dir, exist_ok=True)

    @staticmethod
    def _read_text_file(path: str) -> Optional[str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = f.read().strip()
                return data if data else None
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error("Error reading text file %s: %s", path, e, exc_info=True)
            return None

    @staticmethod
    def _read_json_file(path: str) -> Optional[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else None
        except FileNotFoundError:
            return None
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON file %s: %s", path, e)
            return None
        except Exception as e:
            logger.error("Error reading JSON file %s: %s", path, e, exc_info=True)
            return None

    @staticmethod
    def _required_keys_from_response_schema(schema: Dict[str, Any]) -> set[str]:
        try:
            inner_schema = schema.get("json_schema", {}).get("schema", {})
            required = inner_schema.get("required", [])
            if isinstance(required, list):
                return {str(k) for k in required}
        except Exception:
            pass
        return set()

    @staticmethod
    def _extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
        text = (raw_text or "").strip()
        if not text:
            return None

        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        try:
            parsed = json.loads(text[start:end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    def _memory_key(self, user_id: int) -> str:
        return f"user_memory:{user_id}"

    def _memory_pipeline_audit_key(self) -> str:
        return "user_memory_pipeline:recent"

    def _user_dir_path(self, user_id: int) -> str:
        return os.path.join(self.user_profiles_dir, str(user_id))

    def _user_file_path(self, user_id: int) -> str:
        return os.path.join(self._user_dir_path(user_id), self._MEMORY_FILENAME)

    def _user_label_file_path(self, user_id: int) -> str:
        return os.path.join(self._user_dir_path(user_id), self._USER_LABEL_FILENAME)

    def _style_key(self, user_id: int) -> str:
        return f"user_style:{user_id}"

    def _style_file_path(self, user_id: int) -> str:
        return os.path.join(self._user_dir_path(user_id), self._STYLE_FILENAME)

    def _style_reassess_counter_key(self, user_id: int) -> str:
        return f"user_style_reassess_counter:{user_id}"

    def _expertise_key(self, user_id: int) -> str:
        return f"user_expertise:{user_id}"

    def _expertise_file_path(self, user_id: int) -> str:
        return os.path.join(self._user_dir_path(user_id), self._EXPERTISE_FILENAME)

    def _expertise_reassess_counter_key(self, user_id: int) -> str:
        return f"user_expertise_reassess_counter:{user_id}"

    def _history_profile_bootstrap_attempted_key(self, user_id: int) -> str:
        return f"user_profile_history_bootstrap_attempted:{user_id}"

    def _lock_for_user(self, user_id: int) -> asyncio.Lock:
        lock = self._user_locks.get(user_id)
        if lock is None:
            lock = asyncio.Lock()
            self._user_locks[user_id] = lock
        return lock

    def _normalize_memory(self, memory: str, max_chars: Optional[int] = None) -> str:
        raw = (memory or "").replace("\r\n", "\n").replace("\r", "\n")
        normalized_lines = []
        for line in raw.split("\n"):
            compact = re.sub(r"\s+", " ", line).strip()
            if compact:
                normalized_lines.append(compact)
        clean = "\n".join(normalized_lines).strip()
        if max_chars and max_chars > 0 and len(clean) > max_chars:
            clean = clean[: max_chars].rstrip()
        return clean

    @staticmethod
    def _normalize_user_label(user_label: Optional[str]) -> str:
        return re.sub(r"\s+", " ", str(user_label or "")).strip()

    @classmethod
    def _is_formatted_user_label(cls, user_label: Optional[str]) -> bool:
        label = cls._normalize_user_label(user_label)
        if not label:
            return False
        open_idx = label.rfind("(")
        close_idx = label.endswith(")")
        return open_idx > 0 and close_idx and open_idx < len(label) - 2

    def _ensure_user_label_file(
        self,
        user_id: int,
        *,
        user_label: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> None:
        path = self._user_label_file_path(user_id)
        if not overwrite and os.path.isfile(path):
            return

        label_source = user_label
        if not label_source and payload:
            label_source = payload.get("user_label")
            if not self._is_formatted_user_label(label_source):
                label_source = None

        label = self._normalize_user_label(label_source)
        if not label:
            return

        os.makedirs(self._user_dir_path(user_id), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(label)

    @staticmethod
    def _normalize_style_traits(raw_traits: List[str]) -> List[str]:
        seen = set()
        normalized: List[str] = []
        for raw in raw_traits:
            trait = re.sub(r"\s+", " ", str(raw or "")).strip().lower()
            trait = trait.strip(".,;:!\"'` ")
            if not trait:
                continue
            if len(trait) > 24:
                trait = trait[:24].rstrip()
            if trait in seen:
                continue
            if not re.match(r"^[a-záéíóúñü0-9][a-záéíóúñü0-9\- ]*$", trait):
                continue
            seen.add(trait)
            normalized.append(trait)
            if len(normalized) >= 5:
                break
        return normalized

    @classmethod
    def _build_style_line(cls, traits: List[str]) -> str:
        valid_traits = cls._normalize_style_traits(traits)
        if not valid_traits:
            return cls._DEFAULT_STYLE_TRAITS_LINE
        return f"be {', '.join(valid_traits)}."

    @classmethod
    def _normalize_expertise_level(cls, raw_level: Optional[str]) -> Optional[str]:
        normalized = re.sub(r"\s+", " ", str(raw_level or "")).strip().lower()
        if normalized not in cls._VALID_EXPERTISE_LEVELS:
            return None
        return normalized

    @classmethod
    def _normalize_injection_confidence(cls, raw_confidence: Optional[str]) -> str:
        normalized = re.sub(r"\s+", " ", str(raw_confidence or "")).strip().lower()
        if normalized not in cls._INJECTION_CONFIDENCE_LEVELS:
            return "low"
        return normalized

    @staticmethod
    def _flatten_simple_json_object_text(text: str) -> str:
        try:
            data = json.loads(text)
        except Exception:
            return text
        if not isinstance(data, dict) or not data:
            return text

        parts = []
        for key, value in data.items():
            k = re.sub(r"\s+", " ", str(key)).strip()
            v = re.sub(r"\s+", " ", str(value)).strip()
            if k and v:
                parts.append(f"{k}: {v}")
        return "; ".join(parts) if parts else text

    def _sanitize_memory_piece(self, memory_piece: str) -> str:
        text = (memory_piece or "").strip()
        if not text:
            return ""

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.strip("`\"' ")
        text = re.sub(r"^\s*[-*•]+\s*", "", text)

        if text.startswith("{") and text.endswith("}"):
            text = self._flatten_simple_json_object_text(text)

        text = re.sub(r"\s+", " ", text).strip()
        text = text.strip("`\"' ")
        text = text.strip(" .,:;")

        lowered = text.lower()
        if text in {"{}", "[]"}:
            return ""
        if lowered in {"none", "n/a", "na"}:
            return ""

        words = self._extract_words(lowered)
        if len(words) < 2:
            return ""

        if len(text) > self._MAX_MEMORY_PIECE_CHARS:
            text = text[: self._MAX_MEMORY_PIECE_CHARS].rstrip(" ,;:-")

        return text

    def _sanitize_core_concepts(self, raw_concepts: Any) -> list[str]:
        if not isinstance(raw_concepts, list):
            return []

        concepts: list[str] = []
        seen: set[str] = set()
        for raw_item in raw_concepts:
            text = self._sanitize_memory_piece(str(raw_item))
            if not text:
                continue

            # Keep concepts compact and non-conversational.
            text = text.strip(" .,:;")
            if not text:
                continue
            words = self._extract_words(text.lower())
            if not words:
                continue
            if len(words) < 2:
                continue
            if len(text) > 90:
                text = text[:90].rstrip(" ,;:-")

            lowered = text.lower()
            if re.fullmatch(r"(memoria|memory|concepto|concept|dato|item|valor)[ _-]?\d+", lowered):
                if self.debug_classification_logs:
                    logger.info(
                        "[MEMDBG] tiny_extract_reject user_concept=%s reason=placeholder_label",
                        text
                    )
                continue
            if lowered in {"memoria", "memory", "concepto", "concepto principal", "dato", "item"}:
                if self.debug_classification_logs:
                    logger.info(
                        "[MEMDBG] tiny_extract_reject user_concept=%s reason=placeholder_generic",
                        text
                    )
                continue
            if len(words) <= 3 and lowered.startswith(("el ", "la ", "los ", "las ", "un ", "una ", "unos ", "unas ")):
                continue
            if any(lowered.startswith(prefix) for prefix in ("como asistente", "tu ", "tú ", "usted ")):
                continue
            if lowered in seen:
                continue
            seen.add(lowered)
            concepts.append(text)
            if len(concepts) >= 4:
                break

        return concepts

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Lightweight approximation to avoid tokenizer dependency.
        if not text:
            return 0
        return max(1, int(len(text) / 4))

    def _append_memory_piece(self, existing_memory: str, memory_piece: str) -> str:
        existing_clean = self._normalize_memory(existing_memory, max_chars=self.max_storage_chars)
        piece_clean = self._sanitize_memory_piece(memory_piece)
        if not piece_clean:
            return existing_clean

        existing_lower = existing_clean.lower()
        piece_lower = piece_clean.lower()
        if piece_lower and piece_lower in existing_lower:
            return existing_clean

        if not existing_clean:
            return f"- {piece_clean}"
        return f"{existing_clean}\n- {piece_clean}"

    @staticmethod
    def _extract_words(normalized_lower_text: str) -> list[str]:
        stripped = re.sub(r"https?://\S+", " ", normalized_lower_text)
        stripped = re.sub(r"<@!?\d+>", " ", stripped)
        stripped = re.sub(r"<a?:\w+:\d+>", " ", stripped)
        stripped = re.sub(r"[`*_~#>\-\[\]\(\)]+", " ", stripped)
        stripped = re.sub(r"\s+", " ", stripped).strip()
        return re.findall(r"[a-zA-Z0-9áéíóúñü]+", stripped, flags=re.IGNORECASE)

    @staticmethod
    def _preview_message(raw_message: str, max_chars: int = 80) -> str:
        clean = re.sub(r"\s+", " ", (raw_message or "")).strip()
        if len(clean) <= max_chars:
            return clean
        return clean[:max_chars] + "..."

    def _preview_raw_llm_content(self, raw_content: Any, max_chars: int = 500) -> str:
        if raw_content is None:
            return "<None>"
        text = str(raw_content)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "...[truncated]"

    async def _record_pipeline_audit(
        self,
        *,
        user_id: int,
        guild_id: Optional[int],
        channel_id: Optional[int],
        message_id: Optional[int],
        stage: str,
        decision: str,
        reason: str,
        model: str,
        latency_ms: Optional[float] = None
    ) -> None:
        if not self.redis_client:
            return

        payload = {
            "ts": time.time(),
            "user_id": user_id,
            "guild_id": guild_id,
            "channel_id": channel_id,
            "message_id": message_id,
            "pipeline_mode": self.pipeline_mode,
            "stage": stage,
            "decision": decision,
            "reason": reason,
            "model": model,
            "latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
        }
        try:
            key = self._memory_pipeline_audit_key()
            await asyncio.to_thread(self.redis_client.lpush, key, json.dumps(payload, ensure_ascii=False))
            await asyncio.to_thread(
                self.redis_client.ltrim,
                key,
                0,
                self.memory_audit_max_entries - 1
            )
        except Exception as e:
            logger.error("Failed to record memory pipeline audit: %s", e, exc_info=True)

    async def _record_llm_call_audit(
        self,
        *,
        user_id: Optional[int],
        guild_id: Optional[int],
        channel_id: Optional[int],
        message_id: Optional[int],
        interaction_case: str,
        model: str,
        messages: list[dict[str, str]],
        content: Optional[str],
        usage: Any = None,
        call_metadata: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        if not self.redis_client:
            return
        try:
            target_guild_id = int(guild_id or 0)
            payload = {
                "ts": time.time(),
                "guild_id": target_guild_id,
                "user_id": int(user_id or 0),
                "channel_id": int(channel_id or 0),
                "message_id": int(message_id or 0) if message_id else 0,
                "model": model,
                "interaction_case": interaction_case,
                "was_random": False,
                "is_topic_thread": False,
                "memory_injected": False,
                "memory_update_status": "memory_pipeline",
                "tokens": {
                    "prompt": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "completion": getattr(usage, "completion_tokens", 0) if usage else 0,
                    "total": getattr(usage, "total_tokens", 0) if usage else 0,
                },
                "call_count": len(call_metadata) if call_metadata else 1,
                "llm_calls": call_metadata or [],
                "context_messages": self.litellm_client._truncate_for_call_audit(messages),
                "response_type": "json",
                "response_text": (content[:800] + "...") if content and len(content) > 800 else (content or ""),
                "response_data": "",
            }
            key = f"llm_calls:recent:{target_guild_id}"
            await asyncio.to_thread(self.redis_client.lpush, key, json.dumps(payload, ensure_ascii=False))
            await asyncio.to_thread(
                self.redis_client.ltrim,
                key,
                0,
                max(1, int(self.memory_audit_max_entries)) - 1
            )
        except Exception as e:
            logger.error("Failed to record memory LLM audit: %s", e, exc_info=True)

    async def get_memory(self, user_id: int, user_label: Optional[str] = None) -> str:
        if self.redis_client:
            try:
                cached = await asyncio.to_thread(self.redis_client.get, self._memory_key(user_id))
                if cached:
                    if user_label:
                        await asyncio.to_thread(
                            self._ensure_user_label_file,
                            user_id,
                            user_label=user_label,
                        )
                    return str(cached).strip()
            except Exception as e:
                logger.error("Error reading user memory from Redis (%s): %s", user_id, e)

        payload = await asyncio.to_thread(self._read_json_file, self._user_file_path(user_id))
        if not payload:
            return ""

        await asyncio.to_thread(
            self._ensure_user_label_file,
            user_id,
            user_label=user_label,
            payload=payload,
        )

        memory = self._normalize_memory(str(payload.get("memory", "")), max_chars=self.max_storage_chars)
        if not memory:
            return ""

        if self.redis_client:
            try:
                await asyncio.to_thread(self.redis_client.set, self._memory_key(user_id), memory)
            except Exception as e:
                logger.error("Error writing user memory to Redis (%s): %s", user_id, e)

        return memory

    async def get_memory_items(self, user_id: int) -> List[str]:
        """Return memory as an ordered list of individual items (one per line)."""
        memory = await self.get_memory(user_id)
        if not memory:
            return []
        return [line for line in memory.split("\n") if line.strip()]

    async def delete_memory_item(self, user_id: int, index: int) -> bool:
        """Remove the item at the given 0-based index. Returns True on success."""
        async with self._lock_for_user(user_id):
            payload = await asyncio.to_thread(self._read_json_file, self._user_file_path(user_id))
            if not payload:
                return False
            items = [l for l in str(payload.get("memory", "")).split("\n") if l.strip()]
            if index < 0 or index >= len(items):
                return False
            del items[index]
            new_memory = "\n".join(items)
            payload["memory"] = new_memory
            await asyncio.to_thread(self._write_user_file, user_id, payload)
            if self.redis_client:
                try:
                    if new_memory:
                        await asyncio.to_thread(self.redis_client.set, self._memory_key(user_id), new_memory)
                    else:
                        await asyncio.to_thread(self.redis_client.delete, self._memory_key(user_id))
                except Exception as e:
                    logger.error("Error updating Redis after memory delete (%s): %s", user_id, e)
            return True

    async def clear_memory(self, user_id: int) -> bool:
        """Delete all stored memory for a user. Returns True on success."""
        async with self._lock_for_user(user_id):
            payload = await asyncio.to_thread(self._read_json_file, self._user_file_path(user_id))
            if payload is None:
                return True
            payload["memory"] = ""
            await asyncio.to_thread(self._write_user_file, user_id, payload)
            if self.redis_client:
                try:
                    await asyncio.to_thread(self.redis_client.delete, self._memory_key(user_id))
                except Exception as e:
                    logger.error("Error clearing Redis memory (%s): %s", user_id, e)
            return True

    async def get_user_style_traits(self, user_id: int) -> List[str]:
        if self.redis_client:
            try:
                cached = await asyncio.to_thread(self.redis_client.get, self._style_key(user_id))
                if cached:
                    parsed = json.loads(str(cached))
                    if isinstance(parsed, dict):
                        traits = parsed.get("traits", [])
                    elif isinstance(parsed, list):
                        traits = parsed
                    else:
                        traits = []
                    if isinstance(traits, list):
                        normalized = self._normalize_style_traits([str(t) for t in traits])
                        if normalized:
                            return normalized
            except Exception as e:
                logger.error("Error reading user style from Redis (%s): %s", user_id, e)

        payload = await asyncio.to_thread(self._read_json_file, self._style_file_path(user_id))
        if not payload:
            return []
        raw_traits = payload.get("traits", [])
        if not isinstance(raw_traits, list):
            return []
        traits = self._normalize_style_traits([str(t) for t in raw_traits])
        if not traits:
            return []

        if self.redis_client:
            try:
                await asyncio.to_thread(
                    self.redis_client.set,
                    self._style_key(user_id),
                    json.dumps({"traits": traits}, ensure_ascii=False),
                )
            except Exception as e:
                logger.error("Error writing user style to Redis (%s): %s", user_id, e)
        return traits

    async def get_user_style_line(self, user_id: int) -> Optional[str]:
        traits = await self.get_user_style_traits(user_id)
        if not traits:
            return None
        return self._build_style_line(traits)

    async def get_user_expertise_level(self, user_id: int) -> Optional[str]:
        if self.redis_client:
            try:
                cached = await asyncio.to_thread(self.redis_client.get, self._expertise_key(user_id))
                if cached:
                    normalized_cached = self._normalize_expertise_level(str(cached))
                    if normalized_cached:
                        return normalized_cached
            except Exception as e:
                logger.error("Error reading user expertise from Redis (%s): %s", user_id, e)

        payload = await asyncio.to_thread(self._read_json_file, self._expertise_file_path(user_id))
        if not payload:
            return None
        normalized_level = self._normalize_expertise_level(str(payload.get("expertise_level", "")))
        if not normalized_level:
            return None

        if self.redis_client:
            try:
                await asyncio.to_thread(
                    self.redis_client.set,
                    self._expertise_key(user_id),
                    normalized_level,
                )
            except Exception as e:
                logger.error("Error writing user expertise to Redis (%s): %s", user_id, e)
        return normalized_level

    async def _increment_style_reassess_counter(self, user_id: int) -> int:
        if self.redis_client:
            try:
                return int(await asyncio.to_thread(self.redis_client.incr, self._style_reassess_counter_key(user_id)))
            except Exception as e:
                logger.error("Error incrementing style reassess counter (%s): %s", user_id, e)
        count = int(self._style_reassess_counter_local.get(user_id, 0)) + 1
        self._style_reassess_counter_local[user_id] = count
        return count

    async def _reset_style_reassess_counter(self, user_id: int) -> None:
        if self.redis_client:
            try:
                await asyncio.to_thread(self.redis_client.delete, self._style_reassess_counter_key(user_id))
                return
            except Exception as e:
                logger.error("Error resetting style reassess counter (%s): %s", user_id, e)
        self._style_reassess_counter_local[user_id] = 0

    async def _increment_expertise_reassess_counter(self, user_id: int) -> int:
        if self.redis_client:
            try:
                return int(await asyncio.to_thread(self.redis_client.incr, self._expertise_reassess_counter_key(user_id)))
            except Exception as e:
                logger.error("Error incrementing expertise reassess counter (%s): %s", user_id, e)
        count = int(self._expertise_reassess_counter_local.get(user_id, 0)) + 1
        self._expertise_reassess_counter_local[user_id] = count
        return count

    async def _reset_expertise_reassess_counter(self, user_id: int) -> None:
        if self.redis_client:
            try:
                await asyncio.to_thread(self.redis_client.delete, self._expertise_reassess_counter_key(user_id))
                return
            except Exception as e:
                logger.error("Error resetting expertise reassess counter (%s): %s", user_id, e)
        self._expertise_reassess_counter_local[user_id] = 0

    async def set_user_style_traits(self, user_id: int, traits: List[str]) -> None:
        normalized = self._normalize_style_traits(traits)
        if not normalized:
            return

        payload = {
            "user_id": user_id,
            "traits": normalized,
            "line": self._build_style_line(normalized),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        if self.redis_client:
            try:
                await asyncio.to_thread(
                    self.redis_client.set,
                    self._style_key(user_id),
                    json.dumps({"traits": normalized}, ensure_ascii=False),
                )
            except Exception as e:
                logger.error("Error writing user style to Redis (%s): %s", user_id, e)

        try:
            await asyncio.to_thread(self._write_style_file, user_id, payload)
        except Exception as e:
            logger.error("Error writing user style file (%s): %s", user_id, e, exc_info=True)

    async def set_user_expertise_level(self, user_id: int, level: str) -> None:
        normalized_level = self._normalize_expertise_level(level)
        if not normalized_level:
            return

        payload = {
            "user_id": user_id,
            "expertise_level": normalized_level,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        if self.redis_client:
            try:
                await asyncio.to_thread(
                    self.redis_client.set,
                    self._expertise_key(user_id),
                    normalized_level,
                )
            except Exception as e:
                logger.error("Error writing user expertise to Redis (%s): %s", user_id, e)

        try:
            await asyncio.to_thread(self._write_expertise_file, user_id, payload)
        except Exception as e:
            logger.error("Error writing user expertise file (%s): %s", user_id, e, exc_info=True)

    async def set_memory(
        self,
        user_id: int,
        memory: str,
        max_chars: Optional[int] = None,
        user_label: Optional[str] = None,
    ) -> None:
        effective_max_chars = self.max_storage_chars if max_chars is None else max_chars
        normalized = self._normalize_memory(memory, max_chars=effective_max_chars)
        if not normalized:
            return

        if self.redis_client:
            try:
                await asyncio.to_thread(self.redis_client.set, self._memory_key(user_id), normalized)
            except Exception as e:
                logger.error("Error writing user memory to Redis (%s): %s", user_id, e)

        await self._schedule_file_write(user_id, normalized, user_label=user_label)

    async def _schedule_file_write(self, user_id: int, memory: str, user_label: Optional[str] = None) -> None:
        payload = {
            "user_id": user_id,
            "memory": memory,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        normalized_user_label = self._normalize_user_label(user_label)
        if normalized_user_label:
            payload["user_label"] = normalized_user_label
        self._pending_file_payloads[user_id] = payload
        task = self._file_writer_tasks.get(user_id)
        if task and not task.done():
            return
        self._file_writer_tasks[user_id] = asyncio.create_task(self._flush_user_file_loop(user_id))

    async def _flush_user_file_loop(self, user_id: int) -> None:
        try:
            while True:
                payload = self._pending_file_payloads.pop(user_id, None)
                if payload is None:
                    return
                await asyncio.to_thread(self._write_user_file, user_id, payload)
        except Exception as e:
            logger.error("Error persisting user memory file (%s): %s", user_id, e, exc_info=True)
        finally:
            self._file_writer_tasks.pop(user_id, None)

    def _write_user_file(self, user_id: int, payload: Dict[str, Any]) -> None:
        path = self._user_file_path(user_id)
        os.makedirs(self._user_dir_path(user_id), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self._ensure_user_label_file(user_id, payload=payload, overwrite=True)

    def _write_style_file(self, user_id: int, payload: Dict[str, Any]) -> None:
        path = self._style_file_path(user_id)
        os.makedirs(self._user_dir_path(user_id), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _write_expertise_file(self, user_id: int, payload: Dict[str, Any]) -> None:
        path = self._expertise_file_path(user_id)
        os.makedirs(self._user_dir_path(user_id), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _extract_user_id_from_payload_or_path(payload: Dict[str, Any], file_path: str) -> Optional[int]:
        user_id_raw = payload.get("user_id")
        if user_id_raw is not None:
            try:
                return int(user_id_raw)
            except Exception:
                pass

        parent_name = os.path.basename(os.path.dirname(file_path))
        if parent_name.isdigit():
            return int(parent_name)

        stem = os.path.splitext(os.path.basename(file_path))[0]
        if stem.isdigit():
            return int(stem)
        return None

    def _collect_profile_files(self) -> Tuple[List[str], List[str], List[str]]:
        memory_files: List[str] = []
        style_files: List[str] = []
        expertise_files: List[str] = []

        if os.path.isdir(self.user_profiles_dir):
            try:
                for entry in os.scandir(self.user_profiles_dir):
                    if entry.is_dir():
                        if not entry.name.isdigit():
                            continue

                        memory_path = os.path.join(entry.path, self._MEMORY_FILENAME)
                        if os.path.isfile(memory_path):
                            memory_files.append(memory_path)

                        style_path = os.path.join(entry.path, self._STYLE_FILENAME)
                        if os.path.isfile(style_path):
                            style_files.append(style_path)
                        expertise_path = os.path.join(entry.path, self._EXPERTISE_FILENAME)
                        if os.path.isfile(expertise_path):
                            expertise_files.append(expertise_path)
            except Exception as e:
                logger.error("Error scanning user profile directory %s: %s", self.user_profiles_dir, e)

        return (memory_files, style_files, expertise_files)

    async def hydrate_redis_from_disk(self) -> Tuple[int, int]:
        memory_files, style_files, expertise_files = self._collect_profile_files()
        scanned_memory = len(memory_files)
        scanned_style = len(style_files)
        scanned_expertise = len(expertise_files)
        scanned = scanned_memory + scanned_style + scanned_expertise
        if scanned == 0:
            return (0, 0)

        if not self.redis_client:
            return (0, scanned)

        loaded = 0
        loaded_memory = 0
        loaded_style = 0
        loaded_expertise = 0

        for file_path in memory_files:
            payload = await asyncio.to_thread(self._read_json_file, file_path)
            if not payload:
                continue

            user_id = self._extract_user_id_from_payload_or_path(payload, file_path)
            if user_id is None:
                continue

            await asyncio.to_thread(
                self._ensure_user_label_file,
                user_id,
                payload=payload,
            )

            memory = self._normalize_memory(str(payload.get("memory", "")), max_chars=self.max_storage_chars)
            if not memory:
                continue

            try:
                key = self._memory_key(user_id)
                existing = await asyncio.to_thread(self.redis_client.get, key)
                if not existing:
                    await asyncio.to_thread(self.redis_client.set, key, memory)
                    loaded += 1
                    loaded_memory += 1
            except Exception as e:
                logger.error("Error hydrating user memory (%s) from %s: %s", user_id, file_path, e)
                continue

        for file_path in style_files:
            payload = await asyncio.to_thread(self._read_json_file, file_path)
            if not payload:
                continue

            user_id = self._extract_user_id_from_payload_or_path(payload, file_path)
            if user_id is None:
                continue

            raw_traits = payload.get("traits", [])
            if not isinstance(raw_traits, list):
                continue
            traits = self._normalize_style_traits([str(t) for t in raw_traits])
            if not traits:
                continue

            try:
                key = self._style_key(user_id)
                existing = await asyncio.to_thread(self.redis_client.get, key)
                if not existing:
                    await asyncio.to_thread(
                        self.redis_client.set,
                        key,
                        json.dumps({"traits": traits}, ensure_ascii=False),
                    )
                    loaded += 1
                    loaded_style += 1
            except Exception as e:
                logger.error("Error hydrating user style (%s) from %s: %s", user_id, file_path, e)
                continue

        for file_path in expertise_files:
            payload = await asyncio.to_thread(self._read_json_file, file_path)
            if not payload:
                continue

            user_id = self._extract_user_id_from_payload_or_path(payload, file_path)
            if user_id is None:
                continue

            normalized_level = self._normalize_expertise_level(str(payload.get("expertise_level", "")))
            if not normalized_level:
                continue

            try:
                key = self._expertise_key(user_id)
                existing = await asyncio.to_thread(self.redis_client.get, key)
                if not existing:
                    await asyncio.to_thread(self.redis_client.set, key, normalized_level)
                    loaded += 1
                    loaded_expertise += 1
            except Exception as e:
                logger.error("Error hydrating user expertise (%s) from %s: %s", user_id, file_path, e)
                continue

        logger.info(
            "User profile hydration details: scanned_memory=%s scanned_style=%s scanned_expertise=%s loaded_memory=%s loaded_style=%s loaded_expertise=%s",
            scanned_memory,
            scanned_style,
            scanned_expertise,
            loaded_memory,
            loaded_style,
            loaded_expertise,
        )
        return (loaded, scanned)

    def should_capture_message(self, raw_message: str) -> Tuple[bool, str]:
        if not self.enabled:
            return (False, "memory_disabled")

        if self.pipeline_mode == self.PIPELINE_MODE_DISABLED:
            return (False, "pipeline_disabled")

        text = (raw_message or "").strip()
        if not text:
            return (False, "empty")

        normalized = re.sub(r"\s+", " ", text).strip()
        words = self._extract_words(normalized.lower())

        # Always skip ultra-short chatter.
        if len(words) <= 3:
            return (False, "very_short")

        if random.random() > self.update_chance:
            return (False, "random_skip")

        return (True, "selected")

    async def _call_ollama_for_json(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_schema: Optional[Dict[str, Any]],
        temperature: float,
        user_id: Optional[int] = None,
        guild_id: Optional[int] = None,
        channel_id: Optional[int] = None,
        message_id: Optional[int] = None,
        stage: str = "user_memory_ollama",
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        if not self.ollama_client:
            return None, "ollama_not_configured"

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "timeout": self.ollama_timeout_seconds,
            "tools": [],
            "tool_choice": "none",
        }

        try:
            if response_schema:
                kwargs["response_format"] = response_schema

            response = await self.ollama_client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content if response and response.choices else None
            await self._record_llm_call_audit(
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                interaction_case=stage,
                model=model,
                messages=messages,
                content=content,
                usage=getattr(response, "usage", None),
            )
            if self.debug_classification_logs:
                logger.info(
                    "[MEMDBG] tiny_raw model=%s content=%s",
                    model,
                    self._preview_raw_llm_content(content)
                )
            parsed = self._extract_json_object(content or "")
            if parsed:
                if self.debug_classification_logs:
                    logger.info(
                        "[MEMDBG] tiny_parsed model=%s parsed=yes keys=%s",
                        model,
                        sorted(list(parsed.keys()))
                    )
                return parsed, "ok"
            if self.debug_classification_logs:
                logger.info(
                    "[MEMDBG] tiny_parsed model=%s parsed=no reason=invalid_json_response",
                    model
                )
            return None, "invalid_json_response"
        except Exception as schema_error:
            # Some Ollama setups may reject json_schema response_format; fallback to plain JSON request.
            if response_schema:
                logger.warning(
                    "Ollama JSON-schema call failed (%s). Retrying without response_format.",
                    schema_error
                )
                fallback_kwargs = dict(kwargs)
                fallback_kwargs.pop("response_format", None)
                try:
                    response = await self.ollama_client.chat.completions.create(**fallback_kwargs)
                    content = response.choices[0].message.content if response and response.choices else None
                    await self._record_llm_call_audit(
                        user_id=user_id,
                        guild_id=guild_id,
                        channel_id=channel_id,
                        message_id=message_id,
                        interaction_case=f"{stage}:fallback_no_schema",
                        model=model,
                        messages=messages,
                        content=content,
                        usage=getattr(response, "usage", None),
                    )
                    if self.debug_classification_logs:
                        logger.info(
                            "[MEMDBG] tiny_raw_no_schema model=%s content=%s",
                            model,
                            self._preview_raw_llm_content(content)
                        )
                    parsed = self._extract_json_object(content or "")
                    if parsed:
                        if self.debug_classification_logs:
                            logger.info(
                                "[MEMDBG] tiny_parsed_no_schema model=%s parsed=yes keys=%s",
                                model,
                                sorted(list(parsed.keys()))
                            )
                        return parsed, "ok_no_schema"
                    if self.debug_classification_logs:
                        logger.info(
                            "[MEMDBG] tiny_parsed_no_schema model=%s parsed=no reason=invalid_json_response",
                            model
                        )
                    return None, "invalid_json_response"
                except Exception as plain_error:
                    await self._record_llm_call_audit(
                        user_id=user_id,
                        guild_id=guild_id,
                        channel_id=channel_id,
                        message_id=message_id,
                        interaction_case=f"{stage}:fallback_no_schema:error",
                        model=model,
                        messages=messages,
                        content=str(plain_error),
                        usage=None,
                    )
                    logger.error("Ollama memory call failed for model %s: %s", model, plain_error, exc_info=True)
                    return None, "ollama_error"

            await self._record_llm_call_audit(
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                interaction_case=f"{stage}:error",
                model=model,
                messages=messages,
                content=str(schema_error),
                usage=None,
            )
            logger.error("Ollama memory call failed for model %s: %s", model, schema_error, exc_info=True)
            return None, "ollama_error"

    def _extract_memory_text_from_tiny_response(self, raw_text: str) -> str:
        text = (raw_text or "").strip()
        if not text:
            return ""

        def explode_candidates(candidate: str) -> list[str]:
            c = (candidate or "").replace("\r\n", "\n").replace("\r", "\n")
            if not c.strip():
                return []
            pieces: list[str] = []
            for line in re.split(r"\n+", c):
                line = line.strip()
                if not line:
                    continue
                for sub in re.split(r"\s*;\s*|\s+\|\s+|\s+-\s+|\.\s+", line):
                    sub = sub.strip()
                    if sub:
                        pieces.append(sub)
            return pieces

        def normalize_candidate(candidate: str) -> str:
            c = (candidate or "").strip()
            if not c:
                return ""
            c = c.strip("`\"' ")
            c = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", c)
            return c

        def first_valid(candidates: list[str]) -> str:
            for raw_candidate in candidates:
                for exploded in explode_candidates(raw_candidate):
                    candidate = normalize_candidate(exploded)
                    if not candidate:
                        continue
                    clean = self._sanitize_memory_piece(candidate)
                    if clean:
                        return clean
            return ""

        candidates: list[str] = []
        parsed = self._extract_json_object(text)
        if parsed:
            core_memories = parsed.get("core_memories")
            if isinstance(core_memories, list):
                candidates.extend(str(x).strip() for x in core_memories if str(x).strip())

            core_concepts = parsed.get("core_concepts")
            if isinstance(core_concepts, list):
                candidates.extend(str(x).strip() for x in core_concepts if str(x).strip())

            memory_piece = parsed.get("memory_piece")
            if memory_piece is not None:
                candidates.append(str(memory_piece).strip())

            memory = parsed.get("memory")
            if memory is not None:
                candidates.append(str(memory).strip())

            decision_raw = str(parsed.get("decision", "")).strip().lower()
            if decision_raw == "no":
                return ""

        selected_from_parsed = first_valid(candidates)
        if selected_from_parsed:
            return selected_from_parsed

        clean = text.strip("`\"' ").strip()
        lowered = clean.lower()
        if lowered in {"none", "no", "n/a", "na"}:
            return ""

        # Handle plain-text lists/multi-lines from weak models: keep first valid memory only.
        text_candidates: list[str] = []
        for line in re.split(r"[\r\n]+", clean):
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"\s*;\s*|\s+\|\s+|\s+-\s+", line)
            for part in parts:
                part = part.strip()
                if part:
                    text_candidates.append(part)

        selected = first_valid(text_candidates or [clean])
        if self.debug_classification_logs and selected and len(text_candidates) > 1:
            logger.info(
                "[MEMDBG] tiny_extract_selected candidate=%s candidates_count=%s",
                self._preview_message(selected, max_chars=100),
                len(text_candidates)
            )
        return selected

    async def _call_frontier_for_json(
        self,
        *,
        messages: list[dict[str, str]],
        response_schema: Optional[Dict[str, Any]],
        temperature: float,
        user_id: Optional[int] = None,
        guild_id: Optional[int] = None,
        channel_id: Optional[int] = None,
        message_id: Optional[int] = None,
        stage: str = "user_memory_frontier",
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        try:
            response_or_tuple = await self.litellm_client.chat_completion(
                messages=messages,
                tools=None,
                use_structured_output=bool(response_schema),
                track_calls=True,
                response_schema_override=response_schema,
                call_context={
                    "user_name": f"user_{user_id}" if user_id is not None else "memory_pipeline",
                    "channel_name": "user_memory_pipeline",
                    "guild_name": "n/a",
                    "source": "user_memory",
                    "interaction_case": stage,
                }
            )
            if isinstance(response_or_tuple, tuple):
                response, call_metadata = response_or_tuple
            else:
                response = response_or_tuple
                call_metadata = []
            content = response.choices[0].message.content if response and response.choices else None
            await self._record_llm_call_audit(
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                interaction_case=stage,
                model=self.litellm_client.model,
                messages=messages,
                content=content,
                usage=getattr(response, "usage", None),
                call_metadata=call_metadata,
            )
            parsed = self._extract_json_object(content or "")
            if parsed:
                return parsed, "ok"
            return None, "invalid_json_response"
        except Exception as e:
            await self._record_llm_call_audit(
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                interaction_case=f"{stage}:error",
                model=self.litellm_client.model,
                messages=messages,
                content=str(e),
                usage=None,
            )
            logger.error("Frontier memory call failed for user-memory update: %s", e, exc_info=True)
            return None, "frontier_error"

    async def _update_user_style_from_message(
        self,
        *,
        user_id: int,
        user_label: Optional[str],
        message_content: str,
        guild_id: Optional[int],
        channel_id: Optional[int],
        message_id: Optional[int],
        stage: str = "user_style:tiny_extract",
    ) -> Tuple[bool, str]:
        system_prompt = self._read_text_file(self.style_extract_system_prompt_path)
        user_template = self._read_text_file(self.style_extract_user_prompt_path)
        schema = self._read_json_file(self.style_extract_schema_path)
        if not system_prompt or not user_template or not schema:
            return (False, "missing_style_prompt_or_schema")

        existing_traits = await self.get_user_style_traits(user_id)
        user_prompt = (
            user_template
            .replace("{{USER_MESSAGE}}", message_content or "")
            .replace("{{EXISTING_STYLE_TRAITS}}", ", ".join(existing_traits) if existing_traits else "none")
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        parsed, reason = await self._call_ollama_for_json(
            model=self.tiny_model_classifier,
            messages=messages,
            response_schema=schema,
            temperature=0.0,
            user_id=user_id,
            guild_id=guild_id,
            channel_id=channel_id,
            message_id=message_id,
            stage=stage,
        )
        if not parsed:
            return (False, reason)

        raw_traits = parsed.get("style_traits", [])
        if not isinstance(raw_traits, list):
            return (False, "invalid_style_traits")
        traits = self._normalize_style_traits([str(t) for t in raw_traits])
        if not traits:
            return (False, "empty_style_traits")

        await self.set_user_style_traits(user_id, traits)
        logger.info(
            "[STYLE] user=%s(%s) traits=%s line=\"%s\"",
            user_label or "unknown",
            user_id,
            traits,
            self._build_style_line(traits)
        )
        return (True, "updated")

    async def _maybe_reassess_user_style(
        self,
        *,
        user_id: int,
        user_label: Optional[str],
        message_content: str,
        guild_id: Optional[int],
        channel_id: Optional[int],
        message_id: Optional[int],
        stage_label: str,
    ) -> Tuple[bool, str]:
        existing_traits = await self.get_user_style_traits(user_id)
        counter = await self._increment_style_reassess_counter(user_id)
        should_reassess = (not existing_traits) or (counter >= self._PROFILE_REASSESS_EVERY_WORTHWHILE)
        if not should_reassess:
            return (False, f"skipped:counter={counter}/{self._PROFILE_REASSESS_EVERY_WORTHWHILE}")

        ok, reason = await self._update_user_style_from_message(
            user_id=user_id,
            user_label=user_label,
            message_content=message_content,
            guild_id=guild_id,
            channel_id=channel_id,
            message_id=message_id,
        )
        if ok:
            await self._reset_style_reassess_counter(user_id)

        logger.info(
            "[STYLE] stage=%s user=%s(%s) reassess_counter=%s threshold=%s ok=%s reason=%s",
            stage_label,
            user_label or "unknown",
            user_id,
            counter,
            self._PROFILE_REASSESS_EVERY_WORTHWHILE,
            ok,
            reason,
        )
        return (ok, reason)

    async def _update_user_expertise_from_message(
        self,
        *,
        user_id: int,
        user_label: Optional[str],
        message_content: str,
        guild_id: Optional[int],
        channel_id: Optional[int],
        message_id: Optional[int],
        stage: str = "user_expertise:tiny_extract",
    ) -> Tuple[bool, str]:
        system_prompt = self._read_text_file(self.expertise_extract_system_prompt_path)
        user_template = self._read_text_file(self.expertise_extract_user_prompt_path)
        schema = self._read_json_file(self.expertise_extract_schema_path)
        if not system_prompt or not user_template or not schema:
            return (False, "missing_expertise_prompt_or_schema")

        existing_level = await self.get_user_expertise_level(user_id)
        user_prompt = (
            user_template
            .replace("{{USER_MESSAGE}}", message_content or "")
            .replace("{{EXISTING_EXPERTISE_LEVEL}}", existing_level or "none")
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        parsed, reason = await self._call_ollama_for_json(
            model=self.tiny_model_classifier,
            messages=messages,
            response_schema=schema,
            temperature=0.0,
            user_id=user_id,
            guild_id=guild_id,
            channel_id=channel_id,
            message_id=message_id,
            stage=stage,
        )
        if not parsed:
            return (False, reason)

        normalized_level = self._normalize_expertise_level(str(parsed.get("expertise_level", "")))
        if not normalized_level:
            return (False, "invalid_expertise_level")

        await self.set_user_expertise_level(user_id, normalized_level)
        logger.info(
            "[EXPERTISE] user=%s(%s) level=%s",
            user_label or "unknown",
            user_id,
            normalized_level,
        )
        return (True, "updated")

    async def _maybe_reassess_user_expertise(
        self,
        *,
        user_id: int,
        user_label: Optional[str],
        message_content: str,
        guild_id: Optional[int],
        channel_id: Optional[int],
        message_id: Optional[int],
        stage_label: str,
    ) -> Tuple[bool, str]:
        existing_level = await self.get_user_expertise_level(user_id)
        counter = await self._increment_expertise_reassess_counter(user_id)
        should_reassess = (not existing_level) or (counter >= self._PROFILE_REASSESS_EVERY_WORTHWHILE)
        if not should_reassess:
            return (False, f"skipped:counter={counter}/{self._PROFILE_REASSESS_EVERY_WORTHWHILE}")

        ok, reason = await self._update_user_expertise_from_message(
            user_id=user_id,
            user_label=user_label,
            message_content=message_content,
            guild_id=guild_id,
            channel_id=channel_id,
            message_id=message_id,
        )
        if ok:
            await self._reset_expertise_reassess_counter(user_id)

        current_level = await self.get_user_expertise_level(user_id)
        logger.info(
            "[EXPERTISE] stage=%s user=%s(%s) reassess_counter=%s threshold=%s ok=%s reason=%s level=%s",
            stage_label,
            user_label or "unknown",
            user_id,
            counter,
            self._PROFILE_REASSESS_EVERY_WORTHWHILE,
            ok,
            reason,
            current_level or "none",
        )
        return (ok, reason)

    async def _has_history_profile_bootstrap_attempted(self, user_id: int) -> bool:
        if self.redis_client:
            try:
                raw = await asyncio.to_thread(
                    self.redis_client.get,
                    self._history_profile_bootstrap_attempted_key(user_id),
                )
                return str(raw).strip() == "1" if raw is not None else False
            except Exception as e:
                logger.error("Error reading history profile bootstrap flag (%s): %s", user_id, e)
        return bool(self._history_profile_bootstrap_attempted_local.get(user_id, False))

    async def _mark_history_profile_bootstrap_attempted(self, user_id: int) -> None:
        if self.redis_client:
            try:
                await asyncio.to_thread(
                    self.redis_client.set,
                    self._history_profile_bootstrap_attempted_key(user_id),
                    "1",
                )
            except Exception as e:
                logger.error("Error writing history profile bootstrap flag (%s): %s", user_id, e)
        self._history_profile_bootstrap_attempted_local[user_id] = True

    async def should_attempt_history_profile_bootstrap(self, user_id: int) -> bool:
        attempted = await self._has_history_profile_bootstrap_attempted(user_id)
        if attempted:
            return False
        style_traits = await self.get_user_style_traits(user_id)
        expertise_level = await self.get_user_expertise_level(user_id)
        return (not style_traits) or (not expertise_level)

    async def bootstrap_missing_profile_from_history(
        self,
        *,
        user_id: int,
        user_label: Optional[str],
        history_messages: List[str],
        guild_id: Optional[int],
        channel_id: Optional[int],
        message_id: Optional[int],
    ) -> Tuple[bool, str]:
        lock = self._lock_for_user(user_id)
        async with lock:
            if not self.enabled or self.pipeline_mode == self.PIPELINE_MODE_DISABLED:
                return (False, "skipped:memory_pipeline_disabled")

            existing_traits = await self.get_user_style_traits(user_id)
            existing_level = await self.get_user_expertise_level(user_id)
            if existing_traits and existing_level:
                return (False, "skipped:no_missing_profile")

            if await self._has_history_profile_bootstrap_attempted(user_id):
                return (False, "skipped:already_attempted")

            await self._mark_history_profile_bootstrap_attempted(user_id)

            cleaned_messages: List[str] = []
            for raw in history_messages:
                cleaned = re.sub(r"\s+", " ", str(raw or "")).strip()
                if not cleaned:
                    continue
                cleaned_messages.append(cleaned[:400])
                if len(cleaned_messages) >= 8:
                    break

            if not cleaned_messages:
                logger.info(
                    "[PROFILE_BOOTSTRAP] user=%s(%s) skipped=no_history_messages",
                    user_label or "unknown",
                    user_id,
                )
                return (False, "skipped:no_history_messages")

            transcript = "Recent messages from the same user in this channel (oldest to newest):\n"
            transcript += "\n".join(f"- {msg}" for msg in cleaned_messages)

            style_ok = False
            style_reason = "skipped:already_present"
            if not existing_traits:
                style_ok, style_reason = await self._update_user_style_from_message(
                    user_id=user_id,
                    user_label=user_label,
                    message_content=transcript,
                    guild_id=guild_id,
                    channel_id=channel_id,
                    message_id=message_id,
                    stage="user_style:history_bootstrap",
                )
                if style_ok:
                    await self._reset_style_reassess_counter(user_id)

            expertise_ok = False
            expertise_reason = "skipped:already_present"
            if not existing_level:
                expertise_ok, expertise_reason = await self._update_user_expertise_from_message(
                    user_id=user_id,
                    user_label=user_label,
                    message_content=transcript,
                    guild_id=guild_id,
                    channel_id=channel_id,
                    message_id=message_id,
                    stage="user_expertise:history_bootstrap",
                )
                if expertise_ok:
                    await self._reset_expertise_reassess_counter(user_id)

            logger.info(
                "[PROFILE_BOOTSTRAP] user=%s(%s) history_messages=%s style_ok=%s style_reason=%s expertise_ok=%s expertise_reason=%s",
                user_label or "unknown",
                user_id,
                len(cleaned_messages),
                style_ok,
                style_reason,
                expertise_ok,
                expertise_reason,
            )
            return (
                style_ok or expertise_ok,
                f"style={style_reason};expertise={expertise_reason}",
            )

    async def _run_injection_guard_check(
        self,
        *,
        user_id: int,
        user_label: Optional[str],
        message_content: str,
        guild_id: Optional[int],
        channel_id: Optional[int],
        message_id: Optional[int],
    ) -> Tuple[bool, str, str, str]:
        system_prompt = self._read_text_file(self.injection_guard_system_prompt_path)
        user_template = self._read_text_file(self.injection_guard_user_prompt_path)
        schema = self._read_json_file(self.injection_guard_schema_path)
        if not system_prompt or not user_template or not schema:
            logger.warning("[MEMSAFE] Injection guard prompt/schema missing; skipping guard check")
            return (False, "low", "guard_unavailable", "")

        existing_style_traits = await self.get_user_style_traits(user_id)
        existing_expertise_level = await self.get_user_expertise_level(user_id)

        user_prompt = (
            user_template
            .replace("{{USER_MESSAGE}}", message_content or "")
            .replace("{{EXISTING_STYLE_TRAITS}}", ", ".join(existing_style_traits) if existing_style_traits else "none")
            .replace("{{EXISTING_EXPERTISE_LEVEL}}", existing_expertise_level or "none")
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        parsed, reason = await self._call_frontier_for_json(
            messages=messages,
            response_schema=schema,
            temperature=0.0,
            user_id=user_id,
            guild_id=guild_id,
            channel_id=channel_id,
            message_id=message_id,
            stage="user_memory:injection_guard",
        )
        if not parsed:
            logger.info(
                "[MEMSAFE] user=%s(%s) guard_parse_failed reason=%s",
                user_label or "unknown",
                user_id,
                reason,
            )
            return (False, "low", reason, "")

        is_injection = bool(parsed.get("is_injection", False))
        confidence = self._normalize_injection_confidence(str(parsed.get("confidence", "low")))
        guard_reason = re.sub(r"\s+", " ", str(parsed.get("reason", "")).strip())[:180]
        user_notice = re.sub(r"\s+", " ", str(parsed.get("user_notice", "")).strip())[:280]
        blocked = is_injection and confidence == self._INJECTION_BLOCK_CONFIDENCE

        logger.info(
            "[MEMSAFE] user=%s(%s) injection=%s confidence=%s blocked=%s reason=%s notice_chars=%s",
            user_label or "unknown",
            user_id,
            is_injection,
            confidence,
            blocked,
            guard_reason or "n/a",
            len(user_notice),
        )
        return (blocked, confidence, guard_reason, user_notice)

    async def _update_memory_from_tiny_extract(
        self,
        *,
        user_id: int,
        user_label: Optional[str],
        message_content: str,
        existing_memory: str,
        guild_id: Optional[int],
        channel_id: Optional[int],
        message_id: Optional[int]
    ) -> Tuple[bool, str]:
        system_prompt = self._read_text_file(self.tiny_extract_system_prompt_path)
        user_template = self._read_text_file(self.tiny_extract_user_prompt_path)
        schema = self._read_json_file(self.tiny_extract_schema_path)
        if not system_prompt or not user_template or not schema:
            await self._record_pipeline_audit(
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                stage="tiny_extract",
                decision="skip",
                reason="missing_tiny_extract_prompt_or_schema",
                model=self.tiny_model_extract,
            )
            return (False, "missing_tiny_extract_prompt_or_schema")

        user_prompt = user_template.replace("{{USER_MESSAGE}}", message_content or "")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        started_at = time.time()
        parsed, reason = await self._call_ollama_for_json(
            model=self.tiny_model_extract,
            messages=messages,
            response_schema=schema,
            temperature=0.0,
            user_id=user_id,
            guild_id=guild_id,
            channel_id=channel_id,
            message_id=message_id,
            stage="user_memory:tiny_extract",
        )
        latency_ms = (time.time() - started_at) * 1000
        if not parsed:
            await self._record_pipeline_audit(
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                stage="tiny_extract",
                decision="error",
                reason=reason,
                model=self.tiny_model_extract,
                latency_ms=latency_ms,
            )
            return (False, reason)

        required_keys = self._required_keys_from_response_schema(schema)
        if required_keys and not required_keys.issubset(parsed.keys()):
            await self._record_pipeline_audit(
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                stage="tiny_extract",
                decision="skip",
                reason="schema_required_keys_missing",
                model=self.tiny_model_extract,
                latency_ms=latency_ms,
            )
            return (False, "schema_required_keys_missing")

        if self.debug_classification_logs:
            logger.info(
                "[MEMDBG] tiny_extract_parsed user=%s(%s) payload=%s",
                user_label or "unknown",
                user_id,
                self._preview_raw_llm_content(json.dumps(parsed, ensure_ascii=False))
            )

        extracted_text = self._extract_memory_text_from_tiny_response(
            str(parsed.get("memory", "")).strip()
        )
        if not extracted_text:
            await self._record_pipeline_audit(
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                stage="tiny_extract",
                decision="skip",
                reason="empty_memory",
                model=self.tiny_model_extract,
                latency_ms=latency_ms,
            )
            return (True, "no_update")

        memory_piece = self._sanitize_memory_piece(extracted_text)
        if not memory_piece:
            await self._record_pipeline_audit(
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                stage="tiny_extract",
                decision="skip",
                reason="invalid_memory",
                model=self.tiny_model_extract,
                latency_ms=latency_ms,
            )
            return (True, "no_update")

        if self.debug_classification_logs:
            logger.info(
                "[MEMDBG] tiny_extract user=%s(%s) message=\"%s\" worthwhile=yes extracted_text=\"%s\" piece=\"%s\"",
                user_label or "unknown",
                user_id,
                self._preview_message(message_content),
                self._preview_message(extracted_text, max_chars=120),
                self._preview_message(memory_piece, max_chars=120)
            )

        combined_memory = self._append_memory_piece(existing_memory, memory_piece)
        if combined_memory == self._normalize_memory(existing_memory, max_chars=self.max_storage_chars):
            await self._record_pipeline_audit(
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                stage="tiny_extract",
                decision="skip",
                reason="unchanged",
                model=self.tiny_model_extract,
                latency_ms=latency_ms,
            )
            return (True, "unchanged")

        combined_tokens = self._estimate_tokens(combined_memory)
        if combined_tokens >= self.tiny_accumulate_max_tokens:
            compact_ok, compact_reason = await self._compact_tiny_memory(
                user_id=user_id,
                user_label=user_label,
                accumulated_memory=combined_memory,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id
            )
            if compact_ok:
                await self._record_pipeline_audit(
                    user_id=user_id,
                    guild_id=guild_id,
                    channel_id=channel_id,
                    message_id=message_id,
                    stage="persist",
                    decision="updated",
                    reason="tiny_extract_compacted",
                    model=self.tiny_model_extract,
                    latency_ms=latency_ms,
                )
                return (True, "updated_compacted")
            await self._record_pipeline_audit(
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                stage="tiny_compact",
                decision="error",
                reason=compact_reason,
                model=self.tiny_model_extract,
                latency_ms=latency_ms,
            )
            return (False, f"compact_failed:{compact_reason}")

        await self.set_memory(
            user_id,
            combined_memory,
            max_chars=self.max_storage_chars,
            user_label=user_label,
        )
        await self._record_pipeline_audit(
            user_id=user_id,
            guild_id=guild_id,
            channel_id=channel_id,
            message_id=message_id,
            stage="persist",
            decision="updated",
            reason="tiny_extract_append",
            model=self.tiny_model_extract,
            latency_ms=latency_ms,
        )
        return (True, "updated")

    async def _compact_tiny_memory(
        self,
        *,
        user_id: int,
        user_label: Optional[str],
        accumulated_memory: str,
        guild_id: Optional[int],
        channel_id: Optional[int],
        message_id: Optional[int]
    ) -> Tuple[bool, str]:
        system_prompt = self._read_text_file(self.tiny_compact_system_prompt_path)
        user_template = self._read_text_file(self.tiny_compact_user_prompt_path)
        schema = self._read_json_file(self.tiny_compact_schema_path)
        if not system_prompt or not user_template or not schema:
            return (False, "missing_tiny_compact_prompt_or_schema")

        user_prompt = user_template.replace("{{ACCUMULATED_MEMORY}}", accumulated_memory or "(vacío)")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        started_at = time.time()
        parsed, reason = await self._call_ollama_for_json(
            model=self.tiny_model_extract,
            messages=messages,
            response_schema=schema,
            temperature=0.2,
            user_id=user_id,
            guild_id=guild_id,
            channel_id=channel_id,
            message_id=message_id,
            stage="user_memory:tiny_compact",
        )
        latency_ms = (time.time() - started_at) * 1000
        if not parsed:
            await self._record_pipeline_audit(
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                stage="tiny_compact",
                decision="error",
                reason=reason,
                model=self.tiny_model_extract,
                latency_ms=latency_ms,
            )
            return (False, reason)

        required_keys = self._required_keys_from_response_schema(schema)
        if required_keys and not required_keys.issubset(parsed.keys()):
            await self._record_pipeline_audit(
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                stage="tiny_compact",
                decision="skip",
                reason="schema_required_keys_missing",
                model=self.tiny_model_extract,
                latency_ms=latency_ms,
            )
            return (False, "schema_required_keys_missing")

        compact_memory = self._normalize_memory(str(parsed.get("memory", "")), max_chars=self.max_memory_chars)
        if not compact_memory:
            await self._record_pipeline_audit(
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                stage="tiny_compact",
                decision="skip",
                reason="empty_memory",
                model=self.tiny_model_extract,
                latency_ms=latency_ms,
            )
            return (False, "empty_memory")

        await self.set_memory(
            user_id,
            compact_memory,
            max_chars=self.max_memory_chars,
            user_label=user_label,
        )
        if self.debug_classification_logs:
            logger.info(
                "[MEMDBG] tiny_compact user=%s(%s) compacted=yes tokens_before~%s",
                user_label or "unknown",
                user_id,
                self._estimate_tokens(accumulated_memory)
            )
        await self._record_pipeline_audit(
            user_id=user_id,
            guild_id=guild_id,
            channel_id=channel_id,
            message_id=message_id,
            stage="tiny_compact",
            decision="updated",
            reason="compacted",
            model=self.tiny_model_extract,
            latency_ms=latency_ms,
        )
        return (True, "updated")

    async def _run_tiny_worthwhile_check(
        self,
        *,
        user_id: int,
        user_label: Optional[str],
        message_content: str,
        guild_id: Optional[int],
        channel_id: Optional[int],
        message_id: Optional[int]
    ) -> Tuple[bool, str]:
        system_prompt = self._read_text_file(self.tiny_worthwhile_system_prompt_path)
        user_template = self._read_text_file(self.tiny_worthwhile_user_prompt_path)
        schema = self._read_json_file(self.tiny_worthwhile_schema_path)
        if not system_prompt or not user_template or not schema:
            await self._record_pipeline_audit(
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                stage="tiny_worthwhile",
                decision="skip",
                reason="missing_tiny_worthwhile_prompt_or_schema",
                model=self.tiny_model_classifier,
            )
            return (False, "missing_tiny_worthwhile_prompt_or_schema")

        user_prompt = user_template.replace("{{USER_MESSAGE}}", message_content or "")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        started_at = time.time()
        parsed, reason = await self._call_ollama_for_json(
            model=self.tiny_model_classifier,
            messages=messages,
            response_schema=schema,
            temperature=0.0,
            user_id=user_id,
            guild_id=guild_id,
            channel_id=channel_id,
            message_id=message_id,
            stage="user_memory:tiny_worthwhile",
        )
        latency_ms = (time.time() - started_at) * 1000
        if not parsed:
            await self._record_pipeline_audit(
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                stage="tiny_worthwhile",
                decision="error",
                reason=reason,
                model=self.tiny_model_classifier,
                latency_ms=latency_ms,
            )
            return (False, reason)

        required_keys = self._required_keys_from_response_schema(schema)
        if required_keys and not required_keys.issubset(parsed.keys()):
            await self._record_pipeline_audit(
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                stage="tiny_worthwhile",
                decision="skip",
                reason="schema_required_keys_missing",
                model=self.tiny_model_classifier,
                latency_ms=latency_ms,
            )
            return (False, "schema_required_keys_missing")

        if self.debug_classification_logs:
            logger.info(
                "[MEMDBG] tiny_worthwhile_parsed user=%s(%s) payload=%s",
                user_label or "unknown",
                user_id,
                self._preview_raw_llm_content(json.dumps(parsed, ensure_ascii=False))
            )

        decision_raw = str(parsed.get("decision", "")).strip().lower()
        if decision_raw not in {"yes", "no"}:
            await self._record_pipeline_audit(
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                stage="tiny_worthwhile",
                decision="skip",
                reason="invalid_decision",
                model=self.tiny_model_classifier,
                latency_ms=latency_ms,
            )
            return (False, "invalid_decision")

        decision = decision_raw == "yes"
        reason_text = "yes" if decision else "no"
        logger.info(
            "[MEMFLOW] stage=tiny_worthwhile user=%s(%s) decision=%s",
            user_label or "unknown",
            user_id,
            reason_text
        )
        if self.debug_classification_logs:
            logger.info(
                "[MEMDBG] tiny_worthwhile user=%s(%s) message=\"%s\" worthwhile=%s",
                user_label or "unknown",
                user_id,
                self._preview_message(message_content),
                "yes" if decision else "no"
            )
        await self._record_pipeline_audit(
            user_id=user_id,
            guild_id=guild_id,
            channel_id=channel_id,
            message_id=message_id,
            stage="tiny_worthwhile",
            decision="selected" if decision else "rejected",
            reason=reason_text,
            model=self.tiny_model_classifier,
            latency_ms=latency_ms,
        )
        return (decision, reason_text if decision else "not_worthwhile")

    async def _update_memory_from_frontier(
        self,
        *,
        user_id: int,
        user_label: Optional[str],
        message_content: str,
        existing_memory: str,
        guild_id: Optional[int],
        channel_id: Optional[int],
        message_id: Optional[int]
    ) -> Tuple[bool, str]:
        system_prompt = self._read_text_file(self.memory_update_system_prompt_path)
        user_template = self._read_text_file(self.memory_update_user_prompt_path)
        schema = self._read_json_file(self.memory_update_schema_path)
        if not system_prompt or not user_template or not schema:
            await self._record_pipeline_audit(
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                stage="frontier_extract",
                decision="skip",
                reason="missing_memory_prompt_or_schema",
                model=self.litellm_client.model,
            )
            return (False, "missing_memory_prompt_or_schema")

        user_prompt = (
            user_template
            .replace("{{CURRENT_MEMORY}}", existing_memory or "(vacío)")
            .replace("{{USER_MESSAGE}}", message_content or "")
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        started_at = time.time()
        parsed, reason = await self._call_frontier_for_json(
            messages=messages,
            response_schema=schema,
            temperature=0.3,
            user_id=user_id,
            guild_id=guild_id,
            channel_id=channel_id,
            message_id=message_id,
            stage="user_memory:frontier_update",
        )
        latency_ms = (time.time() - started_at) * 1000
        if not parsed:
            await self._record_pipeline_audit(
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                stage="frontier_extract",
                decision="error",
                reason=reason,
                model=self.litellm_client.model,
                latency_ms=latency_ms,
            )
            return (False, reason)

        required_keys = self._required_keys_from_response_schema(schema)
        if required_keys and not required_keys.issubset(parsed.keys()):
            await self._record_pipeline_audit(
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                stage="frontier_extract",
                decision="skip",
                reason="schema_required_keys_missing",
                model=self.litellm_client.model,
                latency_ms=latency_ms,
            )
            return (False, "schema_required_keys_missing")

        new_memory = self._normalize_memory(str(parsed.get("memory", "")), max_chars=self.max_memory_chars)
        if not new_memory:
            await self._record_pipeline_audit(
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                stage="frontier_extract",
                decision="skip",
                reason="empty_memory",
                model=self.litellm_client.model,
                latency_ms=latency_ms,
            )
            return (False, "empty_memory")

        if new_memory == self._normalize_memory(existing_memory):
            if self.debug_classification_logs:
                logger.info(
                    "[MEMDBG] frontier_update user=%s(%s) message=\"%s\" update=no reason=unchanged",
                    user_label or "unknown",
                    user_id,
                    self._preview_message(message_content)
                )
            await self._record_pipeline_audit(
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                stage="frontier_extract",
                decision="skip",
                reason="unchanged",
                model=self.litellm_client.model,
                latency_ms=latency_ms,
            )
            return (True, "unchanged")

        await self.set_memory(
            user_id,
            new_memory,
            max_chars=self.max_memory_chars,
            user_label=user_label,
        )
        if self.debug_classification_logs:
            logger.info(
                "[MEMDBG] frontier_update user=%s(%s) message=\"%s\" update=yes",
                user_label or "unknown",
                user_id,
                self._preview_message(message_content)
            )
        await self._record_pipeline_audit(
            user_id=user_id,
            guild_id=guild_id,
            channel_id=channel_id,
            message_id=message_id,
            stage="persist",
            decision="updated",
            reason="frontier_extract",
            model=self.litellm_client.model,
            latency_ms=latency_ms,
        )
        return (True, "updated")

    async def _update_memory_from_frontier_core_extract(
        self,
        *,
        user_id: int,
        user_label: Optional[str],
        message_content: str,
        existing_memory: str,
        guild_id: Optional[int],
        channel_id: Optional[int],
        message_id: Optional[int]
    ) -> Tuple[bool, str]:
        system_prompt = self._read_text_file(self.frontier_core_extract_system_prompt_path)
        user_template = self._read_text_file(self.frontier_core_extract_user_prompt_path)
        schema = self._read_json_file(self.frontier_core_extract_schema_path)
        if not system_prompt or not user_template or not schema:
            await self._record_pipeline_audit(
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                stage="frontier_core_extract",
                decision="skip",
                reason="missing_frontier_core_extract_prompt_or_schema",
                model=self.litellm_client.model,
            )
            return (False, "missing_frontier_core_extract_prompt_or_schema")

        user_prompt = user_template.replace("{{USER_MESSAGE}}", message_content or "")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        started_at = time.time()
        parsed, reason = await self._call_frontier_for_json(
            messages=messages,
            response_schema=schema,
            temperature=0.0,
            user_id=user_id,
            guild_id=guild_id,
            channel_id=channel_id,
            message_id=message_id,
            stage="user_memory:frontier_core_extract",
        )
        latency_ms = (time.time() - started_at) * 1000
        if not parsed:
            await self._record_pipeline_audit(
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                stage="frontier_core_extract",
                decision="error",
                reason=reason,
                model=self.litellm_client.model,
                latency_ms=latency_ms,
            )
            return (False, reason)

        required_keys = self._required_keys_from_response_schema(schema)
        if required_keys and not required_keys.issubset(parsed.keys()):
            await self._record_pipeline_audit(
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                stage="frontier_core_extract",
                decision="skip",
                reason="schema_required_keys_missing",
                model=self.litellm_client.model,
                latency_ms=latency_ms,
            )
            return (False, "schema_required_keys_missing")

        extracted_text = self._extract_memory_text_from_tiny_response(
            str(parsed.get("memory", "")).strip()
        )
        if not extracted_text:
            await self._record_pipeline_audit(
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                stage="frontier_core_extract",
                decision="skip",
                reason="empty_memory",
                model=self.litellm_client.model,
                latency_ms=latency_ms,
            )
            return (True, "no_update")

        memory_piece = self._sanitize_memory_piece(extracted_text)
        if not memory_piece:
            await self._record_pipeline_audit(
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                stage="frontier_core_extract",
                decision="skip",
                reason="invalid_memory",
                model=self.litellm_client.model,
                latency_ms=latency_ms,
            )
            return (True, "no_update")

        if self.debug_classification_logs:
            logger.info(
                "[MEMDBG] frontier_core_extract user=%s(%s) message=\"%s\" extracted_text=\"%s\" piece=\"%s\"",
                user_label or "unknown",
                user_id,
                self._preview_message(message_content),
                self._preview_message(extracted_text, max_chars=120),
                self._preview_message(memory_piece, max_chars=120)
            )

        combined_memory = self._append_memory_piece(existing_memory, memory_piece)
        normalized_existing = self._normalize_memory(existing_memory, max_chars=self.max_memory_chars)
        normalized_combined = self._normalize_memory(combined_memory, max_chars=self.max_memory_chars)
        if normalized_combined == normalized_existing:
            await self._record_pipeline_audit(
                user_id=user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                stage="frontier_core_extract",
                decision="skip",
                reason="unchanged",
                model=self.litellm_client.model,
                latency_ms=latency_ms,
            )
            return (True, "unchanged")

        await self.set_memory(
            user_id,
            normalized_combined,
            max_chars=self.max_memory_chars,
            user_label=user_label,
        )
        await self._record_pipeline_audit(
            user_id=user_id,
            guild_id=guild_id,
            channel_id=channel_id,
            message_id=message_id,
            stage="persist",
            decision="updated",
            reason="frontier_core_extract",
            model=self.litellm_client.model,
            latency_ms=latency_ms,
        )
        return (True, "updated")

    async def update_memory_from_message(
        self,
        user_id: int,
        message_content: str,
        current_memory: Optional[str] = None,
        user_label: Optional[str] = None,
        guild_id: Optional[int] = None,
        channel_id: Optional[int] = None,
        message_id: Optional[int] = None
    ) -> Tuple[bool, str]:
        lock = self._lock_for_user(user_id)
        async with lock:
            if not self.enabled:
                return (False, "memory_disabled")

            if self.pipeline_mode == self.PIPELINE_MODE_DISABLED:
                return (False, "pipeline_disabled")

            existing_memory = (
                current_memory
                if current_memory is not None
                else await self.get_memory(user_id, user_label=user_label)
            )

            if self.pipeline_mode == self.PIPELINE_MODE_TINY_EXTRACT:
                logger.info(
                    "[MEMFLOW] stage=tiny_extract user=%s(%s) step=worthwhile_check_start",
                    user_label or "unknown",
                    user_id
                )
                worthwhile, worthwhile_reason = await self._run_tiny_worthwhile_check(
                    user_id=user_id,
                    user_label=user_label,
                    message_content=message_content,
                    guild_id=guild_id,
                    channel_id=channel_id,
                    message_id=message_id
                )
                logger.info(
                    "[MEMFLOW] stage=tiny_extract user=%s(%s) step=worthwhile_check_end worthwhile=%s reason=%s",
                    user_label or "unknown",
                    user_id,
                    "yes" if worthwhile else "no",
                    worthwhile_reason
                )
                if not worthwhile:
                    return (False, f"not_worthwhile:{worthwhile_reason}")

                blocked, confidence, guard_reason, user_notice = await self._run_injection_guard_check(
                    user_id=user_id,
                    user_label=user_label,
                    message_content=message_content,
                    guild_id=guild_id,
                    channel_id=channel_id,
                    message_id=message_id,
                )
                if blocked:
                    blocked_payload = {
                        "confidence": confidence,
                        "reason": guard_reason,
                        "user_notice": user_notice,
                    }
                    return (
                        False,
                        "injection_blocked:" + json.dumps(blocked_payload, ensure_ascii=False),
                    )

                style_ok, style_reason = await self._maybe_reassess_user_style(
                    user_id=user_id,
                    user_label=user_label,
                    message_content=message_content,
                    guild_id=guild_id,
                    channel_id=channel_id,
                    message_id=message_id,
                    stage_label="tiny_extract",
                )
                logger.info(
                    "[STYLE] stage=tiny_extract user=%s(%s) ok=%s reason=%s",
                    user_label or "unknown",
                    user_id,
                    style_ok,
                    style_reason
                )
                expertise_ok, expertise_reason = await self._maybe_reassess_user_expertise(
                    user_id=user_id,
                    user_label=user_label,
                    message_content=message_content,
                    guild_id=guild_id,
                    channel_id=channel_id,
                    message_id=message_id,
                    stage_label="tiny_extract",
                )
                logger.info(
                    "[EXPERTISE] stage=tiny_extract user=%s(%s) ok=%s reason=%s",
                    user_label or "unknown",
                    user_id,
                    expertise_ok,
                    expertise_reason
                )

                logger.info(
                    "[MEMFLOW] stage=tiny_extract user=%s(%s) step=extract_start",
                    user_label or "unknown",
                    user_id
                )
                ok, extract_reason = await self._update_memory_from_tiny_extract(
                    user_id=user_id,
                    user_label=user_label,
                    message_content=message_content,
                    existing_memory=existing_memory,
                    guild_id=guild_id,
                    channel_id=channel_id,
                    message_id=message_id
                )
                logger.info(
                    "[MEMFLOW] stage=tiny_extract user=%s(%s) step=extract_end ok=%s reason=%s",
                    user_label or "unknown",
                    user_id,
                    ok,
                    extract_reason
                )
                return (ok, extract_reason)

            if self.pipeline_mode == self.PIPELINE_MODE_FRONTIER:
                worthwhile, worthwhile_reason = await self._run_tiny_worthwhile_check(
                    user_id=user_id,
                    user_label=user_label,
                    message_content=message_content,
                    guild_id=guild_id,
                    channel_id=channel_id,
                    message_id=message_id
                )
                if not worthwhile:
                    return (False, f"not_worthwhile:{worthwhile_reason}")

                blocked, confidence, guard_reason, user_notice = await self._run_injection_guard_check(
                    user_id=user_id,
                    user_label=user_label,
                    message_content=message_content,
                    guild_id=guild_id,
                    channel_id=channel_id,
                    message_id=message_id,
                )
                if blocked:
                    blocked_payload = {
                        "confidence": confidence,
                        "reason": guard_reason,
                        "user_notice": user_notice,
                    }
                    return (
                        False,
                        "injection_blocked:" + json.dumps(blocked_payload, ensure_ascii=False),
                    )

                style_ok, style_reason = await self._maybe_reassess_user_style(
                    user_id=user_id,
                    user_label=user_label,
                    message_content=message_content,
                    guild_id=guild_id,
                    channel_id=channel_id,
                    message_id=message_id,
                    stage_label="frontier_pipeline",
                )
                logger.info(
                    "[STYLE] stage=frontier_pipeline user=%s(%s) ok=%s reason=%s",
                    user_label or "unknown",
                    user_id,
                    style_ok,
                    style_reason
                )
                expertise_ok, expertise_reason = await self._maybe_reassess_user_expertise(
                    user_id=user_id,
                    user_label=user_label,
                    message_content=message_content,
                    guild_id=guild_id,
                    channel_id=channel_id,
                    message_id=message_id,
                    stage_label="frontier_pipeline",
                )
                logger.info(
                    "[EXPERTISE] stage=frontier_pipeline user=%s(%s) ok=%s reason=%s",
                    user_label or "unknown",
                    user_id,
                    expertise_ok,
                    expertise_reason
                )

                return await self._update_memory_from_frontier(
                    user_id=user_id,
                    user_label=user_label,
                    message_content=message_content,
                    existing_memory=existing_memory,
                    guild_id=guild_id,
                    channel_id=channel_id,
                    message_id=message_id
                )

            if self.pipeline_mode == self.PIPELINE_MODE_TINY_GATE_FRONTIER_CORE:
                logger.info(
                    "[MEMFLOW] stage=tiny_gate_frontier_core user=%s(%s) step=worthwhile_check_start",
                    user_label or "unknown",
                    user_id
                )
                worthwhile, worthwhile_reason = await self._run_tiny_worthwhile_check(
                    user_id=user_id,
                    user_label=user_label,
                    message_content=message_content,
                    guild_id=guild_id,
                    channel_id=channel_id,
                    message_id=message_id
                )
                logger.info(
                    "[MEMFLOW] stage=tiny_gate_frontier_core user=%s(%s) step=worthwhile_check_end worthwhile=%s reason=%s",
                    user_label or "unknown",
                    user_id,
                    "yes" if worthwhile else "no",
                    worthwhile_reason
                )
                if not worthwhile:
                    return (False, f"not_worthwhile:{worthwhile_reason}")

                blocked, confidence, guard_reason, user_notice = await self._run_injection_guard_check(
                    user_id=user_id,
                    user_label=user_label,
                    message_content=message_content,
                    guild_id=guild_id,
                    channel_id=channel_id,
                    message_id=message_id,
                )
                if blocked:
                    blocked_payload = {
                        "confidence": confidence,
                        "reason": guard_reason,
                        "user_notice": user_notice,
                    }
                    return (
                        False,
                        "injection_blocked:" + json.dumps(blocked_payload, ensure_ascii=False),
                    )

                style_ok, style_reason = await self._maybe_reassess_user_style(
                    user_id=user_id,
                    user_label=user_label,
                    message_content=message_content,
                    guild_id=guild_id,
                    channel_id=channel_id,
                    message_id=message_id,
                    stage_label="tiny_gate_frontier_core",
                )
                logger.info(
                    "[STYLE] stage=tiny_gate_frontier_core user=%s(%s) ok=%s reason=%s",
                    user_label or "unknown",
                    user_id,
                    style_ok,
                    style_reason
                )
                expertise_ok, expertise_reason = await self._maybe_reassess_user_expertise(
                    user_id=user_id,
                    user_label=user_label,
                    message_content=message_content,
                    guild_id=guild_id,
                    channel_id=channel_id,
                    message_id=message_id,
                    stage_label="tiny_gate_frontier_core",
                )
                logger.info(
                    "[EXPERTISE] stage=tiny_gate_frontier_core user=%s(%s) ok=%s reason=%s",
                    user_label or "unknown",
                    user_id,
                    expertise_ok,
                    expertise_reason
                )

                logger.info(
                    "[MEMFLOW] stage=tiny_gate_frontier_core user=%s(%s) step=extract_start",
                    user_label or "unknown",
                    user_id
                )
                ok, extract_reason = await self._update_memory_from_frontier_core_extract(
                    user_id=user_id,
                    user_label=user_label,
                    message_content=message_content,
                    existing_memory=existing_memory,
                    guild_id=guild_id,
                    channel_id=channel_id,
                    message_id=message_id
                )
                logger.info(
                    "[MEMFLOW] stage=tiny_gate_frontier_core user=%s(%s) step=extract_end ok=%s reason=%s",
                    user_label or "unknown",
                    user_id,
                    ok,
                    extract_reason
                )
                return (ok, extract_reason)

            return (False, "invalid_pipeline_mode")

    async def close(self) -> None:
        tasks = [t for t in self._file_writer_tasks.values() if not t.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
