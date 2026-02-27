# cogs/daily_topic_cog.py
import asyncio
import json
import logging
import os
import random
import time
import unicodedata
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional, Set, Tuple

import discord
from discord import app_commands
from discord.ext import commands, tasks

if TYPE_CHECKING:
    from bot import AIBot

logger = logging.getLogger(__name__)


TOPIC_OF_DAY_EMBED_TITLE = "Topic of the day"
DAILY_TOPIC_CATEGORIES = [
    "Algoritmos",
    "Estructuras de Datos",
    "Arquitectura",
    "DB Relacionales",
    "DB NoSQL",
    "Sistemas Operativos",
    "Redes y Protocolos",
    "Ciberseguridad",
    "Criptografía",
    "Compiladores",
    "Lenguajes Teoría",
    "Ingeniería Software",
    "IA",
    "Machine Learning",
    "Cloud Computing",
    "Sists Distribuidos",
    "Teoría Computación",
    "Concurrencia",
    "Desarrollo Móvil",
    "DevOps CI-CD",
]


class DailyTopicCog(commands.Cog):
    """Daily/topic-interval educational thread workflow with admin approval."""

    def __init__(self, bot: "AIBot"):
        self.bot = bot
        self.redis_client = self.bot.redis_client
        self.daily_prompts_dir = os.path.join(self.bot.prompts_root_path, "daily_topic")
        self.topic_generation_system_prompt_path = os.path.join(
            self.daily_prompts_dir, "topic_generation_system_prompt.txt"
        )
        self.topic_generation_user_prompt_path = os.path.join(
            self.daily_prompts_dir, "topic_generation_user_prompt.txt"
        )
        self.body_generation_system_prompt_path = os.path.join(
            self.daily_prompts_dir, "body_generation_system_prompt.txt"
        )
        self.body_generation_user_prompt_path = os.path.join(
            self.daily_prompts_dir, "body_generation_user_prompt.txt"
        )
        self.topic_generation_schema_path = os.path.join(
            self.daily_prompts_dir, "topic_generation_response_schema.json"
        )
        self.body_generation_schema_path = os.path.join(
            self.daily_prompts_dir, "body_generation_response_schema.json"
        )
        self._pending_cache: Dict[int, Dict[str, Any]] = {}
        self._approval_msg_id_cache: Dict[int, int] = {}
        self._last_run_ts_cache: Dict[int, float] = {}
        self._last_run_date_cache: Dict[int, str] = {}
        self._category_counts_cache: Dict[int, Dict[str, int]] = {}
        self._guild_locks: Dict[int, asyncio.Lock] = {}

        if self.bot.daily_topic_enabled and self.bot.daily_topic_check_interval_seconds > 0:
            self.daily_topic_scheduler.change_interval(
                seconds=self.bot.daily_topic_check_interval_seconds
            )
            self.daily_topic_scheduler.start()
            logger.info(
                "DailyTopic workflow enabled: check_interval=%ss, interval_mode=%ss, approval_hour_utc=%s",
                self.bot.daily_topic_check_interval_seconds,
                self.bot.daily_topic_interval_seconds,
                self.bot.daily_topic_approval_hour_utc
            )
        else:
            logger.info("DailyTopic workflow disabled")

    @staticmethod
    def _read_text_file(path: str) -> Optional[str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                return content if content else None
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

    def _load_daily_prompt(self, path: str, label: str) -> Optional[str]:
        content = self._read_text_file(path)
        if not content:
            logger.error(
                "Missing/empty daily topic prompt file for %s: %s",
                label,
                path
            )
            return None
        return content

    def _load_daily_schema(self, path: str, label: str) -> Optional[Dict[str, Any]]:
        schema = self._read_json_file(path)
        if not schema:
            logger.error(
                "Missing/invalid daily topic response schema for %s: %s",
                label,
                path
            )
            return None
        return schema

    @staticmethod
    def _required_keys_from_response_schema(schema: Dict[str, Any]) -> Set[str]:
        try:
            inner_schema = schema.get("json_schema", {}).get("schema", {})
            required = inner_schema.get("required", [])
            if isinstance(required, list):
                return {str(k) for k in required}
        except Exception:
            pass
        return set()

    def _lock_for_guild(self, guild_id: int) -> asyncio.Lock:
        lock = self._guild_locks.get(guild_id)
        if lock is None:
            lock = asyncio.Lock()
            self._guild_locks[guild_id] = lock
        return lock

    def _pending_key(self, guild_id: int) -> str:
        return f"daily_topic:pending:{guild_id}"

    def _approval_msg_key(self, guild_id: int) -> str:
        return f"daily_topic:approval_message_id:{guild_id}"

    def _last_run_ts_key(self, guild_id: int) -> str:
        return f"daily_topic:last_run_ts:{guild_id}"

    def _last_run_date_key(self, guild_id: int) -> str:
        return f"daily_topic:last_run_date:{guild_id}"

    def _category_counts_key(self, guild_id: int) -> str:
        return f"daily_topic:category_counts:{guild_id}"

    async def _redis_get(self, key: str) -> Optional[str]:
        if not self.redis_client:
            return None
        try:
            return await asyncio.to_thread(self.redis_client.get, key)
        except Exception as e:
            logger.error("Redis GET failed (%s): %s", key, e)
            return None

    async def _redis_set(self, key: str, value: str) -> None:
        if not self.redis_client:
            return
        try:
            await asyncio.to_thread(self.redis_client.set, key, value)
        except Exception as e:
            logger.error("Redis SET failed (%s): %s", key, e)

    async def _redis_delete(self, key: str) -> None:
        if not self.redis_client:
            return
        try:
            await asyncio.to_thread(self.redis_client.delete, key)
        except Exception as e:
            logger.error("Redis DELETE failed (%s): %s", key, e)

    async def _get_pending(self, guild_id: int) -> Optional[Dict[str, Any]]:
        if self.redis_client:
            raw = await self._redis_get(self._pending_key(guild_id))
            if not raw:
                return None
            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                logger.error("Invalid JSON in pending state for guild %s", guild_id)
                return None
            return None

        return self._pending_cache.get(guild_id)

    async def _set_pending(self, guild_id: int, pending: Dict[str, Any]) -> None:
        if self.redis_client:
            await self._redis_set(self._pending_key(guild_id), json.dumps(pending, ensure_ascii=False))
            return
        self._pending_cache[guild_id] = pending

    async def _clear_pending(self, guild_id: int) -> None:
        if self.redis_client:
            await self._redis_delete(self._pending_key(guild_id))
        self._pending_cache.pop(guild_id, None)

    async def _get_approval_message_id(self, guild_id: int) -> Optional[int]:
        if self.redis_client:
            raw = await self._redis_get(self._approval_msg_key(guild_id))
            if raw and raw.isdigit():
                return int(raw)
            return None
        return self._approval_msg_id_cache.get(guild_id)

    async def _set_approval_message_id(self, guild_id: int, message_id: int) -> None:
        if self.redis_client:
            await self._redis_set(self._approval_msg_key(guild_id), str(message_id))
            return
        self._approval_msg_id_cache[guild_id] = message_id

    async def _get_last_run_ts(self, guild_id: int) -> float:
        if self.redis_client:
            raw = await self._redis_get(self._last_run_ts_key(guild_id))
            try:
                return float(raw) if raw else 0.0
            except (TypeError, ValueError):
                return 0.0
        return self._last_run_ts_cache.get(guild_id, 0.0)

    async def _set_last_run_ts(self, guild_id: int, run_ts: float) -> None:
        if self.redis_client:
            await self._redis_set(self._last_run_ts_key(guild_id), str(run_ts))
            return
        self._last_run_ts_cache[guild_id] = run_ts

    async def _get_last_run_date(self, guild_id: int) -> str:
        if self.redis_client:
            raw = await self._redis_get(self._last_run_date_key(guild_id))
            return raw or ""
        return self._last_run_date_cache.get(guild_id, "")

    async def _set_last_run_date(self, guild_id: int, run_date: str) -> None:
        if self.redis_client:
            await self._redis_set(self._last_run_date_key(guild_id), run_date)
            return
        self._last_run_date_cache[guild_id] = run_date

    async def _get_category_counts(self, guild_id: int) -> Dict[str, int]:
        if self.redis_client:
            raw = await self._redis_get(self._category_counts_key(guild_id))
            if raw:
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        result: Dict[str, int] = {}
                        for category in DAILY_TOPIC_CATEGORIES:
                            try:
                                result[category] = int(parsed.get(category, 0))
                            except Exception:
                                result[category] = 0
                        return result
                except Exception:
                    logger.error("Invalid category counts JSON for guild %s", guild_id)

        existing = self._category_counts_cache.get(guild_id, {})
        result: Dict[str, int] = {}
        for category in DAILY_TOPIC_CATEGORIES:
            try:
                result[category] = int(existing.get(category, 0))
            except Exception:
                result[category] = 0
        return result

    async def _set_category_counts(self, guild_id: int, counts: Dict[str, int]) -> None:
        safe_counts = {category: int(counts.get(category, 0)) for category in DAILY_TOPIC_CATEGORIES}
        if self.redis_client:
            await self._redis_set(self._category_counts_key(guild_id), json.dumps(safe_counts, ensure_ascii=False))
            return
        self._category_counts_cache[guild_id] = safe_counts

    async def _increment_category_count(self, guild_id: int, category: str) -> None:
        if category not in DAILY_TOPIC_CATEGORIES:
            return
        counts = await self._get_category_counts(guild_id)
        counts[category] = int(counts.get(category, 0)) + 1
        await self._set_category_counts(guild_id, counts)

    async def _choose_balanced_category(self, guild_id: int) -> str:
        counts = await self._get_category_counts(guild_id)
        min_count = min(int(counts.get(category, 0)) for category in DAILY_TOPIC_CATEGORIES)
        least_used = [c for c in DAILY_TOPIC_CATEGORIES if int(counts.get(c, 0)) == min_count]
        return random.choice(least_used)

    @staticmethod
    def _normalize_tag_name(value: str) -> str:
        if not value:
            return ""
        text = unicodedata.normalize("NFKD", value)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = text.lower().strip()
        text = text.replace("_", " ").replace("-", " ")
        text = " ".join(text.split())
        return text

    @staticmethod
    def _is_forum_or_media_channel(channel: Optional[discord.abc.GuildChannel]) -> bool:
        """Compatibility-safe check for forum-like channels across discord.py versions."""
        if channel is None:
            return False

        # ForumChannel exists in supported discord.py versions.
        if isinstance(channel, discord.ForumChannel):
            return True

        # Media channels may exist only in newer discord.py versions.
        channel_type = getattr(channel, "type", None)
        forum_type = getattr(discord.ChannelType, "forum", None)
        media_type = getattr(discord.ChannelType, "media", None)

        if forum_type is not None and channel_type == forum_type:
            return True
        if media_type is not None and channel_type == media_type:
            return True

        return False

    def _find_tag_for_category(
        self,
        channel: discord.abc.GuildChannel,
        category: str
    ) -> Optional[discord.ForumTag]:
        if not self._is_forum_or_media_channel(channel):
            return None

        normalized_category = self._normalize_tag_name(category)
        if not normalized_category:
            return None

        available_tags = getattr(channel, "available_tags", []) or []
        if not available_tags:
            return None

        # Exact normalized match first.
        for tag in available_tags:
            if self._normalize_tag_name(tag.name) == normalized_category:
                return tag

        # Fallback partial matching.
        for tag in available_tags:
            normalized_tag = self._normalize_tag_name(tag.name)
            if normalized_tag and (
                normalized_tag in normalized_category or normalized_category in normalized_tag
            ):
                return tag

        return None

    @staticmethod
    def _channel_type_name(channel: Optional[discord.abc.GuildChannel]) -> str:
        if channel is None:
            return "None"
        return type(channel).__name__

    async def _resolve_channel(self, channel_id: Optional[int]) -> Optional[discord.abc.GuildChannel]:
        if not channel_id:
            return None

        channel = self.bot.get_channel(channel_id)
        if channel:
            return channel  # type: ignore

        try:
            fetched = await self.bot.fetch_channel(channel_id)
            return fetched  # type: ignore
        except discord.Forbidden as e:
            logger.error(
                "Could not resolve channel %s due to missing permissions: %s",
                channel_id,
                e
            )
            return None
        except discord.NotFound:
            logger.error("Could not resolve channel %s: channel not found.", channel_id)
            return None
        except Exception as e:
            logger.error("Could not resolve channel %s: %s", channel_id, e)
            return None

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

        candidate = text[start:end + 1]
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    async def _ask_llm_for_json(
        self,
        messages: list[dict[str, str]],
        response_schema: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        try:
            kwargs: Dict[str, Any] = {
                "model": self.bot.litellm_client.model,
                "messages": messages,
                "temperature": 0.7,
                "timeout": 60.0
            }
            if response_schema:
                kwargs["response_format"] = response_schema

            response = await self.bot.litellm_client.client.chat.completions.create(
                **kwargs
            )
            content = response.choices[0].message.content if response and response.choices else None
            if not content:
                return None
            return self._extract_json_object(content)
        except Exception as e:
            logger.error("LLM JSON request failed: %s", e)
            return None

    @staticmethod
    def _build_fallback_topic_for_category(category: str, previous_title: Optional[str] = None) -> Dict[str, str]:
        base_title = f"Fundamentos avanzados de {category}"
        if previous_title and previous_title.strip().lower() == base_title.strip().lower():
            base_title = f"{category}: invariantes, límites y tradeoffs en producción"
        return {
            "topic_title": base_title,
            "topic_description": (
                f"Análisis técnico de {category} con foco en mecanismos internos, "
                "costes asintóticos e implicaciones de diseño en sistemas reales."
            ),
            "importance_reasoning": (
                f"Dominar {category} impacta directamente en rendimiento, "
                "escalabilidad, mantenibilidad y decisiones arquitectónicas críticas."
            )
        }

    async def _generate_topic(self, category: str, previous_title: Optional[str] = None) -> Dict[str, str]:
        avoid_line = f"No repitas el tema anterior: {previous_title}." if previous_title else ""
        system_prompt = self._load_daily_prompt(
            self.topic_generation_system_prompt_path,
            "topic_generation_system_prompt"
        )
        user_template = self._load_daily_prompt(
            self.topic_generation_user_prompt_path,
            "topic_generation_user_prompt"
        )
        response_schema = self._load_daily_schema(
            self.topic_generation_schema_path,
            "topic_generation_response_schema"
        )

        if not system_prompt or not user_template or not response_schema:
            return self._build_fallback_topic_for_category(category, previous_title=previous_title)

        user_prompt = (
            user_template
            .replace("{{PREVIOUS_TOPIC_RULE}}", avoid_line)
            .replace("{{CATEGORY}}", category)
        )
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        parsed = await self._ask_llm_for_json(messages, response_schema=response_schema)
        if parsed:
            required_keys = self._required_keys_from_response_schema(response_schema)
            if not required_keys or required_keys.issubset(parsed.keys()):
                title = str(parsed.get("topic_title", "")).strip()
                description = str(parsed.get("topic_description", "")).strip()
                reasoning = str(parsed.get("importance_reasoning", "")).strip()
                if title and description and reasoning:
                    return {
                        "topic_title": title,
                        "topic_description": description,
                        "importance_reasoning": reasoning
                    }

        return self._build_fallback_topic_for_category(category, previous_title=previous_title)

    async def _generate_topic_body(self, pending: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = self._load_daily_prompt(
            self.body_generation_system_prompt_path,
            "body_generation_system_prompt"
        )
        user_template = self._load_daily_prompt(
            self.body_generation_user_prompt_path,
            "body_generation_user_prompt"
        )
        response_schema = self._load_daily_schema(
            self.body_generation_schema_path,
            "body_generation_response_schema"
        )

        if not system_prompt or not user_template or not response_schema:
            fallback_body = (
                f"## {pending.get('topic_title', 'Tema técnico')}\n\n"
                f"{pending.get('topic_description', '')}\n\n"
                f"**Importancia técnica:** {pending.get('importance_reasoning', '')}\n\n"
                "### Error común\n"
                "Memorizar definiciones sin analizar invariantes, costos y tradeoffs en escenarios reales."
            )
            return str(pending.get("topic_title", "Tema del día")), fallback_body

        user_prompt = (
            user_template
            .replace("{{TOPIC_CATEGORY}}", str(pending.get("topic_category", "")))
            .replace("{{TOPIC_TITLE}}", str(pending.get("topic_title", "")))
            .replace("{{TOPIC_DESCRIPTION}}", str(pending.get("topic_description", "")))
            .replace("{{IMPORTANCE_REASONING}}", str(pending.get("importance_reasoning", "")))
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        parsed = await self._ask_llm_for_json(messages, response_schema=response_schema)
        if parsed:
            required_keys = self._required_keys_from_response_schema(response_schema)
            if not required_keys or required_keys.issubset(parsed.keys()):
                thread_title = str(parsed.get("thread_title", "")).strip()
                body = str(parsed.get("body", "")).strip()
                if thread_title and body:
                    return thread_title, body

        fallback_body = (
            f"Hoy veremos **{pending.get('topic_title', 'un tema de informática')}**.\n\n"
            f"{pending.get('topic_description', '')}\n\n"
            f"¿Por qué importa? {pending.get('importance_reasoning', '')}\n\n"
            "Ejemplo rápido: piensa en un caso real de tu código actual y evalúa cómo este concepto "
            "te ayuda a tomar decisiones mejores.\n\n"
            "Error común: memorizar definiciones sin aplicarlas a problemas concretos."
        )
        return str(pending.get("topic_title", "Tema del día")), fallback_body

    def _build_approval_embed(self, pending: Dict[str, Any], status_text: str) -> discord.Embed:
        deadline_ts = int(float(pending.get("auto_publish_ts", time.time())))
        embed = discord.Embed(
            title=TOPIC_OF_DAY_EMBED_TITLE,
            description=f"**{pending.get('topic_title', 'Tema pendiente')}**",
            color=discord.Color.blurple(),
            timestamp=datetime.now(timezone.utc)
        )
        embed.add_field(
            name="Descripción",
            value=str(pending.get("topic_description", "Sin descripción"))[:1024],
            inline=False
        )
        embed.add_field(
            name="Categoría",
            value=str(pending.get("topic_category", "Sin categoría"))[:1024],
            inline=False
        )
        embed.add_field(
            name="Por qué es importante",
            value=str(pending.get("importance_reasoning", "Sin justificación"))[:1024],
            inline=False
        )
        embed.add_field(name="Estado", value=status_text[:1024], inline=False)
        embed.set_footer(
            text=f"Auto-publicación: <t:{deadline_ts}:t> • Comando: /approve_topic response:<yes|no>"
        )
        return embed

    async def _upsert_approval_message(
        self,
        guild_id: int,
        pending: Dict[str, Any],
        status_text: str
    ) -> Optional[discord.Message]:
        channel = await self._resolve_channel(self.bot.daily_topic_approval_channel_id)
        if not channel or not isinstance(channel, discord.TextChannel):
            logger.error("Approval channel invalid or not text channel: %s", self.bot.daily_topic_approval_channel_id)
            return None
        if channel.guild.id != guild_id:
            logger.error("Approval channel %s is not in guild %s", channel.id, guild_id)
            return None

        embed = self._build_approval_embed(pending, status_text=status_text)
        existing_id = await self._get_approval_message_id(guild_id)

        if existing_id:
            try:
                msg = await channel.fetch_message(existing_id)
                await msg.edit(embed=embed, content=None)
                return msg
            except discord.NotFound:
                logger.warning("Stored approval message %s was not found; creating a new one.", existing_id)
            except Exception as e:
                logger.error("Failed editing approval message %s: %s", existing_id, e)

        try:
            msg = await channel.send(embed=embed)
            await self._set_approval_message_id(guild_id, msg.id)
            return msg
        except Exception as e:
            logger.error("Failed creating approval message in channel %s: %s", channel.id, e)
            return None

    async def _publish_topic_thread(
        self,
        guild_id: int,
        pending: Dict[str, Any],
        approved_by: Optional[str],
        auto_published: bool
    ) -> Tuple[bool, str]:
        # Enforced forum post behavior:
        # - auto-hide after 1 week of inactivity
        # - slowmode disabled
        post_auto_archive_minutes = 10080
        post_slowmode_delay = 0

        publish_channel = await self._resolve_channel(self.bot.daily_topic_publish_channel_id)
        if isinstance(publish_channel, discord.Thread):
            logger.warning(
                "Configured publish channel %s is a thread; using parent channel %s.",
                publish_channel.id,
                publish_channel.parent_id
            )
            publish_channel = publish_channel.parent  # type: ignore[assignment]

        if not publish_channel:
            return False, f"Publish channel unresolved: {self.bot.daily_topic_publish_channel_id}"
        if publish_channel.guild.id != guild_id:
            return False, "Publish channel does not belong to this guild."
        if not self._is_forum_or_media_channel(publish_channel):
            return False, (
                f"Publish channel must be Forum/Media: id={self.bot.daily_topic_publish_channel_id} "
                f"type={self._channel_type_name(publish_channel)}"
            )

        topic_category = str(pending.get("topic_category", "")).strip()
        matched_tag = self._find_tag_for_category(publish_channel, topic_category)
        if not topic_category:
            return False, "Pending topic missing category; cannot assign forum tag."
        if not matched_tag:
            available = [t.name for t in getattr(publish_channel, "available_tags", [])]
            return False, (
                f"No forum/media tag matches category '{topic_category}' in channel {publish_channel.id}. "
                f"Available tags: {available}"
            )

        generated_title, generated_body = await self._generate_topic_body(pending)
        approved_topic_title = str(pending.get("topic_title", "")).strip()
        post_title = (approved_topic_title or generated_title or "Tema del día").strip()[:100]
        if not post_title:
            post_title = "Tema del día"

        post_content = (
            f"{pending.get('topic_description', '')}\n\n"
            f"**Importancia:** {pending.get('importance_reasoning', '')}\n\n"
            f"{generated_body}"
        ).strip()

        chunks = []
        cursor = post_content
        while len(cursor) > 1900:
            split_at = cursor.rfind("\n", 0, 1900)
            if split_at == -1:
                split_at = 1900
            chunks.append(cursor[:split_at])
            cursor = cursor[split_at:].lstrip("\n")
        if cursor:
            chunks.append(cursor)

        if not chunks:
            return False, "Generated empty topic body; nothing to publish."

        try:
            create_kwargs: Dict[str, Any] = {
                "name": post_title,
                "content": chunks[0],
                "auto_archive_duration": post_auto_archive_minutes,
                "slowmode_delay": post_slowmode_delay,
            }
            if matched_tag:
                create_kwargs["applied_tags"] = [matched_tag]

            created = await publish_channel.create_thread(
                **create_kwargs
            )
            thread: Optional[discord.Thread] = created.thread if hasattr(created, "thread") else created  # type: ignore[assignment]
        except Exception as e:
            return False, f"Could not create forum/media post: {e}"

        # Best-effort hardening in case server defaults override creation parameters.
        try:
            if thread:
                await thread.edit(
                    auto_archive_duration=post_auto_archive_minutes,
                    slowmode_delay=post_slowmode_delay
                )
        except Exception as e:
            logger.warning("Could not enforce post thread settings for %s: %s", thread.id if thread else "unknown", e)

        try:
            for chunk in chunks[1:]:
                await thread.send(chunk)  # type: ignore[union-attr]
        except Exception as e:
            return False, f"Post created but sending remaining content failed: {e}"

        publish_note = "Auto-publicado por timeout" if auto_published else f"Aprobado por {approved_by or 'admin'}"
        await self._upsert_approval_message(
            guild_id=guild_id,
            pending=pending,
            status_text=f"✅ {publish_note}. Publicado en <#{publish_channel.id}>."
        )
        await self._clear_pending(guild_id)
        await self._increment_category_count(guild_id, topic_category)
        return True, f"Publicado en post: {thread.mention if thread else 'post_sin_referencia'}"

    def _guild_matches_configured_channels(self, guild: discord.Guild) -> bool:
        approval_id = self.bot.daily_topic_approval_channel_id
        publish_id = self.bot.daily_topic_publish_channel_id
        if not approval_id or not publish_id:
            return False
        return guild.get_channel(approval_id) is not None and guild.get_channel(publish_id) is not None

    async def _create_or_refresh_pending(
        self,
        guild_id: int,
        previous: Optional[Dict[str, Any]] = None,
        preserve_deadline: bool = False
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        previous_title = str(previous.get("topic_title")) if previous else None
        selected_category = await self._choose_balanced_category(guild_id)
        topic = await self._generate_topic(category=selected_category, previous_title=previous_title)
        now_ts = time.time()
        auto_publish_ts = (
            float(previous.get("auto_publish_ts"))
            if previous and preserve_deadline and previous.get("auto_publish_ts")
            else now_ts + self.bot.daily_topic_approval_timeout_seconds
        )

        pending = {
            "topic_title": topic["topic_title"],
            "topic_description": topic["topic_description"],
            "importance_reasoning": topic["importance_reasoning"],
            "topic_category": selected_category,
            "created_ts": now_ts,
            "auto_publish_ts": auto_publish_ts,
            "approval_channel_id": self.bot.daily_topic_approval_channel_id,
            "publish_channel_id": self.bot.daily_topic_publish_channel_id,
        }

        await self._set_pending(guild_id, pending)
        msg = await self._upsert_approval_message(
            guild_id=guild_id,
            pending=pending,
            status_text="⏳ Pendiente de aprobación de admins."
        )
        if not msg:
            await self._clear_pending(guild_id)
            return False, "No se pudo publicar el mensaje de aprobación.", None

        pending["approval_message_id"] = msg.id
        await self._set_pending(guild_id, pending)
        return True, "Tema generado y enviado para aprobación.", pending

    async def _should_create_new_topic(self, guild_id: int, now_utc: datetime) -> bool:
        interval_seconds = self.bot.daily_topic_interval_seconds
        if interval_seconds > 0:
            last_ts = await self._get_last_run_ts(guild_id)
            return (time.time() - last_ts) >= interval_seconds

        if now_utc.hour < self.bot.daily_topic_approval_hour_utc:
            return False
        last_date = await self._get_last_run_date(guild_id)
        return last_date != now_utc.date().isoformat()

    async def _mark_topic_cycle_started(self, guild_id: int, now_utc: datetime) -> None:
        await self._set_last_run_ts(guild_id, time.time())
        await self._set_last_run_date(guild_id, now_utc.date().isoformat())

    async def trigger_on_demand_proposal(self, guild: discord.Guild) -> Tuple[bool, str]:
        """Create daily topic proposal immediately (admin-triggered), reusing normal approval workflow."""
        if not self.bot.daily_topic_enabled:
            return False, "Flujo de tema diario deshabilitado (DAILY_TOPIC_ENABLED=False)."

        if not self._guild_matches_configured_channels(guild):
            return False, "Este servidor no coincide con los canales configurados del flujo diario."

        lock = self._lock_for_guild(guild.id)
        async with lock:
            pending = await self._get_pending(guild.id)
            if pending:
                msg = await self._upsert_approval_message(
                    guild_id=guild.id,
                    pending=pending,
                    status_text="⏳ Pendiente de aprobación de admins."
                )
                if not msg:
                    return False, (
                        "Hay un tema pendiente, pero no se pudo actualizar/publicar el mensaje de aprobación."
                    )
                pending["approval_message_id"] = msg.id
                await self._set_pending(guild.id, pending)
                return True, (
                    "Ya existe un tema pendiente de aprobación: "
                    f"**{pending.get('topic_title', 'Tema')}** "
                    f"(Categoría: **{pending.get('topic_category', 'N/A')}**)"
                )

            ok, status_msg, created_pending = await self._create_or_refresh_pending(guild_id=guild.id)
            if not ok or not created_pending:
                return False, status_msg

            await self._mark_topic_cycle_started(guild.id, datetime.now(timezone.utc))
            return True, (
                "Tema generado en modo on-demand y enviado para aprobación: "
                f"**{created_pending.get('topic_title', 'Tema')}** "
                f"(Categoría: **{created_pending.get('topic_category', 'N/A')}**)"
            )

    @tasks.loop(seconds=60)
    async def daily_topic_scheduler(self):
        if not self.bot.daily_topic_enabled:
            return
        if not self.bot.daily_topic_approval_channel_id or not self.bot.daily_topic_publish_channel_id:
            return

        now_utc = datetime.now(timezone.utc)
        for guild in self.bot.guilds:
            if not self._guild_matches_configured_channels(guild):
                continue

            lock = self._lock_for_guild(guild.id)
            async with lock:
                pending = await self._get_pending(guild.id)

                if pending:
                    auto_publish_ts = float(pending.get("auto_publish_ts", 0))
                    if auto_publish_ts and time.time() >= auto_publish_ts:
                        ok, msg = await self._publish_topic_thread(
                            guild_id=guild.id,
                            pending=pending,
                            approved_by=None,
                            auto_published=True
                        )
                        if ok:
                            logger.info("Daily topic auto-published for guild %s: %s", guild.id, msg)
                        else:
                            logger.error("Daily topic auto-publish failed for guild %s: %s", guild.id, msg)
                            pending["auto_publish_ts"] = time.time() + 300
                            await self._set_pending(guild.id, pending)
                    continue

                should_create = await self._should_create_new_topic(guild.id, now_utc)
                if not should_create:
                    continue

                ok, msg, _ = await self._create_or_refresh_pending(guild_id=guild.id)
                if ok:
                    await self._mark_topic_cycle_started(guild.id, now_utc)
                    logger.info("Daily topic proposal created for guild %s", guild.id)
                else:
                    logger.error("Failed creating daily topic proposal for guild %s: %s", guild.id, msg)

    @daily_topic_scheduler.before_loop
    async def before_daily_topic_scheduler(self):
        await self.bot.wait_until_ready()

    @app_commands.default_permissions(administrator=True)
    @app_commands.guild_only()
    @app_commands.command(
        name="refresh_topic",
        description="Create/refresh today's pending topic for admin approval (Admin only)"
    )
    async def refresh_topic_command(self, interaction: discord.Interaction):
        if not interaction.user.guild_permissions.administrator:  # type: ignore
            await interaction.response.send_message(
                "❌ This command requires administrator permissions.",
                ephemeral=True
            )
            return

        if not interaction.guild:
            await interaction.response.send_message("❌ This command can only be used in a server.", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)

        ok, msg = await self.trigger_on_demand_proposal(interaction.guild)
        await interaction.followup.send(
            f"{'✅' if ok else '❌'} {msg}",
            ephemeral=True
        )

    @app_commands.default_permissions(administrator=True)
    @app_commands.guild_only()
    @app_commands.command(name="approve_topic", description="Approve or reject today's topic proposal (Admin only)")
    @app_commands.describe(response="yes = publish, no = regenerate another topic")
    @app_commands.choices(
        response=[
            app_commands.Choice(name="yes", value="yes"),
            app_commands.Choice(name="no", value="no"),
        ]
    )
    async def approve_topic_command(self, interaction: discord.Interaction, response: app_commands.Choice[str]):
        if not interaction.user.guild_permissions.administrator:  # type: ignore
            await interaction.response.send_message(
                "❌ This command requires administrator permissions.",
                ephemeral=True
            )
            return

        if not interaction.guild:
            await interaction.response.send_message("❌ This command can only be used in a server.", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)

        guild_id = interaction.guild.id
        if not self._guild_matches_configured_channels(interaction.guild):
            await interaction.followup.send(
                "❌ Este servidor no coincide con los canales configurados para el flujo diario.",
                ephemeral=True
            )
            return

        lock = self._lock_for_guild(guild_id)
        async with lock:
            pending = await self._get_pending(guild_id)
            if not pending:
                await interaction.followup.send("ℹ️ No hay tema pendiente de aprobación.", ephemeral=True)
                return

            if response.value == "yes":
                ok, msg = await self._publish_topic_thread(
                    guild_id=guild_id,
                    pending=pending,
                    approved_by=str(interaction.user),
                    auto_published=False
                )
                if ok:
                    await interaction.followup.send(f"✅ {msg}", ephemeral=True)
                else:
                    await interaction.followup.send(f"❌ {msg}", ephemeral=True)
                return

            # response == "no"
            ok, msg, new_pending = await self._create_or_refresh_pending(
                guild_id=guild_id,
                previous=pending,
                preserve_deadline=True
            )
            if ok and new_pending:
                    await interaction.followup.send(
                        (
                            "🔁 Tema rechazado. Nuevo tema generado para aprobación:\n"
                            f"**{new_pending.get('topic_title', 'Tema')}** "
                            f"(Categoría: **{new_pending.get('topic_category', 'N/A')}**)"
                        ),
                        ephemeral=True
                    )
            else:
                await interaction.followup.send(f"❌ No se pudo regenerar el tema: {msg}", ephemeral=True)


async def setup(bot: "AIBot"):
    await bot.add_cog(DailyTopicCog(bot))
