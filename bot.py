# bot.py
import discord
from discord.ext import commands
import logging
from typing import List, Optional, Dict, Any, Set, Tuple
import time
import psutil
import os
import signal
import asyncio
from datetime import datetime, timezone

import redis
from utils.litellm_client import LiteLLMClient
from utils.message_handler import MessageHandler
from utils.prompt_manager import PromptManager
from utils.user_memory_manager import UserMemoryManager

logger = logging.getLogger(__name__)

class AIBot(commands.Bot):
    def __init__(self,
                 chat_system_prompt: Optional[str],
                 prompts_root_path: str,
                 core_prompt_fallback: Optional[str],
                 default_persona_prompt_fallback: Optional[str],
                 response_chance: float,
                 max_history_per_context: int,
                 litellm_api_url: str,
                 litellm_model: str,
                 litellm_api_key: Optional[str],
                 mcp_servers: List[str],
                 redis_config: Dict[str, Any],
                 ignored_role_ids: List[int],
                 rate_limit_count: int,
                 rate_limit_window_seconds: int,
                 token_rate_limit_count: int,
                 restricted_user_role_id: Optional[int],
                 restricted_channel_id: Optional[int],
                 rate_limit_message_user_template: str,
                 restricted_channel_message_user_template: str,
                 super_role_ids: List[int],
                 restriction_duration_seconds: int,
                 restriction_check_interval_seconds: int,
                 random_response_delivery_chance: float,
                 base_activity_system_prompt: Optional[str],
                 activity_update_interval_seconds: int,
                 activity_schedule_enabled: bool,
                 activity_active_start_hour_utc: int,
                 activity_active_end_hour_utc: int,
                 activity_active_days_utc: Set[int],
                 context_history_ttl_seconds: int,
                 context_message_max_age_seconds: int,
                 default_context_messages: int,
                 stats_report_channel_id: Optional[int],
                 stats_report_interval_seconds: int,
                 stats_report_top_users: int,
                 daily_topic_enabled: bool,
                 daily_topic_approval_channel_id: Optional[int],
                 daily_topic_publish_channel_id: Optional[int],
                 daily_topic_interval_seconds: int,
                 daily_topic_approval_hour_utc: int,
                 daily_topic_approval_timeout_seconds: int,
                 daily_topic_check_interval_seconds: int,
                 daily_topic_thread_auto_archive_minutes: int,
                 daily_topic_thread_context_messages: int,
                 user_memory_enabled: bool,
                 user_memory_root_path: str,
                 user_memory_update_chance: float,
                 user_memory_min_message_chars: int,
                 user_memory_min_message_words: int,
                 user_memory_max_chars: int,
                 user_memory_pipeline_mode: str,
                 user_memory_ollama_base_url: str,
                 user_memory_ollama_api_key: str,
                 user_memory_ollama_timeout_seconds: float,
                 user_memory_tiny_model: str,
                 user_memory_tiny_model_extract: Optional[str],
                 user_memory_tiny_model_classifier: Optional[str],
                 user_memory_tiny_accumulate_max_tokens: int,
                 user_memory_audit_max_entries: int,
                 user_memory_debug_classification: bool,
                 llm_call_audit_enabled: bool,
                 llm_call_audit_max_entries: int,
                 debug_context_super_users: bool,
                 intents: discord.Intents):
        super().__init__(command_prefix="!", intents=intents, help_command=None)

        # Core configuration
        self.chat_system_prompt = chat_system_prompt
        self.prompts_root_path = prompts_root_path
        self.response_chance = response_chance
        self.max_history_per_context = max_history_per_context

        # Prompt manager (default + per-channel prompts)
        self.prompt_manager = PromptManager(
            prompts_root_path=self.prompts_root_path,
            core_prompt_fallback=core_prompt_fallback or "",
            default_persona_fallback=default_persona_prompt_fallback or ""
        )
        prompt_reload_success, prompt_reload_message = self.prompt_manager.reload()
        if not prompt_reload_success:
            logger.warning(f"Prompt manager startup reload warning: {prompt_reload_message}")
        
        # LiteLLM configuration
        self.litellm_api_url = litellm_api_url
        self.litellm_model = litellm_model
        self.litellm_api_key = litellm_api_key
        self.mcp_servers = mcp_servers

        # Role-based access control
        self.ignored_role_ids: List[int] = ignored_role_ids
        self.ignored_role_ids_set: Set[int] = set(ignored_role_ids)
        self.super_role_ids: List[int] = super_role_ids
        self.super_role_ids_set: Set[int] = set(super_role_ids)

        # Rate limiting & restrictions
        self.rate_limit_count = rate_limit_count
        self.rate_limit_window_seconds = rate_limit_window_seconds
        self.token_rate_limit_count = token_rate_limit_count
        self.restricted_user_role_id = restricted_user_role_id
        self.restricted_channel_id = restricted_channel_id
        self.rate_limit_message_user_template = rate_limit_message_user_template
        self.restricted_channel_message_user_template = restricted_channel_message_user_template
        self.restriction_duration_seconds = restriction_duration_seconds
        self.restriction_check_interval_seconds = restriction_check_interval_seconds

        # Bot behavior
        self.random_response_delivery_chance = random_response_delivery_chance

        # Activity/Presence
        self.base_activity_system_prompt = base_activity_system_prompt
        self.activity_update_interval_seconds = activity_update_interval_seconds
        self.activity_schedule_enabled = activity_schedule_enabled
        self.activity_active_start_hour_utc = activity_active_start_hour_utc
        self.activity_active_end_hour_utc = activity_active_end_hour_utc
        self.activity_active_days_utc = activity_active_days_utc

        # Context decay
        self.context_history_ttl_seconds = context_history_ttl_seconds
        self.context_message_max_age_seconds = context_message_max_age_seconds
        self.default_context_messages = default_context_messages

        # Stats/Reporting
        self.stats_report_channel_id = stats_report_channel_id
        self.stats_report_interval_seconds = stats_report_interval_seconds
        self.stats_report_top_users = stats_report_top_users
        
        # Daily topic workflow
        self.daily_topic_enabled = daily_topic_enabled
        self.daily_topic_approval_channel_id = daily_topic_approval_channel_id
        self.daily_topic_publish_channel_id = daily_topic_publish_channel_id
        self.daily_topic_interval_seconds = daily_topic_interval_seconds
        self.daily_topic_approval_hour_utc = daily_topic_approval_hour_utc
        self.daily_topic_approval_timeout_seconds = daily_topic_approval_timeout_seconds
        self.daily_topic_check_interval_seconds = daily_topic_check_interval_seconds
        self.daily_topic_thread_auto_archive_minutes = daily_topic_thread_auto_archive_minutes
        self.daily_topic_thread_context_messages = daily_topic_thread_context_messages
        
        # User memory
        self.user_memory_enabled = user_memory_enabled
        self.user_memory_root_path = user_memory_root_path
        self.user_memory_update_chance = user_memory_update_chance
        self.user_memory_min_message_chars = user_memory_min_message_chars
        self.user_memory_min_message_words = user_memory_min_message_words
        self.user_memory_max_chars = user_memory_max_chars
        self.user_memory_pipeline_mode = user_memory_pipeline_mode
        self.user_memory_ollama_base_url = user_memory_ollama_base_url
        self.user_memory_ollama_api_key = user_memory_ollama_api_key
        self.user_memory_ollama_timeout_seconds = user_memory_ollama_timeout_seconds
        self.user_memory_tiny_model = user_memory_tiny_model
        self.user_memory_tiny_model_extract = user_memory_tiny_model_extract
        self.user_memory_tiny_model_classifier = user_memory_tiny_model_classifier
        self.user_memory_tiny_accumulate_max_tokens = user_memory_tiny_accumulate_max_tokens
        self.user_memory_audit_max_entries = user_memory_audit_max_entries
        self.user_memory_debug_classification = user_memory_debug_classification

        # LLM call audit
        self.llm_call_audit_enabled = llm_call_audit_enabled
        self.llm_call_audit_max_entries = llm_call_audit_max_entries

        # Debug mode
        self.debug_context_super_users = debug_context_super_users

        # Stats cog reference (set after cog loads)
        self.stats_cog = None

        # Initialize LiteLLM client
        self.litellm_client = LiteLLMClient(
            model=self.litellm_model,
            base_url=self.litellm_api_url,
            api_key=self.litellm_api_key,
            redis_client=None,  # Will be set after Redis connection
            max_history_messages=self.max_history_per_context,
            context_history_ttl_seconds=self.context_history_ttl_seconds,
            context_message_max_age_seconds=self.context_message_max_age_seconds,
            mcp_servers=self.mcp_servers
        )

        # Initialize Redis client for rate limiting and restrictions
        self.redis_client: Optional[redis.Redis] = None  # type: ignore
        if redis_config and redis_config.get('host'):
            try:
                self.redis_client = redis.Redis(**redis_config, decode_responses=True, socket_connect_timeout=3)  # type: ignore
                self.redis_client.ping()  # type: ignore
                logger.info(f"AIBot: Successfully connected Redis client to {redis_config.get('host')}:{redis_config.get('port')}")
                # Set Redis client on LiteLLM client now that it's connected
                self.litellm_client.redis_client = self.redis_client
            except redis.exceptions.ConnectionError as e:  # type: ignore
                logger.error(f"AIBot: Failed to connect Redis client: {e}. Rate limiting may not work.", exc_info=True)
                self.redis_client = None
            except Exception as e:
                logger.error(f"AIBot: Unexpected error initializing Redis client: {e}", exc_info=True)
                self.redis_client = None
        else:
            logger.warning("AIBot: Redis not configured; rate limiting and restrictions disabled.")

        # Initialize persistent user memory manager
        self.user_memory_manager = UserMemoryManager(
            prompts_root_path=self.prompts_root_path,
            memory_root_path=self.user_memory_root_path,
            redis_client=self.redis_client,
            litellm_client=self.litellm_client,
            enabled=self.user_memory_enabled,
            update_chance=self.user_memory_update_chance,
            min_message_chars=self.user_memory_min_message_chars,
            min_message_words=self.user_memory_min_message_words,
            max_memory_chars=self.user_memory_max_chars,
            pipeline_mode=self.user_memory_pipeline_mode,
            ollama_base_url=self.user_memory_ollama_base_url,
            ollama_api_key=self.user_memory_ollama_api_key,
            ollama_timeout_seconds=self.user_memory_ollama_timeout_seconds,
            tiny_model=self.user_memory_tiny_model,
            tiny_model_extract=self.user_memory_tiny_model_extract,
            tiny_model_classifier=self.user_memory_tiny_model_classifier,
            tiny_accumulate_max_tokens=self.user_memory_tiny_accumulate_max_tokens,
            memory_audit_max_entries=self.user_memory_audit_max_entries,
            debug_classification_logs=self.user_memory_debug_classification
        )

        # Initialize MessageHandler
        self.message_handler = MessageHandler(bot=self)

        # Track bot start time for uptime calculation
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())
        self.shutdown_signal = None
        self._commands_synced_once = False

        # Log configuration
        logger.info("=== AIBot Configuration ===")
        logger.info(f"LiteLLM: {self.litellm_api_url} | Model: {self.litellm_model}")
        logger.info(f"MCP Servers: {len(self.mcp_servers)} configured")
        logger.info(f"Response Chance: {self.response_chance*100:.1f}%")
        logger.info(f"Random Response Delivery Chance: {self.random_response_delivery_chance*100:.1f}%")
        logger.info(f"Max History Per User/Channel: {self.max_history_per_context}")
        logger.info(f"Context TTL: {self.context_history_ttl_seconds}s | Message Max Age: {self.context_message_max_age_seconds}s | Context Messages: {self.default_context_messages}")
        logger.info(f"Rate Limits: {self.rate_limit_count} msgs/{self.rate_limit_window_seconds}s, {self.token_rate_limit_count} tokens/{self.rate_limit_window_seconds}s")
        logger.info(f"Prompts: root={self.prompts_root_path} | channel_overrides={self.prompt_manager.channel_prompt_count}")
        logger.info(
            "User Memory: enabled=%s | root=%s | update_chance=%.2f | min_chars=%s | min_words=%s | max_chars=%s | pipeline_mode=%s",
            self.user_memory_enabled,
            self.user_memory_root_path,
            self.user_memory_update_chance,
            self.user_memory_min_message_chars,
            self.user_memory_min_message_words,
            self.user_memory_max_chars,
            self.user_memory_pipeline_mode
        )
        logger.info(
            "User Memory Tiny Model: base_url=%s | default_model=%s | extract_model=%s | classifier_model=%s | timeout=%.1fs | compact_threshold_tokens=%s | audit_max_entries=%s | debug_classification=%s",
            self.user_memory_ollama_base_url,
            self.user_memory_tiny_model,
            self.user_memory_tiny_model_extract or self.user_memory_tiny_model,
            self.user_memory_tiny_model_classifier or self.user_memory_tiny_model,
            self.user_memory_ollama_timeout_seconds,
            self.user_memory_tiny_accumulate_max_tokens,
            self.user_memory_audit_max_entries,
            self.user_memory_debug_classification,
        )
        logger.info(
            "LLM Call Audit: enabled=%s | max_entries=%s",
            self.llm_call_audit_enabled,
            self.llm_call_audit_max_entries
        )
        logger.info(
            "Daily Topic: enabled=%s | approval_channel=%s | publish_channel=%s | interval=%ss | hour_utc=%s | timeout=%ss",
            self.daily_topic_enabled,
            self.daily_topic_approval_channel_id,
            self.daily_topic_publish_channel_id,
            self.daily_topic_interval_seconds,
            self.daily_topic_approval_hour_utc,
            self.daily_topic_approval_timeout_seconds
        )
        
        if self.super_role_ids_set:
            logger.info(f"Super Roles (bypass limits): {self.super_role_ids_set}")
        if self.ignored_role_ids_set:
            logger.info(f"Ignored Roles: {self.ignored_role_ids_set}")
        
        logger.info(f"Activity Updates: Every {self.activity_update_interval_seconds}s")
        if self.activity_schedule_enabled:
            logger.info(f"Activity Schedule: Active Hours {self.activity_active_start_hour_utc:02d}-{self.activity_active_end_hour_utc:02d} UTC, Days: {sorted(list(self.activity_active_days_utc))}")
        
        logger.info("AIBot initialized successfully")

    def get_chat_system_prompt(
        self,
        channel_id: Optional[int] = None,
        parent_channel_id: Optional[int] = None
    ) -> str:
        """Return channel-specific prompt with optional parent-channel fallback for threads."""
        if channel_id is not None and self.prompt_manager.has_channel_override(channel_id):
            return self.prompt_manager.get_prompt(channel_id)
        if parent_channel_id is not None and self.prompt_manager.has_channel_override(parent_channel_id):
            return self.prompt_manager.get_prompt(parent_channel_id)
        return self.prompt_manager.get_prompt(channel_id)

    def reload_prompts(self) -> Tuple[bool, str]:
        """Reload default and channel prompts from disk."""
        return self.prompt_manager.reload()

    async def setup_hook(self):
        """Load cogs during bot setup"""
        extensions = [
            'cogs.message_cog',
            'cogs.activity_cog',
            'cogs.stats_cog',
            'cogs.daily_topic_cog',
            'cogs.user_memory_cog'
        ]
        
        for extension in extensions:
            try:
                await self.load_extension(extension)
                logger.info(f"Loaded extension: {extension}")
            except commands.ExtensionNotFound:
                logger.critical(f"Extension not found: {extension}")
            except commands.NoEntryPointError:
                logger.critical(f"Extension missing 'setup' function: {extension}")
            except commands.ExtensionFailed as e:
                logger.critical(f"Extension failed to load: {extension} - {e.original}", exc_info=True)  # type: ignore
            except Exception as e:
                logger.critical(f"Unexpected error loading {extension}: {e}", exc_info=True)
        
        # Hydrate Redis with missing user memory from disk at startup.
        loaded_count, scanned_count = await self.user_memory_manager.hydrate_redis_from_disk()
        if scanned_count > 0:
            logger.info(
                "User memory hydration complete: loaded_missing=%s, scanned_files=%s",
                loaded_count,
                scanned_count
            )

        # Store reference to stats_cog for message_handler access
        self.stats_cog = self.get_cog('StatsCog')
        if self.stats_cog:
            logger.info("StatsCog reference stored in bot")
        
        # Preload MCP tools at startup (cache them for all future requests)
        if self.mcp_servers:
            logger.info(f"🔧 Preloading MCP tools from {len(self.mcp_servers)} servers...")
            try:
                mcp_tools = await self.litellm_client.get_mcp_tools()
                if mcp_tools:
                    logger.info(f"✅ Cached {len(mcp_tools)} MCP tools for session (startup preload)")
                else:
                    logger.warning("⚠️  No MCP tools loaded from any server")
            except Exception as e:
                logger.error(f"❌ Failed to preload MCP tools: {e}", exc_info=True)
        
        logger.info("Setup hook completed")

    async def on_ready(self):
        """Called when bot is ready"""
        logger.info(f'Logged in as {self.user.name} (ID: {self.user.id})')  # type: ignore
        logger.info('========== Bot is Ready ==========')

        # Sync slash commands once per process start.
        # We sync globally and also per guild for fast command visibility.
        synced_count = 0
        if not self._commands_synced_once:
            try:
                synced = await self.tree.sync()
                synced_count = len(synced)
                logger.info(f"Synced {synced_count} global slash commands")

                for guild in self.guilds:
                    try:
                        self.tree.copy_global_to(guild=guild)
                        guild_synced = await self.tree.sync(guild=guild)
                        logger.info(
                            "Synced %s guild slash commands for %s (%s)",
                            len(guild_synced),
                            guild.name,
                            guild.id
                        )
                    except Exception as guild_sync_error:
                        logger.error(
                            "Failed guild command sync for %s (%s): %s",
                            guild.name,
                            guild.id,
                            guild_sync_error
                        )

                self._commands_synced_once = True
            except Exception as e:
                logger.error(f"Failed to sync slash commands: {e}")

        # Send startup notification
        # await self.send_startup_notification(synced_count)



    async def close(self):
        """Override close to send shutdown notification"""
        logger.info("Bot shutdown initiated...")

        # Send shutdown notification before closing
        # await self.send_shutdown_notification()

        # Close Redis connection
        if self.redis_client:
            try:
                self.redis_client.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis: {e}")

        # Flush pending user-memory file writes
        try:
            await self.user_memory_manager.close()
        except Exception as e:
            logger.error(f"Error flushing user memory files: {e}")

        # Call parent close
        await super().close()
        logger.info("Bot shutdown complete")
