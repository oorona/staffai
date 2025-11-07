# bot.py
import discord
from discord.ext import commands
import logging
from typing import List, Optional, Dict, Any, Set

import redis
from utils.litellm_client import LiteLLMClient
from utils.message_handler import MessageHandler

logger = logging.getLogger(__name__)

class AIBot(commands.Bot):
    def __init__(self,
                 chat_system_prompt: Optional[str],
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
                 stats_report_channel_id: Optional[int],
                 stats_report_interval_seconds: int,
                 stats_report_top_users: int,
                 intents: discord.Intents):
        super().__init__(command_prefix="!", intents=intents, help_command=None)

        # Core configuration
        self.chat_system_prompt = chat_system_prompt
        self.response_chance = response_chance
        self.max_history_per_context = max_history_per_context
        
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

        # Stats/Reporting
        self.stats_report_channel_id = stats_report_channel_id
        self.stats_report_interval_seconds = stats_report_interval_seconds
        self.stats_report_top_users = stats_report_top_users
        
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

        # Initialize MessageHandler
        self.message_handler = MessageHandler(bot=self)

        # Log configuration
        logger.info("=== AIBot Configuration ===")
        logger.info(f"LiteLLM: {self.litellm_api_url} | Model: {self.litellm_model}")
        logger.info(f"MCP Servers: {len(self.mcp_servers)} configured")
        logger.info(f"Response Chance: {self.response_chance*100:.1f}%")
        logger.info(f"Random Response Delivery Chance: {self.random_response_delivery_chance*100:.1f}%")
        logger.info(f"Max History Per User/Channel: {self.max_history_per_context}")
        logger.info(f"Context TTL: {self.context_history_ttl_seconds}s | Message Max Age: {self.context_message_max_age_seconds}s")
        logger.info(f"Rate Limits: {self.rate_limit_count} msgs/{self.rate_limit_window_seconds}s, {self.token_rate_limit_count} tokens/{self.rate_limit_window_seconds}s")
        
        if self.super_role_ids_set:
            logger.info(f"Super Roles (bypass limits): {self.super_role_ids_set}")
        if self.ignored_role_ids_set:
            logger.info(f"Ignored Roles: {self.ignored_role_ids_set}")
        
        logger.info(f"Activity Updates: Every {self.activity_update_interval_seconds}s")
        if self.activity_schedule_enabled:
            logger.info(f"Activity Schedule: Active Hours {self.activity_active_start_hour_utc:02d}-{self.activity_active_end_hour_utc:02d} UTC, Days: {sorted(list(self.activity_active_days_utc))}")
        
        logger.info("AIBot initialized successfully")

    async def setup_hook(self):
        """Load cogs during bot setup"""
        extensions = [
            'cogs.message_cog',
            'cogs.activity_cog',
            'cogs.stats_cog'
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
        
        # Sync slash commands
        try:
            synced = await self.tree.sync()
            logger.info(f"Synced {len(synced)} slash commands")
        except Exception as e:
            logger.error(f"Failed to sync slash commands: {e}")
