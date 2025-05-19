# bot.py
import discord
from discord.ext import commands
import logging
from typing import List, Optional, Dict, Any, Set

import redis # For the general redis client
from utils.webui_api import WebUIAPI # Ensure this path is correct

logger = logging.getLogger(__name__)

class AIBot(commands.Bot):
    def __init__(self,
                 welcome_channel_id: Optional[int],
                 welcome_system_prompt: Optional[str],
                 welcome_user_prompt: Optional[str],
                 chat_system_prompt: Optional[str],
                 response_chance: float,
                 max_history_per_context: int,
                 api_url: str,
                 model: str,
                 api_key: Optional[str],
                 list_tools: List[str],
                 knowledge_id: Optional[str],
                 redis_config: Dict[str, Any], # This is for WebUIAPI's history Redis
                 general_redis_config: Dict[str, Any], # Separate config for general Redis if needed, or use same
                 ignored_role_ids: List[int],
                 rate_limit_count: int,
                 rate_limit_window_seconds: int,
                 token_rate_limit_count: int,
                 restricted_user_role_id: Optional[int],
                 restricted_channel_id: Optional[int],
                 rate_limit_message_user_template: str, # Renamed from rate_limit_message_user
                 restricted_channel_message_user_template: str, # Renamed from restricted_channel_message_user
                 rate_limit_exempt_role_ids: List[int],
                 restriction_duration_seconds: int,
                 restriction_check_interval_seconds: int,
                 intents: discord.Intents):
        super().__init__(command_prefix="!", intents=intents, help_command=None)

        self.welcome_channel_id = welcome_channel_id
        self.welcome_system_prompt = welcome_system_prompt
        self.welcome_user_prompt = welcome_user_prompt
        self.chat_system_prompt = chat_system_prompt
        self.response_chance = response_chance
        self.max_history_per_context = max_history_per_context
        self.api_url = api_url
        self.model = model
        self.api_key = api_key
        self.list_tools = list_tools
        self.knowledge_id = knowledge_id
        
        self.ignored_role_ids: List[int] = ignored_role_ids # Storing the list
        self.ignored_role_ids_set: Set[int] = set(ignored_role_ids)

        self.rate_limit_count = rate_limit_count
        self.rate_limit_window_seconds = rate_limit_window_seconds
        self.token_rate_limit_count = token_rate_limit_count
        self.restricted_user_role_id = restricted_user_role_id
        self.restricted_channel_id = restricted_channel_id
        self.rate_limit_message_user_template = rate_limit_message_user_template
        self.restricted_channel_message_user_template = restricted_channel_message_user_template
        self.rate_limit_exempt_role_ids: List[int] = rate_limit_exempt_role_ids # Storing the list
        self.rate_limit_exempt_role_ids_set: Set[int] = set(rate_limit_exempt_role_ids)

        self.restriction_duration_seconds = restriction_duration_seconds
        self.restriction_check_interval_seconds = restriction_check_interval_seconds

        # Instantiate WebUIAPI client here and store it on the bot instance
        # It uses 'redis_config' for its dedicated history Redis connection
        self.api_client = WebUIAPI(
            base_url=self.api_url,
            model=self.model,
            api_key=self.api_key,
            welcome_system=self.welcome_system_prompt, # WebUIAPI uses this for its welcome
            welcome_prompt=self.welcome_user_prompt, # WebUIAPI uses this for its welcome
            max_history_per_user=self.max_history_per_context,
            knowledge_id=self.knowledge_id,
            list_tools=self.list_tools,
            redis_config=redis_config # This is for WebUIAPI's history
        )
        
        # General Redis client for cogs/other parts of the bot (e.g. rate limits, restriction expiry tracking)
        # This uses 'general_redis_config'
        self.redis_client_general: Optional[redis.Redis] = None
        if general_redis_config and general_redis_config.get('host'):
            try:
                self.redis_client_general = redis.Redis(**general_redis_config, decode_responses=True, socket_connect_timeout=3) # type: ignore
                self.redis_client_general.ping() # type: ignore
                logger.info(f"AIBot: Successfully connected general Redis client to {general_redis_config.get('host')}:{general_redis_config.get('port')}")
            except redis.exceptions.ConnectionError as e: # type: ignore
                logger.error(f"AIBot: Failed to connect general Redis client: {e}. Rate limiting and restriction expiry may not work.", exc_info=True)
                self.redis_client_general = None
            except Exception as e:
                logger.error(f"AIBot: Unexpected error initializing general Redis client: {e}", exc_info=True)
                self.redis_client_general = None
        else:
            logger.warning("AIBot: General Redis not configured or host not specified; general Redis client not initialized. Rate limiting and restriction expiry will be disabled if it relies on this.")

        if self.rate_limit_exempt_role_ids_set:
            logger.info(f"AIBot: Rate limits will be EXEMPT for Role IDs: {self.rate_limit_exempt_role_ids_set}")
        if self.ignored_role_ids_set:
            logger.info(f"AIBot: Messages from users with these Role IDs will be IGNORED: {self.ignored_role_ids_set}")

        logger.info("AIBot instance configured with WebUIAPI client.")

    async def setup_hook(self):
        initial_extensions = ['cogs.listener_cog'] # Ensure this path is correct if cogs is a top-level dir
        for extension in initial_extensions:
            try:
                await self.load_extension(extension)
                logger.info(f"Successfully loaded extension: {extension}")
            except Exception as e:
                 logger.critical(f"FATAL: An unexpected error occurred loading extension {extension}: {e}", exc_info=True)
                 # Consider re-raising or sys.exit if critical
        logger.info("Setup hook completed.")

    async def on_ready(self):
        logger.info(f'Logged in as {self.user.name} (ID: {self.user.id})') # type: ignore
        logger.info('------ Bot is Ready ------')