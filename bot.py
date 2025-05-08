# bot.py
import discord
from discord.ext import commands
import logging
from typing import List, Optional, Dict, Any
import redis # For the general redis client

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
                 redis_config: Dict[str, Any],
                 ignored_role_ids: List[int],
                 rate_limit_count: int,
                 rate_limit_window_seconds: int,
                 token_rate_limit_count: int,
                 restricted_user_role_id: Optional[int],
                 restricted_channel_id: Optional[int],
                 rate_limit_message_user: str,
                 restricted_channel_message_user: str,
                 rate_limit_exempt_role_ids: List[int],
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
        self.redis_config = redis_config # For WebUIAPI
        self.ignored_role_ids = ignored_role_ids
        self.ignored_role_ids_set = set(ignored_role_ids)

        self.rate_limit_count = rate_limit_count
        self.rate_limit_window_seconds = rate_limit_window_seconds
        self.token_rate_limit_count = token_rate_limit_count
        self.restricted_user_role_id = restricted_user_role_id
        self.restricted_channel_id = restricted_channel_id
        self.rate_limit_message_user_template = rate_limit_message_user
        self.restricted_channel_message_user_template = restricted_channel_message_user
        self.rate_limit_exempt_role_ids_set = set(rate_limit_exempt_role_ids)
        if self.rate_limit_exempt_role_ids_set:
            logger.info(f"Rate limits will be EXEMPT for Role IDs: {self.rate_limit_exempt_role_ids_set}")

        # General Redis client for cogs/other parts of the bot
        self.redis_client_general: Optional[redis.Redis] = None
        if self.redis_config and self.redis_config.get('host'):
            try:
                self.redis_client_general = redis.Redis(**self.redis_config, decode_responses=True, socket_connect_timeout=3)
                self.redis_client_general.ping()
                logger.info(f"AIBot: Successfully connected general Redis client to {self.redis_config.get('host')}")
            except redis.exceptions.ConnectionError as e:
                logger.error(f"AIBot: Failed to connect general Redis client: {e}. Some features like rate limiting may not work.", exc_info=True)
                self.redis_client_general = None
            except Exception as e:
                logger.error(f"AIBot: Unexpected error initializing general Redis client: {e}", exc_info=True)
                self.redis_client_general = None
        else:
            logger.warning("AIBot: Redis not configured or host not specified; general Redis client not initialized. Rate limiting will be disabled.")


        logger.info("AIBot instance configured.")
        # Detailed logging of configs is now mostly in main.py

    async def setup_hook(self):
        initial_extensions = ['cogs.listener_cog']
        for extension in initial_extensions:
            try:
                await self.load_extension(extension)
                logger.info(f"Successfully loaded extension: {extension}")
            except commands.ExtensionNotFound:
                logger.critical(f"FATAL: Extension not found: {extension}.", exc_info=True)
            except commands.NoEntryPointError:
                logger.critical(f"FATAL: Extension {extension} does not have a 'setup' function.", exc_info=True)
            except commands.ExtensionFailed as e:
                logger.critical(f"FATAL: Failed to load extension {extension}: {e.original}", exc_info=True)
            except Exception as e:
                 logger.critical(f"FATAL: An unexpected error occurred loading extension {extension}: {e}", exc_info=True)
        logger.info("Setup hook completed.")

    async def on_ready(self):
        logger.info(f'Logged in as {self.user.name} (ID: {self.user.id})')
        # Redis connection test for general client is implicitly done in __init__
        # WebUIAPI also does its own Redis test if config is passed to it.
        logger.info('------ Bot is Ready ------')