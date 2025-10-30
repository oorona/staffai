# bot.py
import discord
from discord.ext import commands
import logging
from typing import List, Optional, Dict, Any, Set

import redis
from utils.webui_api import WebUIAPI # Keep this import

import spacy

logger = logging.getLogger(__name__)

class AIBot(commands.Bot):
    def __init__(self,
                 chat_system_prompt: Optional[str],
                 # NEW: Add sentiment_system_prompt
                 sentiment_system_prompt: Optional[str],
                 response_chance: float,
                 max_history_per_context: int,
                 api_url: str,
                 model: str,
                 api_key: Optional[str],
                 list_tools: List[str],
                 restricted_list_tools: List[str],
                 knowledge_id: Optional[str],
                 redis_config: Dict[str, Any],
                 general_redis_config: Dict[str, Any],
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
                 profile_max_scored_messages: int,
                 spacy_en_model_name: Optional[str],
                 spacy_es_model_name: Optional[str],
                 random_response_delivery_chance: float,
                 worthiness_min_length: int,
                 worthiness_min_significant_words: int,
                 base_activity_system_prompt: Optional[str],
                 activity_update_interval_seconds: int,
                 activity_schedule_enabled: bool,
                 activity_active_start_hour_utc: int,
                 activity_active_end_hour_utc: int,
                 activity_active_days_utc: Set[int],
                 llm_response_validation_retries: int,
                 intents: discord.Intents,
                 context_history_ttl_seconds: int = 1800,
                 context_message_max_age_seconds: int = 1800):
        super().__init__(command_prefix="!", intents=intents, help_command=None)

        self.chat_system_prompt = chat_system_prompt
        # NEW: Store sentiment_system_prompt
        self.sentiment_system_prompt = sentiment_system_prompt
        self.response_chance = response_chance
        # ... (rest of the __init__ method remains the same until WebUIAPI instantiation) ...

        self.max_history_per_context = max_history_per_context
        self.api_url = api_url
        self.model = model
        self.api_key = api_key
        
        self.list_tools: List[str] = list_tools
        self.restricted_list_tools: List[str] = restricted_list_tools
        
        self.knowledge_id = knowledge_id
        self.guild_id_for_sync = None # Make sure this is defined if used in on_ready

        self.ignored_role_ids: List[int] = ignored_role_ids
        self.ignored_role_ids_set: Set[int] = set(ignored_role_ids)

        self.rate_limit_count = rate_limit_count
        self.rate_limit_window_seconds = rate_limit_window_seconds
        self.token_rate_limit_count = token_rate_limit_count
        self.restricted_user_role_id = restricted_user_role_id
        self.restricted_channel_id = restricted_channel_id
        self.rate_limit_message_user_template = rate_limit_message_user_template
        self.restricted_channel_message_user_template = restricted_channel_message_user_template
        
        self.super_role_ids: List[int] = super_role_ids
        self.super_role_ids_set: Set[int] = set(super_role_ids)

        self.restriction_duration_seconds = restriction_duration_seconds
        self.restriction_check_interval_seconds = restriction_check_interval_seconds

        self.profile_max_scored_messages = profile_max_scored_messages
        self.random_response_delivery_chance = random_response_delivery_chance
        self.spacy_models: Dict[str, spacy.Language] = {}
        self.worthiness_min_length = worthiness_min_length
        self.worthiness_min_significant_words = worthiness_min_significant_words

        self.base_activity_system_prompt = base_activity_system_prompt
        self.activity_update_interval_seconds = activity_update_interval_seconds
        self.activity_schedule_enabled = activity_schedule_enabled
        self.activity_active_start_hour_utc = activity_active_start_hour_utc
        self.activity_active_end_hour_utc = activity_active_end_hour_utc
        self.activity_active_days_utc = activity_active_days_utc
        
        self.llm_response_validation_retries = llm_response_validation_retries
        # TTL (seconds) to set on per-user/channel history keys in Redis. If <= 0, do not set expiry.
        self.context_history_ttl_seconds = context_history_ttl_seconds
        # Max age (seconds) for individual messages in context before they are purged
        self.context_message_max_age_seconds = context_message_max_age_seconds
        
        if spacy_en_model_name:
            try:
                self.spacy_models["en"] = spacy.load(spacy_en_model_name)
                logger.info(f"AIBot: Successfully loaded SpaCy English model '{spacy_en_model_name}'.")
            except OSError:
                logger.error(f"AIBot: Could not load SpaCy English model '{spacy_en_model_name}'. Ensure it's downloaded. English NLP features may be affected.")
            except Exception as e:
                logger.error(f"AIBot: Unexpected error loading SpaCy English model '{spacy_en_model_name}': {e}", exc_info=True)
        else:
            logger.info("AIBot: No English SpaCy model name provided. English NLP features disabled.")

        if spacy_es_model_name:
            try:
                self.spacy_models["es"] = spacy.load(spacy_es_model_name)
                logger.info(f"AIBot: Successfully loaded SpaCy Spanish model '{spacy_es_model_name}'.")
            except OSError:
                logger.error(f"AIBot: Could not load SpaCy Spanish model '{spacy_es_model_name}'. Ensure it's downloaded. Spanish NLP features may be affected.")
            except Exception as e:
                logger.error(f"AIBot: Unexpected error loading SpaCy Spanish model '{spacy_es_model_name}': {e}", exc_info=True)
        else:
            logger.info("AIBot: No Spanish SpaCy model name provided. Spanish NLP features disabled.")

        if not self.spacy_models:
            logger.warning("AIBot: No SpaCy models were successfully loaded. Language-specific worthiness scoring will be disabled.")


        # WebUIAPI instantiation - does not need direct knowledge of sentiment_system_prompt
        # as it will be passed to its new method by MessageHandler.
        self.api_client = WebUIAPI(
            base_url=self.api_url,
            model=self.model,
            api_key=self.api_key,
            max_history_per_user=self.max_history_per_context,
            list_tools_default=self.list_tools, # Default tools for general responses
            knowledge_id=self.knowledge_id,
            redis_config=redis_config, # For conversation history
            context_history_ttl_seconds=self.context_history_ttl_seconds,
            context_message_max_age_seconds=self.context_message_max_age_seconds,
            llm_response_validation_retries=self.llm_response_validation_retries
        )

        self.redis_client_general: Optional[redis.Redis] = None # type: ignore
        if general_redis_config and general_redis_config.get('host'):
            try:
                self.redis_client_general = redis.Redis(**general_redis_config, decode_responses=True, socket_connect_timeout=3) # type: ignore
                self.redis_client_general.ping() # type: ignore
                logger.info(f"AIBot: Successfully connected general Redis client to {general_redis_config.get('host')}:{general_redis_config.get('port')}")
            except redis.exceptions.ConnectionError as e: # type: ignore
                logger.error(f"AIBot: Failed to connect general Redis client: {e}. Some features may not work.", exc_info=True)
                self.redis_client_general = None
            except Exception as e:
                logger.error(f"AIBot: Unexpected error initializing general Redis client: {e}", exc_info=True)
                self.redis_client_general = None
        else:
            logger.warning("AIBot: General Redis not configured or host not specified; general Redis client not initialized.")

        # ... (rest of __init__ logging and setup_hook, on_ready remain the same)
        if self.super_role_ids_set:
            logger.info(f"AIBot: SUPER ROLES (special access & exemptions): {self.super_role_ids_set}")
        if self.ignored_role_ids_set:
            logger.info(f"AIBot: Messages from users with these Role IDs will be IGNORED: {self.ignored_role_ids_set}")
        if self.profile_max_scored_messages > 0:
            logger.info(f"AIBot: User profile storing up to {self.profile_max_scored_messages} scored messages per user.")
        else:
            logger.info("AIBot: User profile message storage is disabled (profile_max_scored_messages <= 0).")
        logger.info(f"AIBot: Random response delivery chance set to {self.random_response_delivery_chance*100:.1f}%.")
        logger.info(f"AIBot: Worthiness min message length set to {self.worthiness_min_length}.")
        logger.info(f"AIBot: Worthiness min significant words set to {self.worthiness_min_significant_words}.")
        if self.sentiment_system_prompt:
            logger.info("AIBot: Sentiment Analysis System Prompt has been loaded.")
        else:
            logger.warning("AIBot: Sentiment Analysis System Prompt is MISSING. Sentiment analysis will likely fail.")
        logger.info(f"AIBot: LLM Response Validation Retries set to: {self.llm_response_validation_retries}")
        logger.info("AIBot instance configured.")

    async def setup_hook(self):
        initial_extensions = [
            'cogs.listener_cog',
            'cogs.profile_cog',
            'cogs.activity_cog'
        ]
        for extension in initial_extensions:
            try:
                await self.load_extension(extension)
                logger.info(f"Successfully loaded extension: {extension}")
            except commands.ExtensionNotFound as e:
                logger.critical(f"FATAL: Extension '{extension}' not found. Ensure the file exists at the correct path. Error: {e}", exc_info=True)
            except commands.NoEntryPointError as e:
                 logger.critical(f"FATAL: Extension '{extension}' does not have a 'setup' function. Error: {e}", exc_info=True)
            except commands.ExtensionFailed as e:
                logger.critical(f"FATAL: Extension '{extension}' failed to load during its setup. Error: {e.original}", exc_info=True) # type: ignore
            except Exception as e:
                 logger.critical(f"FATAL: An unexpected error occurred loading extension {extension}: {e}", exc_info=True)
        logger.info("Setup hook completed.")

    async def on_ready(self):
        logger.info(f'Logged in as {self.user.name} (ID: {self.user.id})') # type: ignore
        logger.info('------ Bot is Ready ------')
        try:
            if self.guild_id_for_sync: # Ensure guild_id_for_sync is set if you use this block
                guild_obj = discord.Object(id=self.guild_id_for_sync) # type: ignore
                self.tree.copy_global_to(guild=guild_obj)
                synced = await self.tree.sync(guild=guild_obj)
            else:
                synced = await self.tree.sync()
            logger.info(f"Synced {len(synced)} slash commands.")
        except Exception as e:
            logger.error(f"Failed to sync slash commands: {e}")