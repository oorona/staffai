# bot.py
import discord
from discord.ext import commands
import logging
from typing import List, Optional, Dict, Any, Set

import redis 
from utils.webui_api import WebUIAPI 

# NEW IMPORTS FOR PHASE 3
import spacy
# langdetect will be used in MessageHandler, not directly in AIBot typically

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
                 general_redis_config: Dict[str, Any], 
                 ignored_role_ids: List[int],
                 rate_limit_count: int,
                 rate_limit_window_seconds: int,
                 token_rate_limit_count: int,
                 restricted_user_role_id: Optional[int],
                 restricted_channel_id: Optional[int],
                 rate_limit_message_user_template: str, 
                 restricted_channel_message_user_template: str, 
                 rate_limit_exempt_role_ids: List[int],
                 restriction_duration_seconds: int,
                 restriction_check_interval_seconds: int,
                 profile_max_scored_messages: int,
                 
                 # NEW PARAMETERS FOR PHASE 3
                 spacy_en_model_name: Optional[str],
                 spacy_es_model_name: Optional[str],
                 random_response_delivery_chance: float,

                 # NEW PARAMETERS for Worthiness
                 worthiness_min_length: int,
                 worthiness_min_significant_words: int,
                 
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
        self.guild_id_for_sync = None
        
        self.ignored_role_ids: List[int] = ignored_role_ids 
        self.ignored_role_ids_set: Set[int] = set(ignored_role_ids)

        self.rate_limit_count = rate_limit_count
        self.rate_limit_window_seconds = rate_limit_window_seconds
        self.token_rate_limit_count = token_rate_limit_count
        self.restricted_user_role_id = restricted_user_role_id
        self.restricted_channel_id = restricted_channel_id
        self.rate_limit_message_user_template = rate_limit_message_user_template
        self.restricted_channel_message_user_template = restricted_channel_message_user_template
        self.rate_limit_exempt_role_ids: List[int] = rate_limit_exempt_role_ids 
        self.rate_limit_exempt_role_ids_set: Set[int] = set(rate_limit_exempt_role_ids)

        self.restriction_duration_seconds = restriction_duration_seconds
        self.restriction_check_interval_seconds = restriction_check_interval_seconds
        
        self.profile_max_scored_messages = profile_max_scored_messages
        
        # NEW ATTRIBUTES FOR PHASE 3
        self.random_response_delivery_chance = random_response_delivery_chance
        self.spacy_models: Dict[str, spacy.Language] = {} # Stores loaded SpaCy models
        # NEW ATTRIBUTES for Worthiness
        self.worthiness_min_length = worthiness_min_length
        self.worthiness_min_significant_words = worthiness_min_significant_words

        # Load SpaCy models
        if spacy_en_model_name:
            try:
                self.spacy_models["en"] = spacy.load(spacy_en_model_name)
                logger.info(f"AIBot: Successfully loaded SpaCy English model '{spacy_en_model_name}'.")
            except OSError:
                logger.error(f"AIBot: Could not load SpaCy English model '{spacy_en_model_name}'. "
                             f"Ensure it's downloaded (e.g., python -m spacy download {spacy_en_model_name}). "
                             "English worthiness scoring will be affected.")
            except Exception as e: # Catch any other loading error
                logger.error(f"AIBot: Unexpected error loading SpaCy English model '{spacy_en_model_name}': {e}", exc_info=True)
        else:
            logger.info("AIBot: No English SpaCy model name provided. English NLP features disabled.")
        
        if spacy_es_model_name:
            try:
                self.spacy_models["es"] = spacy.load(spacy_es_model_name)
                logger.info(f"AIBot: Successfully loaded SpaCy Spanish model '{spacy_es_model_name}'.")
            except OSError:
                logger.error(f"AIBot: Could not load SpaCy Spanish model '{spacy_es_model_name}'. "
                             f"Ensure it's downloaded (e.g., python -m spacy download {spacy_es_model_name}). "
                             "Spanish worthiness scoring will be affected.")
            except Exception as e:
                logger.error(f"AIBot: Unexpected error loading SpaCy Spanish model '{spacy_es_model_name}': {e}", exc_info=True)
        else:
            logger.info("AIBot: No Spanish SpaCy model name provided. Spanish NLP features disabled.")

        if not self.spacy_models:
            logger.warning("AIBot: No SpaCy models were successfully loaded. Language-specific worthiness scoring will be disabled.")

        # Initialize WebUIAPI client
        self.api_client = WebUIAPI(
            base_url=self.api_url,
            model=self.model,
            api_key=self.api_key,
            welcome_system=self.welcome_system_prompt,
            welcome_prompt=self.welcome_user_prompt,
            max_history_per_user=self.max_history_per_context,
            knowledge_id=self.knowledge_id,
            list_tools=self.list_tools,
            redis_config=redis_config
        )
        
        # Initialize general Redis client
        self.redis_client_general: Optional[redis.Redis] = None
        if general_redis_config and general_redis_config.get('host'):
            try:
                # Ensure decode_responses=True if you expect strings from Redis
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

        # Logging other configurations
        if self.rate_limit_exempt_role_ids_set:
            logger.info(f"AIBot: Rate limits will be EXEMPT for Role IDs: {self.rate_limit_exempt_role_ids_set}")
        if self.ignored_role_ids_set:
            logger.info(f"AIBot: Messages from users with these Role IDs will be IGNORED: {self.ignored_role_ids_set}")
        if self.profile_max_scored_messages > 0:
            logger.info(f"AIBot: User profile storing up to {self.profile_max_scored_messages} scored messages per user.")
        else:
            logger.info("AIBot: User profile message storage is disabled (profile_max_scored_messages <= 0).")
        logger.info(f"AIBot: Random response delivery chance set to {self.random_response_delivery_chance*100:.1f}%.")
        logger.info(f"AIBot: Worthiness min message length set to {self.worthiness_min_length}.")
        logger.info(f"AIBot: Worthiness min significant words set to {self.worthiness_min_significant_words}.")
        logger.info("AIBot instance configured.")

    async def setup_hook(self):
        initial_extensions = [
            'cogs.listener_cog',
            'cogs.profile_cog'  # NEW COG FOR PROFILE COMMAND
        ] 
        for extension in initial_extensions:
            try:
                await self.load_extension(extension)
                logger.info(f"Successfully loaded extension: {extension}")
            except commands.ExtensionNotFound as e: # Corrected variable name
                logger.critical(f"FATAL: Extension '{extension}' not found. Ensure the file exists at the correct path (e.g., ./cogs/profile_cog.py). Error: {e}", exc_info=True)
            except commands.NoEntryPointError as e: # Corrected variable name
                 logger.critical(f"FATAL: Extension '{extension}' does not have a 'setup' function. Error: {e}", exc_info=True)
            except commands.ExtensionFailed as e: # Corrected variable name
                # For ExtensionFailed, the original exception is in e.original
                logger.critical(f"FATAL: Extension '{extension}' failed to load during its setup. Error: {e.original}", exc_info=True)
            except Exception as e: # General fallback
                 logger.critical(f"FATAL: An unexpected error occurred loading extension {extension}: {e}", exc_info=True)
        logger.info("Setup hook completed.")


    async def on_ready(self):
        logger.info(f'Logged in as {self.user.name} (ID: {self.user.id})') # type: ignore
        logger.info('------ Bot is Ready ------')
        # If you're using slash commands and have a command tree, you might sync it here.
        # Example:
        try:
            if self.guild_id_for_sync: # Assuming you have a specific guild for testing/syncing
                guild_obj = discord.Object(id=self.guild_id_for_sync)
                self.tree.copy_global_to(guild=guild_obj)
                synced = await self.tree.sync(guild=guild_obj)
            else:
                synced = await self.tree.sync()
            logger.info(f"Synced {len(synced)} slash commands.")
        except Exception as e:
            logger.error(f"Failed to sync slash commands: {e}")