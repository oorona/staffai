# main.py
import os
import sys
import logging
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
import discord
# import json # Not directly used here, but good if you need to handle JSON directly in main

# --- Dynamic Log Level Configuration ---
dotenv_path_check = load_dotenv()

LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO").upper()
log_level_map = {
    "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING,
    "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL,
}
numeric_log_level = log_level_map.get(LOG_LEVEL_STR, logging.INFO)

logging.basicConfig(
    level=numeric_log_level,
    format='%(asctime)s:%(levelname)s:%(name)s: %(message)s',
    handlers=[
        logging.FileHandler("bot.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__) # Define logger here

if not dotenv_path_check:
    logger.warning("Could not find .env file during initial check. Relying on environment variables or later load_dotenv call if any.")
logger.info(f"Logging level set to: {LOG_LEVEL_STR} ({numeric_log_level})")

def load_prompt_from_file(file_path: str) -> Optional[str]:
    try:
        # Construct absolute path relative to this file's directory
        abs_file_path = os.path.join(os.path.dirname(__file__), file_path)
        with open(abs_file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.error(f"FATAL: Prompt file not found: {file_path} (resolved to {abs_file_path})")
        return None
    except Exception as e:
        logger.error(f"FATAL: Error reading prompt file {file_path} (resolved to {abs_file_path}): {e}", exc_info=True)
        return None

logger.info("Loading environment variables from .env file (secondary check)...")
if not load_dotenv(): 
    logger.warning("Could not find .env file after basicConfig or it was already loaded. Ensure it exists or env vars are set.")

logger.info("Retrieving and validating configuration...")

prompt_dir_relative_to_main = os.path.join('utils', 'prompts')
welcome_system_path = os.path.join(prompt_dir_relative_to_main, 'welcome_system.txt')
welcome_prompt_path = os.path.join(prompt_dir_relative_to_main, 'welcome_prompt.txt')
personality_prompt_path = os.path.join(prompt_dir_relative_to_main, 'personality_prompt.txt')


WELCOME_SYSTEM = load_prompt_from_file(welcome_system_path)
WELCOME_PROMPT = load_prompt_from_file(welcome_prompt_path)
PERSONALITY_PROMPT = load_prompt_from_file(personality_prompt_path)

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

try:
    WELCOME_CHANNEL_ID = int(os.getenv("WELCOME_CHANNEL_ID")) if os.getenv("WELCOME_CHANNEL_ID") else None
    RESPONSE_CHANCE = float(os.getenv("RESPONSE_CHANCE", "0.05"))
    MAX_HISTORY_PER_USER = int(os.getenv("MAX_HISTORY_PER_USER", "20")) 

    RATE_LIMIT_COUNT = int(os.getenv("RATE_LIMIT_COUNT", "15"))
    RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
    TOKEN_RATE_LIMIT_COUNT = int(os.getenv("TOKEN_RATE_LIMIT_COUNT", "20000"))
    RESTRICTED_USER_ROLE_ID = int(os.getenv("RESTRICTED_USER_ROLE_ID")) if os.getenv("RESTRICTED_USER_ROLE_ID") else None
    RESTRICTED_CHANNEL_ID = int(os.getenv("RESTRICTED_CHANNEL_ID")) if os.getenv("RESTRICTED_CHANNEL_ID") else None

    RESTRICTION_DURATION_SECONDS = int(os.getenv("RESTRICTION_DURATION_SECONDS", "86400"))
    RESTRICTION_CHECK_INTERVAL_SECONDS = int(os.getenv("RESTRICTION_CHECK_INTERVAL_SECONDS", "300"))
    
    PROFILE_MAX_SCORED_MESSAGES = int(os.getenv("PROFILE_MAX_SCORED_MESSAGES", "50")) 
    
    SPACY_EN_MODEL_NAME = os.getenv("SPACY_EN_MODEL_NAME", "en_core_web_sm")
    SPACY_ES_MODEL_NAME = os.getenv("SPACY_ES_MODEL_NAME", "es_core_news_sm")
    RANDOM_RESPONSE_DELIVERY_CHANCE = float(os.getenv("RANDOM_RESPONSE_DELIVERY_CHANCE", "0.3"))
     # NEW CONFIGURATIONS for Worthiness Checks (Phase 3)
    WORTHINESS_MIN_LENGTH = int(os.getenv("WORTHINESS_MIN_LENGTH", "10"))
    WORTHINESS_MIN_SIGNIFICANT_WORDS = int(os.getenv("WORTHINESS_MIN_SIGNIFICANT_WORDS", "2"))


except ValueError as e:
    logger.critical(f"Invalid integer/float format in critical environment variables: {e}", exc_info=True)
    sys.exit("Exiting due to critical configuration error in numeric environment variables.")

RATE_LIMIT_MESSAGE_USER_TEMPLATE = os.getenv("RATE_LIMIT_MESSAGE_USER", "You've sent messages too frequently. Please use <#{channel_id}> for bot interactions.")
RESTRICTED_CHANNEL_MESSAGE_USER_TEMPLATE = os.getenv("RESTRICTED_CHANNEL_MESSAGE_USER", "As a restricted user, please use <#{channel_id}> for bot interactions.")

RATE_LIMIT_EXEMPT_ROLE_IDS_STR = os.getenv("RATE_LIMIT_EXEMPT_ROLE_IDS", "")
rate_limit_exempt_role_ids: List[int] = []
if RATE_LIMIT_EXEMPT_ROLE_IDS_STR:
    try:
        rate_limit_exempt_role_ids = [int(role_id.strip()) for role_id in RATE_LIMIT_EXEMPT_ROLE_IDS_STR.split(',') if role_id.strip()]
    except ValueError:
        logger.error(f"Invalid format for RATE_LIMIT_EXEMPT_ROLE_IDS: '{RATE_LIMIT_EXEMPT_ROLE_IDS_STR}'. Expected comma-separated numbers. No roles will be exempt due to this error.")

OPENWEBUI_API_URL = os.getenv("OPENWEBUI_API_URL", "http://localhost:8080") 
OPENWEBUI_MODEL = os.getenv("OPENWEBUI_MODEL")
OPENWEBUI_API_KEY = os.getenv("OPENWEBUI_API_KEY")
LIST_TOOLS_STR = os.getenv("LIST_TOOLS")
list_tools_parsed: List[str] = [tool.strip() for tool in LIST_TOOLS_STR.split(',')] if LIST_TOOLS_STR else []
KNOWLEDGE_ID = os.getenv("KNOWLEDGE_ID") 

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379")) 
REDIS_DB = int(os.getenv("REDIS_DB", "0")) 
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") 

redis_connection_config: Dict[str, Any] = {"host": REDIS_HOST, "port": REDIS_PORT, "db": REDIS_DB}
if REDIS_PASSWORD:
    redis_connection_config["password"] = REDIS_PASSWORD

IGNORED_ROLE_IDS_STR = os.getenv("IGNORED_ROLE_IDS", "")
ignored_role_ids: List[int] = []
if IGNORED_ROLE_IDS_STR:
    try:
        ignored_role_ids = [int(role_id.strip()) for role_id in IGNORED_ROLE_IDS_STR.split(',') if role_id.strip()]
    except ValueError:
        logger.error(f"Invalid format for IGNORED_ROLE_IDS: '{IGNORED_ROLE_IDS_STR}'. No roles will be ignored due to this error.")

config_errors = []
if not DISCORD_BOT_TOKEN: config_errors.append("DISCORD_BOT_TOKEN is missing.") #
if not WELCOME_SYSTEM: config_errors.append(f"Failed to load WELCOME_SYSTEM from: {welcome_system_path}")
if not WELCOME_PROMPT: config_errors.append(f"Failed to load WELCOME_PROMPT from: {welcome_prompt_path}")
if not PERSONALITY_PROMPT: config_errors.append(f"Failed to load PERSONALITY_PROMPT from: {personality_prompt_path}")

if not WELCOME_CHANNEL_ID: logger.warning("WELCOME_CHANNEL_ID not set. Welcome messages will be disabled.")
if not OPENWEBUI_MODEL: config_errors.append("OPENWEBUI_MODEL is missing.") #

if not RESTRICTED_USER_ROLE_ID:
    logger.warning("RESTRICTED_USER_ROLE_ID not set. Rate limiting and restriction system will be largely disabled.")
if not RESTRICTED_CHANNEL_ID and RESTRICTED_USER_ROLE_ID :
    logger.warning("RESTRICTED_CHANNEL_ID not set, but RESTRICTED_USER_ROLE_ID is. Restricted channel enforcement will be disabled.")

if RESTRICTION_DURATION_SECONDS > 0 and RESTRICTION_CHECK_INTERVAL_SECONDS <= 0:
    config_errors.append("RESTRICTION_CHECK_INTERVAL_SECONDS must be > 0 if RESTRICTION_DURATION_SECONDS is enabled.")
if RESTRICTION_DURATION_SECONDS > 0 and not RESTRICTED_USER_ROLE_ID:
    logger.warning("RESTRICTION_DURATION_SECONDS is set, but RESTRICTED_USER_ROLE_ID is not. Automatic restriction expiry is effectively disabled.")
if PROFILE_MAX_SCORED_MESSAGES <= 0:
    logger.warning("PROFILE_MAX_SCORED_MESSAGES is <= 0. User message scoring and profiling will be disabled.")
if not SPACY_EN_MODEL_NAME and not SPACY_ES_MODEL_NAME:
     logger.warning("Both SPACY_EN_MODEL_NAME and SPACY_ES_MODEL_NAME are unset. SpaCy-based worthiness scoring will be limited/disabled.")


if config_errors:
    logger.critical("FATAL: Critical configuration errors found:")
    for error in config_errors:
        logger.critical(f" -> {error}")
    sys.exit("Exiting due to critical configuration errors.")
else:
    logger.info("Core configuration loaded and validated successfully.")

logger.info(f"OpenWebUI API URL: {OPENWEBUI_API_URL}, Model: {OPENWEBUI_MODEL}") #
logger.info(f"Redis Connection Config: Host={REDIS_HOST}, Port={REDIS_PORT}, DB={REDIS_DB}")
if ignored_role_ids: logger.info(f"Ignoring Role IDs: {ignored_role_ids}")
if rate_limit_exempt_role_ids: logger.info(f"Rate Limit Exempt Role IDs: {rate_limit_exempt_role_ids}")
logger.info(f"Message Rate Limit: {RATE_LIMIT_COUNT}/{RATE_LIMIT_WINDOW_SECONDS}s, Token Rate Limit: {TOKEN_RATE_LIMIT_COUNT}/{RATE_LIMIT_WINDOW_SECONDS}s")
if RESTRICTED_USER_ROLE_ID: logger.info(f"Restricted User Role ID: {RESTRICTED_USER_ROLE_ID}")
if RESTRICTED_CHANNEL_ID: logger.info(f"Restricted Channel ID: {RESTRICTED_CHANNEL_ID}")
if RESTRICTION_DURATION_SECONDS > 0 and RESTRICTED_USER_ROLE_ID:
    logger.info(f"Automatic Restriction Expiry: Enabled. Duration: {RESTRICTION_DURATION_SECONDS}s, Check Interval: {RESTRICTION_CHECK_INTERVAL_SECONDS}s")
elif RESTRICTED_USER_ROLE_ID: 
    logger.info("Automatic Restriction Expiry: Disabled.")
if PROFILE_MAX_SCORED_MESSAGES > 0:
    logger.info(f"User Profile: Storing last {PROFILE_MAX_SCORED_MESSAGES} scored messages per user.")
logger.info(f"SpaCy English Model: {SPACY_EN_MODEL_NAME if SPACY_EN_MODEL_NAME else 'Disabled/Not Set'}")
logger.info(f"SpaCy Spanish Model: {SPACY_ES_MODEL_NAME if SPACY_ES_MODEL_NAME else 'Disabled/Not Set'}")
logger.info(f"Random Response Delivery Chance: {RANDOM_RESPONSE_DELIVERY_CHANCE*100:.1f}%")


# --- Bot Initialization ---
try:
    from bot import AIBot
    logger.info("Successfully imported AIBot from bot.py")
except ImportError:
    logger.critical("FATAL: Could not import AIBot from bot.py. Check PYTHONPATH or file location.", exc_info=True)
    sys.exit(1)

intents = discord.Intents.default()
intents.guilds = True
intents.messages = True 
intents.members = True  
intents.message_content = True 

logger.info("Initializing the bot instance...")
try:
    the_bot = AIBot(
        welcome_channel_id=WELCOME_CHANNEL_ID,
        welcome_system_prompt=WELCOME_SYSTEM,
        welcome_user_prompt=WELCOME_PROMPT,
        chat_system_prompt=PERSONALITY_PROMPT,
        response_chance=RESPONSE_CHANCE,
        max_history_per_context=MAX_HISTORY_PER_USER,
        api_url=OPENWEBUI_API_URL,
        model=OPENWEBUI_MODEL,
        api_key=OPENWEBUI_API_KEY,
        list_tools=list_tools_parsed,
        knowledge_id=KNOWLEDGE_ID,
        
        redis_config=redis_connection_config,
        general_redis_config=redis_connection_config,
        
        ignored_role_ids=ignored_role_ids,
        rate_limit_count=RATE_LIMIT_COUNT,
        rate_limit_window_seconds=RATE_LIMIT_WINDOW_SECONDS,
        token_rate_limit_count=TOKEN_RATE_LIMIT_COUNT,
        restricted_user_role_id=RESTRICTED_USER_ROLE_ID,
        restricted_channel_id=RESTRICTED_CHANNEL_ID,
        
        rate_limit_message_user_template=RATE_LIMIT_MESSAGE_USER_TEMPLATE,
        restricted_channel_message_user_template=RESTRICTED_CHANNEL_MESSAGE_USER_TEMPLATE,
        
        rate_limit_exempt_role_ids=rate_limit_exempt_role_ids,
        restriction_duration_seconds=RESTRICTION_DURATION_SECONDS,
        restriction_check_interval_seconds=RESTRICTION_CHECK_INTERVAL_SECONDS,
        
        profile_max_scored_messages=PROFILE_MAX_SCORED_MESSAGES,
        spacy_en_model_name=SPACY_EN_MODEL_NAME,
        spacy_es_model_name=SPACY_ES_MODEL_NAME,
        random_response_delivery_chance=RANDOM_RESPONSE_DELIVERY_CHANCE,
        worthiness_min_length=WORTHINESS_MIN_LENGTH,
        worthiness_min_significant_words=WORTHINESS_MIN_SIGNIFICANT_WORDS,
        intents=intents
    )
except Exception as e: # Catch any exception during AIBot initialization
    logger.critical(f"FATAL: Failed to initialize the AIBot: {e}", exc_info=True)
    sys.exit(1) # Exit if bot initialization fails

# --- Bot Execution ---
def main_run():
    logger.info("Attempting to run the bot...")
    if not DISCORD_BOT_TOKEN:
        logger.critical("FATAL: DISCORD_BOT_TOKEN is not set. Cannot start the bot.")
        sys.exit(1)
        
    try:
        the_bot.run(DISCORD_BOT_TOKEN)
    except discord.LoginFailure:
        logger.critical("FATAL: Login failed. Check DISCORD_BOT_TOKEN's validity.")
        sys.exit(1)
    except discord.PrivilegedIntentsRequired as e: #
        logger.critical(f"FATAL: Required Privileged Intents (e.g., Server Members or Message Content) are not enabled. Details: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"FATAL: An unexpected error occurred while running the bot: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Bot process has concluded or been interrupted.")

if __name__ == "__main__":
    main_run()