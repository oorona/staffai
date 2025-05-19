# main.py

import os
import sys
import logging
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
import discord

# --- Dynamic Log Level Configuration ---
# It's good practice to load .env as early as possible, especially for logging.
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
logger = logging.getLogger(__name__)

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
if not load_dotenv(): # Calling load_dotenv() again is fine, it won't override already set env vars unless override=True
    logger.warning("Could not find .env file after basicConfig or it was already loaded. Ensure it exists or env vars are set.")

logger.info("Retrieving and validating configuration...")

# Define prompt paths relative to the project structure expected by load_prompt_from_file
# Assuming 'utils/prompts/' is at the same level as main.py or main.py is in root and utils is a subdir.
# If main.py is in root, path should be 'utils/prompts/file.txt'
# If main.py is elsewhere, adjust base path or how load_prompt_from_file constructs paths.
# For now, assuming load_prompt_from_file handles paths correctly from where it's called or main.py location.
# The original `load_prompt_from_file` in main.py directly used `os.path.join(os.path.dirname(__file__), 'utils', 'prompts')`
# Let's stick to that pattern by passing relative paths to it.

prompt_dir_relative_to_main = os.path.join('utils', 'prompts')
welcome_system_path = os.path.join(prompt_dir_relative_to_main, 'welcome_system.txt')
welcome_prompt_path = os.path.join(prompt_dir_relative_to_main, 'welcome_prompt.txt')
personality_prompt_path = os.path.join(prompt_dir_relative_to_main, 'personality_prompt.txt')


WELCOME_SYSTEM = load_prompt_from_file(welcome_system_path)
WELCOME_PROMPT = load_prompt_from_file(welcome_prompt_path)
PERSONALITY_PROMPT = load_prompt_from_file(personality_prompt_path)

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# Configuration variables conversion and validation
try:
    WELCOME_CHANNEL_ID = int(os.getenv("WELCOME_CHANNEL_ID")) if os.getenv("WELCOME_CHANNEL_ID") else None
    RESPONSE_CHANCE = float(os.getenv("RESPONSE_CHANCE", "0.05"))
    MAX_HISTORY_PER_USER = int(os.getenv("MAX_HISTORY_PER_USER", "20")) # From specs.txt

    RATE_LIMIT_COUNT = int(os.getenv("RATE_LIMIT_COUNT", "15"))
    RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
    TOKEN_RATE_LIMIT_COUNT = int(os.getenv("TOKEN_RATE_LIMIT_COUNT", "20000"))
    RESTRICTED_USER_ROLE_ID = int(os.getenv("RESTRICTED_USER_ROLE_ID")) if os.getenv("RESTRICTED_USER_ROLE_ID") else None
    RESTRICTED_CHANNEL_ID = int(os.getenv("RESTRICTED_CHANNEL_ID")) if os.getenv("RESTRICTED_CHANNEL_ID") else None

    RESTRICTION_DURATION_SECONDS = int(os.getenv("RESTRICTION_DURATION_SECONDS", "86400"))
    RESTRICTION_CHECK_INTERVAL_SECONDS = int(os.getenv("RESTRICTION_CHECK_INTERVAL_SECONDS", "300"))

except ValueError as e:
    logger.critical(f"Invalid integer/float format in critical environment variables: {e}", exc_info=True)
    sys.exit("Exiting due to critical configuration error in numeric environment variables.")

# String configurations (templates and optional values)
RATE_LIMIT_MESSAGE_USER_TEMPLATE = os.getenv("RATE_LIMIT_MESSAGE_USER", "You've sent messages too frequently. Please use <#{channel_id}> for bot interactions.")
RESTRICTED_CHANNEL_MESSAGE_USER_TEMPLATE = os.getenv("RESTRICTED_CHANNEL_MESSAGE_USER", "As a restricted user, please use <#{channel_id}> for bot interactions.")

RATE_LIMIT_EXEMPT_ROLE_IDS_STR = os.getenv("RATE_LIMIT_EXEMPT_ROLE_IDS", "")
rate_limit_exempt_role_ids: List[int] = []
if RATE_LIMIT_EXEMPT_ROLE_IDS_STR:
    try:
        rate_limit_exempt_role_ids = [int(role_id.strip()) for role_id in RATE_LIMIT_EXEMPT_ROLE_IDS_STR.split(',') if role_id.strip()]
    except ValueError:
        logger.error(f"Invalid format for RATE_LIMIT_EXEMPT_ROLE_IDS: '{RATE_LIMIT_EXEMPT_ROLE_IDS_STR}'. Expected comma-separated numbers. No roles will be exempt due to this error.")
        # rate_limit_exempt_role_ids remains empty

OPENWEBUI_API_URL = os.getenv("OPENWEBUI_API_URL", "http://localhost:8080") # Default from original main.py
OPENWEBUI_MODEL = os.getenv("OPENWEBUI_MODEL")
OPENWEBUI_API_KEY = os.getenv("OPENWEBUI_API_KEY") # Optional
LIST_TOOLS_STR = os.getenv("LIST_TOOLS")
list_tools_parsed: List[str] = [tool.strip() for tool in LIST_TOOLS_STR.split(',')] if LIST_TOOLS_STR else []
KNOWLEDGE_ID = os.getenv("KNOWLEDGE_ID") # Optional

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379")) # Ensure conversion if not done above
REDIS_DB = int(os.getenv("REDIS_DB", "0")) # Ensure conversion if not done above
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") # Optional

# This single Redis connection configuration will be used for both
# WebUIAPI's history and the bot's general Redis client (for rate limits, expiry).
# This aligns with the original spec of REDIS_DB being for all operational data.
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
        # ignored_role_ids remains empty

# --- Configuration Validation ---
config_errors = []
if not DISCORD_BOT_TOKEN: config_errors.append("DISCORD_BOT_TOKEN is missing.")
if not WELCOME_SYSTEM: config_errors.append(f"Failed to load WELCOME_SYSTEM from: {welcome_system_path}")
if not WELCOME_PROMPT: config_errors.append(f"Failed to load WELCOME_PROMPT from: {welcome_prompt_path}")
if not PERSONALITY_PROMPT: config_errors.append(f"Failed to load PERSONALITY_PROMPT from: {personality_prompt_path}")

if not WELCOME_CHANNEL_ID: logger.warning("WELCOME_CHANNEL_ID not set. Welcome messages will be disabled.")
if not OPENWEBUI_MODEL: config_errors.append("OPENWEBUI_MODEL is missing.")
# Check for OPENWEBUI_API_URL is implicitly handled by its default, but you could add an explicit check if empty is invalid.

if not RESTRICTED_USER_ROLE_ID:
    logger.warning("RESTRICTED_USER_ROLE_ID not set. Rate limiting will not assign roles or enforce restrictions. Automatic restriction expiry will also be disabled.")
if not RESTRICTED_CHANNEL_ID and RESTRICTED_USER_ROLE_ID :
    logger.warning("RESTRICTED_CHANNEL_ID not set, but RESTRICTED_USER_ROLE_ID is. Restricted channel enforcement will be disabled.")

if RESTRICTION_DURATION_SECONDS > 0 and RESTRICTION_CHECK_INTERVAL_SECONDS <= 0:
    config_errors.append("RESTRICTION_CHECK_INTERVAL_SECONDS must be greater than 0 if RESTRICTION_DURATION_SECONDS is enabled.")
if RESTRICTION_DURATION_SECONDS > 0 and not RESTRICTED_USER_ROLE_ID:
    logger.warning("RESTRICTION_DURATION_SECONDS is set, but RESTRICTED_USER_ROLE_ID is not. Automatic restriction expiry is effectively disabled.")

if config_errors:
    logger.critical("FATAL: Critical configuration errors found:")
    for error in config_errors:
        logger.critical(f" -> {error}")
    sys.exit("Exiting due to critical configuration errors.")
else:
    logger.info("Core configuration loaded and validated successfully.")

# Log some key configurations
logger.info(f"OpenWebUI API URL: {OPENWEBUI_API_URL}, Model: {OPENWEBUI_MODEL}")
logger.info(f"Redis Connection Config (for history & general): Host={REDIS_HOST}, Port={REDIS_PORT}, DB={REDIS_DB}")
if ignored_role_ids: logger.info(f"Ignoring Role IDs: {ignored_role_ids}")
if rate_limit_exempt_role_ids: logger.info(f"Rate Limit Exempt Role IDs: {rate_limit_exempt_role_ids}")
logger.info(f"Message Rate Limit: {RATE_LIMIT_COUNT}/{RATE_LIMIT_WINDOW_SECONDS}s, Token Rate Limit: {TOKEN_RATE_LIMIT_COUNT}/{RATE_LIMIT_WINDOW_SECONDS}s")
if RESTRICTED_USER_ROLE_ID: logger.info(f"Restricted User Role ID: {RESTRICTED_USER_ROLE_ID}")
if RESTRICTED_CHANNEL_ID: logger.info(f"Restricted Channel ID: {RESTRICTED_CHANNEL_ID}")
if RESTRICTION_DURATION_SECONDS > 0 and RESTRICTED_USER_ROLE_ID:
    logger.info(f"Automatic Restriction Expiry: Enabled. Duration: {RESTRICTION_DURATION_SECONDS}s, Check Interval: {RESTRICTION_CHECK_INTERVAL_SECONDS}s")
elif RESTRICTED_USER_ROLE_ID: # Only log if restriction system is somewhat active
    logger.info("Automatic Restriction Expiry: Disabled (RESTRICTION_DURATION_SECONDS is 0 or not set).")


# --- Bot Initialization ---
try:
    # Assuming bot.py is in the same directory or Python's path can find it.
    # If main.py is in root, and bot.py is in root: from bot import AIBot
    # If main.py is in root, and bot.py is in a subdir 'src': from src.bot import AIBot
    # For now, assuming 'from bot import AIBot' works based on typical project structure.
    from bot import AIBot
    logger.info("Successfully imported AIBot from bot.py")
except ImportError:
    logger.critical("FATAL: Could not import AIBot from bot.py. Check PYTHONPATH or file location.", exc_info=True)
    sys.exit(1)

intents = discord.Intents.default()
intents.guilds = True
intents.messages = True # Includes DMs and Guild messages if not further restricted
intents.members = True  # Privileged: Required for on_member_join, role changes, get_member
intents.message_content = True # Privileged: Required to read message content

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
        
        # Updated Redis config parameters for AIBot constructor
        redis_config=redis_connection_config,  # For WebUIAPI history
        general_redis_config=redis_connection_config, # For bot's general use (rate limits, expiry)
        
        ignored_role_ids=ignored_role_ids,
        rate_limit_count=RATE_LIMIT_COUNT,
        rate_limit_window_seconds=RATE_LIMIT_WINDOW_SECONDS,
        token_rate_limit_count=TOKEN_RATE_LIMIT_COUNT,
        restricted_user_role_id=RESTRICTED_USER_ROLE_ID,
        restricted_channel_id=RESTRICTED_CHANNEL_ID,
        
        # Using updated template names
        rate_limit_message_user_template=RATE_LIMIT_MESSAGE_USER_TEMPLATE,
        restricted_channel_message_user_template=RESTRICTED_CHANNEL_MESSAGE_USER_TEMPLATE,
        
        rate_limit_exempt_role_ids=rate_limit_exempt_role_ids,
        restriction_duration_seconds=RESTRICTION_DURATION_SECONDS,
        restriction_check_interval_seconds=RESTRICTION_CHECK_INTERVAL_SECONDS,
        intents=intents
    )
except Exception as e:
    logger.critical(f"FATAL: Failed to initialize the AIBot: {e}", exc_info=True)
    sys.exit(1)

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
    except discord.PrivilegedIntentsRequired as e:
        logger.critical(f"FATAL: Required Privileged Intents (e.g., Server Members or Message Content) are not enabled in Discord Developer Portal or are missing in code. Details: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"FATAL: An unexpected error occurred while running the bot: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # This will run if the bot stops, whether normally or due to an error caught above or a keyboard interrupt.
        logger.info("Bot process has concluded or been interrupted.")

if __name__ == "__main__":
    main_run()