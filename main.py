# main.py
import os
import sys
import logging
import warnings
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List, Set
import discord

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
logger = logging.getLogger(__name__)

# Suppress noisy HTTP connection warnings and tracebacks from MCP servers
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*SSE.*")
warnings.filterwarnings("ignore", message=".*httpx.*")
warnings.filterwarnings(
    "ignore",
    message=r".*parameter 'timeout' of type 'float' is deprecated.*ClientWSTimeout.*",
    category=DeprecationWarning
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.CRITICAL)  # Completely suppress MCP library errors
logging.getLogger("mcp.client").setLevel(logging.CRITICAL)  # Suppress MCP client errors
logging.getLogger("mcp.client.streamable_http").setLevel(logging.CRITICAL)  # Suppress specific SSE errors

# Add a custom filter to block specific MCP error messages
class MCPErrorFilter(logging.Filter):
    def filter(self, record):
        # Block SSE stream errors and connection errors from MCP
        if any(x in record.getMessage().lower() for x in [
            'error reading sse stream',
            'peer closed connection',
            'remoteprotocolerror',
            'incomplete chunked read'
        ]):
            return False
        return True

# Apply the filter to all loggers that might show MCP errors
for logger_name in ['mcp', 'mcp.client', 'mcp.client.streamable_http', 'httpx', 'httpcore']:
    logging.getLogger(logger_name).addFilter(MCPErrorFilter())

if not dotenv_path_check:
    logger.warning("Could not find .env file during initial check. Relying on environment variables or later load_dotenv call if any.")
logger.info(f"Logging level set to: {LOG_LEVEL_STR} ({numeric_log_level})")

def load_prompt_from_file(file_path: str) -> Optional[str]:
    try:
        abs_file_path = os.path.join(os.path.dirname(__file__), file_path)
        with open(abs_file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.error(f"FATAL: Prompt file not found: {file_path} (resolved to {abs_file_path})")
        return None
    except Exception as e:
        logger.error(f"FATAL: Error reading prompt file {file_path} (resolved to {abs_file_path}): {e}", exc_info=True)
        return None


def compose_system_prompt(core_prompt: str, persona_prompt: str) -> str:
    """Build final chat system prompt by injecting persona into core template."""
    if "{PERSONA_INSTRUCTIONS}" in core_prompt:
        return core_prompt.replace("{PERSONA_INSTRUCTIONS}", persona_prompt).strip()
    return f"{core_prompt}\n\n{persona_prompt}".strip()

logger.info("Loading environment variables from .env file (secondary check)...")
if not load_dotenv():
    logger.warning("Could not find .env file after basicConfig or it was already loaded. Ensure it exists or env vars are set.")

logger.info("Retrieving and validating configuration...")


def _read_docker_secret(secret_name: str) -> Optional[str]:
    """Read a Docker secret mounted at /run/secrets/<secret_name> if present.
    Returns the secret string (stripped) or None if not available.
    """
    secret_path = f"/run/secrets/{secret_name}"
    try:
        if os.path.exists(secret_path):
            with open(secret_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
    except Exception:
        logger.debug(f"Could not read docker secret {secret_name} at {secret_path}")
    return None

prompt_dir_relative_to_main = os.path.join('utils', 'prompts')
prompts_root_path = os.path.join(os.path.dirname(__file__), prompt_dir_relative_to_main)
personality_core_prompt_path = os.path.join(prompt_dir_relative_to_main, 'personality_core_prompt.txt')
personality_prompt_path = os.path.join(prompt_dir_relative_to_main, 'personality_prompt.txt')
base_activity_system_prompt_path_env = os.getenv("BASE_ACTIVITY_SYSTEM_PROMPT_PATH", os.path.join(prompt_dir_relative_to_main, 'base_activity_system_prompt.txt'))

PERSONALITY_CORE_PROMPT = load_prompt_from_file(personality_core_prompt_path)
PERSONALITY_PROMPT = load_prompt_from_file(personality_prompt_path)
BASE_ACTIVITY_SYSTEM_PROMPT = load_prompt_from_file(base_activity_system_prompt_path_env)
DEFAULT_CHAT_SYSTEM_PROMPT = (
    compose_system_prompt(PERSONALITY_CORE_PROMPT, PERSONALITY_PROMPT)
    if PERSONALITY_CORE_PROMPT and PERSONALITY_PROMPT
    else None
)


# Prefer Docker secrets when available (mounted to /run/secrets/<name> by docker-compose).
DISCORD_BOT_TOKEN = _read_docker_secret('discord_bot_token') or os.getenv("DISCORD_BOT_TOKEN")

try:
    RESPONSE_CHANCE = float(os.getenv("RESPONSE_CHANCE", "0.05"))
    MAX_HISTORY_PER_USER = int(os.getenv("MAX_HISTORY_PER_USER", "20"))

    RATE_LIMIT_COUNT = int(os.getenv("RATE_LIMIT_COUNT", "15"))
    RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
    TOKEN_RATE_LIMIT_COUNT = int(os.getenv("TOKEN_RATE_LIMIT_COUNT", "20000"))
    RESTRICTED_USER_ROLE_ID = int(os.getenv("RESTRICTED_USER_ROLE_ID")) if os.getenv("RESTRICTED_USER_ROLE_ID") else None
    RESTRICTED_CHANNEL_ID = int(os.getenv("RESTRICTED_CHANNEL_ID")) if os.getenv("RESTRICTED_CHANNEL_ID") else None

    RESTRICTION_DURATION_SECONDS = int(os.getenv("RESTRICTION_DURATION_SECONDS", "86400"))
    RESTRICTION_CHECK_INTERVAL_SECONDS = int(os.getenv("RESTRICTION_CHECK_INTERVAL_SECONDS", "300"))

    RANDOM_RESPONSE_DELIVERY_CHANCE = float(os.getenv("RANDOM_RESPONSE_DELIVERY_CHANCE", "0.3"))
    ACTIVITY_UPDATE_INTERVAL_SECONDS = int(os.getenv("ACTIVITY_UPDATE_INTERVAL_SECONDS", "300"))
    # How long to keep per-user/channel conversation history (in seconds) before it decays
    CONTEXT_HISTORY_TTL_SECONDS = int(os.getenv("CONTEXT_HISTORY_TTL_SECONDS", "1800"))
    # How old (in seconds) individual messages can be before being purged from context (default: 30 minutes)
    CONTEXT_MESSAGE_MAX_AGE_SECONDS = int(os.getenv("CONTEXT_MESSAGE_MAX_AGE_SECONDS", "1800"))
    # Number of recent messages to fetch per user for context (default: 5)
    DEFAULT_CONTEXT_MESSAGES = int(os.getenv("DEFAULT_CONTEXT_MESSAGES", "5"))

    ACTIVITY_SCHEDULE_ENABLED = os.getenv("ACTIVITY_SCHEDULE_ENABLED", "False").lower() in ('true', '1', 't')
    ACTIVITY_ACTIVE_START_HOUR_UTC = int(os.getenv("ACTIVITY_ACTIVE_START_HOUR_UTC", "0"))
    ACTIVITY_ACTIVE_END_HOUR_UTC = int(os.getenv("ACTIVITY_ACTIVE_END_HOUR_UTC", "23"))
    ACTIVITY_ACTIVE_DAYS_STR = os.getenv("ACTIVITY_ACTIVE_DAYS_UTC", "0,1,2,3,4,5,6")

    # Daily topic workflow (approval + publish)
    DAILY_TOPIC_ENABLED = os.getenv("DAILY_TOPIC_ENABLED", "False").lower() in ('true', '1', 't')
    DAILY_TOPIC_APPROVAL_CHANNEL_ID = int(os.getenv("DAILY_TOPIC_APPROVAL_CHANNEL_ID")) if os.getenv("DAILY_TOPIC_APPROVAL_CHANNEL_ID") else None
    DAILY_TOPIC_PUBLISH_CHANNEL_ID = int(os.getenv("DAILY_TOPIC_PUBLISH_CHANNEL_ID")) if os.getenv("DAILY_TOPIC_PUBLISH_CHANNEL_ID") else None
    DAILY_TOPIC_INTERVAL_SECONDS = int(os.getenv("DAILY_TOPIC_INTERVAL_SECONDS", "0"))
    DAILY_TOPIC_APPROVAL_HOUR_UTC = int(os.getenv("DAILY_TOPIC_APPROVAL_HOUR_UTC", "8"))
    DAILY_TOPIC_APPROVAL_TIMEOUT_SECONDS = int(os.getenv("DAILY_TOPIC_APPROVAL_TIMEOUT_SECONDS", "14400"))
    DAILY_TOPIC_CHECK_INTERVAL_SECONDS = int(os.getenv("DAILY_TOPIC_CHECK_INTERVAL_SECONDS", "60"))
    DAILY_TOPIC_THREAD_AUTO_ARCHIVE_MINUTES = int(os.getenv("DAILY_TOPIC_THREAD_AUTO_ARCHIVE_MINUTES", "1440"))
    DAILY_TOPIC_THREAD_CONTEXT_MESSAGES = int(os.getenv("DAILY_TOPIC_THREAD_CONTEXT_MESSAGES", "40"))

    # User memory + LLM call auditing
    USER_MEMORY_ENABLED = os.getenv("USER_MEMORY_ENABLED", "True").lower() in ('true', '1', 't')
    USER_MEMORY_UPDATE_CHANCE = float(os.getenv("USER_MEMORY_UPDATE_CHANCE", "0.25"))
    USER_MEMORY_MIN_MESSAGE_CHARS = int(os.getenv("USER_MEMORY_MIN_MESSAGE_CHARS", "50"))
    USER_MEMORY_MIN_MESSAGE_WORDS = int(os.getenv("USER_MEMORY_MIN_MESSAGE_WORDS", "10"))
    USER_MEMORY_MAX_CHARS = int(os.getenv("USER_MEMORY_MAX_CHARS", "420"))
    USER_MEMORY_PIPELINE_MODE = os.getenv("USER_MEMORY_PIPELINE_MODE", "tiny_gate_frontier_core").strip().lower()
    USER_MEMORY_OLLAMA_BASE_URL = os.getenv("USER_MEMORY_OLLAMA_BASE_URL", "http://localhost:11434").strip()
    USER_MEMORY_OLLAMA_API_KEY = _read_docker_secret('user_memory_ollama_api_key') or os.getenv("USER_MEMORY_OLLAMA_API_KEY", "ollama")
    USER_MEMORY_OLLAMA_TIMEOUT_SECONDS = float(os.getenv("USER_MEMORY_OLLAMA_TIMEOUT_S", "30"))
    USER_MEMORY_TINY_MODEL = os.getenv("USER_MEMORY_TINY_MODEL", "SmolLM2-1.7B-Instruct").strip()
    USER_MEMORY_TINY_MODEL_EXTRACT = os.getenv("USER_MEMORY_TINY_MODEL_EXTRACT", "").strip() or None
    USER_MEMORY_TINY_MODEL_CLASSIFIER = os.getenv("USER_MEMORY_TINY_MODEL_CLASSIFIER", "").strip() or None
    USER_MEMORY_TINY_ACCUMULATE_MAX_TOKENS = int(os.getenv("USER_MEMORY_TINY_ACCUMULATE_MAX_TOKENS", "4000"))
    USER_MEMORY_AUDIT_MAX_ENTRIES = int(os.getenv("USER_MEMORY_AUDIT_MAX_ENTRIES", "200"))
    USER_MEMORY_DEBUG_CLASSIFICATION = os.getenv("USER_MEMORY_DEBUG_CLASSIFICATION", "False").lower() in ('true', '1', 't')
    LLM_CALL_AUDIT_ENABLED = os.getenv("LLM_CALL_AUDIT_ENABLED", "True").lower() in ('true', '1', 't')
    LLM_CALL_AUDIT_MAX_ENTRIES = int(os.getenv("LLM_CALL_AUDIT_MAX_ENTRIES", "100"))

    activity_active_days_utc: Set[int] = set()
    if ACTIVITY_ACTIVE_DAYS_STR:
        try:
            activity_active_days_utc = {int(day.strip()) for day in ACTIVITY_ACTIVE_DAYS_STR.split(',') if day.strip() and 0 <= int(day.strip()) <= 6}
        except ValueError:
            logger.error(f"Invalid format or value in ACTIVITY_ACTIVE_DAYS_UTC: '{ACTIVITY_ACTIVE_DAYS_STR}'. Defaulting to all days.")
            activity_active_days_utc = set(range(7))
    else:
        activity_active_days_utc = set(range(7))

except ValueError as e:
    logger.critical(f"Invalid integer/float format in critical environment variables: {e}", exc_info=True)
    sys.exit("Exiting due to critical configuration error in numeric environment variables.")

RATE_LIMIT_MESSAGE_USER_TEMPLATE = os.getenv("RATE_LIMIT_MESSAGE_USER", "You've sent messages too frequently. Please use <#{channel_id}> for bot interactions.")
RESTRICTED_CHANNEL_MESSAGE_USER_TEMPLATE = os.getenv("RESTRICTED_CHANNEL_MESSAGE_USER", "As a restricted user, please use <#{channel_id}> for bot interactions.")

SUPER_ROLE_IDS_STR = os.getenv("SUPER_ROLE_IDS", "")
super_role_ids: List[int] = []
if SUPER_ROLE_IDS_STR:
    try:
        super_role_ids = [int(role_id.strip()) for role_id in SUPER_ROLE_IDS_STR.split(',') if role_id.strip()]
    except ValueError:
        logger.error(f"Invalid format for SUPER_ROLE_IDS: '{SUPER_ROLE_IDS_STR}'. Expected comma-separated numbers. No roles will be super users due to this error.")

# LiteLLM configuration (replaces OpenWebUI)
LITELLM_API_URL = os.getenv("LITELLM_API_URL", "http://localhost:4000")
LITELLM_MODEL = os.getenv("LITELLM_MODEL")
LITELLM_API_KEY = _read_docker_secret('litellm_api_key') or os.getenv("LITELLM_API_KEY", "sk-1234")  # Prefer Docker secret, fallback to env var

# MCP servers configuration
MCP_SERVERS_STR = os.getenv("MCP_SERVERS", "")
mcp_servers_parsed: List[str] = [server.strip() for server in MCP_SERVERS_STR.split(',') if server.strip()] if MCP_SERVERS_STR else []

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

# Stats/Reporting configuration
STATS_REPORT_CHANNEL_ID = int(os.getenv("STATS_REPORT_CHANNEL_ID")) if os.getenv("STATS_REPORT_CHANNEL_ID") else None
STATS_REPORT_INTERVAL_SECONDS = int(os.getenv("STATS_REPORT_INTERVAL_SECONDS", "86400"))
STATS_REPORT_TOP_USERS = int(os.getenv("STATS_REPORT_TOP_USERS", "10"))

# Debug mode for super users
DEBUG_CONTEXT_SUPER_USERS = os.getenv("DEBUG_CONTEXT_SUPER_USERS", "False").lower() in ('true', '1', 't')
USER_MEMORY_ROOT_PATH_ENV = os.getenv("USER_MEMORY_ROOT_PATH", os.path.join("data", "user_memory"))
USER_MEMORY_ROOT_PATH = (
    USER_MEMORY_ROOT_PATH_ENV
    if os.path.isabs(USER_MEMORY_ROOT_PATH_ENV)
    else os.path.join(os.path.dirname(__file__), USER_MEMORY_ROOT_PATH_ENV)
)

config_errors = []
if not DISCORD_BOT_TOKEN: config_errors.append("DISCORD_BOT_TOKEN is missing.")
if not PERSONALITY_CORE_PROMPT: config_errors.append(f"Failed to load PERSONALITY_CORE_PROMPT from: {personality_core_prompt_path}")
if not PERSONALITY_PROMPT: config_errors.append(f"Failed to load default persona prompt from: {personality_prompt_path}")
if not BASE_ACTIVITY_SYSTEM_PROMPT: config_errors.append(f"Failed to load BASE_ACTIVITY_SYSTEM_PROMPT from: {base_activity_system_prompt_path_env}")
if not LITELLM_MODEL: config_errors.append("LITELLM_MODEL is missing.")


if not RESTRICTED_USER_ROLE_ID:
    logger.warning("RESTRICTED_USER_ROLE_ID not set. Rate limiting and restriction system will be largely disabled.")
if not RESTRICTED_CHANNEL_ID and RESTRICTED_USER_ROLE_ID :
    logger.warning("RESTRICTED_CHANNEL_ID not set, but RESTRICTED_USER_ROLE_ID is. Restricted channel enforcement will be disabled.")

if RESTRICTION_DURATION_SECONDS > 0 and RESTRICTION_CHECK_INTERVAL_SECONDS <= 0:
    config_errors.append("RESTRICTION_CHECK_INTERVAL_SECONDS must be > 0 if RESTRICTION_DURATION_SECONDS is enabled.")
if RESTRICTION_DURATION_SECONDS > 0 and not RESTRICTED_USER_ROLE_ID:
    logger.warning("RESTRICTION_DURATION_SECONDS is set, but RESTRICTED_USER_ROLE_ID is not. Automatic restriction expiry is effectively disabled.")

if DAILY_TOPIC_ENABLED:
    if not DAILY_TOPIC_APPROVAL_CHANNEL_ID:
        config_errors.append("DAILY_TOPIC_ENABLED is true but DAILY_TOPIC_APPROVAL_CHANNEL_ID is missing.")
    if not DAILY_TOPIC_PUBLISH_CHANNEL_ID:
        config_errors.append("DAILY_TOPIC_ENABLED is true but DAILY_TOPIC_PUBLISH_CHANNEL_ID is missing.")
    if DAILY_TOPIC_INTERVAL_SECONDS < 0:
        config_errors.append("DAILY_TOPIC_INTERVAL_SECONDS must be >= 0.")
    if not (0 <= DAILY_TOPIC_APPROVAL_HOUR_UTC <= 23):
        config_errors.append("DAILY_TOPIC_APPROVAL_HOUR_UTC must be between 0 and 23.")
    if DAILY_TOPIC_APPROVAL_TIMEOUT_SECONDS <= 0:
        config_errors.append("DAILY_TOPIC_APPROVAL_TIMEOUT_SECONDS must be > 0.")
    if DAILY_TOPIC_CHECK_INTERVAL_SECONDS <= 0:
        config_errors.append("DAILY_TOPIC_CHECK_INTERVAL_SECONDS must be > 0.")
    if DAILY_TOPIC_THREAD_CONTEXT_MESSAGES <= 0:
        config_errors.append("DAILY_TOPIC_THREAD_CONTEXT_MESSAGES must be > 0.")
    if DAILY_TOPIC_THREAD_AUTO_ARCHIVE_MINUTES not in (60, 1440, 4320, 10080):
        logger.warning(
            "DAILY_TOPIC_THREAD_AUTO_ARCHIVE_MINUTES=%s is not a standard Discord value (60, 1440, 4320, 10080).",
            DAILY_TOPIC_THREAD_AUTO_ARCHIVE_MINUTES
        )

if not (0.0 <= USER_MEMORY_UPDATE_CHANCE <= 1.0):
    config_errors.append("USER_MEMORY_UPDATE_CHANCE must be between 0.0 and 1.0.")
if USER_MEMORY_MIN_MESSAGE_CHARS <= 0:
    config_errors.append("USER_MEMORY_MIN_MESSAGE_CHARS must be > 0.")
if USER_MEMORY_MIN_MESSAGE_WORDS <= 0:
    config_errors.append("USER_MEMORY_MIN_MESSAGE_WORDS must be > 0.")
if USER_MEMORY_MAX_CHARS <= 0:
    config_errors.append("USER_MEMORY_MAX_CHARS must be > 0.")
if USER_MEMORY_PIPELINE_MODE not in ("tiny_extract", "frontier_pipeline", "tiny_gate_frontier_core", "disabled"):
    config_errors.append(
        "USER_MEMORY_PIPELINE_MODE must be one of: tiny_extract, frontier_pipeline, tiny_gate_frontier_core, disabled."
    )
if USER_MEMORY_OLLAMA_TIMEOUT_SECONDS <= 0:
    config_errors.append("USER_MEMORY_OLLAMA_TIMEOUT_S must be > 0.")
if USER_MEMORY_AUDIT_MAX_ENTRIES <= 0:
    config_errors.append("USER_MEMORY_AUDIT_MAX_ENTRIES must be > 0.")
if USER_MEMORY_TINY_ACCUMULATE_MAX_TOKENS <= 0:
    config_errors.append("USER_MEMORY_TINY_ACCUMULATE_MAX_TOKENS must be > 0.")
if USER_MEMORY_PIPELINE_MODE != "disabled":
    if not USER_MEMORY_OLLAMA_BASE_URL:
        config_errors.append("USER_MEMORY_OLLAMA_BASE_URL is required when USER_MEMORY_PIPELINE_MODE is not disabled.")
    if not USER_MEMORY_TINY_MODEL and not USER_MEMORY_TINY_MODEL_EXTRACT:
        config_errors.append("Set USER_MEMORY_TINY_MODEL (or USER_MEMORY_TINY_MODEL_EXTRACT) when USER_MEMORY_PIPELINE_MODE is not disabled.")
if LLM_CALL_AUDIT_MAX_ENTRIES <= 0:
    config_errors.append("LLM_CALL_AUDIT_MAX_ENTRIES must be > 0.")

try:
    os.makedirs(USER_MEMORY_ROOT_PATH, exist_ok=True)
except Exception as e:
    config_errors.append(f"Could not create USER_MEMORY_ROOT_PATH ({USER_MEMORY_ROOT_PATH}): {e}")

if config_errors:
    logger.critical("FATAL: Critical configuration errors found:")
    for error in config_errors:
        logger.critical(f" -> {error}")
    sys.exit("Exiting due to critical configuration errors.")
else:
    logger.info("Core configuration loaded and validated successfully.")


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
        chat_system_prompt=DEFAULT_CHAT_SYSTEM_PROMPT,
        prompts_root_path=prompts_root_path,
        core_prompt_fallback=PERSONALITY_CORE_PROMPT,
        default_persona_prompt_fallback=PERSONALITY_PROMPT,
        response_chance=RESPONSE_CHANCE,
        max_history_per_context=MAX_HISTORY_PER_USER,
        litellm_api_url=LITELLM_API_URL,
        litellm_model=LITELLM_MODEL,
        litellm_api_key=LITELLM_API_KEY,
        mcp_servers=mcp_servers_parsed,
        redis_config=redis_connection_config,
        ignored_role_ids=ignored_role_ids,
        rate_limit_count=RATE_LIMIT_COUNT,
        rate_limit_window_seconds=RATE_LIMIT_WINDOW_SECONDS,
        token_rate_limit_count=TOKEN_RATE_LIMIT_COUNT,
        restricted_user_role_id=RESTRICTED_USER_ROLE_ID,
        restricted_channel_id=RESTRICTED_CHANNEL_ID,
        rate_limit_message_user_template=RATE_LIMIT_MESSAGE_USER_TEMPLATE,
        restricted_channel_message_user_template=RESTRICTED_CHANNEL_MESSAGE_USER_TEMPLATE,
        super_role_ids=super_role_ids,
        restriction_duration_seconds=RESTRICTION_DURATION_SECONDS,
        restriction_check_interval_seconds=RESTRICTION_CHECK_INTERVAL_SECONDS,
        random_response_delivery_chance=RANDOM_RESPONSE_DELIVERY_CHANCE,
        base_activity_system_prompt=BASE_ACTIVITY_SYSTEM_PROMPT,
        activity_update_interval_seconds=ACTIVITY_UPDATE_INTERVAL_SECONDS,
        activity_schedule_enabled=ACTIVITY_SCHEDULE_ENABLED,
        activity_active_start_hour_utc=ACTIVITY_ACTIVE_START_HOUR_UTC,
        activity_active_end_hour_utc=ACTIVITY_ACTIVE_END_HOUR_UTC,
        activity_active_days_utc=activity_active_days_utc,
        context_history_ttl_seconds=CONTEXT_HISTORY_TTL_SECONDS,
        context_message_max_age_seconds=CONTEXT_MESSAGE_MAX_AGE_SECONDS,
        default_context_messages=DEFAULT_CONTEXT_MESSAGES,
        stats_report_channel_id=STATS_REPORT_CHANNEL_ID,
        stats_report_interval_seconds=STATS_REPORT_INTERVAL_SECONDS,
        stats_report_top_users=STATS_REPORT_TOP_USERS,
        daily_topic_enabled=DAILY_TOPIC_ENABLED,
        daily_topic_approval_channel_id=DAILY_TOPIC_APPROVAL_CHANNEL_ID,
        daily_topic_publish_channel_id=DAILY_TOPIC_PUBLISH_CHANNEL_ID,
        daily_topic_interval_seconds=DAILY_TOPIC_INTERVAL_SECONDS,
        daily_topic_approval_hour_utc=DAILY_TOPIC_APPROVAL_HOUR_UTC,
        daily_topic_approval_timeout_seconds=DAILY_TOPIC_APPROVAL_TIMEOUT_SECONDS,
        daily_topic_check_interval_seconds=DAILY_TOPIC_CHECK_INTERVAL_SECONDS,
        daily_topic_thread_auto_archive_minutes=DAILY_TOPIC_THREAD_AUTO_ARCHIVE_MINUTES,
        daily_topic_thread_context_messages=DAILY_TOPIC_THREAD_CONTEXT_MESSAGES,
        user_memory_enabled=USER_MEMORY_ENABLED,
        user_memory_root_path=USER_MEMORY_ROOT_PATH,
        user_memory_update_chance=USER_MEMORY_UPDATE_CHANCE,
        user_memory_min_message_chars=USER_MEMORY_MIN_MESSAGE_CHARS,
        user_memory_min_message_words=USER_MEMORY_MIN_MESSAGE_WORDS,
        user_memory_max_chars=USER_MEMORY_MAX_CHARS,
        user_memory_pipeline_mode=USER_MEMORY_PIPELINE_MODE,
        user_memory_ollama_base_url=USER_MEMORY_OLLAMA_BASE_URL,
        user_memory_ollama_api_key=USER_MEMORY_OLLAMA_API_KEY,
        user_memory_ollama_timeout_seconds=USER_MEMORY_OLLAMA_TIMEOUT_SECONDS,
        user_memory_tiny_model=USER_MEMORY_TINY_MODEL,
        user_memory_tiny_model_extract=USER_MEMORY_TINY_MODEL_EXTRACT,
        user_memory_tiny_model_classifier=USER_MEMORY_TINY_MODEL_CLASSIFIER,
        user_memory_tiny_accumulate_max_tokens=USER_MEMORY_TINY_ACCUMULATE_MAX_TOKENS,
        user_memory_audit_max_entries=USER_MEMORY_AUDIT_MAX_ENTRIES,
        user_memory_debug_classification=USER_MEMORY_DEBUG_CLASSIFICATION,
        llm_call_audit_enabled=LLM_CALL_AUDIT_ENABLED,
        llm_call_audit_max_entries=LLM_CALL_AUDIT_MAX_ENTRIES,
        debug_context_super_users=DEBUG_CONTEXT_SUPER_USERS,
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
        logger.critical(f"FATAL: Required Privileged Intents (e.g., Server Members or Message Content) are not enabled. Details: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"FATAL: An unexpected error occurred while running the bot: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Bot process has concluded or been interrupted.")

if __name__ == "__main__":
    main_run()
