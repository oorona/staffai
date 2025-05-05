# main.py

import os
import sys
import logging
import discord
from dotenv import load_dotenv

# --- Logging Setup (No changes needed) ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(name)s: %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- Helper Function to Read Prompts ---
def load_prompt_from_file(file_path):
    """Reads content from a specified file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.error(f"FATAL: Prompt file not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"FATAL: Error reading prompt file {file_path}: {e}")
        return None

# --- Load Environment Variables (No changes needed) ---
logger.info("Loading environment variables from .env file...")
if not load_dotenv():
    logger.warning("Could not find .env file. Ensure it exists in the root directory.")

# --- Configuration Retrieval and Validation (UPDATED) ---
logger.info("Retrieving and validating configuration...")

# --- Load Prompts from Files ---
logger.info("Loading prompts from files...")
# Define the base path for prompts relative to the script's location
# Adjust 'utils/prompts' if your structure is different
prompt_dir = os.path.join(os.path.dirname(__file__), 'utils', 'prompts')
welcome_system_path = os.path.join(prompt_dir, 'welcome_system.txt')
welcome_prompt_path = os.path.join(prompt_dir, 'welcome_prompt.txt')

WELCOME_SYSTEM = load_prompt_from_file(welcome_system_path)
WELCOME_PROMPT = load_prompt_from_file(welcome_prompt_path)

# --- Load other configurations from .env ---
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
WELCOME_CHANNEL_ID = int(os.getenv("WELCOME_CHANNEL_ID") ) # Keep validation if needed
RESPONSE_CHANCE = float(os.getenv("RESPONSE_CHANCE"))
MAX_HISTORY_PER_USER = int(os.getenv("MAX_HISTORY_PER_USER"))
TEST_API_URL = os.getenv("OPENWEBUI_API_URL", "http://localhost:3000")
TEST_MODEL = os.getenv("OPENWEBUI_MODEL")
TEST_API_KEY = os.getenv("OPENWEBUI_API_KEY")
LIST_TOOLS= os.getenv("LIST_TOOLS")
KNOWLEDGE_ID= os.getenv("KNOWLEDGE_ID")

# --- Validation ---
config_errors = []

if not DISCORD_BOT_TOKEN: config_errors.append("DISCORD_BOT_TOKEN is missing.")
if not WELCOME_SYSTEM: config_errors.append("Failed to load WELCOME_SYSTEM from file.") # Check if loaded
if not WELCOME_PROMPT: config_errors.append("Failed to load WELCOME_PROMPT from file.") # Check if loaded
# ... (Keep other existing validation checks for TEST_API_URL, TEST_MODEL, etc.) ...
if not TEST_API_URL:
    config_errors.append("TEST_API_URL missing")
if not TEST_MODEL:
     config_errors.append("TEST_MODEL missing")
# Add checks for other required variables like WELCOME_CHANNEL_ID, etc. if necessary

if config_errors:
    logger.error("FATAL: Configuration errors: %s", config_errors)
    # Print specific errors if needed
    if not WELCOME_SYSTEM: logger.error(f" -> Check path: {welcome_system_path}")
    if not WELCOME_PROMPT: logger.error(f" -> Check path: {welcome_prompt_path}")
    sys.exit(1)
else:
    logger.info("Configuration loaded successfully (including prompts from files).")


print("\n--- Starting WebUI API Test (Endpoint: /api/chat/completions) ---")
print(f"Using API URL: {TEST_API_URL}")
print(f"Using Model: {TEST_MODEL}")
print(f"API Key Provided: {'Yes' if TEST_API_KEY else 'No'}")


# --- Bot Initialization (No changes needed here, uses the loaded variables) ---
try:
    from bot import AIBot
    logger.info("Successfully imported AIBot from bot.py")
except ImportError:
    logger.error("FATAL: Could not import AIBot from bot.py.")
    logger.error("Make sure bot.py exists in the same directory and defines the AIBot class.")
    sys.exit(1)
except Exception as e:
     logger.error(f"FATAL: An unexpected error occurred during import: {e}")
     sys.exit(1)

# Define necessary intents (No changes needed)
intents = discord.Intents.default()
intents.guilds = True
intents.messages = True
intents.members = True
intents.message_content = True # Ensure this is enabled in your Discord Dev Portal

logger.info("Initializing the bot instance...")
try:
    # Instantiate the bot, passing the configuration values loaded from files/env
    the_bot = AIBot(
        welcome_channel_id=WELCOME_CHANNEL_ID,
        welcome_system=WELCOME_SYSTEM, # Now loaded from file
        welcome_prompt=WELCOME_PROMPT, # Now loaded from file
        response_chance=RESPONSE_CHANCE,
        max_history=MAX_HISTORY_PER_USER,
        api_url=TEST_API_URL,
        model=TEST_MODEL,
        api_key=TEST_API_KEY,
        list_tools=LIST_TOOLS.split(",") if LIST_TOOLS else [], # Handle case where LIST_TOOLS might be empty
        knowledge_id=KNOWLEDGE_ID,
        intents=intents)
except Exception as e:
    logger.exception(f"FATAL: Failed to initialize the AIBot: {e}")
    sys.exit(1)


# --- Run the Bot (No changes needed) ---
def main():
    """Runs the Discord bot."""
    logger.info("Attempting to run the bot...")
    try:
        the_bot.run(DISCORD_BOT_TOKEN)
    except discord.LoginFailure:
        logger.error("FATAL: Login failed. Check DISCORD_BOT_TOKEN.")
        sys.exit(1)
    except discord.PrivilegedIntentsRequired:
        logger.error("FATAL: Message Content Intent is required but not enabled.")
        logger.error("Enable it in the Discord Developer Portal -> Bot -> Privileged Gateway Intents.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"FATAL: An unexpected error occurred while running the bot: {e}")
        sys.exit(1)
    finally:
        logger.info("Bot process has stopped.")

if __name__ == "__main__":
    main()

# --- End of main.py ---