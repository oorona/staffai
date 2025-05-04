# main.py

import os
import sys
import logging
import discord
from dotenv import load_dotenv

# --- Logging Setup (No changes needed) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(name)s: %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- Load Environment Variables (No changes needed) ---
logger.info("Loading environment variables from .env file...")
if not load_dotenv():
    logger.warning("Could not find .env file. Ensure it exists in the root directory.")

# --- Configuration Retrieval and Validation (No changes needed in logic) ---
logger.info("Retrieving and validating configuration...")
# ... (Keep the existing validation block for DISCORD_BOT_TOKEN, etc.) ...
# ... Ensure variables INPUT_CHANNEL_ID, OUTPUT_CHANNEL_ID, RESPONSE_CHANCE, MAX_HISTORY_PER_USER are defined from env vars ...
# --- Start Example Validation Block (ensure yours is complete as before) ---
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
INPUT_CHANNEL_ID = int(os.getenv("INPUT_CHANNEL_ID"))
OUTPUT_CHANNEL_ID = int(os.getenv("OUTPUT_CHANNEL_ID"))
WELCOME_CHANNEL_ID = int(os.getenv("WELCOME_CHANNEL_ID") )
RESPONSE_CHANCE = float(os.getenv("RESPONSE_CHANCE"))
MAX_HISTORY_PER_USER = int(os.getenv("MAX_HISTORY_PER_USER"))
TEST_API_URL = os.getenv("OPENWEBUI_API_URL", "http://localhost:3000")
TEST_MODEL = os.getenv("OPENWEBUI_MODEL")
TEST_API_KEY = os.getenv("OPENWEBUI_API_KEY")
LIST_TOOLS= os.getenv("LIST_TOOLS")
KNOWLEDGE_ID= os.getenv("KNOWLEDGE_ID")
WELCOME_SYSTEM= os.getenv("WELCOME_SYSTEM")
WELCOME_PROMPT= os.getenv("WELCOME_PROMPT")

config_errors = []

if not DISCORD_BOT_TOKEN: config_errors.append("DISCORD_BOT_TOKEN is missing.")

try: INPUT_CHANNEL_ID = int(INPUT_CHANNEL_ID) if INPUT_CHANNEL_ID else config_errors.append("INPUT_CHANNEL_ID missing")
except ValueError: config_errors.append("INPUT_CHANNEL_ID invalid") 

if not TEST_API_URL:
    print("ERROR: OPENWEBUI_API_URL not set in .env file or default. Cannot run tests.")
    config_errors.append("TEST_API_URL missing")

if not TEST_MODEL:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!! ERROR: OPENWEBUI_MODEL not found in .env file.       !!!")
    print("!!!          Please set it in .env to run tests.          !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    config_errors.append("TEST_MODEL missing")



if config_errors: # Simplified error exit
    logger.error("FATAL: Configuration errors: %s", config_errors)
    sys.exit(1)
else:
    logger.info("Configuration loaded successfully.")




print("\n--- Starting WebUI API Test (Endpoint: /api/chat/completions) ---")
print(f"Using API URL: {TEST_API_URL}")
print(f"Using Model: {TEST_MODEL}")
print(f"API Key Provided: {'Yes' if TEST_API_KEY else 'No'}")


# --- Bot Initialization (UPDATED) ---
try:
    from bot import AIBot 
    logger.info("Successfully imported AIBot from bot.py") 
except ImportError:
    # --- UPDATED ERROR MESSAGE ---
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

intents.message_content = True

logger.info("Initializing the bot instance...")
try:
    # Instantiate the bot, passing the configuration values
    the_bot = AIBot( # Changed from InteractionBot
        input_channel_id=INPUT_CHANNEL_ID,
        output_channel_id=OUTPUT_CHANNEL_ID,
        welcome_channel_id=WELCOME_CHANNEL_ID, 
        welcome_system=WELCOME_SYSTEM,
        welcome_prompt=WELCOME_PROMPT,
        response_chance=RESPONSE_CHANCE,
        max_history=MAX_HISTORY_PER_USER,
        api_url=TEST_API_URL, 
        model=TEST_MODEL, 
        api_key=TEST_API_KEY,
        list_tools=LIST_TOOLS.split (","),
        knowledge_id=KNOWLEDGE_ID,
        intents=intents)
except Exception as e:
    # --- UPDATED ERROR MESSAGE ---
    logger.exception(f"FATAL: Failed to initialize the AIBot: {e}") # Updated name
    sys.exit(1)


# --- Run the Bot (No changes needed) ---
def main():
    """Runs the Discord bot."""
    logger.info("Attempting to run the bot...")
    try:
        # The variable name 'the_bot' is fine, it holds the AIBot instance
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