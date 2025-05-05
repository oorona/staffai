# bot.py

import discord


from discord.ext import commands
import logging
from collections import deque
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# --- RENAMED CLASS ---
class AIBot(commands.Bot): # Renamed from InteractionBot
    """
    The main class for the Discord AI Bot.
    Handles configuration, user history storage, and loading cogs.
    """
    def __init__(self,  welcome_channel_id: int,
                 welcome_system: str, welcome_prompt: str,
                 response_chance: float, max_history: int, 
                 api_url:str, model: str, api_key: str,
                 list_tools:List[str], knowledge_id: str,
                 intents: discord.Intents):
        """
        Initializes the bot instance.

        Args:
           
            response_chance (float): The probability (0.0 to 1.0) of responding to a message.
            max_history (int): The maximum number of messages to store per user.
            intents (discord.Intents): The intents to use for the bot connection.
        """
        super().__init__(command_prefix="!", intents=intents, help_command=None)


        self.response_chance = response_chance
        self.welcome_channel_id = welcome_channel_id 
        self.max_history_per_user = max_history
        self.api_url = api_url
        self.model = model
        self.api_key = api_key        
        self.user_histories = {}
        self.list_tools=list_tools
        self.knowledge_id=knowledge_id
        self.welcome_system = welcome_system
        self.welcome_prompt = welcome_prompt

        # --- UPDATED LOG MESSAGE ---
        logger.info("AIBot instance created.") # Updated name
        logger.info(f"Response Chance: {self.response_chance * 100:.1f}%")
        logger.info(f"Max History per User: {self.max_history_per_user}")
        logger.info(f"Text Api Url: {self.api_url}")
        logger.info(f"Text model: {self.model}")
        logger.info(f"Text Api key: {self.api_key}")
        logger.info(f"List Tools: {self.list_tools}")
        logger.info(f"Knowledge ID: {self.knowledge_id}")
        logger.info(f"Welcome System: {self.welcome_system}")
        logger.info(f"Welcome Prompt: {self.welcome_prompt}")   
        


    async def setup_hook(self):
        """
        Asynchronous setup hook called after login but before connecting to WebSocket.
        Used to load extensions (cogs).
        """
        initial_extensions = [
            'cogs.listener_cog'
        ]
        for extension in initial_extensions:
            try:
                await self.load_extension(extension)
                logger.info(f"Successfully loaded extension: {extension}")
            except commands.ExtensionNotFound:
                logger.error(f"FATAL: Extension not found: {extension}. Make sure the file exists (e.g., cogs/listener_cog.py).")
            except commands.ExtensionAlreadyLoaded:
                logger.warning(f"Extension already loaded: {extension}")
            except commands.NoEntryPointError:
                logger.error(f"FATAL: Extension {extension} does not have a 'setup' function.")
            except commands.ExtensionFailed as e:
                logger.exception(f"FATAL: Failed to load extension {extension}: {e.original}")
            except Exception as e:
                 logger.exception(f"FATAL: An unexpected error occurred loading extension {extension}: {e}")

        logger.info("Setup hook completed.")


    async def on_ready(self):
        """
        Called when the bot is ready and connected to Discord.
        """
        logger.info(f'Logged in as {self.user.name} (ID: {self.user.id})')
        logger.info('------ Bot is Ready ------')



    ''' 
    def get_user_history(self, user_id: int) -> deque:
        """
        Retrieves the message history deque for a given user.
        Creates a new deque if the user is not already tracked.

        Args:
            user_id (int): The Discord user ID.

        Returns:
            deque: The message history deque for the user.
        """
        if user_id not in self.user_histories:
            logger.debug(f"Creating new history deque for user ID: {user_id}")
            self.user_histories[user_id] = deque(maxlen=self.max_history_per_user)
        return self.user_histories[user_id]

    def add_message_to_history(self, user_id: int, message_content: str):
        """
        Adds a message to the specified user's history deque.

        Args:
            user_id (int): The Discord user ID.
            message_content (str): The content of the message to add.
        """
        history_deque = self.get_user_history(user_id)
        history_deque.append(message_content)
        logger.debug(f"Added message to history for user ID: {user_id}. New length: {len(history_deque)}")
    '''
