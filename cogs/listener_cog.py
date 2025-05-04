# cogs/listener_cog.py

import discord
from discord.ext import commands
import random
import logging
from utils.webui_api import WebUIAPI 


# Import the bot class definition for type hinting
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # --- UPDATED IMPORT FOR TYPE HINTING ---
    from bot import AIBot # Changed from InteractionBot

logger = logging.getLogger(__name__)

class ListenerCog(commands.Cog):
    """
    Cog responsible for listening to messages in the input channel,
    managing user history, and triggering responses.
    """
    def __init__(self, bot: 'AIBot'): # --- UPDATED TYPE HINT ---
        """
        Initializes the ListenerCog.

        Args:
            bot (AIBot): The instance of the main bot class. # --- UPDATED TYPE HINT DOC ---
        """
        self.bot = bot
        self.api_client = WebUIAPI(base_url=self.bot.api_url, model=self.bot.model, api_key=self.bot.api_key,
                                   welcome_system=self.bot.welcome_system,
                                   welcome_prompt=self.bot.welcome_prompt,
                                   max_history_per_user=self.bot.max_history_per_user, 
                                   knowledge_id=self.bot.knowledge_id,
                                   list_tools=self.bot.list_tools)

        logger.info("ListenerCog initialized.")

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """ Handles incoming messages. """
        # 1. Ignore self
        if message.author == self.bot.user:
            return

        if message.author.bot:
            return
    
        # 2. Check channel
        if message.channel.id != self.bot.input_channel_id:
            return

        # --- Logic continues below (no changes needed in the core logic) ---
        user_id = message.author.id
        message_content = message.content
        logger.debug(f"Received message from user {message.author.name} ({user_id}) in input channel: '{message_content}'")

        # 3. Add to history
        #self.bot.add_message_to_history(user_id, message_content)

        should_respond = False
        is_reply_to_bot = False

        # Check if the message is a reply and if it's replying to the bot
        if message.reference and message.reference.resolved:
            # message.reference.resolved contains the Message object being replied to (if cached)
            referenced_message = message.reference.resolved
            if isinstance(referenced_message, discord.Message) and referenced_message.author == self.bot.user:
                is_reply_to_bot = True
                should_respond = True # Force response if it's a reply to the bot
                logger.info(f"Message is a reply to the bot. Forcing response for user {user_id}.")

        if self.bot.user in message.mentions:
            is_reply_to_bot = True
            should_respond = True
            logger.info(f"Message mentions the bot. Forcing response for user {user_id}.")

        # If not forced by reply, check random chance (Requirement #6)
        if not should_respond:
            if random.random() < self.bot.response_chance:
                should_respond = True
                logger.info(f"Random chance met ({self.bot.response_chance * 100:.1f}%). Preparing response for user {user_id}.")
            else: # Optional logging for when random chance fails
                logger.debug(f"Random chance not met for user {user_id}. No response sent.")

        # 4. Random response check
        if should_respond:

            # 5. Retrieve & Format history
            #user_history = self.bot.get_user_history(user_id)
            #if not user_history:
            #    logger.warning(f"Attempted to respond to user {user_id} but their history was empty.")
            #    return

            #response_content = "\n".join(user_history)
            #response_content = f"History for {message.author.mention}:\n---\n{response_content}"
        

            # 6. Get output channel & Send
            output_channel = self.bot.get_channel(self.bot.output_channel_id)
            if output_channel is None:
                logger.error(f"Could not find output channel ID: {self.bot.output_channel_id}. Response aborted.")
                return
            if not output_channel.permissions_for(output_channel.guild.me).send_messages:
                 logger.error(f"Bot lacks 'Send Messages' permission in output channel: {output_channel.name} ({self.bot.output_channel_id}).")
                 return

            try:
                #user_id_1 = 123
                #prompt_1 = "Why is the sky blue?"
                #system_prompt_1 = "You are a helpful stafff expert in computer science."
                response_content, error_1 = await self.api_client.generate_response(user_id, message_content)
                logger.debug(response_content)
                if len(response_content) > 2000:
                    response_content = response_content[:1997] + "..."
                    logger.warning(f"Response content for user {user_id} exceeded 2000 chars, truncated.")
                #await output_channel.send(response_content)
                await message.reply(response_content)
                await self.bot.process_commands(message)
                logger.info(f"Sent response for user {user_id} to output channel {self.bot.output_channel_id}.")
            except discord.Forbidden:
                 logger.error(f"Forbidden: Cannot send message to output channel ID: {self.bot.output_channel_id}.")
            except discord.HTTPException as e:
                logger.error(f"Failed to send message to output channel ID {self.bot.output_channel_id}: {e}")
            except Exception as e:
                 logger.exception(f"Unexpected error sending response for user {user_id}: {e}")


    @commands.Cog.listener()
    async def on_member_join(self, member: discord.Member):
        """
        Handles actions when a new member joins the server.
        Sends a welcome message if configured.
        """
        logger.info(f"Member joined: {member.name}#{member.discriminator} (ID: {member.id}) in server {member.guild.name}")

        # Check if welcome messages are enabled (channel ID is set)
        if not self.bot.welcome_channel_id:
            logger.debug("Welcome channel ID not configured, skipping welcome message.")
            return

        # Get the welcome channel object
        welcome_channel = self.bot.get_channel(self.bot.welcome_channel_id)

        if welcome_channel is None:
            logger.error(f"Could not find the configured welcome channel with ID: {self.bot.welcome_channel_id}. Cannot send welcome message.")
            return

        # Optional: Check if the bot has permissions in the welcome channel
        if not welcome_channel.permissions_for(member.guild.me).send_messages:
            logger.error(f"Bot lacks 'Send Messages' permission in the welcome channel: {welcome_channel.name} ({self.bot.welcome_channel_id}). Cannot send welcome message.")
            return

        # Create and send the welcome message
        response_content, error_1 = await self.api_client.generate_welcome_message(member_name=member.display_name,guild_name=member.guild.name)
        response_content=f"{member.mention}\n"+response_content
        print(response_content)

        try:
            await welcome_channel.send(response_content)
            logger.info(f"Sent welcome message for {member.name} to channel {welcome_channel.name}.")
        except discord.Forbidden:
            logger.error(f"Forbidden: Bot lacks permissions to send welcome message to channel ID: {self.bot.welcome_channel_id}.")
        except discord.HTTPException as e:
            logger.error(f"Failed to send welcome message to channel ID {self.bot.welcome_channel_id}: {e}")
        except Exception as e:
            logger.exception(f"An unexpected error occurred while sending welcome message for {member.name}: {e}")

# Mandatory setup function for the cog
async def setup(bot: 'AIBot'): 
    await bot.add_cog(ListenerCog(bot))
    logger.info("ListenerCog added to the bot.")

# --- End of cogs/listener_cog.py ---