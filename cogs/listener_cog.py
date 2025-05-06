# cogs/listener_cog.py

import discord
from discord.ext import commands
import random
import logging
import re # Import regex module for potentially cleaner removal
from utils.webui_api import WebUIAPI

# Import the bot class definition for type hinting
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from bot import AIBot

logger = logging.getLogger(__name__)

class ListenerCog(commands.Cog):
    """
    Cog responsible for listening to messages, removing self-mentions before
    sending to LLM, managing user history, triggering responses in the same channel,
    and welcoming members.
    """
    def __init__(self, bot: 'AIBot'):
        """
        Initializes the ListenerCog.

        Args:
            bot (AIBot): The instance of the main bot class.
        """
        self.bot = bot
        # Initialize API client (ensure parameters match your bot's attributes)
        self.api_client = WebUIAPI(
            base_url=self.bot.api_url,
            model=self.bot.model,
            api_key=self.bot.api_key,
            welcome_system=self.bot.welcome_system,
            welcome_prompt=self.bot.welcome_prompt,
            max_history_per_user=self.bot.max_history_per_user,
            knowledge_id=self.bot.knowledge_id,
            list_tools=self.bot.list_tools
        )
        logger.info("ListenerCog initialized.")

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """ Handles incoming messages. """
        # 1. Ignore self
        if message.author == self.bot.user:
            return

        # Ignore other bots
        if message.author.bot:
            return

        # Bot now listens in any channel it can read/write in.

        user_id = message.author.id
        original_message_content = message.content # Store original content
        logger.debug(f"Received message from {message.author.name} ({user_id}) in #{message.channel.name}: '{original_message_content}'")

        # --- Prepare content for LLM: Remove bot's own mention ---
        content_for_llm = original_message_content # Start with original

        # Construct the bot mention string
        bot_mention_string = f'<@{self.bot.user.id}>'

        # Remove all occurrences of the bot mention string
        # Using simple replace first
        content_for_llm = content_for_llm.replace(bot_mention_string, '')

        # Clean up potential resulting double spaces or leading/trailing spaces
        content_for_llm = re.sub(r'\s+', ' ', content_for_llm).strip()
        # Example: "<@BOT_ID> hello <@OTHER_ID>" becomes " hello <@OTHER_ID>"
        # Then after re.sub/strip: "hello <@OTHER_ID>"

        # --- Determine if the bot should respond ---
        should_respond = False

        # Force response if the message is a reply to the bot
        if message.reference and message.reference.resolved:
            referenced_message = message.reference.resolved
            if isinstance(referenced_message, discord.Message) and referenced_message.author == self.bot.user:
                should_respond = True
                logger.info(f"Message in #{message.channel.name} is a reply to the bot. Forcing response for user {user_id}.")

        # Force response if the bot was mentioned in the *original* message
        # We check message.mentions which contains the list of mentioned user objects
        if self.bot.user in message.mentions:
             should_respond = True
             logger.info(f"Message in #{message.channel.name} mentions the bot. Forcing response for user {user_id}.")

        # If not forced, check random chance
        if not should_respond:
            if random.random() < self.bot.response_chance:
                should_respond = True
                logger.info(f"Random chance ({self.bot.response_chance * 100:.1f}%) met in #{message.channel.name}. Preparing response for {user_id}.")
            else:
                logger.debug(f"Random chance not met for user {user_id} in #{message.channel.name}. No response.")

        # --- Send response if needed ---
        if should_respond:
            # Check if the cleaned message is empty (e.g., user only mentioned the bot)
            if not content_for_llm:
                logger.info(f"Message content is empty after removing bot mention for user {user_id}. Not sending to LLM.")
                # Optionally send a default reply here if just mentioning the bot should do something
                # await message.reply("Yes?")
                return

            output_channel = message.channel # Respond in the same channel

            # Check bot's permissions
            if not output_channel.permissions_for(output_channel.guild.me).send_messages:
                 logger.error(f"Bot lacks 'Send Messages' permission in channel: #{output_channel.name} ({output_channel.id}). Cannot reply.")
                 return

            try:
                # Indicate typing while generating response
                async with output_channel.typing():
                    logger.debug(f"Sending content to LLM for user {user_id} (Bot mentions removed): '{content_for_llm}'")
                    response_content, error_message = await self.api_client.generate_response(user_id, content_for_llm) # Use cleaned content

                    if error_message:
                        logger.error(f"API Error for user {user_id} in #{output_channel.name}: {error_message}")
                        reply_text = response_content if response_content else "Sorry, I encountered an error trying to respond."
                        await message.reply(reply_text)
                        return

                    if not response_content:
                         logger.warning(f"API returned no content for user {user_id} in #{output_channel.name}. No message sent.")
                         return

                    if len(response_content) > 2000:
                        response_content = response_content[:1997] + "..."
                        logger.warning(f"Response content for {user_id} in #{output_channel.name} truncated.")

                    await message.reply(response_content)
                    logger.info(f"Sent reply for user {user_id} to channel #{output_channel.name}.")

            except discord.Forbidden:
                 logger.error(f"Forbidden: Cannot send message to channel #{output_channel.name} ({output_channel.id}). Check permissions.")
            except discord.HTTPException as e:
                logger.error(f"Failed to send message to channel #{output_channel.name} ({output_channel.id}): {e}")
            except Exception as e:
                 logger.exception(f"Unexpected error sending response for user {user_id} in #{output_channel.name}: {e}")


    @commands.Cog.listener()
    async def on_member_join(self, member: discord.Member):
        """
        Handles actions when a new member joins the server.
        Sends a welcome message if configured. (No changes here from previous)
        """
        logger.info(f"Member joined: {member.name}#{member.discriminator} (ID: {member.id}) in server {member.guild.name}")

        if not self.bot.welcome_channel_id:
            logger.debug("Welcome channel ID not configured, skipping welcome message.")
            return

        welcome_channel = self.bot.get_channel(self.bot.welcome_channel_id)
        if welcome_channel is None:
            logger.error(f"Could not find welcome channel ID: {self.bot.welcome_channel_id}.")
            return

        if not welcome_channel.permissions_for(member.guild.me).send_messages:
            logger.error(f"Bot lacks 'Send Messages' permission in welcome channel: {welcome_channel.name}.")
            return

        try:
            async with welcome_channel.typing():
                welcome_message_content, error_msg = await self.api_client.generate_welcome_message(
                    member
                )

                if error_msg:
                    logger.error(f"API Error generating welcome for {member.name}: {error_msg}")
                    await welcome_channel.send(f"Welcome {member.mention}! Had a bit of trouble generating a special greeting.")
                    return

                if not welcome_message_content:
                    logger.warning(f"API returned no content for welcome message for {member.name}. Sending generic.")
                    await welcome_channel.send(f"Welcome {member.mention} to {member.guild.name}!")
                    return

                final_welcome_text = welcome_message_content
                logger.info(f"Final welcome message for {member.name}: {final_welcome_text}")
                if len(final_welcome_text) > 2000:
                    final_welcome_text = final_welcome_text[:1997] + "..."
                    logger.warning(f"Welcome message for {member.name} truncated.")

                await welcome_channel.send(final_welcome_text)
                logger.info(f"Sent welcome message for {member.name} to channel {welcome_channel.name}.")

        except discord.Forbidden:
            logger.error(f"Forbidden: Cannot send welcome message to channel ID: {self.bot.welcome_channel_id}.")
        except discord.HTTPException as e:
            logger.error(f"Failed to send welcome message to channel ID {self.bot.welcome_channel_id}: {e}")
        except Exception as e:
            logger.exception(f"An unexpected error sending welcome message for {member.name}: {e}")


# Mandatory setup function for the cog
async def setup(bot: 'AIBot'):
    await bot.add_cog(ListenerCog(bot))
    logger.info("ListenerCog added to the bot.")

# --- End of cogs/listener_cog.py ---