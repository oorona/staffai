# cogs/listener_cog.py

import discord
from discord.ext import commands
import random
import logging
import re
import time
from utils.webui_api import WebUIAPI
from typing import TYPE_CHECKING, Optional, Set, Any

# Attempt to import redis for type hinting, but don't crash if not installed
# Rate limiting features depend on it being installed and configured.
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    # Define redis.Redis as Any if not available, for type hints to work without error
    if TYPE_CHECKING:
        redis = Any

if TYPE_CHECKING:
    from bot import AIBot


logger = logging.getLogger(__name__)

class ListenerCog(commands.Cog):
    """
    Cog handling message listening, rate limiting, restrictions, LLM interaction,
    and welcoming new members. Uses public replies for notifications.
    """
    def __init__(self, bot: 'AIBot'):
        self.bot = bot
        self.api_client = WebUIAPI(
            base_url=self.bot.api_url,
            model=self.bot.model,
            api_key=self.bot.api_key,
            welcome_system=self.bot.welcome_system_prompt,
            welcome_prompt=self.bot.welcome_user_prompt,
            max_history_per_user=self.bot.max_history_per_context, # Name alignment
            knowledge_id=self.bot.knowledge_id,
            list_tools=self.bot.list_tools,
            redis_config=self.bot.redis_config # Pass Redis config for history persistence
        )

        # Store Redis client from bot instance for easier access in this cog
        self.redis_client: Optional[redis.Redis] = getattr(self.bot, 'redis_client_general', None)

        logger.info("ListenerCog initialized.")
        if not self.redis_client:
             logger.warning("ListenerCog: Bot's general Redis client unavailable. Rate limiting will be disabled.")

        if self.bot.ignored_role_ids_set:
            logger.info(f"ListenerCog will globally ignore users with roles: {self.bot.ignored_role_ids_set}")
        if self.bot.rate_limit_exempt_role_ids_set:
            logger.info(f"ListenerCog: Rate limits EXEMPT for Role IDs: {self.bot.rate_limit_exempt_role_ids_set}")
        if self.bot.restricted_user_role_id:
            logger.info(f"Restriction system configured: RoleID={self.bot.restricted_user_role_id}, ChannelID={self.bot.restricted_channel_id}")
            logger.info(f"Msg Limit: {self.bot.rate_limit_count}/{self.bot.rate_limit_window_seconds}s, Token Limit: {self.bot.token_rate_limit_count}/{self.bot.rate_limit_window_seconds}s")
        else:
            logger.info("Restriction system (role assignment) not fully configured (RESTRICTED_USER_ROLE_ID not set).")


    async def _apply_restriction(self, member: discord.Member, guild: discord.Guild, reason: str) -> bool:
        """
        Helper to apply restriction role. Logs action.
        Returns True if the role was newly applied, False otherwise.
        Does NOT send notifications.
        """
        if not isinstance(member, discord.Member): return False
        if not self.bot.restricted_user_role_id: return False

        restricted_role = guild.get_role(self.bot.restricted_user_role_id)
        if not restricted_role:
            logger.error(f"Configured RESTRICTED_USER_ROLE_ID {self.bot.restricted_user_role_id} not found in guild {guild.name}.")
            return False

        role_applied = False
        try:
            if restricted_role not in member.roles:
                await member.add_roles(restricted_role, reason=reason)
                logger.info(f"Assigned role '{restricted_role.name}' to {member.name} ({member.id}) for: {reason}")
                role_applied = True
            else:
                logger.info(f"{member.name} ({member.id}) already has restricted role. Current trigger reason: {reason}")
                # In Phase 2 (Decay), would potentially extend expiry here.
        except discord.Forbidden:
            logger.error(f"Bot lacks 'Manage Roles' permission to assign role ID {self.bot.restricted_user_role_id} to {member.name} ({member.id}) in guild {guild.name}.")
        except discord.HTTPException as e:
            logger.error(f"Failed to assign role to {member.name} ({member.id}): {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error assigning role to {member.name} ({member.id}): {e}", exc_info=True)
        return role_applied

    def _format_notification(self, template: str, channel_id: Optional[int]) -> str:
        """Formats notification templates, replacing channel placeholder."""
        channel_mention = f"<#{channel_id}>" if channel_id else "the designated channel"
        # Use a placeholder unlikely to be in the template naturally
        return template.replace("<#{channel_id}>", channel_mention)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        # === 1. Initial Checks & Setup ===
        if message.author == self.bot.user or message.author.bot or not message.guild:
            return

        member: discord.Member = message.author
        current_time = time.time()
        guild_id = message.guild.id
        user_id = member.id
        author_role_ids: Set[int] = {role.id for role in member.roles}

        # === 2. Pre-processing Checks ===

        # --- Globally Ignored Roles Check ---
        if self.bot.ignored_role_ids_set and not self.bot.ignored_role_ids_set.isdisjoint(author_role_ids):
            matched_roles = [role.name for role in member.roles if role.id in self.bot.ignored_role_ids_set]
            logger.info(f"Ignoring message from {member.name} ({user_id}) due to globally ignored role(s): {', '.join(matched_roles)}")
            return

        # --- Rate Limit Exemption Check ---
        is_rate_limit_exempt = False
        if self.bot.rate_limit_exempt_role_ids_set and not self.bot.rate_limit_exempt_role_ids_set.isdisjoint(author_role_ids):
            is_rate_limit_exempt = True
            exempt_roles = [role.name for role in member.roles if role.id in self.bot.rate_limit_exempt_role_ids_set]
            logger.debug(f"User {member.name} ({user_id}) is EXEMPT from rate limits due to role(s): {', '.join(exempt_roles)}.")

        # --- Restricted User & Channel Enforcement ---
        is_currently_restricted_by_role = self.bot.restricted_user_role_id and (self.bot.restricted_user_role_id in author_role_ids)
        if is_currently_restricted_by_role and self.bot.restricted_channel_id:
            if message.channel.id != self.bot.restricted_channel_id:
                logger.info(f"Restricted user {member.name} ({user_id}) used bot in disallowed channel {message.channel.name} ({message.channel.id}).")
                notification_content = self._format_notification(
                    self.bot.restricted_channel_message_user_template,
                    self.bot.restricted_channel_id
                )
                try:
                    if message.channel.permissions_for(message.guild.me).send_messages:
                        await message.reply(notification_content, mention_author=True)
                    else: logger.warning(f"Cannot send restricted channel notification reply in {message.channel.name}: No permission.")
                except Exception as e: logger.error(f"Failed to send restricted channel notification reply: {e}", exc_info=True)
                return # Stop processing

        # === 3. Determine if Bot Should Engage ===
        original_message_content = message.content
        content_for_llm = original_message_content
        bot_mention_strings = [f'<@{self.bot.user.id}>', f'<@!{self.bot.user.id}>']
        for mention_str in bot_mention_strings:
            content_for_llm = content_for_llm.replace(mention_str, '')
        content_for_llm = re.sub(r'\s+', ' ', content_for_llm).strip()

        should_respond = False
        is_reply = message.reference and message.reference.resolved and isinstance(message.reference.resolved, discord.Message) and message.reference.resolved.author == self.bot.user
        is_mention = self.bot.user in message.mentions
        meets_random_chance = (random.random() < self.bot.response_chance)

        if is_reply:
            should_respond = True
            logger.debug(f"Responding to {member.name} (reply).")
        elif is_mention:
            if content_for_llm:
                should_respond = True
                logger.debug(f"Responding to {member.name} (mention with content).")
            else:
                logger.info(f"User {member.name} only mentioned the bot; ignoring LLM interaction.")
                # Maybe send simple ack: await message.reply("Yes?", mention_author=False)
                pass # should_respond remains False
        elif meets_random_chance:
            should_respond = True
            logger.debug(f"Responding to {member.name} (random chance).")

        if not should_respond:
            # logger.debug(f"Not responding to message from {member.name} (should_respond=False).")
            return

        # === 4. Process Bot Interaction (If Applicable) ===
        logger.debug(f"Processing interaction for {member.name} in {message.channel.name}")

        # --- Rate Limiting Checks (Only if should_respond is True, skip if exempt) ---
        perform_token_check = False # Flag for later

        # Check Redis availability first
        if not self.redis_client:
            if not is_rate_limit_exempt:
                 logger.warning(f"Rate limiting checks skipped for {member.name}: Redis client unavailable.")
        
        elif not is_rate_limit_exempt and self.bot.restricted_user_role_id:
            # Only check/apply limits if not exempt AND restriction system is configured
            
            # A. Message Rate Limit Check (Apply only if not already restricted by role)
            if self.bot.rate_limit_count > 0 and not is_currently_restricted_by_role:
                msg_rl_key = f"msg_rl:{guild_id}:{user_id}"
                try:
                    self.redis_client.lpush(msg_rl_key, current_time)
                    min_time_for_window = current_time - self.bot.rate_limit_window_seconds
                    timestamps_in_list = self.redis_client.lrange(msg_rl_key, 0, -1)
                    
                    messages_in_window_timestamps = []
                    valid_timestamps_str = []
                    for ts_str in timestamps_in_list:
                        try:
                            ts = float(ts_str)
                            if ts > min_time_for_window:
                                messages_in_window_timestamps.append(ts)
                                valid_timestamps_str.append(ts_str)
                        except ValueError: logger.warning(f"Non-float value '{ts_str}' in Redis list {msg_rl_key}")
                    
                    # Prune: Replace list with only valid entries
                    pipe = self.redis_client.pipeline()
                    pipe.delete(msg_rl_key)
                    if valid_timestamps_str:
                        pipe.rpush(msg_rl_key, *valid_timestamps_str)
                        pipe.expire(msg_rl_key, self.bot.rate_limit_window_seconds + 120)
                    pipe.execute()

                    # Check limit
                    if len(messages_in_window_timestamps) > self.bot.rate_limit_count:
                        logger.info(f"User {member.name} ({user_id}) exceeded MESSAGE rate limit: {len(messages_in_window_timestamps)} msgs.")
                        await self._apply_restriction(member, message.guild, "Exceeded message rate limit")
                        notification_content = self._format_notification(
                            self.bot.rate_limit_message_user_template,
                            self.bot.restricted_channel_id
                        )
                        try:
                            if message.channel.permissions_for(message.guild.me).send_messages:
                                await message.reply(notification_content, mention_author=True)
                            else: logger.warning(f"Cannot send rate limit notification reply in {message.channel.name}: No permission.")
                        except Exception as e: logger.error(f"Failed to send rate limit notification reply: {e}", exc_info=True)
                        return # Stop processing THIS message (no LLM call)

                except redis.exceptions.RedisError as e: logger.error(f"Redis error (message rate limit) for {member.name}: {e}", exc_info=True)
                except Exception as e: logger.error(f"Unexpected error (message rate limit) for {member.name}: {e}", exc_info=True)
            
            # B. Setup Token Check Flag (Only if message limit didn't hit/apply and not already restricted)
            # Note: newly_restricted_this_message flag removed as msg limit check now returns early.
            perform_token_check = (
                self.redis_client is not None and # Double check redis client
                self.bot.token_rate_limit_count > 0 and
                self.bot.restricted_user_role_id and
                not is_currently_restricted_by_role # Only check tokens if not already restricted
                # No need to check exempt flag again as it's handled by outer 'if not is_rate_limit_exempt'
            )


        # --- Proceed with LLM Call (only if no prior return/rate limit stop) ---
        if not message.channel.permissions_for(message.guild.me).send_messages:
            logger.error(f"Cannot respond in {message.channel.name}: Missing 'Send Messages' permission.")
            return

        try:
            async with message.channel.typing():
                chat_system_prompt = self.bot.chat_system_prompt
                logger.info(f"Requesting LLM response for {member.name} in {message.channel.name}.")
                
                # Call LLM API
                response_content, error_message, tokens_used = await self.api_client.generate_response(
                    user_id, message.channel.id, content_for_llm, chat_system_prompt
                )

                # Handle API/LLM Response Errors or No Content
                if error_message or not response_content:
                    logger.error(f"API Error/No Content for {member.name}: Err='{error_message}', Content='{response_content}'")
                    reply_text = response_content or "Sorry, I encountered an error or couldn't generate a response."
                    try:
                        await message.reply(reply_text, mention_author=False) # Reply with error/status
                    except Exception as e:
                        logger.error(f"Failed sending error reply: {e}", exc_info=True)
                    return # Stop processing

                # --- Send LLM Response FIRST ---
                llm_reply_sent = False
                try:
                    final_response_content = response_content # Already stripped in api_client
                    if len(final_response_content) > 2000:
                         final_response_content = final_response_content[:1997] + "..."
                         logger.warning(f"Response for {member.name} truncated.")
                    await message.reply(final_response_content, mention_author=False) # Regular reply
                    llm_reply_sent = True
                    logger.info(f"Sent LLM reply to {member.name}. Tokens used: {tokens_used or 'Unknown'}")
                except discord.Forbidden: logger.error(f"Forbidden: Cannot send LLM reply to #{message.channel.name}.")
                except discord.HTTPException as e: logger.error(f"HTTPException sending LLM reply in #{message.channel.name}: {e}", exc_info=True)
                except Exception as e: logger.exception(f"Unexpected error sending LLM reply for {member.name}: {e}")

                # --- Perform Token Rate Limit Check (if applicable and LLM reply was successful) ---
                if llm_reply_sent and perform_token_check and tokens_used is not None and tokens_used > 0:
                    token_rl_key = f"token_rl:{guild_id}:{user_id}"
                    try:
                        self.redis_client.lpush(token_rl_key, f"{current_time}:{tokens_used}")
                        min_time_token_window = current_time - self.bot.rate_limit_window_seconds
                        
                        entries_in_list = self.redis_client.lrange(token_rl_key, 0, -1)
                        total_tokens_in_window = 0
                        valid_entries_for_trim = []
                        for entry in entries_in_list:
                            try:
                                ts_str, tk_str = entry.split(":", 1); ts = float(ts_str); tk = int(tk_str)
                                if ts > min_time_token_window:
                                    total_tokens_in_window += tk
                                    valid_entries_for_trim.append(entry)
                            except (ValueError, IndexError): logger.warning(f"Malformed entry in {token_rl_key}: {entry}")

                        # Prune token list
                        pipe_token = self.redis_client.pipeline()
                        pipe_token.delete(token_rl_key)
                        if valid_entries_for_trim:
                            pipe_token.rpush(token_rl_key, *valid_entries_for_trim)
                            pipe_token.expire(token_rl_key, self.bot.rate_limit_window_seconds + 120)
                        pipe_token.execute()

                        # Check if limit exceeded
                        if total_tokens_in_window > self.bot.token_rate_limit_count:
                            logger.info(f"User {member.name} ({user_id}) exceeded TOKEN rate limit: {total_tokens_in_window} tokens.")
                            # Apply restriction role
                            if await self._apply_restriction(member, message.guild, "Exceeded token usage rate limit"):
                                # If role was newly applied, send separate notification
                                notification_content = self._format_notification(
                                    self.bot.rate_limit_message_user_template,
                                    self.bot.restricted_channel_id
                                )
                                try:
                                    # Send as a new message in the channel, mentioning the user
                                    if message.channel.permissions_for(message.guild.me).send_messages:
                                        await message.channel.send(f"{member.mention} {notification_content}")
                                        logger.info(f"Sent separate token rate limit notification for {member.name} in #{message.channel.name}.")
                                    else: logger.warning(f"Cannot send token rate limit notification in {message.channel.name}: No permission.")
                                except Exception as e: logger.error(f"Failed to send token rate limit notification: {e}", exc_info=True)

                    except redis.exceptions.RedisError as e: logger.error(f"Redis error (token rate limit) for {member.name}: {e}", exc_info=True)
                    except Exception as e: logger.error(f"Unexpected error (token rate limit) for {member.name}: {e}", exc_info=True)

        except Exception as e: # Catch-all for the interaction block
             logger.exception(f"Outer unexpected error processing message for {member.name}: {e}")


    @commands.Cog.listener()
    async def on_member_join(self, member: discord.Member):
        # --- Welcome message logic ---
        logger.info(f"Member joined: {member.name} (ID: {member.id}) in guild {member.guild.name}")
        if not self.bot.welcome_channel_id or not self.bot.welcome_system_prompt or not self.bot.welcome_user_prompt:
            logger.debug("Welcome feature not fully configured, skipping welcome message.")
            return

        welcome_channel = self.bot.get_channel(self.bot.welcome_channel_id)
        if welcome_channel is None:
            try:
                welcome_channel = await self.bot.fetch_channel(self.bot.welcome_channel_id)
            except (discord.NotFound, discord.Forbidden):
                logger.error(f"Failed to fetch welcome channel ID: {self.bot.welcome_channel_id}. Invalid or no permissions.")
                return
            if welcome_channel is None: # Still None after fetch
                 logger.error(f"Welcome channel {self.bot.welcome_channel_id} not found even after fetch.")
                 return

        if not isinstance(welcome_channel, discord.TextChannel): # Ensure it's a text channel
             logger.error(f"Welcome channel {self.bot.welcome_channel_id} ({welcome_channel.name}) is not a text channel.")
             return

        if not welcome_channel.permissions_for(member.guild.me).send_messages:
            logger.error(f"Bot lacks 'Send Messages' permission in welcome channel: {welcome_channel.name}.")
            return

        try:
            async with welcome_channel.typing():
                welcome_message_content, error_msg = await self.api_client.generate_welcome_message(member)

                if error_msg:
                    logger.error(f"API Error generating welcome for {member.name}: {error_msg}")
                    await welcome_channel.send(f"Welcome {member.mention}! Had a bit of trouble generating a special greeting.") # Fallback user message
                    return

                if not welcome_message_content:
                    logger.warning(f"API returned no content for welcome message for {member.name}.")
                    await welcome_channel.send(f"Welcome {member.mention} to {member.guild.name}!") # Generic fallback
                    return

                # Ensure message length is valid
                if len(welcome_message_content) > 2000:
                    welcome_message_content = welcome_message_content[:1997] + "..."
                    logger.warning(f"Welcome message for {member.name} truncated.")

                await welcome_channel.send(welcome_message_content)
                logger.info(f"Sent welcome message for {member.name} to #{welcome_channel.name}.")

        except discord.Forbidden:
            logger.error(f"Forbidden: Cannot send welcome message to channel {welcome_channel.name} ({self.bot.welcome_channel_id}). Check permissions.")
        except discord.HTTPException as e:
            logger.error(f"Failed to send welcome message to channel {self.bot.welcome_channel_id}: {e}", exc_info=True)
        except Exception as e:
            logger.exception(f"An unexpected error sending welcome message for {member.name}: {e}")


async def setup(bot: 'AIBot'):
    # Basic check for redis client
    if not hasattr(bot, 'redis_client_general'):
         logger.warning("AIBot is missing 'redis_client_general'. Rate limiting features in ListenerCog may fail or be disabled.")

    await bot.add_cog(ListenerCog(bot))
    logger.info("ListenerCog added to the bot.")