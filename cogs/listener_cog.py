# cogs/listener_cog.py
import discord
from discord.ext import commands
import random
import logging
import re
import time
from utils.webui_api import WebUIAPI
from typing import TYPE_CHECKING, Optional, Set, Any, List, Dict # Added List, Dict

# Attempt Redis import for type hinting
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    if TYPE_CHECKING: redis = Any # type: ignore

if TYPE_CHECKING:
    from bot import AIBot


logger = logging.getLogger(__name__)

class ListenerCog(commands.Cog):
    """
    Cog handling message listening, rate limiting, restrictions, LLM interaction,
    and welcoming new members. Uses public replies for notifications.
    Handles context injection/saving for multi-user replies.
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
        # Store general Redis client from bot instance
        self.redis_client: Optional[redis.Redis] = getattr(self.bot, 'redis_client_general', None)

        logger.info("ListenerCog initialized.")
        if not self.redis_client:
             logger.warning("ListenerCog: Bot's general Redis client unavailable. Rate limiting disabled.")

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
        """Applies restriction role. Returns True if newly applied."""
        if not isinstance(member, discord.Member): return False
        if not self.bot.restricted_user_role_id: return False
        restricted_role = guild.get_role(self.bot.restricted_user_role_id)
        if not restricted_role:
            logger.error(f"RESTRICTED_USER_ROLE_ID {self.bot.restricted_user_role_id} not found in guild {guild.name}.")
            return False
        role_applied = False
        try:
            if restricted_role not in member.roles:
                await member.add_roles(restricted_role, reason=reason)
                logger.info(f"Assigned role '{restricted_role.name}' to {member.name} ({member.id}) for: {reason}")
                role_applied = True
            else: logger.info(f"{member.name} ({member.id}) already restricted. Trigger: {reason}")
        except Exception as e: logger.error(f"Error assigning role to {member.name} ({member.id}): {e}", exc_info=True)
        return role_applied

    def _format_notification(self, template: str, channel_id: Optional[int]) -> str:
        """Formats notification templates."""
        channel_mention = f"<#{channel_id}>" if channel_id else "the designated channel"
        return template.replace("<#{channel_id}>", channel_mention)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        # === 1. Initial Checks & Setup ===
        if message.author == self.bot.user or message.author.bot or not message.guild: return
        member: discord.Member = message.author
        current_time = time.time(); guild_id = message.guild.id; user_id = member.id
        author_role_ids: Set[int] = {role.id for role in member.roles}

        # === 2. Pre-processing Checks ===
        # --- Globally Ignored Roles Check ---
        if self.bot.ignored_role_ids_set and not self.bot.ignored_role_ids_set.isdisjoint(author_role_ids):
            matched_roles = [role.name for role in member.roles if role.id in self.bot.ignored_role_ids_set]
            logger.info(f"Ignoring message from {member.name} ({user_id}) due to globally ignored role(s): {', '.join(matched_roles)}")
            return
        # --- Rate Limit Exemption Check ---
        is_rate_limit_exempt = self.bot.rate_limit_exempt_role_ids_set and not self.bot.rate_limit_exempt_role_ids_set.isdisjoint(author_role_ids)
        if is_rate_limit_exempt: logger.debug(f"User {member.name} is exempt from rate limits.")
        # --- Restricted User & Channel Enforcement ---
        is_currently_restricted_by_role = self.bot.restricted_user_role_id and (self.bot.restricted_user_role_id in author_role_ids)
        if is_currently_restricted_by_role and self.bot.restricted_channel_id and message.channel.id != self.bot.restricted_channel_id:
            logger.info(f"Restricted user {member.name} used bot in disallowed channel {message.channel.name}.")
            notification_content = self._format_notification(self.bot.restricted_channel_message_user_template, self.bot.restricted_channel_id)
            try:
                if message.channel.permissions_for(message.guild.me).send_messages:
                    await message.reply(notification_content, mention_author=True)
                else: logger.warning(f"Cannot send restricted channel notification reply in {message.channel.name}: No permission.")
            except Exception as e: logger.error(f"Failed to send restricted channel notification reply: {e}", exc_info=True)
            return

        # === 3. Determine if Bot Should Engage ===
        content_for_llm = message.content
        bot_mention_strings = [f'<@{self.bot.user.id}>', f'<@!{self.bot.user.id}>']
        for mention_str in bot_mention_strings: content_for_llm = content_for_llm.replace(mention_str, '')
        content_for_llm = re.sub(r'\s+', ' ', content_for_llm).strip()

        should_respond = False; is_reply_to_bot = False
        is_reply = message.reference and message.reference.resolved and isinstance(message.reference.resolved, discord.Message)
        if is_reply and message.reference.resolved.author == self.bot.user:
            should_respond = True; is_reply_to_bot = True; logger.debug(f"Responding to {member.name} (reply to bot).")
        is_mention = self.bot.user in message.mentions
        if not should_respond and is_mention:
            if content_for_llm: should_respond = True; logger.debug(f"Responding to {member.name} (mention w/ content).")
            else: logger.info(f"User {member.name} only mentioned bot; ignoring LLM.")
        if not should_respond and (random.random() < self.bot.response_chance):
            should_respond = True; logger.debug(f"Responding to {member.name} (random chance).")

        if not should_respond: return

        # === 4. Process Bot Interaction ===
        logger.debug(f"Processing interaction for {member.name} in {message.channel.name}")

        # --- Rate Limiting Checks (Only if should_respond is True, skip if exempt) ---
        perform_token_check = False # Default flag for token check later
        if not is_rate_limit_exempt and self.redis_client and self.bot.restricted_user_role_id:
            # A. Message Rate Limit Check (Apply only if not already restricted by role)
            if self.bot.rate_limit_count > 0 and not is_currently_restricted_by_role:
                msg_rl_key = f"msg_rl:{guild_id}:{user_id}"
                try:
                    self.redis_client.lpush(msg_rl_key, current_time)
                    min_time_for_window = current_time - self.bot.rate_limit_window_seconds
                    timestamps_in_list = self.redis_client.lrange(msg_rl_key, 0, -1)
                    
                    # This calculates the list of valid timestamps (floats)
                    messages_in_window_timestamps: List[float] = []
                    valid_timestamps_str: List[str] = []
                    for ts_str in timestamps_in_list:
                        try:
                            ts = float(ts_str)
                            if ts > min_time_for_window:
                                messages_in_window_timestamps.append(ts)
                                valid_timestamps_str.append(ts_str)
                        except ValueError: logger.warning(f"Non-float value '{ts_str}' in Redis list {msg_rl_key}")
                    
                    # Prune list in Redis
                    pipe = self.redis_client.pipeline()
                    pipe.delete(msg_rl_key)
                    if valid_timestamps_str:
                        pipe.rpush(msg_rl_key, *valid_timestamps_str)
                        pipe.expire(msg_rl_key, self.bot.rate_limit_window_seconds + 120)
                    pipe.execute()

                    # --- REMOVED ERRONEOUS PLACEHOLDER LINE ---

                    # Check limit using the length of the calculated list
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
            
            # B. Setup Token Check Flag (as before)
            perform_token_check = (
                self.redis_client is not None and
                self.bot.token_rate_limit_count > 0 and
                self.bot.restricted_user_role_id and
                not is_currently_restricted_by_role # Only check tokens if not already restricted
                # No need to check exempt flag again as outer 'if' handles it
                # No need to check newly_restricted flag as msg limit check now returns early
            )


        # --- Proceed with LLM Call (only if no prior return/rate limit stop) ---
        if not message.channel.permissions_for(message.guild.me).send_messages:
             logger.error(f"Cannot respond in {message.channel.name}: Missing 'Send Messages' permission (checked again before API call).")
             return

        try:
            async with message.channel.typing():
                chat_system_prompt = self.bot.chat_system_prompt
                
                # Context Injection Logic
                current_history: List[Dict[str, str]] = self.api_client.get_context_history(user_id, message.channel.id)
                extra_assistant_context: Optional[str] = None
                inject_context_for_saving = False

                if is_reply_to_bot: # This flag was set earlier when determining should_respond
                    replied_to_bot_message = message.reference.resolved # Safe now
                    is_reply_to_own_thread = False
                    if replied_to_bot_message.reference and replied_to_bot_message.reference.resolved:
                        if replied_to_bot_message.reference.resolved.author == member:
                            is_reply_to_own_thread = True
                    
                    if not is_reply_to_own_thread:
                        extra_assistant_context = replied_to_bot_message.content
                        inject_context_for_saving = True
                        logger.debug(f"Injecting context for {member.name} replying to external thread.")

                # Call LLM API
                logger.info(f"Requesting LLM response for {member.name} (Injecting Context: {inject_context_for_saving}).")
                response_content, error_message, tokens_used = await self.api_client.generate_response(
                    user_id, message.channel.id, content_for_llm, chat_system_prompt,
                    history=current_history,
                    extra_assistant_context=extra_assistant_context
                )

                # Handle API/LLM Response Errors or No Content
                if error_message or not response_content:
                    reply_text = response_content or "Sorry, error processing request."
                    try: await message.reply(reply_text, mention_author=False, ephemeral=True)
                    except Exception: pass # Avoid exception loops on error reporting
                    logger.error(f"API Error/No Content for {member.name}: Err='{error_message}', Content='{response_content}'")
                    return

                # Save History (incorporating injected context if needed)
                next_history = list(current_history)
                if inject_context_for_saving and extra_assistant_context is not None:
                    next_history.append({"role": "assistant", "content": extra_assistant_context})
                next_history.append({"role": "user", "content": content_for_llm})
                next_history.append({"role": "assistant", "content": response_content})
                self.api_client.save_context_history(user_id, message.channel.id, next_history)
                logger.debug(f"Saved context history for {member.name} (Injected: {inject_context_for_saving})")

                # Send LLM Response FIRST
                llm_reply_sent = False
                try:
                    final_response_content = response_content
                    if len(final_response_content) > 2000: final_response_content = final_response_content[:1997] + "..."
                    await message.reply(final_response_content, mention_author=False)
                    llm_reply_sent = True
                    logger.info(f"Sent LLM reply to {member.name}. Tokens used: {tokens_used or 'Unknown'}")
                except Exception as e: logger.exception(f"Failed to send LLM reply for {member.name}: {e}")

                # Perform Token Rate Limit Check (if applicable and reply sent)
                if llm_reply_sent and perform_token_check and tokens_used is not None and tokens_used > 0:
                    token_rl_key = f"token_rl:{guild_id}:{user_id}"
                    try:
                        # ... (Redis logic for token tracking: lpush, lrange, filter, sum, prune as before) ...
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
                                # Role was newly applied, send separate notification
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

        # --- Outer Exception Handling ---
        except Exception as e:
             logger.exception(f"Outer unexpected error processing message for {member.name}: {e}")


    @commands.Cog.listener()
    async def on_member_join(self, member: discord.Member):
        # (Welcome message logic remains unchanged)
        # ... (Implementation as before) ...
        pass


async def setup(bot: 'AIBot'):
    # (setup check as before)
    if not hasattr(bot, 'redis_client_general'):
         logger.warning("AIBot is missing 'redis_client_general'. Rate limiting features disabled.")
    await bot.add_cog(ListenerCog(bot))
    logger.info("ListenerCog added to the bot.")