# cogs/listener_cog.py
import discord
from discord.ext import commands, tasks
import random
import logging
import re
import time
import asyncio # For asyncio.to_thread
from utils.webui_api import WebUIAPI # Assuming this is in your utils directory
from typing import TYPE_CHECKING, Optional, Set, Any, List, Dict

# Attempt Redis import for type hinting
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    if TYPE_CHECKING: redis = Any # type: ignore

if TYPE_CHECKING:
    from bot import AIBot # Ensure this import points to your AIBot class in bot.py


logger = logging.getLogger(__name__)

class ListenerCog(commands.Cog):
    """
    Cog handling message listening, rate limiting, restrictions, LLM interaction,
    welcoming new members, and automatic restriction expiry.
    """
    def __init__(self, bot: 'AIBot'):
        self.bot = bot
        self.api_client = WebUIAPI(
            base_url=self.bot.api_url,
            model=self.bot.model,
            api_key=self.bot.api_key,
            welcome_system=self.bot.welcome_system_prompt,
            welcome_prompt=self.bot.welcome_user_prompt,
            max_history_per_user=self.bot.max_history_per_context,
            knowledge_id=self.bot.knowledge_id,
            list_tools=self.bot.list_tools,
            redis_config=self.bot.redis_config
        )
        self.redis_client: Optional[redis.Redis] = getattr(self.bot, 'redis_client_general', None)

        logger.info("ListenerCog initialized.")
        if not self.redis_client:
             logger.warning("ListenerCog: Bot's general Redis client unavailable. Rate limiting and restriction expiry disabled.")

        if self.bot.ignored_role_ids_set:
            logger.info(f"ListenerCog will globally ignore users with roles: {self.bot.ignored_role_ids_set}")
        if self.bot.rate_limit_exempt_role_ids_set:
            logger.info(f"ListenerCog: Rate limits EXEMPT for Role IDs: {self.bot.rate_limit_exempt_role_ids_set}")

        # Start restriction expiry loop if configured
        if self.bot.restricted_user_role_id: # Prerequisite for expiry system
            logger.info(f"Restriction system configured: RoleID={self.bot.restricted_user_role_id}, ChannelID={self.bot.restricted_channel_id}")
            logger.info(f"Msg Limit: {self.bot.rate_limit_count}/{self.bot.rate_limit_window_seconds}s, Token Limit: {self.bot.token_rate_limit_count}/{self.bot.rate_limit_window_seconds}s")
            if self.bot.restriction_duration_seconds > 0 and self.bot.restriction_check_interval_seconds > 0:
                if self.redis_client: # Only start if redis is available
                    self.check_restrictions_loop.change_interval(seconds=self.bot.restriction_check_interval_seconds)
                    self.check_restrictions_loop.start()
                    logger.info(f"Automatic restriction expiry check loop started. Duration: {self.bot.restriction_duration_seconds}s, Check Interval: {self.bot.restriction_check_interval_seconds}s")
                else:
                    logger.warning("Automatic restriction expiry configured but Redis is unavailable. Loop not started.")
            elif self.bot.restriction_duration_seconds > 0 :
                 logger.warning(f"Automatic restriction expiry duration is {self.bot.restriction_duration_seconds}s but check interval is not valid ({self.bot.restriction_check_interval_seconds}s). Loop not started.")
            else:
                logger.info("Automatic restriction expiry is disabled (duration is 0 or not set).")
        else:
            logger.info("Restriction system (role assignment) not fully configured (RESTRICTED_USER_ROLE_ID not set). Automatic expiry disabled.")


    async def _apply_restriction(self, member: discord.Member, guild: discord.Guild, reason: str) -> bool:
        """
        Applies the restricted role to a member and sets an expiry in Redis if configured.
        Returns True if the user has the role after this function (either newly applied or already had it and expiry was set/updated).
        Returns False if the role could not be applied or restriction system is disabled.
        """
        if not isinstance(member, discord.Member):
            logger.error(f"_apply_restriction called with non-Member object: {type(member)}")
            return False
        if not self.bot.restricted_user_role_id:
            logger.debug("Restriction system disabled (no RESTRICTED_USER_ROLE_ID). Cannot apply restriction.")
            return False

        restricted_role = guild.get_role(self.bot.restricted_user_role_id)
        if not restricted_role:
            logger.error(f"RESTRICTED_USER_ROLE_ID {self.bot.restricted_user_role_id} not found in guild {guild.name}. Cannot apply restriction.")
            return False

        user_has_role_after_action = False
        try:
            if restricted_role not in member.roles:
                await member.add_roles(restricted_role, reason=reason)
                logger.info(f"Assigned role '{restricted_role.name}' to {member.name} ({member.id}) for: {reason}")
                user_has_role_after_action = True
            else:
                logger.info(f"{member.name} ({member.id}) already had restricted role '{restricted_role.name}'. Reason for check: {reason}")
                user_has_role_after_action = True # User still has the role

        except discord.Forbidden:
            logger.error(f"Bot lacks permissions to assign role '{restricted_role.name}' to {member.name} in {guild.name}.")
            return False # Failed to apply/confirm role
        except discord.HTTPException as e:
            logger.error(f"Failed to assign role to {member.name} ({member.id}) due to HTTP error: {e}")
            return False # Failed to apply/confirm role
        except Exception as e:
            logger.error(f"Unexpected error assigning role to {member.name} ({member.id}): {e}", exc_info=True)
            return False

        # If role was successfully applied or confirmed, and auto-expiry is enabled
        if user_has_role_after_action and self.bot.restriction_duration_seconds > 0 and self.redis_client:
            try:
                expiry_timestamp = time.time() + self.bot.restriction_duration_seconds
                redis_key = f"restricted_until:{guild.id}:{member.id}"
                # Set the expiry time in Redis, with a Redis TTL slightly longer than the duration for cleanup
                self.redis_client.set(redis_key, expiry_timestamp, ex=self.bot.restriction_duration_seconds + 3600) # Add 1hr buffer to key TTL
                logger.info(f"Restriction expiry set for {member.name} ({member.id}) until {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(expiry_timestamp))} (Key: {redis_key})")
            except redis.exceptions.RedisError as e:
                logger.error(f"Redis error setting restriction expiry for {member.name} ({member.id}): {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Unexpected error setting restriction expiry for {member.name} ({member.id}): {e}", exc_info=True)
        elif user_has_role_after_action and self.bot.restriction_duration_seconds > 0 and not self.redis_client:
            logger.warning(f"Cannot set restriction expiry for {member.name}; Redis client unavailable.")

        return user_has_role_after_action


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
        
        logger.debug(f"Message from {member.name} ({user_id}) in guild {guild_id}, channel {message.channel.id}. Content: '{message.content[:50]}...'")

        # === 2. Pre-processing Checks (Initial Filters) ===
        if self.bot.ignored_role_ids_set and not self.bot.ignored_role_ids_set.isdisjoint(author_role_ids):
            logger.debug(f"Message from {member.name} ignored due to globally ignored role(s).")
            return
        is_rate_limit_exempt = self.bot.rate_limit_exempt_role_ids_set and not self.bot.rate_limit_exempt_role_ids_set.isdisjoint(author_role_ids)
        if is_rate_limit_exempt: logger.debug(f"User {member.name} is exempt from rate limits.")

        # NOTE: is_currently_restricted_by_role check is MOVED to after should_respond determination

        # === 3. Determine if Bot Should Engage ===
        content_for_llm = message.content
        bot_mention_strings = [f'<@{self.bot.user.id}>', f'<@!{self.bot.user.id}>']
        for mention_str in bot_mention_strings: content_for_llm = content_for_llm.replace(mention_str, '')
        content_for_llm = re.sub(r'\s+', ' ', content_for_llm).strip()

        should_respond = False
        interaction_case_debug = "No Interaction Triggered" 
        is_reply = message.reference and message.reference.resolved and isinstance(message.reference.resolved, discord.Message)
        is_reply_to_bot = False
        is_mention = self.bot.user in message.mentions
        
        logger.debug(f"Engagement check for {member.name}: is_reply={is_reply}, is_mention={is_mention}, is_reply_to_bot_initially={is_reply_to_bot}, content_for_llm='{content_for_llm[:30]}...'")

        if is_reply:
            if message.reference.resolved.author == self.bot.user:
                is_reply_to_bot = True; should_respond = True
                interaction_case_debug = "Reply to Bot (Case 2 or 3 pending sub-evaluation)"
                logger.debug(f"Engagement: {member.name} replied to bot. Setting should_respond=True. ({interaction_case_debug})")
            elif is_mention and content_for_llm: 
                should_respond = True
                interaction_case_debug = "USE CASE 4: Reply to User + Mention"
                logger.debug(f"Engagement: {member.name} replied to another user AND mentioned bot with content. Setting should_respond=True. ({interaction_case_debug})")

        if not should_respond and is_mention: 
            if content_for_llm:
                if not is_reply:
                    should_respond = True
                    interaction_case_debug = "USE CASE 1: Direct Mention"
                    logger.debug(f"Engagement: {member.name} directly mentioned bot with content (not a reply). Setting should_respond=True. ({interaction_case_debug})")
            else:
                interaction_case_debug = "Mention Only (No Content)"
                logger.info(f"Engagement: {member.name} only mentioned bot without additional content; ignoring LLM. ({interaction_case_debug})")

        if not should_respond and (random.random() < self.bot.response_chance):
            should_respond = True
            interaction_case_debug = "Random Chance"
            logger.debug(f"Engagement: {member.name} triggered response by random chance. Setting should_respond=True. ({interaction_case_debug})")
        
        # === Moved Restricted Channel Check ===
        # This check now only applies if the bot was going to respond anyway.
        is_currently_restricted_by_role = self.bot.restricted_user_role_id and (self.bot.restricted_user_role_id in author_role_ids) # [cite: 1]
        if is_currently_restricted_by_role: logger.debug(f"User {member.name} has restricted role.") # [cite: 1]

        if should_respond and is_currently_restricted_by_role and \
           self.bot.restricted_channel_id and message.channel.id != self.bot.restricted_channel_id: # [cite: 1]
            logger.info(f"Restricted user {member.name} attempted to interact with bot in disallowed channel {message.channel.name}. Notifying and returning.") # [cite: 1]
            notification_content = self._format_notification(self.bot.restricted_channel_message_user_template, self.bot.restricted_channel_id) # [cite: 1]
            try:
                if message.channel.permissions_for(message.guild.me).send_messages: await message.reply(notification_content, mention_author=True) # [cite: 1]
            except Exception as e: logger.error(f"Failed to send restricted channel notification reply: {e}", exc_info=True) # [cite: 1]
            return # Return AFTER sending notification, but BEFORE rate limiting or LLM call for this interaction type. [cite: 1]
        
        # Fallback: If bot wasn't going to respond, exit now.
        if not should_respond:
            logger.debug(f"Engagement: No interaction criteria met for {member.name}. Final interaction_case_debug: {interaction_case_debug}. Returning.")
            return

        # === 4. Process Bot Interaction (Rate Limiting) ===
        # This section is reached only if should_respond is True AND the restricted channel check (if applicable) didn't return.
        logger.debug(f"Processing interaction for {member.name} (Interaction type initially flagged as: {interaction_case_debug})")
        perform_token_check = False
        if not is_rate_limit_exempt and self.redis_client and self.bot.restricted_user_role_id:
            # Message rate limit check still only applies if user is NOT YET restricted by role.
            # If they become restricted here, subsequent token checks for THIS message might be skipped by perform_token_check logic.
            if self.bot.rate_limit_count > 0 and not is_currently_restricted_by_role:
                msg_rl_key = f"msg_rl:{guild_id}:{user_id}"
                try:
                    self.redis_client.lpush(msg_rl_key, current_time)
                    min_time_for_window = current_time - self.bot.rate_limit_window_seconds
                    timestamps_in_list = self.redis_client.lrange(msg_rl_key, 0, -1)
                    messages_in_window_timestamps: List[float] = []
                    valid_timestamps_str: List[str] = []
                    for ts_str in timestamps_in_list:
                        try: ts = float(ts_str)
                        except ValueError: logger.warning(f"Non-float value '{ts_str}' in Redis list {msg_rl_key}"); continue
                        if ts > min_time_for_window:
                            messages_in_window_timestamps.append(ts)
                            valid_timestamps_str.append(ts_str) 
                    pipe = self.redis_client.pipeline()
                    pipe.delete(msg_rl_key)
                    if valid_timestamps_str: pipe.rpush(msg_rl_key, *valid_timestamps_str)
                    pipe.expire(msg_rl_key, self.bot.rate_limit_window_seconds + 120)
                    pipe.execute()
                    if len(messages_in_window_timestamps) > self.bot.rate_limit_count:
                        logger.info(f"User {member.name} ({user_id}) exceeded MESSAGE rate limit ({len(messages_in_window_timestamps)}/{self.bot.rate_limit_count}). Applying restriction.")
                        # If _apply_restriction is successful, user becomes is_currently_restricted_by_role for subsequent checks
                        if await self._apply_restriction(member, message.guild, "Exceeded message rate limit"):
                            is_currently_restricted_by_role = True # Update local flag
                            notification_content = self._format_notification(self.bot.rate_limit_message_user_template, self.bot.restricted_channel_id)
                            try:
                                if message.channel.permissions_for(message.guild.me).send_messages: await message.reply(notification_content, mention_author=True)
                            except Exception as e: logger.error(f"Failed to send rate limit notification reply: {e}", exc_info=True)
                        return
                except Exception as e: logger.error(f"Error in message rate limit for {member.name}: {e}", exc_info=True)
            
            # Token check should only be performed if user isn't ALREADY restricted (either from before this message, or by message limit above)
            perform_token_check = (self.bot.token_rate_limit_count > 0 and not is_currently_restricted_by_role)


        # === 5. LLM Call & Context Handling ===
        if not message.channel.permissions_for(message.guild.me).send_messages:
             logger.error(f"Cannot respond in {message.channel.name}: Missing 'Send Messages' permission (final check).")
             return
        try:
            async with message.channel.typing():
                chat_system_prompt = self.bot.chat_system_prompt
                current_history: List[Dict[str, str]] = self.api_client.get_context_history(user_id, message.channel.id)
                extra_assistant_context: Optional[str] = None
                inject_context_for_saving = False
                final_interaction_case_log = interaction_case_debug 

                if interaction_case_debug == "USE CASE 4: Reply to User + Mention":
                    replied_to_user_message = message.reference.resolved 
                    if replied_to_user_message and replied_to_user_message.content:
                        context_prefix = f"Context from reply to {replied_to_user_message.author.display_name} (User ID: {replied_to_user_message.author.id}):"
                        extra_assistant_context = f"{context_prefix}\n```\n{replied_to_user_message.content}\n```"
                        inject_context_for_saving = True
                        logger.debug(f"CONTEXT LOGIC ({member.name}): Determined USE CASE 4. Injecting content from {replied_to_user_message.author.name}.")
                    else:
                         logger.warning(f"CONTEXT LOGIC ({member.name}): Determined USE CASE 4 but replied-to user message content missing.")
                
                elif is_reply_to_bot: 
                    replied_to_bot_message = message.reference.resolved
                    is_continuing_own_direct_thread = True 
                    current_case_for_this_block = "Reply to Bot (Sub-case Undetermined)"

                    if replied_to_bot_message.reference and replied_to_bot_message.reference.resolved:
                        original_recipient_of_bot_reply = replied_to_bot_message.reference.resolved.author
                        if original_recipient_of_bot_reply != member:
                            is_continuing_own_direct_thread = False
                            current_case_for_this_block = "USE CASE 3: Reply to Bot (Bot's msg was reply to other)"
                            logger.debug(f"CONTEXT LOGIC ({member.name}): {current_case_for_this_block}. Bot's message was reply to {original_recipient_of_bot_reply.name} (ID: {original_recipient_of_bot_reply.id}).")
                        else:
                            current_case_for_this_block = "USE CASE 2: Reply to Bot (Bot's msg was reply to self)"
                            logger.debug(f"CONTEXT LOGIC ({member.name}): {current_case_for_this_block}.")
                    
                    elif not replied_to_bot_message.reference or \
                         (replied_to_bot_message.reference and not replied_to_bot_message.reference.resolved):
                        
                        log_prefix_detail = "(Bot's msg not formal reply)"
                        if replied_to_bot_message.reference and not replied_to_bot_message.reference.resolved:
                            log_prefix_detail = "(Bot's msg was reply, but original unresolved)"
                            logger.warning(f"CONTEXT LOGIC ({member.name}): Bot's replied-to message's own reference could not be resolved. Message ID: {replied_to_bot_message.id}, Its Ref Message ID: {replied_to_bot_message.reference.message_id if replied_to_bot_message.reference else 'N/A'}")
                        
                        last_bot_message_in_member_history_content = None
                        if current_history: 
                            for i in range(len(current_history) - 1, -1, -1):
                                if current_history[i]['role'] == 'assistant':
                                    last_bot_message_in_member_history_content = current_history[i]['content']
                                    break
                        
                        logger.debug(f"CONTEXT COMPARISON ({member.name}) {log_prefix_detail}: Last bot msg in own history: '{str(last_bot_message_in_member_history_content)[:50]}...'")
                        logger.debug(f"CONTEXT COMPARISON ({member.name}) {log_prefix_detail}: Replied-to bot msg content: '{replied_to_bot_message.content[:50]}...'")

                        if last_bot_message_in_member_history_content != replied_to_bot_message.content:
                            is_continuing_own_direct_thread = False
                            current_case_for_this_block = f"USE CASE 3: Reply to Bot {log_prefix_detail}, content differs)"
                            logger.debug(f"CONTEXT LOGIC ({member.name}): {current_case_for_this_block}.")
                        else:
                            current_case_for_this_block = f"USE CASE 2: Reply to Bot {log_prefix_detail}, content matches)"
                            logger.debug(f"CONTEXT LOGIC ({member.name}): {current_case_for_this_block}.")
                    
                    final_interaction_case_log = current_case_for_this_block 

                    if not is_continuing_own_direct_thread: 
                        extra_assistant_context = replied_to_bot_message.content
                        inject_context_for_saving = True
                
                elif final_interaction_case_log == "USE CASE 1: Direct Mention" or final_interaction_case_log == "Random Chance":
                     logger.debug(f"CONTEXT LOGIC ({member.name}): Confirmed {final_interaction_case_log}. No special context injection beyond history.")

                logger.info(f"Requesting LLM response for {member.name} (Final Case: {final_interaction_case_log}, Extra LLM Context: {extra_assistant_context is not None}, Saving Injected: {inject_context_for_saving}).")
                response_content, error_message, tokens_used = await self.api_client.generate_response(
                    user_id, message.channel.id, content_for_llm, chat_system_prompt,
                    history=current_history, extra_assistant_context=extra_assistant_context)

                if error_message or not response_content:
                    reply_text = response_content or "Sorry, error processing request."
                    try: await message.reply(reply_text, mention_author=False, ephemeral=True)
                    except Exception: logger.error(f"Failed to send error feedback reply to {member.name}.")
                    logger.error(f"API Error/No Content for {member.name}: Err='{error_message}', Content Null/Empty='{not response_content}' (Case: {final_interaction_case_log})")
                    return

                next_history = list(current_history)
                if inject_context_for_saving and extra_assistant_context is not None:
                    next_history.append({"role": "assistant", "content": extra_assistant_context})
                next_history.append({"role": "user", "content": content_for_llm})
                next_history.append({"role": "assistant", "content": response_content})
                self.api_client.save_context_history(user_id, message.channel.id, next_history)
                logger.debug(f"Saved context history for {member.name} (Injected Saved: {inject_context_for_saving and extra_assistant_context is not None}, Case: {final_interaction_case_log})")

                llm_reply_sent = False
                try:
                    final_response_content = response_content
                    if len(final_response_content) > 2000: final_response_content = final_response_content[:1997] + "..."
                    await message.reply(final_response_content, mention_author=False)
                    llm_reply_sent = True
                except Exception as e: logger.exception(f"Failed to send LLM reply for {member.name} (Case: {final_interaction_case_log}): {e}")

                if llm_reply_sent and perform_token_check and self.redis_client and tokens_used is not None and tokens_used > 0:
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
                        pipe_token = self.redis_client.pipeline()
                        pipe_token.delete(token_rl_key)
                        if valid_entries_for_trim: pipe_token.rpush(token_rl_key, *valid_entries_for_trim)
                        pipe_token.expire(token_rl_key, self.bot.rate_limit_window_seconds + 120)
                        pipe_token.execute()
                        if total_tokens_in_window > self.bot.token_rate_limit_count:
                            logger.info(f"User {member.name} ({user_id}) exceeded TOKEN rate limit ({total_tokens_in_window}/{self.bot.token_rate_limit_count}). Applying restriction.")
                            if await self._apply_restriction(member, message.guild, "Exceeded token usage rate limit"):
                                # User is now restricted, update local flag for any subsequent logic in this same event, though unlikely needed here
                                # is_currently_restricted_by_role = True # Not strictly needed as we usually return or complete after this
                                notification_content = self._format_notification(self.bot.rate_limit_message_user_template, self.bot.restricted_channel_id)
                                try:
                                    if message.channel.permissions_for(message.guild.me).send_messages: await message.channel.send(f"{member.mention} {notification_content}")
                                except Exception as e: logger.error(f"Failed to send token rate limit notification: {e}", exc_info=True)
                    except Exception as e: logger.error(f"Error in token rate limit for {member.name}: {e}", exc_info=True)
        except Exception as e:
             logger.exception(f"Outer unexpected error processing message for {member.name} (Initial Case Flag: {interaction_case_debug}): {e}")

    @tasks.loop(seconds=300) 
    async def check_restrictions_loop(self):
        if not self.redis_client or not self.bot.restricted_user_role_id or self.bot.restriction_duration_seconds <= 0:
            if self.check_restrictions_loop.is_running():
                 logger.debug("Restriction loop: Conditions not met (Redis, RoleID, or Duration <= 0). Skipping check.")
            return

        logger.debug("Restriction loop: Checking for expired restrictions...")
        pattern = f"restricted_until:*:*"
        try:
            current_scan_cursor = 0 
            while True:
                if not self.redis_client:
                    logger.error("Restriction loop: Redis client became unavailable during scan.")
                    break
                
                next_cursor_int, keys_batch_strings = await asyncio.to_thread(
                    self.redis_client.scan, current_scan_cursor, match=pattern, count=100
                )
                
                for key in keys_batch_strings: 
                    try:
                        parts = key.split(':')
                        if len(parts) != 3:
                            logger.warning(f"Malformed restriction key found in Redis: {key}. Skipping.")
                            continue
                        
                        guild_id_str, user_id_str = parts[1], parts[2]
                        guild_id = int(guild_id_str)
                        user_id = int(user_id_str)

                        expiry_timestamp_str = await asyncio.to_thread(self.redis_client.get, key)
                        if not expiry_timestamp_str: 
                            continue
                        
                        expiry_timestamp = float(expiry_timestamp_str)

                        if time.time() >= expiry_timestamp:
                            logger.info(f"Restriction expired for user {user_id} in guild {guild_id}. Key: {key}")
                            guild = self.bot.get_guild(guild_id)
                            if not guild:
                                logger.warning(f"Could not find guild {guild_id} for restriction removal of user {user_id}. Deleting key {key}.")
                                await asyncio.to_thread(self.redis_client.delete, key)
                                continue

                            member_obj = guild.get_member(user_id) 
                            if not member_obj:
                                logger.warning(f"Could not find member {user_id} in guild {guild.name} for restriction removal. Deleting key {key}.")
                                await asyncio.to_thread(self.redis_client.delete, key)
                                continue

                            restricted_role = guild.get_role(self.bot.restricted_user_role_id)
                            if not restricted_role:
                                logger.error(f"Could not find restricted role ID {self.bot.restricted_user_role_id} in guild {guild.name}. Cannot remove role for user {user_id}.")
                                continue

                            if restricted_role in member_obj.roles:
                                try:
                                    await member_obj.remove_roles(restricted_role, reason="Restriction period expired.")
                                    logger.info(f"Successfully removed restricted role from {member_obj.name} ({member_obj.id}) in {guild.name}.")
                                except discord.Forbidden:
                                    logger.error(f"Bot lacks permissions to remove role '{restricted_role.name}' from {member_obj.name} in {guild.name}.")
                                except discord.HTTPException as e_http:
                                    logger.error(f"Failed to remove role from {member_obj.name} ({member_obj.id}) due to HTTP error: {e_http}")
                                else: 
                                    await asyncio.to_thread(self.redis_client.delete, key) 
                            else: 
                                logger.info(f"User {member_obj.name} ({member_obj.id}) in {guild.name} no longer had restricted role {restricted_role.name}, but expiry key {key} existed. Deleting key.")
                                await asyncio.to_thread(self.redis_client.delete, key)
                    
                    except ValueError: 
                        logger.warning(f"Could not parse IDs/timestamp from key '{key}' or its value. Skipping.")
                    except Exception as e_key_proc: 
                        logger.error(f"Error processing individual restriction key {key}: {e_key_proc}", exc_info=True)
                
                current_scan_cursor = next_cursor_int 
                if current_scan_cursor == 0: 
                    break
        
        except redis.exceptions.RedisError as e_redis_scan: 
            logger.error(f"Redis error during restriction check scan: {e_redis_scan}", exc_info=True)
        except Exception as e_loop: 
            logger.error(f"Unexpected error in check_restrictions_loop: {e_loop}", exc_info=True)

    @check_restrictions_loop.before_loop
    async def before_check_restrictions_loop(self):
        logger.info("Restriction expiry check loop: Waiting for bot to be ready...")
        await self.bot.wait_until_ready()
        logger.info("Restriction expiry check loop: Bot is ready. Loop will start if conditions met.")

    @commands.Cog.listener()
    async def on_member_join(self, member: discord.Member):
        if not self.bot.welcome_channel_id: return
        welcome_channel = member.guild.get_channel(self.bot.welcome_channel_id)
        if not welcome_channel or not isinstance(welcome_channel, discord.TextChannel): return 
        if not welcome_channel.permissions_for(member.guild.me).send_messages: return 
        logger.info(f"Generating welcome message for {member.name} ({member.id}) in {member.guild.name}")
        try:
            async with welcome_channel.typing():
                response_content, error_message = await self.api_client.generate_welcome_message(member)
            if response_content:
                await welcome_channel.send(response_content[:2000])
                logger.info(f"Sent welcome message for {member.name}.")
            else: logger.error(f"Failed to generate welcome message for {member.name}. Error: {error_message}")
        except Exception as e: logger.exception(f"Error during welcome message for {member.name}: {e}")

async def setup(bot: 'AIBot'):
    if not hasattr(bot, 'redis_client_general'):
         logger.warning("AIBot is missing 'redis_client_general'. Rate limiting and restriction expiry features may be impacted.")
    await bot.add_cog(ListenerCog(bot))
    logger.info("ListenerCog added to the bot.")