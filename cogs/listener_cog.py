# cogs/listener_cog.py
import discord
from discord.ext import commands, tasks
import logging
import time 
import redis 
from typing import TYPE_CHECKING, Optional, Set, Any, List, Dict
import datetime # For formatting timestamps and time differences

from utils.message_handler import MessageHandler, MessageHandlerResult 

if TYPE_CHECKING:
    from bot import AIBot 

logger = logging.getLogger(__name__)

class ListenerCog(commands.Cog):
    # ... (other parts of the class: __init__, _apply_restriction, _format_notification, on_message) ...
    # Ensure those parts are the same as the last correct version.
    def __init__(self, bot: 'AIBot'):
        self.bot: 'AIBot' = bot
        self.redis_client: Optional[redis.Redis] = self.bot.redis_client_general # type: ignore

        logger.info("ListenerCog initialized.")
        if not self.redis_client:
             logger.warning("ListenerCog: Bot's general Redis client unavailable. Rate limiting and restriction expiry might be affected if they rely on it here.")

        if self.bot.restricted_user_role_id:
            logger.info(f"ListenerCog: Restriction system configured: RoleID={self.bot.restricted_user_role_id}, ChannelID={self.bot.restricted_channel_id}")
            if self.bot.restriction_duration_seconds > 0 and self.bot.restriction_check_interval_seconds > 0:
                if self.redis_client:
                    self.check_restrictions_loop.change_interval(seconds=self.bot.restriction_check_interval_seconds)
                    self.check_restrictions_loop.start()
                    logger.info(f"ListenerCog: Automatic restriction expiry check loop started. Duration: {self.bot.restriction_duration_seconds}s, Interval: {self.bot.restriction_check_interval_seconds}s")
                else:
                    logger.warning("ListenerCog: Automatic restriction expiry configured but general Redis client is unavailable. Loop not started.")
            elif self.bot.restriction_duration_seconds > 0 : 
                 logger.warning(f"ListenerCog: Automatic restriction expiry duration is {self.bot.restriction_duration_seconds}s but check interval is not valid ({self.bot.restriction_check_interval_seconds}s). Loop not started.")
            else: 
                logger.info("ListenerCog: Automatic restriction expiry is disabled (duration is 0 or not set).")
        else:
            logger.info("ListenerCog: Restriction system (role assignment) not fully configured. Automatic expiry disabled.")

    async def _apply_restriction(self, member: discord.Member, guild: discord.Guild, reason: str) -> bool:
        if not self.bot.restricted_user_role_id:
            logger.debug("ListenerCog: Restriction system disabled (no RESTRICTED_USER_ROLE_ID). Cannot apply restriction.")
            return False

        restricted_role = guild.get_role(self.bot.restricted_user_role_id)
        if not restricted_role:
            logger.error(f"ListenerCog: RESTRICTED_USER_ROLE_ID {self.bot.restricted_user_role_id} not found in guild {guild.name}. Cannot apply restriction.")
            return False
        
        user_has_role_after_action = False
        try:
            if restricted_role not in member.roles:
                await member.add_roles(restricted_role, reason=reason)
                logger.info(f"ListenerCog: Assigned role '{restricted_role.name}' to {member.name} ({member.id}) for: {reason}")
                user_has_role_after_action = True
            else:
                logger.info(f"ListenerCog: {member.name} ({member.id}) already had restricted role '{restricted_role.name}'. Reason for check: {reason}")
                user_has_role_after_action = True
        except discord.Forbidden:
            logger.error(f"ListenerCog: Bot lacks permissions to assign role '{restricted_role.name}' to {member.name} in {guild.name}.")
            return False
        except discord.HTTPException as e:
            logger.error(f"ListenerCog: Failed to assign role to {member.name} ({member.id}) due to HTTP error: {e}")
            return False
        
        if user_has_role_after_action and self.bot.restriction_duration_seconds > 0 and self.redis_client:
            try:
                expiry_timestamp = time.time() + self.bot.restriction_duration_seconds
                redis_key = f"restricted_until:{guild.id}:{member.id}"
                await discord.utils.asyncio.to_thread(self.redis_client.set, redis_key, expiry_timestamp, ex=self.bot.restriction_duration_seconds + 3600)
                logger.info(f"ListenerCog: Restriction expiry set for {member.name} ({member.id}) until {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(expiry_timestamp))} (Key: {redis_key})")
            except Exception as e: 
                logger.error(f"ListenerCog: Redis error setting restriction expiry for {member.name} ({member.id}): {e}", exc_info=True)
        elif user_has_role_after_action and self.bot.restriction_duration_seconds > 0 and not self.redis_client:
             logger.warning(f"ListenerCog: Cannot set restriction expiry for {member.name}; Redis client (general) unavailable.")
        
        return user_has_role_after_action

    def _format_notification(self, template: str, channel_id: Optional[int]) -> str:
        channel_mention = f"<#{channel_id}>" if channel_id else "the designated channel"
        return template.replace("<#{channel_id}>", channel_mention)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author == self.bot.user or message.author.bot or not message.guild:
            return
        
        if not isinstance(message.author, discord.Member):
            logger.debug(f"ListenerCog: Message author {message.author.id} is not a Member instance. Skipping.")
            return

        handler = MessageHandler(self.bot, message)
        result: MessageHandlerResult
        
        try:
            result = await handler.process()
        except Exception as e:
            logger.error(f"ListenerCog: Unhandled exception during MessageHandler.process() for {message.author.name}: {e}", exc_info=True)
            try:
                await message.reply("Sorry, an unexpected error occurred while I was thinking.", mention_author=False)
            except Exception:
                pass 
            return

        if result.get("log_message"):
            logger.info(f"ListenerCog: From MessageHandler for {message.author.name} - {result['log_message']}")

        action = result.get("action")

        if action == "reply":
            content = result.get("content")
            if content:
                try:
                    final_response_content = content
                    if len(final_response_content) > 2000: 
                        final_response_content = final_response_content[:1997] + "..."
                    await message.reply(final_response_content, mention_author=False)
                except discord.Forbidden:
                    logger.error(f"ListenerCog: No permission to send reply in {message.channel.name} for {message.author.name}.")
                except Exception as e:
                    logger.error(f"ListenerCog: Failed to send LLM reply for {message.author.name}: {e}", exc_info=True)
        
        elif action == "notify_restricted_channel":
            content = result.get("content")
            if content:
                try:
                    if message.channel.permissions_for(message.guild.me).send_messages:
                        await message.reply(content, mention_author=True)
                    else:
                        logger.warning(f"ListenerCog: Missing send_messages permission in {message.channel.name} to notify restricted user.")
                except Exception as e:
                    logger.error(f"ListenerCog: Failed to send restricted channel notification reply: {e}", exc_info=True)

        elif action == "apply_restriction":
            user_id = result.get("user_id_to_restrict")
            guild_id = result.get("guild_id_for_restriction")
            reason = result.get("restriction_reason")
            trigger_case = result.get("triggering_interaction_case") 

            if user_id and guild_id and reason:
                guild = self.bot.get_guild(guild_id)
                if guild:
                    member_to_restrict = guild.get_member(user_id)
                    if member_to_restrict:
                        if await self._apply_restriction(member_to_restrict, guild, reason):                            
                            if trigger_case == "Random Chance": 
                                logger.info(f"ListenerCog: Restriction applied to {member_to_restrict.name} for '{reason}' (triggered by 'Random Chance'). Suppressing notification.")
                            else:
                                logger.info(f"ListenerCog: Restriction applied to {member_to_restrict.name} for '{reason}' (triggered by '{trigger_case}'). Sending notification.")
                                notification_template = self.bot.rate_limit_message_user_template
                                channel_to_mention_in_notif = self.bot.restricted_channel_id
                                notification_content = self._format_notification(notification_template, channel_to_mention_in_notif)
                                try:
                                    if message.channel.permissions_for(message.guild.me).send_messages:
                                        await message.reply(notification_content, mention_author=True)
                                    else:
                                         logger.warning(f"ListenerCog: Missing send_messages permission in {message.channel.name} to send rate limit notification.")
                                except Exception as e:
                                    logger.error(f"ListenerCog: Failed to send rate limit notification reply: {e}", exc_info=True)
                    else:
                        logger.warning(f"ListenerCog: Could not find member {user_id} in guild {guild_id} to apply restriction.")
                else:
                    logger.warning(f"ListenerCog: Could not find guild {guild_id} to apply restriction for user {user_id}.")
            else:
                logger.warning(f"ListenerCog: 'apply_restriction' action received with missing data: user_id={user_id}, guild_id={guild_id}, reason={reason}, trigger={trigger_case}")
        
        elif action == "error":
            error_content = result.get("content", "An unexpected error occurred.") 
            try:
                await message.reply(error_content, mention_author=False) 
            except discord.Forbidden:
                 logger.error(f"ListenerCog: No permission to send error reply in {message.channel.name} for {message.author.name}.")
            except Exception as e:
                logger.error(f"ListenerCog: Failed to send error feedback reply to {message.author.name}: {e}", exc_info=True)

        elif action == "do_nothing":
            pass 
            
    @tasks.loop(seconds=300) # This interval should be respected by discord.ext.tasks
    async def check_restrictions_loop(self):
        if not self.redis_client or not self.bot.restricted_user_role_id or self.bot.restriction_duration_seconds <= 0:
            return

        current_time_unix = time.time()
        current_time_readable = datetime.datetime.fromtimestamp(current_time_unix, tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        logger.info(f"Restriction Loop: Starting check at {current_time_readable} (Unix: {current_time_unix:.2f})")
        
        processed_keys_count = 0
        unrestricted_count = 0
        scan_iterations = 0 # To count how many times SCAN is called in one go

        try:
            current_scan_cursor = "0" # Initial cursor for SCAN
            logger.debug(f"Restriction Loop: Initial SCAN cursor: {current_scan_cursor}")

            while True:
                scan_iterations += 1
                if not self.redis_client:
                    logger.error("Restriction Loop: Redis client became unavailable during scan.")
                    break
                
                logger.debug(f"Restriction Loop: Calling SCAN with cursor: {current_scan_cursor}, iteration: {scan_iterations}")
                next_cursor_str, keys_batch_as_strings = await discord.utils.asyncio.to_thread(
                    self.redis_client.scan, current_scan_cursor, match=f"restricted_until:*:*", count=100
                )
                logger.debug(f"Restriction Loop: SCAN returned next_cursor: {next_cursor_str}, found {len(keys_batch_as_strings)} keys in this batch.")

                if not keys_batch_as_strings and next_cursor_str == "0" and scan_iterations == 1 :
                    logger.info("Restriction Loop: No active restriction keys found in Redis on first scan iteration.")
                    # This break is important if the first scan yields no keys and cursor is already 0
                    break 
                
                for key_str in keys_batch_as_strings: 
                    processed_keys_count += 1
                    # ... (the detailed logging and processing for each key remains the same as before) ...
                    logger.debug(f"Restriction Loop: Processing key: {key_str}")
                    try:
                        parts = key_str.split(':')
                        if len(parts) != 3:
                            logger.warning(f"Restriction Loop: Malformed key '{key_str}'. Skipping.")
                            continue
                        
                        guild_id_str, user_id_str = parts[1], parts[2]
                        guild_id = int(guild_id_str)
                        user_id = int(user_id_str)

                        expiry_timestamp_str = await discord.utils.asyncio.to_thread(self.redis_client.get, key_str)
                        if not expiry_timestamp_str:
                            logger.warning(f"Restriction Loop: Key '{key_str}' found by SCAN but GET returned None. Skipping.")
                            continue
                        
                        expiry_timestamp_unix = float(expiry_timestamp_str)
                        expiry_time_readable = datetime.datetime.fromtimestamp(expiry_timestamp_unix, tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                        remaining_seconds = expiry_timestamp_unix - current_time_unix
                        remaining_timedelta = datetime.timedelta(seconds=max(0, remaining_seconds))

                        logger.info(f"Restriction Loop: User {user_id} (Guild {guild_id}). Key: {key_str}. Current time: {current_time_readable}. Expires at: {expiry_time_readable} (Unix: {expiry_timestamp_unix:.2f}). Remaining: {str(remaining_timedelta).split('.')[0]}.")

                        if current_time_unix >= expiry_timestamp_unix:
                            logger.info(f"Restriction Loop: DECISION - Restriction for User {user_id} (Guild {guild_id}) has EXPIRED. Attempting to remove role.")
                            guild = self.bot.get_guild(guild_id)
                            if not guild:
                                logger.warning(f"Restriction Loop: Guild {guild_id} not found for User {user_id}. Deleting key '{key_str}'.")
                                await discord.utils.asyncio.to_thread(self.redis_client.delete, key_str)
                                continue

                            member_obj = guild.get_member(user_id)
                            if not member_obj:
                                logger.warning(f"Restriction Loop: Member {user_id} not found in Guild {guild.name}. Deleting key '{key_str}'.")
                                await discord.utils.asyncio.to_thread(self.redis_client.delete, key_str)
                                continue

                            restricted_role = guild.get_role(self.bot.restricted_user_role_id) # type: ignore
                            if not restricted_role:
                                logger.error(f"Restriction Loop: Restricted role ID {self.bot.restricted_user_role_id} not found in Guild {guild.name} for User {user_id}. Cannot remove role. Key preserved.")
                                continue 

                            if restricted_role in member_obj.roles:
                                try:
                                    await member_obj.remove_roles(restricted_role, reason="Restriction period expired.")
                                    logger.info(f"Restriction Loop: ACTION - Successfully removed role from {member_obj.name} ({user_id}).")
                                    unrestricted_count += 1
                                    await discord.utils.asyncio.to_thread(self.redis_client.delete, key_str)
                                    logger.debug(f"Restriction Loop: ACTION - Deleted Redis key '{key_str}'.")
                                except discord.Forbidden:
                                    logger.error(f"Restriction Loop: Bot lacks permissions to remove role from {member_obj.name} ({user_id}). Key preserved.")
                                except discord.HTTPException as e_http:
                                    logger.error(f"Restriction Loop: Failed to remove role from {member_obj.name} ({user_id}) HTTP: {e_http}. Key preserved.")
                            else:
                                logger.info(f"Restriction Loop: User {member_obj.name} ({user_id}) no longer had role. Deleting key '{key_str}'.")
                                await discord.utils.asyncio.to_thread(self.redis_client.delete, key_str)
                        else:
                            logger.debug(f"Restriction Loop: DECISION - User {user_id} (Guild {guild_id}) still active. Expires in {str(remaining_timedelta).split('.')[0]}.")
                    
                    except ValueError: 
                        logger.warning(f"Restriction Loop: Could not parse key '{key_str}'. Skipping.")
                    except Exception as e_key_proc: 
                        logger.error(f"Restriction Loop: Error processing key '{key_str}': {e_key_proc}", exc_info=True)
                
                current_scan_cursor = next_cursor_str # Update for the next SCAN call
                if current_scan_cursor == "0": 
                    logger.info(f"Restriction Loop: SCAN iteration complete, final cursor received is '0'. Total SCAN calls in this cycle: {scan_iterations}.")
                    break # Exit the while True loop
                
                # Safety break if it loops too many times for a small number of expected keys
                # This indicates an issue with SCAN cursor not returning "0" as expected.
                if scan_iterations > 10 and processed_keys_count < (scan_iterations * 10): # Arbitrary threshold
                    logger.error(f"Restriction Loop: Excessive SCAN iterations ({scan_iterations}) for few processed keys ({processed_keys_count}). Breaking to prevent runaway loop. Last cursor: {current_scan_cursor}. This indicates a potential issue with Redis SCAN behavior or key population.")
                    break
        
        except redis.exceptions.RedisError as e_redis_scan: 
            logger.error(f"Restriction Loop: Redis error during scan setup or execution: {e_redis_scan}", exc_info=True)
        except Exception as e_loop: 
            logger.error(f"Restriction Loop: Unexpected error in main loop logic: {e_loop}", exc_info=True)
        
        logger.info(f"Restriction Loop: Finished check. Processed {processed_keys_count} total key-value pairs across {scan_iterations} SCAN calls. Unrestricted {unrestricted_count} users in this cycle.")

    @check_restrictions_loop.before_loop
    async def before_check_restrictions_loop(self):
        logger.info("Restriction expiry check loop: Waiting for bot to be ready...")
        await self.bot.wait_until_ready()
        logger.info("Restriction expiry check loop: Bot is ready. Loop will start if conditions are met.")

    @commands.Cog.listener()
    async def on_member_join(self, member: discord.Member):
        # ... (on_member_join logic remains the same) ...
        if not self.bot.welcome_channel_id: return
        welcome_channel = member.guild.get_channel(self.bot.welcome_channel_id)
        if not welcome_channel or not isinstance(welcome_channel, discord.TextChannel):
            logger.warning(f"Welcome channel {self.bot.welcome_channel_id} not found or not text in {member.guild.name}.")
            return
        if not welcome_channel.permissions_for(member.guild.me).send_messages:
            logger.warning(f"Missing send_messages permission in welcome channel {welcome_channel.name}.")
            return
        logger.info(f"Generating welcome message for {member.name} ({member.id}) in {member.guild.name}")
        try:
            async with welcome_channel.typing(): 
                response_content, error_message = await self.bot.api_client.generate_welcome_message(member) # type: ignore
            if response_content:
                await welcome_channel.send(response_content[:2000])
                logger.info(f"Sent welcome message for {member.name}.")
            else:
                logger.error(f"Failed to generate welcome message for {member.name}. Error: {error_message}")
        except Exception as e:
            logger.exception(f"Error during welcome message for {member.name}: {e}")

async def setup(bot: 'AIBot'): 
    # ... (setup logic remains the same) ...
    if not hasattr(bot, 'api_client') or not bot.api_client: # type: ignore
         logger.critical("ListenerCog setup: AIBot is missing 'api_client'. This is required. Cog will not load.")
         return
    if not hasattr(bot, 'redis_client_general') and bot.restricted_user_role_id : # type: ignore
         logger.warning("ListenerCog setup: AIBot general Redis client not available. Rate limiting and restriction expiry may not work as expected.")
    await bot.add_cog(ListenerCog(bot))
    logger.info("ListenerCog added to the bot.")