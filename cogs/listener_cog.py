# cogs/listener_cog.py
import discord
from discord.ext import commands, tasks
import logging
import time
import redis
from typing import TYPE_CHECKING, Optional, Dict, Any
import datetime
import io
import urllib.parse
import aiohttp


from utils.message_handler import MessageHandler, MessageHandlerResult 

if TYPE_CHECKING:
    from bot import AIBot 

logger = logging.getLogger(__name__)

class ListenerCog(commands.Cog):
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
                 logger.warning(f"ListenerCog: Automatic restriction expiry duration is {self.bot.restriction_duration_seconds}s but check interval not valid ({self.bot.restriction_check_interval_seconds}s. Loop not started.")
            else: 
                logger.info("ListenerCog: Automatic restriction expiry is disabled (duration is 0 or not set).")
        else:
            logger.info("ListenerCog: Restriction system (role assignment) not fully configured. Automatic expiry disabled.")

    async def _apply_restriction(self, member: discord.Member, guild: discord.Guild, reason: str) -> bool:
        if not self.bot.restricted_user_role_id: 
            logger.debug("ListenerCog: Restriction system disabled (no RESTRICTED_USER_ROLE_ID). Cannot apply restriction.")
            return False

        try:
            role_id_to_get = int(self.bot.restricted_user_role_id)
        except (ValueError, TypeError):
            logger.error(f"ListenerCog: RESTRICTED_USER_ROLE_ID ('{self.bot.restricted_user_role_id}') is not a valid integer. Cannot apply restriction.")
            return False

        restricted_role = guild.get_role(role_id_to_get)
        if not restricted_role:
            logger.error(f"ListenerCog: RESTRICTED_USER_ROLE_ID {role_id_to_get} not found in guild {guild.name}. Cannot apply restriction.")
            return False
        
        user_has_role_after_action = False
        try:
            if restricted_role not in member.roles:
                await member.add_roles(restricted_role, reason=reason)
                logger.info(f"ListenerCog: Assigned role '{restricted_role.name}' to {member.name} ({member.id}) for: {reason}")
                user_has_role_after_action = True
            else:
                logger.info(f"ListenerCog: {member.name} ({member.id}) already had restricted role '{restricted_role.name}'. Reason for check: {reason}")
                user_has_role_after_action = True # Already has the role
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
                await discord.utils.asyncio.to_thread(self.redis_client.set, redis_key, float(expiry_timestamp), ex=self.bot.restriction_duration_seconds + 3600) 
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
            logger.debug(f"ListenerCog: Message author {message.author.name} is not a discord.Member instance. Attempting to fetch member object.")
            try:
                author_member = await message.guild.fetch_member(message.author.id) # type: ignore
                if not author_member: # Still not found
                     logger.warning(f"ListenerCog: Could not resolve message author {message.author.name} ({message.author.id}) to a guild member. Skipping.")
                     return

            except discord.NotFound:
                logger.warning(f"ListenerCog: Message author {message.author.name} ({message.author.id}) not found in guild. Skipping.")
                return
            except discord.HTTPException as e:
                logger.error(f"ListenerCog: Failed to fetch member {message.author.name} ({message.author.id}): {e}. Skipping.")
                return


        handler = MessageHandler(self.bot, message)
        result: MessageHandlerResult

        try:
            result = await handler.process() 
        except Exception as e:
            logger.error(f"ListenerCog: Unhandled exception during MessageHandler.process() for {message.author.name}: {e}", exc_info=True)
            try: 
                if message.guild and message.channel.permissions_for(message.guild.me).send_messages:
                    await message.reply("Sorry, an unexpected error occurred while I was thinking.", mention_author=False)
            except Exception: pass 
            return

        if result.get("log_message"):
            author_name_for_log = message.author.name if isinstance(message.author, discord.Member) else f"User {message.author.id}"
            logger.info(f"ListenerCog: From MessageHandler for {author_name_for_log} - {result['log_message']}")

        action = result.get("action")
        logger.debug(f"ListenerCog: Received action '{action}' for message ID {message.id} from {message.author.name}. Full result: {result}")


        can_send_messages = False
        if message.guild:
            can_send_messages = message.channel.permissions_for(message.guild.me).send_messages

    
        is_message_sending_action = action and action in [
            "reply_text", "reply_with_url", "reply_with_gif", "reply_with_latex", "reply_with_code",
            "notify_restricted_channel", "error"

        ]

        if not can_send_messages and is_message_sending_action:
            logger.warning(f"ListenerCog: Missing send_messages permission in {message.channel.name} to perform action '{action}'. Bot cannot respond/notify.")
            return
        
        try:
            if action == "reply_text":
                content = result.get("content")
                if content and can_send_messages:
                    final_response_content = content
                    if len(final_response_content) > 2000:
                        final_response_content = final_response_content[:1997] + "..."
                    await message.reply(final_response_content, mention_author=False)
            elif action == "reply_with_gif":
                base_text = result.get("base_response_text")
                gif_url = result.get("gif_data_url")
                if base_text and can_send_messages:
                    if len(base_text) > 2000: base_text = base_text[:1997] + "..."
                    await message.reply(base_text, mention_author=False)
                if gif_url and can_send_messages:
                    embed = discord.Embed(color=discord.Color.purple())
                    embed.title = "GIF!"
                    embed.set_image(url=gif_url)
                    await message.channel.send(embed=embed) 
                elif not base_text and not gif_url:
                    logger.warning(f"ListenerCog: 'reply_with_gif' action with no base_text or gif_url for message ID {message.id}.")

            elif action == "reply_with_url":
                base_text = result.get("base_response_text")
                url_data = result.get("url_data")
                if base_text and can_send_messages:
                    if len(base_text) > 2000: base_text = base_text[:1997] + "..."
                    await message.reply(base_text, mention_author=False)
                if url_data and can_send_messages:
                    embed = discord.Embed(title="Link Provided:", description=url_data, color=discord.Color.blue())
                    if any(url_data.lower().endswith(ext) for ext in ['.gif', '.png', '.jpg', '.jpeg', '.webp']):
                        embed.set_image(url=url_data)
                        if not base_text and embed.description == url_data:
                             embed.description = None
                             embed.title = "Image/GIF"
                    await message.channel.send(embed=embed) 
                elif not base_text and not url_data:
                    logger.warning(f"ListenerCog: 'reply_with_url' action with no base_text or url_data for message ID {message.id}.")

            elif action == "reply_with_latex":
                base_text = result.get("base_response_text")
                latex_string = result.get("latex_data")
                
                # Step 1: Send the base_text as a regular reply if it exists
                if base_text and can_send_messages: # type: ignore
                    if len(base_text) > 2000: base_text = base_text[:1997] + "..."
                    await message.reply(base_text, mention_author=False)

                # Step 2: Send the LaTeX image in an embed as a new message if latex_string exists
                if latex_string and can_send_messages: # type: ignore
                    fg_color = "FFFFFF"
                    bg_color = "40444B"
                    dpi_value = 200
                    url_prefix_commands = f"\\dpi{{{dpi_value}}}\\fg{{{fg_color}}}\\bg{{{bg_color}}}"
                    # Apply prefix commands before encoding the main latex_string
                    encoded_latex_expression = urllib.parse.quote(f"{url_prefix_commands} {latex_string}")
                    latex_api_url = f"https://latex.codecogs.com/png.latex?{encoded_latex_expression}"

                    logger.info(f"Requesting LaTeX image from: {latex_api_url}")
                    async with aiohttp.ClientSession() as http_session:
                        async with http_session.get(latex_api_url) as resp:
                            if resp.status == 200:
                                image_bytes = await resp.read()
                                with io.BytesIO(image_bytes) as img_buffer:
                                    discord_file = discord.File(img_buffer, filename="latex_expression.png")
                                    
                                    embed = discord.Embed(
                                        # title="Rendered Equation", # Optional: Title for the embed itself
                                        color=discord.Color.blue() # Or your preferred color
                                    )
                                    embed.set_image(url="attachment://latex_expression.png")
                                    
                                    # Send as a new message in the channel, not a reply to the user's original message
                                    await message.channel.send(embed=embed, file=discord_file) # type: ignore
                            else:
                                logger.error(f"Failed to fetch LaTeX image. Status: {resp.status}, URL: {latex_api_url}, Original LaTeX: {latex_string[:100]}")
                                # Send a fallback message if image rendering fails
                                await message.channel.send(f"(Sorry, I couldn't render the math expression: `{latex_string[:100].replace('`', '')}...`)") # type: ignore
                
                # If only one of them was missing (and not handled above) or both were missing
                elif not base_text and not latex_string:
                     logger.warning(f"ListenerCog: 'reply_with_latex' action with no base_text or latex_data for message ID {message.id}.")
            

            elif action == "reply_with_code":
                base_text = result.get("base_response_text")
                lang = result.get("code_data_language", "")
                code = result.get("code_data_content")
                if base_text and can_send_messages:
                    if len(base_text) > 2000: base_text = base_text[:1997] + "..."
                    await message.reply(base_text, mention_author=False)
                if code and can_send_messages:
                    formatted_code_block = f"```{lang}\n{code}\n```"
                    if len(formatted_code_block) <= 2000:
                        await message.channel.send(formatted_code_block)
                    else:
                        logger.info(f"Code for {message.author.name} (msg ID {message.id}) is too long. Sending as file.")
                        file_extension = lang if lang and lang.isalnum() else "txt"
                        file_name = f"code_snippet.{file_extension}"
                        try:
                            code_bytes = code.encode('utf-8')
                            with io.BytesIO(code_bytes) as code_buffer: 
                                discord_file_obj = discord.File(code_buffer, filename=file_name)
                                embed_for_file = discord.Embed(
                                    title="Code Snippet Attached",
                                    description=f"The code was too long to display directly. See attached file: `{file_name}`",
                                    color=discord.Color.greyple())
                                await message.channel.send(embed=embed_for_file, file=discord_file_obj) 
                        except Exception as e_file:
                            logger.error(f"Error creating/sending code file for msg ID {message.id}: {e_file}", exc_info=True)
                            truncated_code_block = f"```{lang}\n{code[:1500]}...\n```\n(Code too long, file attach failed)"
                            if can_send_messages: await message.channel.send(truncated_code_block) 
                elif not base_text and not code:
                     logger.warning(f"ListenerCog: 'reply_with_code' action with no base_text or code_data for message ID {message.id}.")

            elif action == "notify_restricted_channel": 
                content = result.get("content")
                if content and can_send_messages:
                    await message.reply(content, mention_author=True)

            elif action == "error":
                error_content = result.get("content", "An unexpected error occurred.")
                if can_send_messages:
                    await message.reply(error_content, mention_author=False)

            elif action == "add_reaction_and_do_nothing":
                reaction_emoji = result.get("content")
                if reaction_emoji:
                    try:
                        await message.add_reaction(reaction_emoji)
                    except discord.Forbidden:
                        logger.warning(f"ListenerCog: Missing permission to add reaction '{reaction_emoji}' to message ID {message.id}.")
                    except discord.HTTPException as e_react:
                        logger.warning(f"ListenerCog: Failed to add reaction '{reaction_emoji}' to message ID {message.id}: {e_react}")

            elif action == "do_nothing":
                pass 


            # Step 2: Check for and apply any pending restriction AFTER the main reply/action
            pending_restriction_details = result.get("pending_restriction")
            if pending_restriction_details and isinstance(message.author, discord.Member) and message.guild : # Ensure author is member and guild exists
                user_id_to_restrict = pending_restriction_details.get("user_id_to_restrict")
                guild_id_for_restriction = pending_restriction_details.get("guild_id_for_restriction")
                reason_for_restriction = pending_restriction_details.get("restriction_reason")
                trigger_case_for_log = pending_restriction_details.get("trigger_case_for_restriction_log", "Unknown (pending)")

                target_guild = self.bot.get_guild(guild_id_for_restriction) 
                if not target_guild:
                    logger.error(f"ListenerCog: Pending restriction: Guild {guild_id_for_restriction} not found.")
                else:
                    target_member = target_guild.get_member(user_id_to_restrict) 
                    if not target_member:
                        logger.error(f"ListenerCog: Pending restriction: Member {user_id_to_restrict} not found in Guild {target_guild.name}.")
                    elif reason_for_restriction:
                        logger.info(f"ListenerCog: Applying PENDING restriction to {target_member.name} due to: {reason_for_restriction}")
                        if await self._apply_restriction(target_member, target_guild, reason_for_restriction):
                            # Send notification publicly in channel AFTER restriction is applied
                            # Notification is NOT sent if the *original interaction case* that *led to this restriction* was Random Chance
                            # The 'triggering_interaction_case' in the main result dict should reflect the original user interaction.
                            # The 'trigger_case_for_restriction_log' inside pending_restriction_details is for logging the restriction itself.
                            original_interaction_case = result.get("triggering_interaction_case")
                            
                            if original_interaction_case != "Random Chance": # Solves Problem 2 for *notification*
                                if can_send_messages:
                                    logger.info(f"ListenerCog: Restriction applied to {target_member.name} ({original_interaction_case}). Sending notification.")
                                    notification_template = self.bot.rate_limit_message_user_template #
                                    channel_to_mention_in_notif = self.bot.restricted_channel_id #
                                    notification_content = self._format_notification(notification_template, channel_to_mention_in_notif)
                                    await message.reply(notification_content, mention_author=True)
                                else:
                                     logger.warning(f"ListenerCog: Missing send_messages permission for 'apply_restriction' notification to {target_member.name}. Role applied silently.")
                            else:
                                logger.info(f"ListenerCog: Restriction applied to {target_member.name} (Original interaction was Random Chance). Notification suppressed as per Problem 2.")
                        else:
                            logger.error(f"ListenerCog: Failed to apply PENDING restriction to {target_member.name}.")
            
            elif action and action not in ["reply_text", "reply_with_url", "reply_with_gif", "reply_with_latex", "reply_with_code", "notify_restricted_channel", "error", "add_reaction_and_do_nothing", "do_nothing"]:
                 logger.warning(f"ListenerCog: Unknown action '{action}' received from MessageHandler for message ID {message.id}. Result: {result}")


        except discord.Forbidden as e_forbidden:
            logger.error(f"ListenerCog: Permission error (Forbidden) performing action '{action}' or applying restriction in {message.channel.name} for message ID {message.id}: {e_forbidden}") 
        except discord.HTTPException as e_discord_http:
            logger.error(f"ListenerCog: Discord HTTP error performing action '{action}' or applying restriction in {message.channel.name} for message ID {message.id}: {e_discord_http}", exc_info=True) 
        except aiohttp.ClientError as e_aio:
            logger.error(f"ListenerCog: HTTP Client error (e.g., for LaTeX) performing action '{action}' for message ID {message.id}: {e_aio}", exc_info=True) 
            if action == "reply_with_latex" and can_send_messages:
                try: await message.channel.send(f"(Sorry, I couldn't render the math expression right now.)") 
                except Exception: pass
        except Exception as e_unhandled:
            logger.error(f"ListenerCog: An unexpected error occurred while performing action '{action}' or applying restriction for message ID {message.id} by {message.author.name}: {e_unhandled}", exc_info=True) 


    @tasks.loop(seconds=300)
    async def check_restrictions_loop(self):
        if not self.redis_client or not self.bot.restricted_user_role_id or self.bot.restriction_duration_seconds <= 0: #
            return

        current_time_unix = time.time()
        current_time_readable = datetime.datetime.fromtimestamp(current_time_unix, tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        logger.info(f"Restriction Loop: Starting check at {current_time_readable} (Unix: {current_time_unix:.2f})")

        processed_keys_count = 0
        unrestricted_count = 0
        scan_iterations = 0

        try:
            current_scan_cursor = "0" # Start with string "0" if decode_responses=True for client

            while True:
                scan_iterations += 1
                if not self.redis_client:
                    logger.error("Restriction Loop: Redis client became unavailable during scan.")
                    break

                next_cursor_val, keys_batch_val = await discord.utils.asyncio.to_thread(
                    self.redis_client.scan, current_scan_cursor, match=f"restricted_until:*:*", count=100 
                )

                current_scan_cursor = next_cursor_val # Update cursor for next iteration
                keys_batch_as_strings = keys_batch_val # Already strings if decode_responses=True

                if not keys_batch_as_strings and current_scan_cursor == "0" and scan_iterations == 1 :
                    logger.info("Restriction Loop: No active restriction keys found in Redis on first scan iteration.")
                    break

                for key_str in keys_batch_as_strings:
                    processed_keys_count += 1
                    try:
                        parts = key_str.split(':')
                        if len(parts) != 3:
                            logger.warning(f"Restriction Loop: Malformed key '{key_str}'. Skipping.")
                            continue

                        guild_id_str, user_id_str = parts[1], parts[2]
                        guild_id = int(guild_id_str)
                        user_id = int(user_id_str)

                        expiry_timestamp_val = await discord.utils.asyncio.to_thread(self.redis_client.get, key_str) 
                        if not expiry_timestamp_val: # Key might have expired
                            logger.warning(f"Restriction Loop: Key '{key_str}' found by SCAN but GET returned None. Skipping.")
                            continue

                        expiry_timestamp_unix = float(expiry_timestamp_val)
                        expiry_time_readable = datetime.datetime.fromtimestamp(expiry_timestamp_unix, tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                        remaining_seconds = expiry_timestamp_unix - current_time_unix
                        remaining_timedelta_str = str(datetime.timedelta(seconds=max(0, remaining_seconds))).split('.')[0]
                        logger.debug(f"Restriction Loop: User {user_id} (Guild {guild_id}). Key: {key_str}. Expires at: {expiry_time_readable}. Remaining: {remaining_timedelta_str}.") #

                        if current_time_unix >= expiry_timestamp_unix:
                            logger.info(f"Restriction Loop: DECISION - Restriction for User {user_id} (Guild {guild_id}) has EXPIRED. Attempting to remove role.")
                            guild = self.bot.get_guild(guild_id)
                            if not guild:
                                logger.warning(f"Restriction Loop: Guild {guild_id} not found. Deleting key '{key_str}'.")
                                await discord.utils.asyncio.to_thread(self.redis_client.delete, key_str) 
                                continue

                            member_obj = guild.get_member(user_id)
                            if not member_obj:
                                logger.warning(f"Restriction Loop: Member {user_id} not found in Guild {guild.name}. Deleting key '{key_str}'.")
                                await discord.utils.asyncio.to_thread(self.redis_client.delete, key_str) 
                                continue

                            role_id_int = int(self.bot.restricted_user_role_id) if self.bot.restricted_user_role_id else 0 
                            restricted_role = guild.get_role(role_id_int)
                            if not restricted_role:
                                logger.error(f"Restriction Loop: Restricted role ID {role_id_int} not found in Guild {guild.name} for User {user_id}. Cannot remove role. Key preserved.")
                                continue

                            if restricted_role in member_obj.roles:
                                try:
                                    await member_obj.remove_roles(restricted_role, reason="Restriction period expired.")
                                    logger.info(f"Restriction Loop: ACTION - Successfully removed role from {member_obj.name} ({user_id}).")
                                    unrestricted_count += 1
                                    await discord.utils.asyncio.to_thread(self.redis_client.delete, key_str) 
                                    logger.debug(f"Restriction Loop: ACTION - Deleted Redis key '{key_str}'.")
                                except discord.Forbidden: #
                                    logger.error(f"Restriction Loop: Bot lacks permissions to remove role from {member_obj.name} ({user_id}). Key preserved.")
                                except discord.HTTPException as e_http:
                                    logger.error(f"Restriction Loop: Failed to remove role from {member_obj.name} ({user_id}) HTTP: {e_http}. Key preserved.")
                            else:
                                logger.info(f"Restriction Loop: User {member_obj.name} ({user_id}) no longer had restricted role. Deleting key '{key_str}'.")
                                await discord.utils.asyncio.to_thread(self.redis_client.delete, key_str) 
                    except ValueError as ve:
                        logger.warning(f"Restriction Loop: Could not parse components for key '{key_str}'. Error: {ve}. Skipping.")
                    except Exception as e_key_proc:
                        logger.error(f"Restriction Loop: Error processing key '{key_str}': {e_key_proc}", exc_info=True)

                if current_scan_cursor == "0": # Cursor will be '0' (string) if decode_responses=True
                    logger.debug(f"Restriction Loop: SCAN iteration complete, final cursor is '0'. Total SCAN calls: {scan_iterations}.")
                    break

                if scan_iterations > 100: # Safety break for very large Redis instances or issues
                    logger.error(f"Restriction Loop: Excessive SCAN iterations ({scan_iterations}). Breaking. Last cursor: {current_scan_cursor}.")
                    break
        except redis.exceptions.RedisError as e_redis_scan:
            logger.error(f"Restriction Loop: Redis error during scan: {e_redis_scan}", exc_info=True)
        except Exception as e_loop:
            logger.error(f"Restriction Loop: Unexpected error: {e_loop}", exc_info=True)

        if processed_keys_count > 0 or unrestricted_count > 0 or scan_iterations > 1:
            logger.info(f"Restriction Loop: Finished. Processed {processed_keys_count} keys ({scan_iterations} SCANs). Unrestricted {unrestricted_count} users.")


    @check_restrictions_loop.before_loop
    async def before_check_restrictions_loop(self): #
        logger.info("Restriction expiry check loop: Waiting for bot to be ready...")
        await self.bot.wait_until_ready()
        logger.info("Restriction expiry check loop: Bot is ready. Loop will start if conditions are met.")

    @commands.Cog.listener()
    async def on_member_join(self, member: discord.Member): #
        if not self.bot.welcome_channel_id: return #
        welcome_channel = member.guild.get_channel(self.bot.welcome_channel_id) #
        if not welcome_channel or not isinstance(welcome_channel, discord.TextChannel):
            logger.warning(f"Welcome channel ID {self.bot.welcome_channel_id} not found or not a text channel in guild {member.guild.name}.") #
            return
        if not welcome_channel.permissions_for(member.guild.me).send_messages: #
            logger.warning(f"Missing send_messages permission in welcome channel {welcome_channel.name}.")
            return

        logger.info(f"Generating welcome message for {member.name} ({member.id}) in {member.guild.name}")
        try:
            async with welcome_channel.typing():
                if hasattr(self.bot, 'api_client') and self.bot.api_client: #
                    response_content, error_message = await self.bot.api_client.generate_welcome_message(member) #
                    if response_content:
                        await welcome_channel.send(response_content[:2000])
                        logger.info(f"Sent welcome message for {member.name}.")
                    else:
                        logger.error(f"Failed to generate welcome message for {member.name}. Error: {error_message}")
                else:
                    logger.error("Welcome message generation skipped: api_client not available on bot instance.")
        except Exception as e:
            logger.error(f"Error during welcome message for {member.name}: {e}", exc_info=True)

async def setup(bot: 'AIBot'):
    # This setup function remains the same
    if not hasattr(bot, 'api_client') or not bot.api_client: #
         logger.critical("ListenerCog setup: AIBot instance is missing 'api_client'. Cog will not load.")
         return
    # Check for general_redis_config if timed restrictions are on
    if bot.restricted_user_role_id and bot.restriction_duration_seconds > 0 and \
       (not hasattr(bot, 'redis_client_general') or not bot.redis_client_general) : #
         logger.warning("ListenerCog setup: AIBot general Redis client not available, but timed restrictions are configured. Expiry may not work.")

    await bot.add_cog(ListenerCog(bot))
    logger.info("ListenerCog added to the bot.")