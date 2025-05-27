# cogs/activity_cog.py
import discord
from discord.ext import commands, tasks
import logging
import random
from typing import TYPE_CHECKING, Optional, Dict, Any, List, Set
import datetime

# For slash commands and permission checks
from discord import app_commands

if TYPE_CHECKING:
    from bot import AIBot

logger = logging.getLogger(__name__)

class ActivityCog(commands.Cog):
    def __init__(self, bot: 'AIBot'):
        self.bot = bot
        self.current_activity_text = "Initializing..."
        self.current_activity_type_name = "Playing"
        self.current_online_status = discord.Status.online

        self.activity_options: List[Dict[str, Any]] = [
            # For these, discord.Game, discord.Activity(type=listening/watching) are appropriate
            {"type_enum": discord.ActivityType.playing, "name_for_discord": "Playing", "prompt_verb": "a game title or a general status update that fits 'Playing X'", "requires_emoji": False, "use_custom_activity_class": False},
            {"type_enum": discord.ActivityType.listening, "name_for_discord": "Listening to", "prompt_verb": "a song title, podcast name, or interesting sound", "requires_emoji": False, "use_custom_activity_class": False},
            {"type_enum": discord.ActivityType.watching, "name_for_discord": "Watching", "prompt_verb": "a show title, movie name, or intriguing video", "requires_emoji": False, "use_custom_activity_class": False},
            # For this one, we want it to be a "custom status" without a "Playing" prefix.
            {"type_enum": discord.ActivityType.custom, "name_for_discord": "Custom", "prompt_verb": "a general custom status message (this will NOT be prefixed by 'Playing', 'Watching' etc.)", "requires_emoji": True, "use_custom_activity_class": True},
        ]

        if not self.bot.base_activity_system_prompt:
            logger.warning("ActivityCog: Bot is missing 'base_activity_system_prompt'. Activity updates will be disabled.")
            return

        if self.bot.activity_update_interval_seconds > 0:
            self.update_bot_activity_loop.change_interval(seconds=self.bot.activity_update_interval_seconds)
            self.update_bot_activity_loop.start()
            logger.info(f"ActivityCog: Bot activity update task scheduled. Interval: {self.bot.activity_update_interval_seconds}s")
            if self.bot.activity_schedule_enabled:
                 logger.info(f"ActivityCog: Schedule enabled. Active Mon-Sun (0-6): {sorted(list(self.bot.activity_active_days_utc))}, Hours UTC: {self.bot.activity_active_start_hour_utc:02d}-{self.bot.activity_active_end_hour_utc:02d}")
        else:
            logger.info("ActivityCog: Bot activity update task disabled (interval is 0 or not set).")

    def is_active_time(self) -> bool:
        # ... (is_active_time method remains the same)
        if not self.bot.activity_schedule_enabled:
            self.current_online_status = discord.Status.online
            return True
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        current_hour_utc = now_utc.hour
        current_weekday_utc = now_utc.weekday()
        if current_weekday_utc not in self.bot.activity_active_days_utc:
            self.current_online_status = discord.Status.idle
            return False
        start_hour = self.bot.activity_active_start_hour_utc
        end_hour = self.bot.activity_active_end_hour_utc
        is_active = False
        if start_hour <= end_hour:
            is_active = start_hour <= current_hour_utc < end_hour
        else:
            is_active = current_hour_utc >= start_hour or current_hour_utc < end_hour
        
        self.current_online_status = discord.Status.online if is_active else discord.Status.idle
        return is_active


    async def _trigger_activity_update(self, interaction: Optional[discord.Interaction] = None) -> bool:
        if not self.bot.base_activity_system_prompt or not self.activity_options:
            logger.warning("ActivityCog: Cannot trigger activity update: base_activity_system_prompt or activity_options are missing.")
            if interaction and not interaction.response.is_done():
                await interaction.response.send_message("Activity update system is not properly configured.", ephemeral=True)
            return False

        try:
            selected_config = random.choice(self.activity_options)
            activity_enum_for_discord = selected_config["type_enum"]
            activity_type_name_for_discord = selected_config["name_for_discord"] # This is our internal name
            prompt_verb_for_llm = selected_config["prompt_verb"]
            needs_emoji_from_llm = selected_config["requires_emoji"]
            use_custom_class = selected_config["use_custom_activity_class"]

            log_source = f"Manual refresh by {interaction.user.name}" if interaction else "Loop"
            logger.info(f"ActivityCog: {log_source} selected type: {activity_type_name_for_discord}")

            dynamic_instruction = (
                f"Your task is to generate ONLY the creative text for a bot's activity status. "
                f"This text should be suitable content for '{prompt_verb_for_llm}'. "
            )
            if use_custom_class: # If it's our "Custom" type that should use CustomActivity
                 dynamic_instruction += "This will be displayed as a general status, not prefixed by 'Playing', 'Watching', etc. "
            else: # For Playing, Listening, Watching
                 dynamic_instruction += f"For example, if the task is to provide a game title, just provide the title like 'Cosmic Chess Masters'. Do NOT include prefixes like '{activity_type_name_for_discord}' yourself. "


            if needs_emoji_from_llm: # This is true if the type is "Custom" (and use_custom_class is true)
                dynamic_instruction += "\nAdditionally, please suggest a single Unicode emoji in the 'data.emoji_suggestion' field that complements this status text. The main status text itself should still be just the creative text without the emoji prepended by you."
            else:
                dynamic_instruction += "\nFor this task, the 'data.emoji_suggestion' field in your JSON response should be null (an emoji is not required for this activity type)."

            final_system_prompt = f"{self.bot.base_activity_system_prompt}\n\nCURRENT TASK:\n{dynamic_instruction}"
            user_trigger_prompt = "Generate an activity status."

            llm_output_dict, error_msg_api, tokens_used = await self.bot.api_client.generate_response(
                user_id="BOT_ACTIVITY_UPDATER_CUSTOM_FIX",
                channel_id="BOT_ACTIVITY_CHANNEL_CUSTOM_FIX",
                prompt=user_trigger_prompt,
                system_message=final_system_prompt,
                history=[],
                extra_assistant_context=None
            )

            error_occurred = False
            error_message_for_user = ""

            if error_msg_api:
                logger.error(f"ActivityCog: LLM API error during activity generation: {error_msg_api}")
                error_occurred = True
                error_message_for_user = f"LLM API error: {error_msg_api}"
            elif not isinstance(llm_output_dict.get("type"), str) or llm_output_dict.get("type") != "text" or \
                 not isinstance(llm_output_dict.get("response"), str):
                logger.error(f"ActivityCog: LLM output missing 'type: text' or 'response' string fields. Output: {llm_output_dict}")
                error_occurred = True
                error_message_for_user = "LLM returned an unexpected data structure."
            
            main_activity_text_from_llm = ""
            if not error_occurred:
                main_activity_text_from_llm = llm_output_dict.get("response", "")
                if not (main_activity_text_from_llm and main_activity_text_from_llm.strip()):
                    logger.warning(f"ActivityCog: LLM did not return a valid main 'response' text. Full output: {llm_output_dict}")
                    error_occurred = True
                    error_message_for_user = "LLM did not provide valid activity text."

            if error_occurred:
                if interaction:
                    if interaction.response.is_done(): await interaction.followup.send(error_message_for_user, ephemeral=True)
                activity_to_set_on_error = discord.Game(name=self.current_activity_text if self.current_activity_text and self.current_activity_text != "Initializing..." else "Error fetching status...")
                await self.bot.change_presence(status=self.current_online_status, activity=activity_to_set_on_error)
                return False

            llm_emoji_suggestion = None
            data_field = llm_output_dict.get("data")
            if isinstance(data_field, dict):
                llm_emoji_suggestion = data_field.get("emoji_suggestion")

            activity_text_to_display_on_discord = main_activity_text_from_llm.strip()[:128]
            activity_object_to_set = None
            final_activity_string_for_user_message = activity_text_to_display_on_discord # Base for ephemeral msg

            # Construct the activity object based on the type
            if use_custom_class: # This is our specific "Custom" type
                activity_args = {"name": activity_text_to_display_on_discord}
                current_activity_display_name = "Custom Status" # For user message
                if needs_emoji_from_llm and llm_emoji_suggestion and isinstance(llm_emoji_suggestion, str) and llm_emoji_suggestion.strip():
                    parsed_emoji = llm_emoji_suggestion.strip().split(" ")[0]
                    if len(parsed_emoji) == 1 or (parsed_emoji.startswith("<:") and parsed_emoji.endswith(">")): # Basic check
                        activity_args["emoji"] = parsed_emoji
                        final_activity_string_for_user_message = f"{parsed_emoji} {activity_text_to_display_on_discord}"
                    else:
                        logger.warning(f"ActivityCog: LLM emoji suggestion '{llm_emoji_suggestion}' not a single char or custom emoji format for CustomActivity. Using text only.")
                        # final_activity_string_for_user_message remains just the text
                
                try:
                    if "emoji" in activity_args:
                        activity_object_to_set = discord.CustomActivity(name=activity_args["name"], emoji=activity_args.get("emoji"))
                    else:
                        activity_object_to_set = discord.CustomActivity(name=activity_args["name"])
                except Exception as e_cust:
                    logger.error(f"ActivityCog: Failed to create CustomActivity: {e_cust}. Falling back to Game.")
                    activity_object_to_set = discord.Game(name=final_activity_string_for_user_message[:128]) # Use the potentially emoji-prepended string
                    current_activity_display_name = "Playing" # Fallback display name

            elif activity_enum_for_discord == discord.ActivityType.playing:
                activity_object_to_set = discord.Game(name=activity_text_to_display_on_discord)
                current_activity_display_name = "Playing"
            elif activity_enum_for_discord == discord.ActivityType.listening:
                activity_object_to_set = discord.Activity(type=discord.ActivityType.listening, name=activity_text_to_display_on_discord)
                current_activity_display_name = "Listening to"
            elif activity_enum_for_discord == discord.ActivityType.watching:
                activity_object_to_set = discord.Activity(type=discord.ActivityType.watching, name=activity_text_to_display_on_discord)
                current_activity_display_name = "Watching"


            if activity_object_to_set:
                status_to_set = discord.Status.online if interaction or self.is_active_time() else self.current_online_status
                if not interaction: status_to_set = self.current_online_status

                await self.bot.change_presence(status=status_to_set, activity=activity_object_to_set)
                self.current_activity_text = activity_text_to_display_on_discord # Store raw text from LLM
                self.current_activity_type_name = current_activity_display_name # Store the display name like "Playing", "Custom Status"
                
                log_text_for_activity = activity_object_to_set.name # This will be the name passed to the activity object
                if hasattr(activity_object_to_set, 'emoji') and activity_object_to_set.emoji: # For CustomActivity
                    log_text_for_activity = f"{activity_object_to_set.emoji} {log_text_for_activity}"


                log_message = f"ActivityCog: Bot activity {'manually refreshed' if interaction else 'updated by loop'} to ({self.current_activity_type_name}): '{log_text_for_activity}'. Status: {status_to_set}. Tokens: {tokens_used or 'N/A'}"
                logger.info(log_message)

                if interaction:
                    # For the ephemeral message, use final_activity_string_for_user_message which includes emoji for custom type.
                    # And current_activity_display_name which is "Custom Status" for that case.
                    user_feedback_activity_text = final_activity_string_for_user_message
                    if not use_custom_class: # For Playing, Listening, Watching, the final string is just the text
                        user_feedback_activity_text = activity_text_to_display_on_discord

                    success_message = f"Bot status refreshed to: **{self.current_activity_type_name} {user_feedback_activity_text}** (Status: {status_to_set})"
                    if interaction.response.is_done(): await interaction.followup.send(success_message, ephemeral=True)
                return True
            else:
                logger.warning(f"ActivityCog: Failed to create an activity object. LLM output: {llm_output_dict}")
                if interaction:
                    err_msg = "Failed to create a valid activity object from LLM response."
                    if interaction.response.is_done(): await interaction.followup.send(err_msg, ephemeral=True)
                return False

        except Exception as e:
            logger.error(f"ActivityCog: Unexpected error in _trigger_activity_update: {e}", exc_info=True)
            if interaction:
                err_msg = f"An unexpected error occurred: {e}"
                if interaction.response.is_done(): await interaction.followup.send(err_msg, ephemeral=True)
            try:
                await self.bot.change_presence(status=self.current_online_status, activity=discord.Game(name="Idle | Error processing"))
            except Exception as presence_error:
                logger.error(f"ActivityCog: Failed to set fallback presence after error: {presence_error}")
            return False

    @tasks.loop()
    async def update_bot_activity_loop(self):
        # ... (loop logic remains the same - calls _trigger_activity_update)
        is_currently_active_time = self.is_active_time()

        if not is_currently_active_time:
            current_loop_count = self.update_bot_activity_loop.current_loop
            if current_loop_count is not None and current_loop_count % 12 == 0 :
                 logger.debug(f"ActivityCog: Outside of active hours/days. Skipping activity update. Status: {self.current_online_status}. Loop: {current_loop_count}")
            elif current_loop_count is None:
                 logger.debug(f"ActivityCog: Outside of active hours/days on first check. Skipping activity update. Status: {self.current_online_status}")

            try:
                current_presence_activity = self.bot.guilds[0].me.activity if self.bot.guilds and self.bot.guilds[0].me else None
                current_presence_status = self.bot.guilds[0].me.status if self.bot.guilds and self.bot.guilds[0].me else None
                resting_activity_name = "Resting... 💤"

                if not (isinstance(current_presence_activity, discord.Game) and current_presence_activity.name == resting_activity_name and current_presence_status == discord.Status.idle):
                    resting_activity = discord.Game(name=resting_activity_name)
                    await self.bot.change_presence(status=discord.Status.idle, activity=resting_activity)
                    logger.info(f"ActivityCog: Bot status set to Idle and activity to '{resting_activity_name}' (inactive period).")
            except Exception as e:
                logger.error(f"ActivityCog: Failed to set resting presence: {e}")
            return

        await self._trigger_activity_update()


    @update_bot_activity_loop.before_loop
    async def before_update_bot_activity_loop(self):
        # ... (before_loop remains the same)
        await self.bot.wait_until_ready()
        logger.info("ActivityCog: Bot is ready. Activity update loop will now start if configured.")
        try:
            self.is_active_time() # Determine initial desired status
            initial_activity_name = self.current_activity_text if self.current_activity_text != "Initializing..." else "Booting up..."
            initial_activity = discord.Game(name=initial_activity_name) # Initial is fine as Game
            await self.bot.change_presence(status=self.current_online_status, activity=initial_activity)
            logger.info(f"ActivityCog: Initial presence set. Status: {self.current_online_status}, Activity: '{initial_activity.name}'")
        except Exception as e:
            logger.error(f"ActivityCog: Failed to set initial presence: {e}")

    async def cog_unload(self):
        # ... (cog_unload remains the same)
        self.update_bot_activity_loop.cancel()
        logger.info("ActivityCog: Unloaded, activity update task cancelled.")

    def is_rate_limit_exempt(): # type: ignore
        # ... (is_rate_limit_exempt remains the same)
        async def predicate(interaction: discord.Interaction) -> bool:
            if not interaction.guild or not isinstance(interaction.user, discord.Member): # Should always be true for guild commands
                return False 
            
            if not hasattr(interaction.client, 'rate_limit_exempt_role_ids_set'): # type: ignore
                logger.error("is_rate_limit_exempt check: Bot is missing 'rate_limit_exempt_role_ids_set'. Denying command.")
                return False

            bot_instance = interaction.client # type: ignore
            user_roles = {role.id for role in interaction.user.roles}
            exempt_roles = bot_instance.rate_limit_exempt_role_ids_set

            if not exempt_roles: 
                return False
            
            return not exempt_roles.isdisjoint(user_roles)
        return app_commands.check(predicate)

    @app_commands.command(name="refresh_status", description="Manually refreshes the bot's activity status.")
    @is_rate_limit_exempt()
    async def refresh_status_command(self, interaction: discord.Interaction):
        # ... (refresh_status_command remains the same)
        logger.info(f"/refresh_status called by {interaction.user.name} ({interaction.user.id})")
        
        await interaction.response.defer(ephemeral=True, thinking=True)

        success = await self._trigger_activity_update(interaction=interaction)
        
        if not success and interaction.response.is_done():
            try:
                # Check if followup was already sent by _trigger_activity_update for an error
                # A bit of a defensive check; ideally _trigger_activity_update sends all followups.
                # If no followup has been sent, send a generic one.
                # This is hard to perfectly check without more complex state.
                # Assuming _trigger_activity_update tried to send a message on error paths.
                pass
            except discord.NotFound: # Should not happen with defer
                 logger.debug("No followup message detected from _trigger_activity_update on failure, sending generic for /refresh_status.")
                 await interaction.followup.send("Status refresh command was processed, but an issue was encountered.", ephemeral=True)


    @refresh_status_command.error
    async def refresh_status_command_error(self, interaction: discord.Interaction, error: app_commands.AppCommandError):
        # ... (command error handler remains the same)
        send_method = interaction.followup.send

        if isinstance(error, app_commands.CheckFailure):
            try:
                await send_method("You do not have the necessary permissions to use this command.", ephemeral=True)
            except discord.HTTPException: pass 
            logger.warning(f"User {interaction.user.name} ({interaction.user.id}) failed permission check for /refresh_status.")
        else: 
            try:
                await send_method(f"An error occurred: {str(error)}", ephemeral=True)
            except discord.HTTPException: pass
            logger.error(f"Error in /refresh_status command by {interaction.user.name}: {error}", exc_info=True)


async def setup(bot: 'AIBot'):
    # ... (setup remains the same)
    if not hasattr(bot, 'api_client') or not bot.api_client:
        logger.critical("ActivityCog cannot be loaded: Bot instance is missing 'api_client'.")
        return
    if not hasattr(bot, 'base_activity_system_prompt'):
        logger.critical("ActivityCog cannot be loaded: Bot instance is missing 'base_activity_system_prompt'.")
        return
    if not hasattr(bot, 'activity_update_interval_seconds'):
        logger.critical("ActivityCog cannot be loaded: Bot instance is missing 'activity_update_interval_seconds'.")
        return
    if hasattr(bot, 'activity_schedule_enabled') and bot.activity_schedule_enabled:
        if not hasattr(bot, 'activity_active_start_hour_utc') or \
           not hasattr(bot, 'activity_active_end_hour_utc') or \
           not hasattr(bot, 'activity_active_days_utc'):
            logger.warning("ActivityCog: Scheduling is enabled, but one or more scheduling time attributes are missing from bot instance.")
    if not hasattr(bot, 'rate_limit_exempt_role_ids_set'):
        logger.warning("ActivityCog: Bot instance is missing 'rate_limit_exempt_role_ids_set'. /refresh_status command checks may fail or deny all.")

    await bot.add_cog(ActivityCog(bot))