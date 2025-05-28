# cogs/activity_cog.py
import discord
from discord.ext import commands, tasks
import logging
import random
from typing import TYPE_CHECKING, Optional, Dict, Any, List, Set # Added Set
import datetime

# For slash commands and permission checks
from discord import app_commands

if TYPE_CHECKING:
    from bot import AIBot
    # from utils.webui_api import WebUIAPI # Not directly used here, but bot.api_client is

logger = logging.getLogger(__name__)

class ActivityCog(commands.Cog):
    def __init__(self, bot: 'AIBot'):
        self.bot = bot
        self.current_activity_text = "Initializing..."
        self.current_activity_type_name = "Playing" # Default display type name
        self.current_online_status = discord.Status.online

        self.activity_options: List[Dict[str, Any]] = [
            {"type_enum": discord.ActivityType.playing, "name_for_discord": "Playing", "prompt_verb": "a game title or a general status update that fits 'Playing X'", "requires_emoji": False, "use_custom_activity_class": False},
            {"type_enum": discord.ActivityType.listening, "name_for_discord": "Listening to", "prompt_verb": "a song title, podcast name, or interesting sound", "requires_emoji": False, "use_custom_activity_class": False},
            {"type_enum": discord.ActivityType.watching, "name_for_discord": "Watching", "prompt_verb": "a show title, movie name, or intriguing video", "requires_emoji": False, "use_custom_activity_class": False},
            {"type_enum": discord.ActivityType.custom, "name_for_discord": "Custom", "prompt_verb": "a general custom status message (this will NOT be prefixed by 'Playing', 'Watching' etc.)", "requires_emoji": True, "use_custom_activity_class": True},
        ]

        if not self.bot.base_activity_system_prompt:
            logger.warning("ActivityCog: Bot is missing 'base_activity_system_prompt'. Activity updates will be disabled.")
            return # Do not start the loop

        if self.bot.activity_update_interval_seconds > 0:
            self.update_bot_activity_loop.change_interval(seconds=self.bot.activity_update_interval_seconds)
            self.update_bot_activity_loop.start()
            logger.info(f"ActivityCog: Bot activity update task scheduled. Interval: {self.bot.activity_update_interval_seconds}s")
            if self.bot.activity_schedule_enabled:
                 logger.info(f"ActivityCog: Schedule enabled. Active Mon-Sun (0-6): {sorted(list(self.bot.activity_active_days_utc))}, Hours UTC: {self.bot.activity_active_start_hour_utc:02d}-{self.bot.activity_active_end_hour_utc:02d}")
        else:
            logger.info("ActivityCog: Bot activity update task disabled (interval is 0 or not set).")

    def is_active_time(self) -> bool:
        if not self.bot.activity_schedule_enabled:
            self.current_online_status = discord.Status.online # Default to online if schedule disabled
            return True
        
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        current_hour_utc = now_utc.hour
        current_weekday_utc = now_utc.weekday() # Monday is 0 and Sunday is 6

        if current_weekday_utc not in self.bot.activity_active_days_utc:
            self.current_online_status = discord.Status.idle
            return False # Not an active day

        start_hour = self.bot.activity_active_start_hour_utc
        end_hour = self.bot.activity_active_end_hour_utc
        
        is_active_hour = False
        if start_hour <= end_hour: # Same day range (e.g., 9-17)
            is_active_hour = start_hour <= current_hour_utc < end_hour
        else: # Overnight range (e.g., 22-06)
            is_active_hour = current_hour_utc >= start_hour or current_hour_utc < end_hour
        
        self.current_online_status = discord.Status.online if is_active_hour else discord.Status.idle
        return is_active_hour

    async def _trigger_activity_update(self, interaction: Optional[discord.Interaction] = None) -> bool:
        if not self.bot.base_activity_system_prompt or not self.activity_options:
            logger.warning("ActivityCog: Cannot trigger activity update: base_activity_system_prompt or activity_options missing.")
            if interaction and not interaction.response.is_done(): 
                try:
                    if interaction.response.is_done(): 
                         await interaction.followup.send("Activity update system not properly configured.", ephemeral=True)
                except discord.HTTPException as e:
                    logger.error(f"ActivityCog: HTTPExc trying to send config error to user {interaction.user if interaction else 'N/A'}: {e}")
            return False

        try:
            selected_config = random.choice(self.activity_options)
            activity_enum_for_discord = selected_config["type_enum"]
            activity_type_name_for_discord = selected_config["name_for_discord"]
            prompt_verb_for_llm = selected_config["prompt_verb"]
            needs_emoji_from_llm = selected_config["requires_emoji"]
            use_custom_class = selected_config["use_custom_activity_class"]

            log_source = f"Manual refresh by {interaction.user.name}" if interaction else "Loop"
            logger.info(f"ActivityCog: {log_source} selected type: {activity_type_name_for_discord}")

            dynamic_instruction = (
                f"Your task is to generate ONLY the creative text for a bot's activity status. "
                f"This text should be suitable content for '{prompt_verb_for_llm}'. "
            )
            if use_custom_class:
                 dynamic_instruction += "This will be displayed as a general status, not prefixed by 'Playing', 'Watching', etc. "
            else:
                 dynamic_instruction += f"For example, if the task is to provide a game title, just provide the title like 'Cosmic Chess Masters'. Do NOT include prefixes like '{activity_type_name_for_discord}' yourself. "

            if needs_emoji_from_llm:
                dynamic_instruction += "\nAdditionally, please suggest a single Unicode emoji in the 'data.emoji_suggestion' field that complements this status text. The main status text itself should still be just the creative text without the emoji prepended by you."
            else:
                dynamic_instruction += "\nFor this task, the 'data.emoji_suggestion' field in your JSON response should be null (an emoji is not required for this activity type)."

            final_system_prompt = f"{self.bot.base_activity_system_prompt}\n\nCURRENT TASK:\n{dynamic_instruction}"
            user_trigger_prompt = "Generate an activity status."
            
            llm_output_dict, error_msg_api, tokens_used = await self.bot.api_client.generate_response(
                user_id="BOT_ACTIVITY_UPDATER_INTERNAL_V2", 
                channel_id="BOT_ACTIVITY_CHANNEL_INTERNAL_V2",
                prompt=user_trigger_prompt,
                system_message=final_system_prompt,
                history=[], 
                extra_assistant_context=None,
                tools_to_use=[] 
            )

            error_occurred = False
            error_message_for_user = "Could not update activity due to an LLM issue."

            if error_msg_api: 
                logger.error(f"ActivityCog: LLM API error during activity generation: {error_msg_api}. LLM output (if any): {llm_output_dict}")
                error_occurred = True
                error_message_for_user = f"LLM API error: {error_msg_api}. Bot's response: {llm_output_dict.get('response') if llm_output_dict else 'N/A'}"
            elif not isinstance(llm_output_dict.get("type"), str) or llm_output_dict.get("type") != "text" or \
                 not isinstance(llm_output_dict.get("response"), str):
                logger.error(f"ActivityCog: LLM output missing 'type: text' or 'response' string fields. Output: {llm_output_dict}")
                error_occurred = True
                error_message_for_user = "LLM returned an unexpected data structure for activity."
            
            main_activity_text_from_llm = ""
            if not error_occurred:
                main_activity_text_from_llm = llm_output_dict.get("response", "")
                if not (main_activity_text_from_llm and main_activity_text_from_llm.strip()):
                    logger.warning(f"ActivityCog: LLM did not return a valid main 'response' text for activity. Full output: {llm_output_dict}")
                    error_occurred = True
                    error_message_for_user = "LLM did not provide valid activity text."

            if error_occurred:
                if interaction and interaction.response.is_done(): await interaction.followup.send(error_message_for_user, ephemeral=True)
                activity_to_set_on_error = discord.Game(name=self.current_activity_text if self.current_activity_text and self.current_activity_text != "Initializing..." else "Error fetching status...")
                await self.bot.change_presence(status=self.current_online_status, activity=activity_to_set_on_error)
                return False

            llm_emoji_suggestion = None
            data_field = llm_output_dict.get("data")
            if isinstance(data_field, dict): llm_emoji_suggestion = data_field.get("emoji_suggestion")

            activity_text_to_display_on_discord = main_activity_text_from_llm.strip()[:128]
            activity_object_to_set = None
            current_activity_display_name = activity_type_name_for_discord

            if use_custom_class:
                activity_args = {"name": activity_text_to_display_on_discord}
                current_activity_display_name = "Custom Status"
                if needs_emoji_from_llm and llm_emoji_suggestion and isinstance(llm_emoji_suggestion, str) and llm_emoji_suggestion.strip():
                    parsed_emoji = llm_emoji_suggestion.strip().split(" ")[0]
                    if len(parsed_emoji) == 1 or (parsed_emoji.startswith("<") and parsed_emoji.endswith(">")):
                         activity_args["emoji"] = parsed_emoji
                    else:
                        logger.warning(f"ActivityCog: LLM emoji suggestion '{llm_emoji_suggestion}' invalid for CustomActivity. Using text only.")
                
                try:
                    if "emoji" in activity_args:
                        activity_object_to_set = discord.CustomActivity(name=activity_args["name"], emoji=activity_args.get("emoji"))
                    else:
                        activity_object_to_set = discord.CustomActivity(name=activity_args["name"])
                except Exception as e_cust_activity:
                    logger.error(f"ActivityCog: Error creating discord.CustomActivity: {e_cust_activity}. Falling back to discord.Game.")
                    activity_object_to_set = discord.Game(name=f"{activity_args.get('emoji', '')} {activity_args['name']}".strip()[:128])
                    current_activity_display_name = "Playing" 

            elif activity_enum_for_discord == discord.ActivityType.playing:
                activity_object_to_set = discord.Game(name=activity_text_to_display_on_discord)
            elif activity_enum_for_discord == discord.ActivityType.listening:
                activity_object_to_set = discord.Activity(type=discord.ActivityType.listening, name=activity_text_to_display_on_discord)
            elif activity_enum_for_discord == discord.ActivityType.watching:
                activity_object_to_set = discord.Activity(type=discord.ActivityType.watching, name=activity_text_to_display_on_discord)

            if activity_object_to_set:
                status_to_set = discord.Status.online if interaction or self.is_active_time() else self.current_online_status
                if not interaction: status_to_set = self.current_online_status

                await self.bot.change_presence(status=status_to_set, activity=activity_object_to_set)
                self.current_activity_text = activity_text_to_display_on_discord
                self.current_activity_type_name = current_activity_display_name
                
                log_text_for_activity = getattr(activity_object_to_set, 'name', 'Unknown Activity')
                if hasattr(activity_object_to_set, 'emoji') and getattr(activity_object_to_set, 'emoji', None): 
                    log_text_for_activity = f"{activity_object_to_set.emoji} {log_text_for_activity}" # type: ignore

                logger.info(f"ActivityCog: Bot activity {'manually refreshed' if interaction else 'updated'} to ({self.current_activity_type_name}): '{log_text_for_activity}'. Status: {status_to_set}. Tokens: {tokens_used or 'N/A'}")

                if interaction and interaction.response.is_done():
                    success_message = f"Bot status refreshed to: **{self.current_activity_type_name} {log_text_for_activity}** (Status: {status_to_set})"
                    await interaction.followup.send(success_message, ephemeral=True)
                return True
            else: 
                logger.warning(f"ActivityCog: Failed to create an activity object even after LLM success. LLM output: {llm_output_dict}")
                if interaction and interaction.response.is_done(): await interaction.followup.send("Failed to create a valid activity object from LLM response.", ephemeral=True)
                return False

        except Exception as e:
            logger.error(f"ActivityCog: Unexpected error in _trigger_activity_update: {e}", exc_info=True)
            if interaction and interaction.response.is_done(): await interaction.followup.send(f"An unexpected error occurred: {e}", ephemeral=True)
            try: await self.bot.change_presence(status=self.current_online_status, activity=discord.Game(name="Idle | Error processing activity"))
            except Exception as presence_error: logger.error(f"ActivityCog: Failed to set fallback presence after error: {presence_error}")
            return False

    @tasks.loop()
    async def update_bot_activity_loop(self):
        is_currently_active_time = self.is_active_time()

        if not is_currently_active_time:
            current_loop_count = self.update_bot_activity_loop.current_loop
            if current_loop_count is not None and current_loop_count % 12 == 0 :
                 logger.debug(f"ActivityCog: Outside of active hours/days. Skipping activity update. Status: {self.current_online_status}. Loop: {current_loop_count}")
            elif current_loop_count is None:
                 logger.debug(f"ActivityCog: Outside of active hours/days on first check. Status: {self.current_online_status}")

            try:
                current_presence_activity = self.bot.guilds[0].me.activity if self.bot.guilds and self.bot.guilds[0].me else None
                current_presence_status = self.bot.guilds[0].me.status if self.bot.guilds and self.bot.guilds[0].me else None
                resting_activity_name = "Resting... 💤"

                if not (isinstance(current_presence_activity, discord.Game) and current_presence_activity.name == resting_activity_name and current_presence_status == discord.Status.idle):
                    resting_activity = discord.Game(name=resting_activity_name)
                    await self.bot.change_presence(status=discord.Status.idle, activity=resting_activity)
                    logger.info(f"ActivityCog: Bot status set to Idle and activity to '{resting_activity_name}' (inactive period).")
            except Exception as e:
                logger.error(f"ActivityCog: Failed to set resting presence: {e}", exc_info=True)
            return

        await self._trigger_activity_update()

    @update_bot_activity_loop.before_loop
    async def before_update_bot_activity_loop(self):
        await self.bot.wait_until_ready()
        logger.info("ActivityCog: Bot is ready. Activity update loop will now start if configured.")
        try:
            self.is_active_time() 
            initial_activity_name = self.current_activity_text if self.current_activity_text != "Initializing..." else "Booting up..."
            initial_activity = discord.Game(name=initial_activity_name)
            await self.bot.change_presence(status=self.current_online_status, activity=initial_activity)
            logger.info(f"ActivityCog: Initial presence set. Status: {self.current_online_status}, Activity: '{initial_activity.name}'")
        except Exception as e:
            logger.error(f"ActivityCog: Failed to set initial presence: {e}", exc_info=True)

    async def cog_unload(self):
        self.update_bot_activity_loop.cancel()
        logger.info("ActivityCog: Unloaded, activity update task cancelled.")

    def is_super_user(): # Renamed from is_rate_limit_exempt
        async def predicate(interaction: discord.Interaction) -> bool:
            if not interaction.guild or not isinstance(interaction.user, discord.Member):
                return False 
            
            if not hasattr(interaction.client, 'super_role_ids_set'): # type: ignore
                logger.error("is_super_user check: Bot is missing 'super_role_ids_set'. Denying command.")
                return False

            bot_instance = interaction.client # type: ignore
            user_roles = {role.id for role in interaction.user.roles} # type: ignore
            privileged_roles = bot_instance.super_role_ids_set # type: ignore

            if not privileged_roles: 
                return False
            
            return not privileged_roles.isdisjoint(user_roles)
        return app_commands.check(predicate)

    @app_commands.command(name="refresh_status", description="Manually refreshes the bot's activity status.")
    @is_super_user() # Uses the renamed predicate
    async def refresh_status_command(self, interaction: discord.Interaction):
        logger.info(f"/refresh_status called by {interaction.user.name} ({interaction.user.id})")
        
        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=True, thinking=True)
        
        await self._trigger_activity_update(interaction=interaction)

    @refresh_status_command.error
    async def refresh_status_command_error(self, interaction: discord.Interaction, error: app_commands.AppCommandError):
        send_method = interaction.followup.send
        if not interaction.response.is_done():
            try: await interaction.response.defer(ephemeral=True)
            except discord.HTTPException: logger.warning(f"Failed to defer for error handling /refresh_status by {interaction.user.name}"); return

        if isinstance(error, app_commands.CheckFailure):
            try: await send_method("You do not have the necessary permissions to use this command.", ephemeral=True)
            except discord.HTTPException: pass 
            logger.warning(f"User {interaction.user.name} failed permission check (is_super_user) for /refresh_status.")
        else: 
            try: await send_method(f"An error occurred: {str(error)}", ephemeral=True)
            except discord.HTTPException: pass
            logger.error(f"Error in /refresh_status command by {interaction.user.name}: {error}", exc_info=True)

async def setup(bot: 'AIBot'):
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
        if not all(hasattr(bot, attr) for attr in ['activity_active_start_hour_utc', 'activity_active_end_hour_utc', 'activity_active_days_utc']):
            logger.warning("ActivityCog: Scheduling enabled, but one or more time attributes are missing.")
    if not hasattr(bot, 'super_role_ids_set'): # Check for the new attribute name
        logger.warning("ActivityCog: Bot instance is missing 'super_role_ids_set'. /refresh_status checks may fail.")

    await bot.add_cog(ActivityCog(bot))
    logger.info("ActivityCog loaded successfully.")