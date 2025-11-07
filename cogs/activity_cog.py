# cogs/activity_cog.py
import discord
from discord.ext import commands, tasks
from discord import app_commands
import logging
import json
import random
import datetime
from typing import TYPE_CHECKING, Optional, List, Dict, Any

if TYPE_CHECKING:
    from bot import AIBot

logger = logging.getLogger(__name__)


class ActivityCog(commands.Cog):
    """Manages bot activity/presence updates using LLM"""
    
    def __init__(self, bot: 'AIBot'):
        self.bot = bot
        self.current_activity_text = "Initializing..."
        self.current_activity_type_name = "Playing"
        self.current_online_status = discord.Status.online
        
        # Activity type options
        self.activity_options: List[Dict[str, Any]] = [
            {
                "type_enum": discord.ActivityType.playing,
                "name": "Playing",
                "description": "a game title or activity"
            },
            {
                "type_enum": discord.ActivityType.listening,
                "name": "Listening to",
                "description": "a song, podcast, or sound"
            },
            {
                "type_enum": discord.ActivityType.watching,
                "name": "Watching",
                "description": "a show, movie, or video"
            },
            {
                "type_enum": discord.ActivityType.custom,
                "name": "Custom",
                "description": "a general status message",
                "use_custom": True
            }
        ]
        
        # Start activity update loop if configured
        if (self.bot.base_activity_system_prompt and 
            self.bot.activity_update_interval_seconds > 0):
            
            self.update_bot_activity_loop.change_interval(
                seconds=self.bot.activity_update_interval_seconds
            )
            self.update_bot_activity_loop.start()
            logger.info(
                f"Activity updates enabled: interval={self.bot.activity_update_interval_seconds}s"
            )
            
            if self.bot.activity_schedule_enabled:
                logger.info(
                    f"Schedule: Days={sorted(list(self.bot.activity_active_days_utc))}, "
                    f"Hours UTC={self.bot.activity_active_start_hour_utc:02d}-"
                    f"{self.bot.activity_active_end_hour_utc:02d}"
                )
        else:
            logger.info("Activity updates disabled")
    
    def is_active_time(self) -> bool:
        """Check if current time is within active schedule"""
        if not self.bot.activity_schedule_enabled:
            self.current_online_status = discord.Status.online
            return True
        
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        current_hour = now_utc.hour
        current_weekday = now_utc.weekday()  # Monday=0, Sunday=6
        
        # Check if active day
        if current_weekday not in self.bot.activity_active_days_utc:
            self.current_online_status = discord.Status.idle
            return False
        
        # Check if active hour
        start_hour = self.bot.activity_active_start_hour_utc
        end_hour = self.bot.activity_active_end_hour_utc
        
        is_active = False
        if start_hour <= end_hour:  # Same day (e.g., 9-17)
            is_active = start_hour <= current_hour < end_hour
        else:  # Overnight (e.g., 22-6)
            is_active = current_hour >= start_hour or current_hour < end_hour
        
        self.current_online_status = discord.Status.online if is_active else discord.Status.idle
        return is_active
    
    async def _trigger_activity_update(
        self, 
        interaction: Optional[discord.Interaction] = None
    ) -> bool:
        """Generate and set new activity status"""
        if not self.bot.base_activity_system_prompt:
            logger.warning("Activity update disabled: no system prompt")
            if interaction and not interaction.response.is_done():
                try:
                    await interaction.response.send_message(
                        "Activity updates not configured",
                        ephemeral=True
                    )
                except Exception:
                    pass
            return False
        
        try:
            # Select random activity type
            selected = random.choice(self.activity_options)
            activity_type = selected["type_enum"]
            activity_name = selected["name"]
            description = selected["description"]
            use_custom = selected.get("use_custom", False)
            
            source = f"Manual by {interaction.user.name}" if interaction else "Loop"
            logger.info(f"Activity update ({source}): type={activity_name}")
            
            # Build prompt for LLM
            user_prompt = (
                f"Generate a creative status text for: {description}. "
                f"Keep it under 30 words, be witty and brief."
            )
            
            messages = [
                {
                    "role": "system",
                    "content": self.bot.base_activity_system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
            
            # Call LLM with standard structured output
            logger.debug("Calling LLM for activity generation")
            try:
                response = await self.bot.litellm_client.chat_completion(
                    messages=messages,
                    use_structured_output=True
                )
                
                if not response:
                    logger.error("LLM returned None response")
                    return False
                
                # Extract the message content (should be JSON from structured output)
                message_content = response.choices[0].message.content
                response_dict = json.loads(message_content) if message_content else None
                
                if not response_dict:
                    logger.error("LLM returned empty response")
                    return False
                    
            except Exception as e:
                logger.error(f"Error calling LLM: {e}", exc_info=True)
                if interaction and not interaction.response.is_done():
                    try:
                        await interaction.response.send_message(
                            f"Activity update failed: {str(e)}",
                            ephemeral=True
                        )
                    except Exception:
                        pass
                return False
            
            # Extract activity text from response
            activity_text = response_dict.get("response", "").strip()
            
            if not activity_text:
                logger.warning("LLM returned empty activity text")
                activity_text = "Thinking..."
            
            # Truncate if needed
            if len(activity_text) > 100:
                activity_text = activity_text[:97] + "..."
            
            # Set Discord activity
            try:
                if use_custom:
                    # Custom activity
                    new_activity = discord.CustomActivity(name=activity_text)
                else:
                    # Standard activity
                    new_activity = discord.Activity(
                        type=activity_type,
                        name=activity_text
                    )
                
                await self.bot.change_presence(
                    activity=new_activity,
                    status=self.current_online_status
                )
                
                self.current_activity_text = activity_text
                self.current_activity_type_name = activity_name
                
                logger.info(f"Activity updated: {activity_name} '{activity_text}'")
                
                # Send confirmation if manual
                if interaction and not interaction.response.is_done():
                    try:
                        await interaction.response.send_message(
                            f"Activity updated: {activity_name} **{activity_text}**",
                            ephemeral=True
                        )
                    except Exception as e:
                        logger.error(f"Failed to send confirmation: {e}")
                
                return True
            
            except discord.HTTPException as e:
                logger.error(f"Discord API error setting activity: {e}")
                if interaction and not interaction.response.is_done():
                    try:
                        await interaction.response.send_message(
                            f"Failed to set activity: {e}",
                            ephemeral=True
                        )
                    except Exception:
                        pass
                return False
        
        except Exception as e:
            logger.error(f"Error in activity update: {e}", exc_info=True)
            if interaction and not interaction.response.is_done():
                try:
                    await interaction.response.send_message(
                        f"Activity update error: {str(e)}",
                        ephemeral=True
                    )
                except Exception:
                    pass
            return False
    
    @tasks.loop(seconds=300)  # Default interval, changed in __init__
    async def update_bot_activity_loop(self):
        """Periodic activity updates"""
        if not self.is_active_time():
            logger.debug("Outside active hours, skipping activity update")
            return
        
        await self._trigger_activity_update()
    
    @update_bot_activity_loop.before_loop
    async def before_activity_loop(self):
        """Wait for bot to be ready"""
        await self.bot.wait_until_ready()
    
    @app_commands.command(name="refresh_status", description="Manually refresh bot activity")
    async def refresh_status_command(self, interaction: discord.Interaction):
        """Slash command to manually refresh bot status"""
        await self._trigger_activity_update(interaction)
    
    @app_commands.command(name="current_status", description="Show current bot activity")
    async def current_status_command(self, interaction: discord.Interaction):
        """Slash command to show current status"""
        await interaction.response.send_message(
            f"**Current Status:**\n"
            f"Type: {self.current_activity_type_name}\n"
            f"Text: {self.current_activity_text}\n"
            f"Online Status: {self.current_online_status}",
            ephemeral=True
        )


async def setup(bot: 'AIBot'):
    await bot.add_cog(ActivityCog(bot))
