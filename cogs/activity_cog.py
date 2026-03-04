# cogs/activity_cog.py
import discord
from discord.ext import commands, tasks
from discord import app_commands
import asyncio
import logging
import json
import os
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
        self.activity_response_schema = self._load_activity_response_schema()
        
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

    def _load_activity_response_schema(self) -> Dict[str, Any]:
        """Load dedicated structured-output schema for activity generation."""
        schema_path = os.path.join(self.bot.prompts_root_path, "activity_status", "schema.json")
        try:
            with open(schema_path, "r", encoding="utf-8") as schema_file:
                schema = json.load(schema_file)
            logger.info("Loaded activity status schema from %s", schema_path)
            return schema
        except Exception as e:
            logger.error("Failed loading activity status schema at %s: %s", schema_path, e, exc_info=True)
            # Safe fallback: single-field schema for one status string.
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "activity_status_response_fallback",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"status_text": {"type": "string"}},
                        "required": ["status_text"],
                        "additionalProperties": False
                    }
                }
            }
    
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

    async def _send_interaction_feedback(
        self,
        interaction: discord.Interaction,
        message: str
    ) -> None:
        """Send interaction feedback safely whether initial response is done or deferred."""
        try:
            if interaction.response.is_done():
                await interaction.followup.send(message, ephemeral=True)
            else:
                await interaction.response.send_message(message, ephemeral=True)
        except Exception as e:
            logger.error(f"Failed to send interaction feedback: {e}")
    
    async def _trigger_activity_update(
        self, 
        interaction: Optional[discord.Interaction] = None,
        send_interaction_feedback: bool = True
    ) -> bool:
        """Generate and set new activity status"""
        if not self.bot.base_activity_system_prompt:
            logger.warning("Activity update disabled: no system prompt")
            if interaction and send_interaction_feedback:
                await self._send_interaction_feedback(interaction, "Activity updates not configured")
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
            user_template = self.bot.activity_user_prompt_template or (
                "Generate a creative status text for: {{ACTIVITY_DESCRIPTION}}. "
                "Keep it under 30 words, be witty and brief."
            )
            user_prompt = user_template.replace("{{ACTIVITY_DESCRIPTION}}", description)
            
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
                response_result = await self.bot.litellm_client.chat_completion(
                    messages=messages,
                    use_structured_output=True,
                    track_calls=True,
                    response_schema_override=self.activity_response_schema,
                    call_context={
                        "user_name": (
                            (getattr(interaction.user, "display_name", None) or interaction.user.name)
                            if interaction else "activity_scheduler"
                        ),
                        "channel_name": interaction.channel.name if interaction and interaction.channel else "presence_loop",
                        "guild_name": interaction.guild.name if interaction and interaction.guild else "n/a",
                        "source": "activity_status",
                        "interaction_case": "activity_status_generation",
                    },
                )

                if isinstance(response_result, tuple):
                    response_obj, call_metadata = response_result
                else:
                    response_obj, call_metadata = response_result, []
                
                if not response_obj:
                    logger.error("LLM returned None response")
                    return False
                
                # Extract and parse JSON content from structured output
                message_content = response_obj.choices[0].message.content
                cleaned_content = (message_content or "").strip()
                if cleaned_content.startswith("```"):
                    lines = cleaned_content.split("\n")
                    if lines and lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    cleaned_content = "\n".join(lines).strip()
                response_dict = json.loads(cleaned_content) if cleaned_content else None
                await self._record_activity_llm_audit(
                    messages=messages,
                    response_content=cleaned_content or message_content,
                    usage=getattr(response_obj, "usage", None),
                    call_metadata=call_metadata,
                )
                
                if not response_dict:
                    logger.error("LLM returned empty response")
                    return False
                    
            except Exception as e:
                logger.error(f"Error calling LLM: {e}", exc_info=True)
                if interaction and send_interaction_feedback:
                    await self._send_interaction_feedback(
                        interaction,
                        f"Activity update failed: {str(e)}"
                    )
                return False

            # Extract activity text from response (new minimal schema + legacy compatibility)
            activity_text = str(
                response_dict.get("status_text")
                or response_dict.get("response")
                or ""
            ).strip()
            
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
                if interaction and send_interaction_feedback:
                    await self._send_interaction_feedback(
                        interaction,
                        f"Activity updated: {activity_name} **{activity_text}**"
                    )
                
                return True
            
            except discord.HTTPException as e:
                logger.error(f"Discord API error setting activity: {e}")
                if interaction and send_interaction_feedback:
                    await self._send_interaction_feedback(
                        interaction,
                        f"Failed to set activity: {e}"
                    )
                return False
        
        except Exception as e:
            logger.error(f"Error in activity update: {e}", exc_info=True)
            if interaction and send_interaction_feedback:
                await self._send_interaction_feedback(
                    interaction,
                    f"Activity update error: {str(e)}"
                )
            return False

    async def _record_activity_llm_audit(
        self,
        *,
        messages: List[Dict[str, str]],
        response_content: Optional[str],
        usage: Any,
        call_metadata: List[Dict[str, Any]],
    ) -> None:
        if not getattr(self.bot, "llm_call_audit_enabled", False):
            return
        redis_client = getattr(self.bot, "redis_client", None)
        if not redis_client:
            return
        guild_id = self.bot.guilds[0].id if self.bot.guilds else 0
        try:
            payload = {
                "ts": datetime.datetime.now(datetime.timezone.utc).timestamp(),
                "guild_id": guild_id,
                "user_id": 0,
                "channel_id": 0,
                "model": self.bot.litellm_client.model,
                "interaction_case": "activity_status_generation",
                "was_random": False,
                "is_topic_thread": False,
                "memory_injected": False,
                "memory_update_status": "n/a",
                "tokens": {
                    "prompt": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "completion": getattr(usage, "completion_tokens", 0) if usage else 0,
                    "total": getattr(usage, "total_tokens", 0) if usage else 0,
                },
                "call_count": len(call_metadata) if call_metadata else 1,
                "llm_calls": call_metadata or [],
                "context_messages": self.bot.litellm_client._truncate_for_call_audit(messages),
                "response_type": "json",
                "response_text": (response_content[:800] + "...") if response_content and len(response_content) > 800 else (response_content or ""),
                "response_data": "",
            }
            key = f"llm_calls:recent:{guild_id}"
            await asyncio.to_thread(redis_client.lpush, key, json.dumps(payload, ensure_ascii=False))
            await asyncio.to_thread(
                redis_client.ltrim,
                key,
                0,
                int(getattr(self.bot, "llm_call_audit_max_entries", 100)) - 1
            )
        except Exception as e:
            logger.error("Failed to record activity LLM audit: %s", e, exc_info=True)
    
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
    
    @app_commands.default_permissions(administrator=True)
    @app_commands.guild_only()
    @app_commands.command(
        name="refresh_status",
        description="Reload prompts and refresh bot activity (Admin only)"
    )
    async def refresh_status_command(self, interaction: discord.Interaction):
        """Slash command to reload prompts and manually refresh bot activity."""
        # Check if user has administrator permissions
        if not interaction.user.guild_permissions.administrator:  # type: ignore
            await interaction.response.send_message(
                "❌ This command requires administrator permissions.",
                ephemeral=True
            )
            return

        await interaction.response.defer(ephemeral=True)

        # Reload prompt files first
        prompt_ok, prompt_msg = self.bot.reload_prompts()
        if not prompt_ok:
            await interaction.followup.send(f"❌ {prompt_msg}", ephemeral=True)
            return

        # Refresh activity without sending intermediate interaction messages
        activity_ok = await self._trigger_activity_update(
            interaction=interaction,
            send_interaction_feedback=False
        )

        if activity_ok:
            await interaction.followup.send(
                (
                    f"✅ {prompt_msg}\n"
                    f"✅ Activity updated: {self.current_activity_type_name} **{self.current_activity_text}**"
                ),
                ephemeral=True
            )
            return

        await interaction.followup.send(
            (
                f"✅ {prompt_msg}\n"
                "⚠️ Activity refresh failed. Check logs for details."
            ),
            ephemeral=True
        )
    
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
