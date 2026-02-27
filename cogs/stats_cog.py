# cogs/stats_cog.py
import discord
from discord.ext import commands, tasks
from discord import app_commands
import asyncio
import logging
import time
import redis
from typing import TYPE_CHECKING, Optional, List, Tuple
from datetime import datetime, timezone

if TYPE_CHECKING:
    from bot import AIBot

logger = logging.getLogger(__name__)

# Model pricing (per 1M tokens)
# Format: "model_name": (input_cost, output_cost)
MODEL_PRICING = {
    # Gemini models
    "gemini/gemini-2.5-flash": (0.0375, 0.15),  # Flash models
    "gemini/gemini-2.5-flash-lite": (0.01875, 0.075),  # Flash-lite (half of flash)
    "gemini/gemini-1.5-pro": (1.25, 5.00),
    "gemini/gemini-1.5-flash": (0.075, 0.30),

    # OpenAI models
    "openai/gpt-5-mini": (0.10, 0.40),
    "openai/gpt-5-nano": (0.05, 0.20),
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4-turbo": (10.00, 30.00),
    "openai/gpt-3.5-turbo": (0.50, 1.50),

    # XAI models
    "xai/grok-3-mini": (0.15, 0.60),
    "xai/grok-3-mini-fast": (0.10, 0.40),

    # Claude models
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
    "claude-3-opus": (15.00, 75.00),

    # Default fallback (if model not found)
    "default": (0.15, 0.60)
}


class StatsCog(commands.Cog):
    """Manages token consumption statistics and admin reporting"""

    def __init__(self, bot: 'AIBot'):
        self.bot: 'AIBot' = bot
        self.redis_client: Optional[redis.Redis] = self.bot.redis_client
        
        logger.info("StatsCog initialized")
        
        # Start token report loop if configured
        if (self.bot.stats_report_channel_id and 
            self.bot.stats_report_interval_seconds > 0 and 
            self.redis_client):
            
            self.send_token_report_loop.change_interval(
                seconds=self.bot.stats_report_interval_seconds
            )
            self.send_token_report_loop.start()
            logger.info(
                f"Token stats reporting enabled: "
                f"channel={self.bot.stats_report_channel_id}, "
                f"interval={self.bot.stats_report_interval_seconds}s, "
                f"top_users={self.bot.stats_report_top_users}"
            )
        else:
            logger.info("Token stats reporting disabled")
    
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int, model_name: str) -> float:
        """
        Calculate the cost for a given token usage.

        Args:
            prompt_tokens: Number of input/prompt tokens
            completion_tokens: Number of output/completion tokens
            model_name: Name of the model used

        Returns:
            Cost in USD
        """
        if model_name in MODEL_PRICING:
            input_cost_per_million, output_cost_per_million = MODEL_PRICING[model_name]
        else:
            input_cost_per_million, output_cost_per_million = MODEL_PRICING["default"]

        prompt_cost = (prompt_tokens / 1_000_000) * input_cost_per_million
        completion_cost = (completion_tokens / 1_000_000) * output_cost_per_million

        return prompt_cost + completion_cost

    async def record_token_usage(
        self,
        user_id: int,
        guild_id: int,
        tokens: int,
        message_type: str = "chat",
        cached_tokens: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0
    ):
        """
        Record token usage for a user and update global cost counter.

        Args:
            user_id: Discord user ID
            guild_id: Discord guild ID
            tokens: Total number of tokens consumed
            message_type: Type of interaction (chat, random, mention, etc.)
            cached_tokens: Number of cached tokens (not billed)
            prompt_tokens: Number of input tokens (for accurate cost calculation)
            completion_tokens: Number of output tokens (for accurate cost calculation)
        """
        if not self.redis_client or tokens <= 0:
            return

        try:
            current_time = time.time()

            # Calculate accurate cost if we have prompt/completion breakdown
            if prompt_tokens > 0 and completion_tokens > 0:
                cost = self.calculate_cost(prompt_tokens, completion_tokens, self.bot.litellm_client.model)
            else:
                # Fallback to average pricing if we don't have the breakdown
                model_name = self.bot.litellm_client.model
                if model_name in MODEL_PRICING:
                    input_cost, output_cost = MODEL_PRICING[model_name]
                    avg_cost_per_million = (input_cost + output_cost) / 2
                else:
                    input_cost, output_cost = MODEL_PRICING["default"]
                    avg_cost_per_million = (input_cost + output_cost) / 2

                billed_tokens = tokens - cached_tokens
                cost = (billed_tokens / 1_000_000) * avg_cost_per_million

            # Update global cost counter
            global_cost_key = f"token_stats:global_cost:{guild_id}"
            await asyncio.to_thread(
                self.redis_client.incrbyfloat,
                global_cost_key,
                cost
            )

            # Update daily global cost
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            daily_global_cost_key = f"token_stats:daily_global_cost:{guild_id}:{today}"
            await asyncio.to_thread(
                self.redis_client.incrbyfloat,
                daily_global_cost_key,
                cost
            )
            # Expire daily cost after 7 days
            await asyncio.to_thread(
                self.redis_client.expire,
                daily_global_cost_key,
                604800  # 7 days
            )
            
            # Key for total token consumption per user
            total_key = f"token_stats:total:{guild_id}:{user_id}"
            total_cached_key = f"token_stats:total_cached:{guild_id}:{user_id}"
            
            # Key for daily token consumption
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            daily_key = f"token_stats:daily:{guild_id}:{today}:{user_id}"
            daily_cached_key = f"token_stats:daily_cached:{guild_id}:{today}:{user_id}"
            
            # Key for detailed usage log (sorted set with timestamps)
            log_key = f"token_stats:log:{guild_id}:{user_id}"
            
            # Increment total tokens
            await asyncio.to_thread(
                self.redis_client.incrby,
                total_key,
                tokens
            )
            
            # Increment total cached tokens
            if cached_tokens > 0:
                await asyncio.to_thread(
                    self.redis_client.incrby,
                    total_cached_key,
                    cached_tokens
                )
            
            # Increment daily tokens
            await asyncio.to_thread(
                self.redis_client.incrby,
                daily_key,
                tokens
            )
            
            # Increment daily cached tokens
            if cached_tokens > 0:
                await asyncio.to_thread(
                    self.redis_client.incrby,
                    daily_cached_key,
                    cached_tokens
                )
                # Daily cached key expires after 7 days
                await asyncio.to_thread(
                    self.redis_client.expire,
                    daily_cached_key,
                    604800  # 7 days
                )
            # Daily key expires after 7 days
            await asyncio.to_thread(
                self.redis_client.expire,
                daily_key,
                604800  # 7 days
            )
            
            # Add to usage log (sorted set: score=timestamp, value=tokens:type)
            log_entry = f"{tokens}:{message_type}"
            await asyncio.to_thread(
                self.redis_client.zadd,
                log_key,
                {log_entry: current_time}
            )
            
            # Keep only last 1000 entries per user
            await asyncio.to_thread(
                self.redis_client.zremrangebyrank,
                log_key,
                0,
                -1001
            )
            
            logger.debug(f"Recorded {tokens} tokens for user {user_id} ({message_type})")
            
        except Exception as e:
            logger.error(f"Error recording token usage: {e}", exc_info=True)

    async def get_global_cost_stats(self, guild_id: int) -> Tuple[float, float]:
        """
        Get global cost statistics for the guild.

        Args:
            guild_id: Discord guild ID

        Returns:
            Tuple of (total_cost, daily_cost)
        """
        if not self.redis_client:
            return (0.0, 0.0)

        try:
            # Get total cost
            global_cost_key = f"token_stats:global_cost:{guild_id}"
            total_cost = await asyncio.to_thread(
                self.redis_client.get,
                global_cost_key
            )
            total_cost = float(total_cost) if total_cost else 0.0

            # Get daily cost
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            daily_global_cost_key = f"token_stats:daily_global_cost:{guild_id}:{today}"
            daily_cost = await asyncio.to_thread(
                self.redis_client.get,
                daily_global_cost_key
            )
            daily_cost = float(daily_cost) if daily_cost else 0.0

            return (total_cost, daily_cost)

        except Exception as e:
            logger.error(f"Error getting global cost stats: {e}", exc_info=True)
            return (0.0, 0.0)

    async def get_user_token_stats(
        self,
        user_id: int,
        guild_id: int
    ) -> Tuple[int, int, int, int, List[Tuple[int, str, float]]]:
        """
        Get token consumption stats for a user.
        
        Args:
            user_id: Discord user ID
            guild_id: Discord guild ID
        
        Returns:
            Tuple of (total_tokens, daily_tokens, total_cached_tokens, daily_cached_tokens, recent_usage_log)
            recent_usage_log is list of (tokens, type, timestamp)
        """
        if not self.redis_client:
            return (0, 0, 0, 0, [])
        
        try:
            # Get total tokens
            total_key = f"token_stats:total:{guild_id}:{user_id}"
            total_tokens = await asyncio.to_thread(
                self.redis_client.get,
                total_key
            )
            total_tokens = int(total_tokens) if total_tokens else 0
            
            # Get total cached tokens
            total_cached_key = f"token_stats:total_cached:{guild_id}:{user_id}"
            total_cached_tokens = await asyncio.to_thread(
                self.redis_client.get,
                total_cached_key
            )
            total_cached_tokens = int(total_cached_tokens) if total_cached_tokens else 0
            
            # Get daily tokens
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            daily_key = f"token_stats:daily:{guild_id}:{today}:{user_id}"
            daily_tokens = await asyncio.to_thread(
                self.redis_client.get,
                daily_key
            )
            daily_tokens = int(daily_tokens) if daily_tokens else 0
            
            # Get daily cached tokens
            daily_cached_key = f"token_stats:daily_cached:{guild_id}:{today}:{user_id}"
            daily_cached_tokens = await asyncio.to_thread(
                self.redis_client.get,
                daily_cached_key
            )
            daily_cached_tokens = int(daily_cached_tokens) if daily_cached_tokens else 0
            
            # Get recent usage log (last 10 entries)
            log_key = f"token_stats:log:{guild_id}:{user_id}"
            log_entries = await asyncio.to_thread(
                self.redis_client.zrevrange,
                log_key,
                0,
                9,
                withscores=True
            )
            
            # Parse log entries
            recent_usage = []
            for entry, timestamp in log_entries:
                entry_str = entry.decode('utf-8') if isinstance(entry, bytes) else entry
                parts = entry_str.split(':', 1)
                tokens = int(parts[0])
                msg_type = parts[1] if len(parts) > 1 else "unknown"
                recent_usage.append((tokens, msg_type, timestamp))
            
            return (total_tokens, daily_tokens, total_cached_tokens, daily_cached_tokens, recent_usage)
            
        except Exception as e:
            logger.error(f"Error getting user token stats: {e}", exc_info=True)
            return (0, 0, 0, 0, [])
    
    async def get_top_users_by_tokens(
        self,
        guild_id: int,
        limit: int = 10,
        period: str = "total"
    ) -> List[Tuple[int, int, int]]:
        """
        Get top users by token consumption.
        
        Args:
            guild_id: Discord guild ID
            limit: Number of top users to return
            period: "total" or "daily"
        
        Returns:
            List of (user_id, tokens, cached_tokens) tuples, sorted by tokens descending
        """
        if not self.redis_client:
            return []
        
        try:
            if period == "daily":
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                pattern = f"token_stats:daily:{guild_id}:{today}:*"
            else:
                pattern = f"token_stats:total:{guild_id}:*"
            
            # Scan for all user token keys
            cursor = 0
            user_tokens = []
            
            while True:
                cursor, keys = await asyncio.to_thread(
                    self.redis_client.scan,
                    cursor,
                    match=pattern,
                    count=100
                )
                
                for key in keys:
                    try:
                        # Extract user_id from key
                        key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                        user_id = int(key_str.split(':')[-1])
                        
                        # Get token count
                        tokens = await asyncio.to_thread(
                            self.redis_client.get,
                            key
                        )
                        tokens = int(tokens) if tokens else 0
                        
                        # Get cached token count
                        if period == "daily":
                            cached_key = f"token_stats:daily_cached:{guild_id}:{today}:{user_id}"
                        else:
                            cached_key = f"token_stats:total_cached:{guild_id}:{user_id}"
                        cached_tokens = await asyncio.to_thread(
                            self.redis_client.get,
                            cached_key
                        )
                        cached_tokens = int(cached_tokens) if cached_tokens else 0
                        
                        if tokens > 0:
                            user_tokens.append((user_id, tokens, cached_tokens))
                    
                    except Exception as e:
                        logger.error(f"Error processing key {key}: {e}")
                
                if cursor == 0:
                    break
            
            # Sort by tokens descending and return top N
            user_tokens.sort(key=lambda x: x[1], reverse=True)
            return user_tokens[:limit]
            
        except Exception as e:
            logger.error(f"Error getting top users: {e}", exc_info=True)
            return []
    
    @app_commands.command(
        name="tokenstats",
        description="View token consumption statistics (admins can check any user)"
    )
    @app_commands.describe(
        user="User to check stats for (defaults to yourself, admins can check anyone)"
    )
    async def tokenstats_command(
        self,
        interaction: discord.Interaction,
        user: Optional[discord.User] = None
    ):
        """Slash command to view user token consumption stats"""
        # Check if user is in a server
        if not isinstance(interaction.user, discord.Member):
            await interaction.response.send_message(
                "This command can only be used in a server.",
                ephemeral=True
            )
            return

        member = interaction.user
        has_admin = False

        # Check if user has super role (which includes admin permissions)
        if self.bot.super_role_ids_set:
            user_role_ids = {role.id for role in member.roles}
            has_admin = not self.bot.super_role_ids_set.isdisjoint(user_role_ids)

        # Determine target user
        # - If no user specified, default to the person who called the command
        # - If user specified and caller is admin, allow checking that user
        # - If user specified and caller is NOT admin, only allow checking themselves
        if user is None:
            target_user = interaction.user
        elif has_admin:
            # Admin can check anyone
            target_user = user
        else:
            # Non-admin tried to check someone else
            if user.id != interaction.user.id:
                await interaction.response.send_message(
                    "⛔ You can only check your own token stats. Admins can check stats for other users.",
                    ephemeral=True
                )
                return
            target_user = user

        guild_id = interaction.guild_id
        
        if not guild_id:
            await interaction.response.send_message(
                "This command can only be used in a server.",
                ephemeral=True
            )
            return
        
        # Get user stats
        total_tokens, daily_tokens, total_cached_tokens, daily_cached_tokens, recent_usage = await self.get_user_token_stats(
            target_user.id,
            guild_id
        )

        # Get global costs
        global_total_cost, global_daily_cost = await self.get_global_cost_stats(guild_id)

        # Build embed
        embed = discord.Embed(
            title=f"📊 Token Consumption Stats",
            description=f"Statistics for {target_user.mention}",
            color=discord.Color.blue(),
            timestamp=datetime.now(timezone.utc)
        )
        
        embed.add_field(
            name="Total Tokens (All Time)",
            value=f"**{total_tokens:,}** tokens\n({total_cached_tokens:,} cached)",
            inline=True
        )
        
        embed.add_field(
            name="Today's Usage",
            value=f"**{daily_tokens:,}** tokens\n({daily_cached_tokens:,} cached)",
            inline=True
        )
        
        # Calculate cost based on model pricing
        # Note: This uses average pricing since we don't track input/output separately
        model_name = self.bot.litellm_client.model
        if model_name in MODEL_PRICING:
            input_cost, output_cost = MODEL_PRICING[model_name]
            # Use average of input and output cost as approximation
            avg_cost_per_million = (input_cost + output_cost) / 2
        else:
            # Fallback to default pricing
            input_cost, output_cost = MODEL_PRICING["default"]
            avg_cost_per_million = (input_cost + output_cost) / 2

        # Cost is based on billed tokens (total - cached)
        billed_total_tokens = total_tokens - total_cached_tokens
        billed_daily_tokens = daily_tokens - daily_cached_tokens
        total_cost = (billed_total_tokens / 1_000_000) * avg_cost_per_million
        daily_cost = (billed_daily_tokens / 1_000_000) * avg_cost_per_million
        
        embed.add_field(
            name="Estimated Cost",
            value=f"Total: ${total_cost:.6f}\nToday: ${daily_cost:.6f}\n\n_Model: {self.bot.litellm_client.model}_",
            inline=True
        )
        
        # Recent usage
        if recent_usage:
            usage_lines = []
            for tokens, msg_type, timestamp in recent_usage[:5]:
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                time_str = dt.strftime("%m/%d %H:%M")
                usage_lines.append(f"`{time_str}` • {tokens:4d} tokens • {msg_type}")
            
            embed.add_field(
                name="Recent Usage (Last 5)",
                value="\n".join(usage_lines),
                inline=False
            )
        
        # Add global cost info
        if has_admin:
            embed.add_field(
                name="🌍 Server Totals (Admin Info)",
                value=f"Total: ${global_total_cost:.2f}\nToday: ${global_daily_cost:.2f}",
                inline=False
            )

        embed.set_thumbnail(url=target_user.display_avatar.url)
        embed.set_footer(text=f"Requested by {interaction.user.name} • Model: {self.bot.litellm_client.model}")
        
        await interaction.response.send_message(embed=embed, ephemeral=True)
    
    @tasks.loop(seconds=3600)  # Default 1 hour, changed in __init__
    async def send_token_report_loop(self):
        """Periodically send token consumption report to notification channel"""
        if not self.redis_client or not self.bot.stats_report_channel_id:
            return
        
        try:
            channel = self.bot.get_channel(self.bot.stats_report_channel_id)
            if not channel or not isinstance(channel, discord.TextChannel):
                logger.warning(f"Stats report channel {self.bot.stats_report_channel_id} not found")
                return
            
            # Get guild ID from channel
            guild_id = channel.guild.id
            
            # Get top users
            top_users = await self.get_top_users_by_tokens(
                guild_id,
                limit=self.bot.stats_report_top_users,
                period="daily"
            )
            
            if not top_users:
                logger.debug("No token usage to report")
                return
            
            # Build embed
            embed = discord.Embed(
                title="📊 Token Consumption Report",
                description=f"Top {len(top_users)} users by daily token usage",
                color=discord.Color.gold(),
                timestamp=datetime.now(timezone.utc)
            )
            
            # Add top users
            rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
            
            for idx, (user_id, tokens, daily_cached_tokens) in enumerate(top_users):
                try:
                    user = await self.bot.fetch_user(user_id)
                    user_name = user.name
                except Exception:
                    user_name = f"User {user_id}"
                
                emoji = rank_emoji[idx] if idx < len(rank_emoji) else f"{idx+1}."
                
                # Get total tokens for comparison
                total_tokens, _, total_cached_tokens, _, _ = await self.get_user_token_stats(user_id, guild_id)

                # Calculate cost using model pricing
                model_name = self.bot.litellm_client.model
                if model_name in MODEL_PRICING:
                    input_cost, output_cost = MODEL_PRICING[model_name]
                    avg_cost_per_million = (input_cost + output_cost) / 2
                else:
                    input_cost, output_cost = MODEL_PRICING["default"]
                    avg_cost_per_million = (input_cost + output_cost) / 2

                billed_daily_tokens = tokens - daily_cached_tokens
                daily_cost = (billed_daily_tokens / 1_000_000) * avg_cost_per_million
                
                embed.add_field(
                    name=f"{emoji} {user_name}",
                    value=f"Today: **{tokens:,}** tokens ({daily_cached_tokens:,} cached, ${daily_cost:.6f})\nTotal: {total_tokens:,} tokens ({total_cached_tokens:,} cached)",
                    inline=False
                )
            
            # Calculate totals
            total_daily = sum(tokens for _, tokens, _ in top_users)
            total_daily_cached = sum(cached for _, _, cached in top_users)
            total_daily_billed = total_daily - total_daily_cached

            # Use model pricing for total cost
            model_name = self.bot.litellm_client.model
            if model_name in MODEL_PRICING:
                input_cost, output_cost = MODEL_PRICING[model_name]
                avg_cost_per_million = (input_cost + output_cost) / 2
            else:
                input_cost, output_cost = MODEL_PRICING["default"]
                avg_cost_per_million = (input_cost + output_cost) / 2

            total_daily_cost = (total_daily_billed / 1_000_000) * avg_cost_per_million
            
            embed.add_field(
                name="📈 Summary",
                value=f"Total (Top {len(top_users)}): **{total_daily:,}** tokens ({total_daily_cached:,} cached)\nEstimated Cost: **${total_daily_cost:.6f}**\n\n_Model: {self.bot.litellm_client.model}_",
                inline=False
            )

            embed.set_footer(text=f"Token stats tracked per user per guild • Model pricing: ${avg_cost_per_million:.4f} per 1M tokens (avg)")
            
            await channel.send(embed=embed)
            logger.info(f"Sent token consumption report to channel {channel.name}")
            
        except Exception as e:
            logger.error(f"Error sending token report: {e}", exc_info=True)
    
    @send_token_report_loop.before_loop
    async def before_token_report(self):
        """Wait for bot to be ready before starting loop"""
        await self.bot.wait_until_ready()

    @app_commands.command(name="refresh_mcp_tools", description="Refresh MCP tools cache (Admin only)")
    async def refresh_mcp_tools(self, interaction: discord.Interaction):
        """Refresh the cached MCP tools from all configured servers"""
        # Check if user has administrator permissions
        if not interaction.user.guild_permissions.administrator:  # type: ignore
            await interaction.response.send_message(
                "❌ This command requires administrator permissions.", 
                ephemeral=True
            )
            return
        
        await interaction.response.defer(ephemeral=True)
        
        try:
            if not self.bot.mcp_servers:
                await interaction.followup.send("⚠️ No MCP servers configured.", ephemeral=True)
                return
            
            # Clear cache to force reload (including failed servers)
            self.bot.litellm_client._mcp_tools_cache = None
            self.bot.litellm_client._mcp_tools_cache_time = 0
            self.bot.litellm_client._mcp_failed_servers.clear()  # Retry all servers, even previously failed ones
            self.bot.litellm_client._tool_to_server_map.clear()  # Clear tool-to-server mapping

            # Load tools fresh
            logger.info(f"Admin {interaction.user} requested MCP tools refresh - retrying ALL servers")
            mcp_tools = await self.bot.litellm_client.get_mcp_tools()

            # Get server status details
            total_configured = len(self.bot.mcp_servers)
            failed_count = len(self.bot.litellm_client._mcp_failed_servers)
            successful_count = total_configured - failed_count

            if mcp_tools:
                status_msg = f"✅ Successfully refreshed {len(mcp_tools)} MCP tools\n"
                status_msg += f"📊 Servers: {successful_count} successful"
                if failed_count > 0:
                    status_msg += f", {failed_count} failed"
                status_msg += f" (out of {total_configured} total)"

                await interaction.followup.send(status_msg, ephemeral=True)
                logger.info(f"✅ MCP tools refreshed by admin: {len(mcp_tools)} tools from {successful_count}/{total_configured} servers")
            else:
                await interaction.followup.send(
                    f"⚠️ No MCP tools were loaded. All {total_configured} servers failed to connect.\nCheck server connectivity and logs.",
                    ephemeral=True
                )
                logger.warning(f"⚠️ MCP tools refresh resulted in no tools loaded ({failed_count}/{total_configured} servers failed)")
                
        except Exception as e:
            await interaction.followup.send(
                f"❌ Error refreshing MCP tools: {str(e)}",
                ephemeral=True
            )
            logger.error(f"❌ Error during MCP tools refresh: {e}", exc_info=True)

    @app_commands.command(name="status", description="Check bot status and statistics (Admin only)")
    async def status_command(self, interaction: discord.Interaction):
        """Check bot status and statistics"""
        # Check if user has administrator permissions
        if not interaction.user.guild_permissions.administrator:  # type: ignore
            await interaction.response.send_message(
                "❌ This command requires administrator permissions.", 
                ephemeral=True
            )
            return
        
        await interaction.response.defer(ephemeral=True)
        
        try:
            # Get system info
            cpu_percent = self.bot.process.cpu_percent()
            memory_mb = self.bot.process.memory_info().rss / 1024 / 1024

            # Calculate uptime
            uptime_seconds = int(time.time() - self.bot.start_time)
            uptime_minutes = uptime_seconds // 60
            uptime_hours = uptime_minutes // 60
            uptime_days = uptime_hours // 24

            if uptime_days > 0:
                uptime_str = f"{uptime_days}d {uptime_hours % 24}h {uptime_minutes % 60}m"
            elif uptime_hours > 0:
                uptime_str = f"{uptime_hours}h {uptime_minutes % 60}m"
            else:
                uptime_str = f"{uptime_minutes}m {uptime_seconds % 60}s"

            # Count guilds and members
            guild_count = len(self.bot.guilds)
            member_count = sum(guild.member_count or 0 for guild in self.bot.guilds)

            # Redis status
            redis_status = "Conectado" if self.bot.redis_client else "Desconectado"

            # Get token stats if available
            total_tokens_str = "N/A"
            total_cost_str = "N/A"
            if self.redis_client and self.bot.guilds:
                try:
                    # Get stats from the current guild
                    guild_id = interaction.guild_id or self.bot.guilds[0].id

                    # Get all-time token usage from Redis
                    global_key = f"tokens:guild:{guild_id}:total"
                    total_tokens = await asyncio.to_thread(self.redis_client.get, global_key)
                    total_tokens = int(total_tokens) if total_tokens else 0

                    # Get cached tokens
                    cached_key = f"tokens:guild:{guild_id}:total_cached"
                    cached_tokens = await asyncio.to_thread(self.redis_client.get, cached_key)
                    cached_tokens = int(cached_tokens) if cached_tokens else 0

                    # Calculate cost (only billed tokens)
                    billed_tokens = total_tokens - cached_tokens
                    
                    # Use model pricing
                    model_name = self.bot.litellm_client.model
                    if model_name in MODEL_PRICING:
                        input_cost, output_cost = MODEL_PRICING[model_name]
                        avg_cost_per_million = (input_cost + output_cost) / 2
                    else:
                        input_cost, output_cost = MODEL_PRICING["default"]
                        avg_cost_per_million = (input_cost + output_cost) / 2
                        
                    total_cost = (billed_tokens / 1_000_000) * avg_cost_per_million

                    total_tokens_str = f"{total_tokens:,} ({cached_tokens:,} cached)"
                    total_cost_str = f"${total_cost:.4f}"
                except Exception as e:
                    logger.error(f"Error fetching token stats for status: {e}")

            # Create embed
            embed = discord.Embed(
                title="🤖 Bot Status",
                description=f"Status report for {self.bot.user.name}",  # type: ignore
                color=discord.Color.blue(),
                timestamp=datetime.now(timezone.utc)
            )

            server_info = (
                f"Servidores: {guild_count}\n"
                f"Usuarios: {member_count:,}\n"
                f"Uptime: {uptime_str}"
            )
            database_info = (
                f"Estado: {redis_status}\n"
                f"MCP Servers: {len(self.bot.mcp_servers)}"
            )
            system_info = (
                f"Discord.py: {discord.__version__}\n"
                f"Modelo: {self.bot.litellm_client.model}\n"
                f"CPU: {cpu_percent:.1f}%\n"
                f"Memoria: {memory_mb:.1f} MB"
            )

            embed.add_field(
                name="📊 Info del Servidor",
                value=server_info,
                inline=True
            )
            embed.add_field(
                name="🗃️ Base de Datos",
                value=database_info,
                inline=True
            )
            embed.add_field(
                name="⚙️ Sistema",
                value=system_info,
                inline=True
            )

            # Token consumption stats
            embed.add_field(
                name="🎯 Consumo de Tokens (Guild)",
                value=(
                    f"Total: {total_tokens_str}\n"
                    f"Costo Estimado: {total_cost_str}"
                ),
                inline=False
            )

            embed.set_footer(text=f"Requested by {interaction.user.name} • Status Check") 

            await interaction.followup.send(embed=embed, ephemeral=True)
            
        except Exception as e:
            await interaction.followup.send(
                f"❌ Error fetching status: {str(e)}",
                ephemeral=True
            )
            logger.error(f"❌ Error during status check: {e}", exc_info=True)


async def setup(bot: 'AIBot'):
    await bot.add_cog(StatsCog(bot))
