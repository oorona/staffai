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
    
    async def record_token_usage(
        self,
        user_id: int,
        guild_id: int,
        tokens: int,
        message_type: str = "chat"
    ):
        """
        Record token usage for a user.
        
        Args:
            user_id: Discord user ID
            guild_id: Discord guild ID
            tokens: Number of tokens consumed
            message_type: Type of interaction (chat, random, mention, etc.)
        """
        if not self.redis_client or tokens <= 0:
            return
        
        try:
            current_time = time.time()
            
            # Key for total token consumption per user
            total_key = f"token_stats:total:{guild_id}:{user_id}"
            
            # Key for daily token consumption
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            daily_key = f"token_stats:daily:{guild_id}:{today}:{user_id}"
            
            # Key for detailed usage log (sorted set with timestamps)
            log_key = f"token_stats:log:{guild_id}:{user_id}"
            
            # Increment total tokens
            await asyncio.to_thread(
                self.redis_client.incrby,
                total_key,
                tokens
            )
            
            # Increment daily tokens
            await asyncio.to_thread(
                self.redis_client.incrby,
                daily_key,
                tokens
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
    
    async def get_user_token_stats(
        self,
        user_id: int,
        guild_id: int
    ) -> Tuple[int, int, List[Tuple[int, str, float]]]:
        """
        Get token consumption stats for a user.
        
        Args:
            user_id: Discord user ID
            guild_id: Discord guild ID
        
        Returns:
            Tuple of (total_tokens, daily_tokens, recent_usage_log)
            recent_usage_log is list of (tokens, type, timestamp)
        """
        if not self.redis_client:
            return (0, 0, [])
        
        try:
            # Get total tokens
            total_key = f"token_stats:total:{guild_id}:{user_id}"
            total_tokens = await asyncio.to_thread(
                self.redis_client.get,
                total_key
            )
            total_tokens = int(total_tokens) if total_tokens else 0
            
            # Get daily tokens
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            daily_key = f"token_stats:daily:{guild_id}:{today}:{user_id}"
            daily_tokens = await asyncio.to_thread(
                self.redis_client.get,
                daily_key
            )
            daily_tokens = int(daily_tokens) if daily_tokens else 0
            
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
            
            return (total_tokens, daily_tokens, recent_usage)
            
        except Exception as e:
            logger.error(f"Error getting user token stats: {e}", exc_info=True)
            return (0, 0, [])
    
    async def get_top_users_by_tokens(
        self,
        guild_id: int,
        limit: int = 10,
        period: str = "total"
    ) -> List[Tuple[int, int]]:
        """
        Get top users by token consumption.
        
        Args:
            guild_id: Discord guild ID
            limit: Number of top users to return
            period: "total" or "daily"
        
        Returns:
            List of (user_id, tokens) tuples, sorted by tokens descending
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
                        
                        if tokens > 0:
                            user_tokens.append((user_id, tokens))
                    
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
        description="View token consumption statistics for a user (Admin only)"
    )
    @app_commands.describe(
        user="User to check stats for (defaults to yourself)"
    )
    async def tokenstats_command(
        self,
        interaction: discord.Interaction,
        user: Optional[discord.User] = None
    ):
        """Slash command to view user token consumption stats"""
        # Check if user has admin role
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
        
        if not has_admin:
            await interaction.response.send_message(
                "⛔ This command requires admin permissions.",
                ephemeral=True
            )
            return
        
        # Default to interaction user if no user specified
        target_user = user or interaction.user
        guild_id = interaction.guild_id
        
        if not guild_id:
            await interaction.response.send_message(
                "This command can only be used in a server.",
                ephemeral=True
            )
            return
        
        # Get stats
        total_tokens, daily_tokens, recent_usage = await self.get_user_token_stats(
            target_user.id,
            guild_id
        )
        
        # Build embed
        embed = discord.Embed(
            title=f"📊 Token Consumption Stats",
            description=f"Statistics for {target_user.mention}",
            color=discord.Color.blue(),
            timestamp=datetime.now(timezone.utc)
        )
        
        embed.add_field(
            name="Total Tokens (All Time)",
            value=f"**{total_tokens:,}** tokens",
            inline=True
        )
        
        embed.add_field(
            name="Today's Usage",
            value=f"**{daily_tokens:,}** tokens",
            inline=True
        )
        
        # Calculate approximate cost (assuming ~$0.15 per 1M tokens for gpt-4o-mini)
        cost_per_million = 0.15
        total_cost = (total_tokens / 1_000_000) * cost_per_million
        daily_cost = (daily_tokens / 1_000_000) * cost_per_million
        
        embed.add_field(
            name="Estimated Cost",
            value=f"Total: ${total_cost:.4f}\nToday: ${daily_cost:.4f}",
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
        
        embed.set_thumbnail(url=target_user.display_avatar.url)
        embed.set_footer(text=f"Requested by {interaction.user.name}")
        
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
            
            for idx, (user_id, tokens) in enumerate(top_users):
                try:
                    user = await self.bot.fetch_user(user_id)
                    user_name = user.name
                except Exception:
                    user_name = f"User {user_id}"
                
                emoji = rank_emoji[idx] if idx < len(rank_emoji) else f"{idx+1}."
                
                # Get total tokens for comparison
                total_tokens, _, _ = await self.get_user_token_stats(user_id, guild_id)
                
                cost_per_million = 0.15
                daily_cost = (tokens / 1_000_000) * cost_per_million
                
                embed.add_field(
                    name=f"{emoji} {user_name}",
                    value=f"Today: **{tokens:,}** tokens (${daily_cost:.4f})\nTotal: {total_tokens:,} tokens",
                    inline=False
                )
            
            # Calculate totals
            total_daily = sum(tokens for _, tokens in top_users)
            total_daily_cost = (total_daily / 1_000_000) * 0.15
            
            embed.add_field(
                name="📈 Summary",
                value=f"Total (Top {len(top_users)}): **{total_daily:,}** tokens\nEstimated Cost: **${total_daily_cost:.4f}**",
                inline=False
            )
            
            embed.set_footer(text="Token stats are tracked per user per guild")
            
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
            
            # Clear cache to force reload
            self.bot.litellm_client._mcp_tools_cache = None
            self.bot.litellm_client._mcp_tools_cache_time = 0
            
            # Load tools fresh
            logger.info(f"Admin {interaction.user} requested MCP tools refresh")
            mcp_tools = await self.bot.litellm_client.get_mcp_tools()
            
            if mcp_tools:
                await interaction.followup.send(
                    f"✅ Successfully refreshed {len(mcp_tools)} MCP tools from {len(self.bot.mcp_servers)} servers.",
                    ephemeral=True
                )
                logger.info(f"✅ MCP tools refreshed by admin: {len(mcp_tools)} tools loaded")
            else:
                await interaction.followup.send(
                    "⚠️ No MCP tools were loaded. Check server connectivity.",
                    ephemeral=True
                )
                logger.warning("⚠️ MCP tools refresh resulted in no tools loaded")
                
        except Exception as e:
            await interaction.followup.send(
                f"❌ Error refreshing MCP tools: {str(e)}",
                ephemeral=True
            )
            logger.error(f"❌ Error during MCP tools refresh: {e}", exc_info=True)


async def setup(bot: 'AIBot'):
    await bot.add_cog(StatsCog(bot))
