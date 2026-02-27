import logging
from typing import TYPE_CHECKING

import discord
from discord import app_commands
from discord.ext import commands

if TYPE_CHECKING:
    from bot import AIBot

logger = logging.getLogger(__name__)


class UserMemoryCog(commands.Cog):
    """User-facing command to inspect personal bot memory."""

    def __init__(self, bot: "AIBot"):
        self.bot = bot

    @app_commands.guild_only()
    @app_commands.command(name="my_memory", description="See what memory the bot has stored about you")
    async def my_memory_command(self, interaction: discord.Interaction):
        memory_manager = getattr(self.bot, "user_memory_manager", None)
        if not memory_manager:
            await interaction.response.send_message(
                "Memory system is not configured.",
                ephemeral=True
            )
            return

        memory = await memory_manager.get_memory(interaction.user.id)
        if not memory:
            await interaction.response.send_message(
                "I don't have notable memory about you yet.",
                ephemeral=True
            )
            return

        if len(memory) > 1800:
            memory = memory[:1800] + "..."

        embed = discord.Embed(
            title="Your Memory",
            description=memory,
            color=discord.Color.blurple()
        )
        embed.set_footer(text="Only visible to you")
        await interaction.response.send_message(embed=embed, ephemeral=True)


async def setup(bot: "AIBot"):
    await bot.add_cog(UserMemoryCog(bot))

