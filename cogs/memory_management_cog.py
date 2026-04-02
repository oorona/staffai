import logging
from typing import TYPE_CHECKING, List, Optional

import discord
from discord import app_commands
from discord.ext import commands

if TYPE_CHECKING:
    from bot import AIBot
    from utils.user_memory_manager import UserMemoryManager

logger = logging.getLogger(__name__)

_ITEMS_PER_PAGE = 5


def _build_embed(items: List[str], page: int) -> discord.Embed:
    total_pages = max(1, (len(items) + _ITEMS_PER_PAGE - 1) // _ITEMS_PER_PAGE)
    start = page * _ITEMS_PER_PAGE
    page_items = items[start: start + _ITEMS_PER_PAGE]

    if page_items:
        lines = "\n".join(f"`{start + i + 1}.` {item}" for i, item in enumerate(page_items))
    else:
        lines = "*No memory stored.*"

    embed = discord.Embed(title="Your Memory", description=lines, color=discord.Color.blurple())
    embed.set_footer(text=f"Page {page + 1}/{total_pages} • {len(items)} item(s) total • Only visible to you")
    return embed


class _DeleteItemModal(discord.ui.Modal, title="Delete a memory item"):
    number = discord.ui.TextInput(
        label="Item number to delete",
        placeholder="e.g. 3",
        min_length=1,
        max_length=4,
    )

    def __init__(self, view: "_MemoryView") -> None:
        super().__init__()
        self._view = view

    async def on_submit(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer()
        try:
            idx = int(self.number.value.strip()) - 1
        except ValueError:
            await interaction.followup.send("Please enter a valid number.", ephemeral=True)
            return

        if idx < 0 or idx >= len(self._view.items):
            await interaction.followup.send(
                f"Number must be between 1 and {len(self._view.items)}.", ephemeral=True
            )
            return

        success = await self._view.manager.delete_memory_item(self._view.user_id, idx)
        if not success:
            await interaction.followup.send("Could not delete that item.", ephemeral=True)
            return

        self._view.items = await self._view.manager.get_memory_items(self._view.user_id)
        total_pages = max(1, (len(self._view.items) + _ITEMS_PER_PAGE - 1) // _ITEMS_PER_PAGE)
        if self._view.page >= total_pages:
            self._view.page = max(0, total_pages - 1)
        self._view._refresh_buttons()

        if self._view.message:
            await self._view.message.edit(embed=_build_embed(self._view.items, self._view.page), view=self._view)


class _MemoryView(discord.ui.View):
    def __init__(self, user_id: int, items: List[str], manager: "UserMemoryManager") -> None:
        super().__init__(timeout=180)
        self.user_id = user_id
        self.items = items
        self.manager = manager
        self.page = 0
        self.message: Optional[discord.Message] = None
        self._refresh_buttons()

    def _total_pages(self) -> int:
        return max(1, (len(self.items) + _ITEMS_PER_PAGE - 1) // _ITEMS_PER_PAGE)

    def _refresh_buttons(self) -> None:
        self.prev_btn.disabled = self.page == 0
        self.next_btn.disabled = self.page >= self._total_pages() - 1
        has_items = bool(self.items)
        self.delete_btn.disabled = not has_items
        self.delete_all_btn.disabled = not has_items

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("This menu is not for you.", ephemeral=True)
            return False
        return True

    async def on_timeout(self) -> None:
        for child in self.children:
            child.disabled = True  # type: ignore[union-attr]
        if self.message:
            try:
                await self.message.edit(view=self)
            except Exception:
                pass

    @discord.ui.button(label="←", style=discord.ButtonStyle.secondary, row=0)
    async def prev_btn(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        self.page = max(0, self.page - 1)
        self._refresh_buttons()
        await interaction.response.edit_message(embed=_build_embed(self.items, self.page), view=self)

    @discord.ui.button(label="→", style=discord.ButtonStyle.secondary, row=0)
    async def next_btn(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        self.page = min(self._total_pages() - 1, self.page + 1)
        self._refresh_buttons()
        await interaction.response.edit_message(embed=_build_embed(self.items, self.page), view=self)

    @discord.ui.button(label="🗑 Delete item", style=discord.ButtonStyle.danger, row=1)
    async def delete_btn(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.send_modal(_DeleteItemModal(self))

    @discord.ui.button(label="Delete All", style=discord.ButtonStyle.danger, row=1)
    async def delete_all_btn(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self.manager.clear_memory(self.user_id)
        self.items = []
        self.page = 0
        self._refresh_buttons()
        embed = discord.Embed(
            title="Your Memory",
            description="All memory cleared.",
            color=discord.Color.red(),
        )
        embed.set_footer(text="Only visible to you")
        await interaction.response.edit_message(embed=embed, view=self)

    @discord.ui.button(label="✕ Exit", style=discord.ButtonStyle.secondary, row=2)
    async def exit_btn(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        for child in self.children:
            child.disabled = True  # type: ignore[union-attr]
        await interaction.response.edit_message(view=self)
        self.stop()


class MemoryManagementCog(commands.Cog):
    """Slash command for users to view and manage their stored memory."""

    def __init__(self, bot: "AIBot"):
        self.bot = bot

    @app_commands.guild_only()
    @app_commands.command(name="memory", description="View and manage your stored memory")
    async def memory_command(self, interaction: discord.Interaction):
        memory_manager = getattr(self.bot, "user_memory_manager", None)
        if not memory_manager:
            await interaction.response.send_message("Memory system is not configured.", ephemeral=True)
            return

        items = await memory_manager.get_memory_items(interaction.user.id)
        view = _MemoryView(user_id=interaction.user.id, items=items, manager=memory_manager)
        embed = _build_embed(items, page=0)
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)
        view.message = await interaction.original_response()


async def setup(bot: "AIBot"):
    await bot.add_cog(MemoryManagementCog(bot))
