# cogs/profile_cog.py
import discord
from discord.ext import commands
from discord import app_commands # For slash commands
import redis # For type hinting, actual client is from bot
import json
import io
import logging
from typing import TYPE_CHECKING, List, Dict, Optional, Union # Added Union
import numpy as np 
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for matplotlib BEFORE pyplot import
import matplotlib.pyplot as plt


if TYPE_CHECKING:
    from bot import AIBot # Your AIBot class

logger = logging.getLogger(__name__)

# Minimum number of scored messages required to generate a profile
MIN_SCORED_MESSAGES_FOR_PROFILE = 5 # Example value, can be made configurable later if needed

class ProfileCog(commands.Cog):
    def __init__(self, bot: 'AIBot'):
        self.bot = bot
        self.score_dimensions = [
            "warmth", "humor", "helpful", "civility", 
            "engagement", "creativity", "insightfulness"
        ]
        # Max score value for each dimension (e.g., if scores are 1-5)
        self.max_score_value = 5 
        self.radar_chart_color = "#7289DA" # Discord blurple

    def _generate_profile_chart(self, average_scores: Dict[str, float], user_name: str) -> Optional[io.BytesIO]:
        labels = np.array(self.score_dimensions)
        num_vars = len(labels)

        stats_values = np.array([average_scores.get(dim, 0.0) for dim in self.score_dimensions])

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        
        # The stats array needs to be repeated for the last point to close the circle.
        stats_values_closed = np.concatenate((stats_values,[stats_values[0]]))
        angles_closed = angles + angles[:1]

        # Define colors based on LaTeX scheme
        fig_bg_color = '#40444B' # Dark gray background (same as LaTeX bg)
        text_color = '#FFFFFF'   # White text (same as LaTeX fg)
        grid_color = '#6A6E73'   # A slightly lighter gray for grid lines for visibility
        plot_line_color = self.radar_chart_color # e.g., '#7289DA' (Discord blurple for the data plot)

        # No specific plt.style.use() here, we are manually setting colors.
        # If you were using a style, ensure it doesn't override these.
        plt.style.use('default') # Start from a basic style to avoid conflicts

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        fig.patch.set_facecolor(fig_bg_color) 
        ax.set_facecolor(fig_bg_color)     

        # Plot data
        ax.plot(angles_closed, stats_values_closed, color=plot_line_color, linewidth=2.5, linestyle='solid', marker='o', markersize=7, label=user_name)
        ax.fill(angles_closed, stats_values_closed, color=plot_line_color, alpha=0.35)

        # Set labels (the dimension names)
        ax.set_thetagrids(np.degrees(angles), labels, color=text_color, fontsize=12, weight='bold')
        
        # Radial Ticks (Y-axis labels)
        ax.set_rlabel_position(35) # Adjust position of radial labels
        tick_values = np.linspace(0, self.max_score_value, num=6 if self.max_score_value == 5 else int(self.max_score_value) + 1) 
        ax.set_yticks(tick_values)
        ax.set_yticklabels([f"{val:.0f}" if val.is_integer() else f"{val:.1f}" for val in tick_values], color=text_color, alpha=0.7, fontsize=9) # Whiteish, slightly transparent
        ax.set_ylim(0, self.max_score_value) 

        # Title
        plt.title(f"Interaction Profile: {user_name}", size=18, color=text_color, y=1.12, weight='bold')
        
        # Grid lines
        ax.grid(color=grid_color, linestyle='--', linewidth=0.6, alpha=0.8)

        # Value Annotations at each point
        for i, txt_val in enumerate(stats_values): 
            # Offset slightly for better readability
            offset_amount = self.max_score_value * 0.06 
            ax.text(angles[i], stats_values[i] + offset_amount, f"{stats_values[i]:.1f}", 
                    ha='center', va='bottom', fontsize=10, color=plot_line_color, weight='bold',
                    bbox=dict(facecolor=fig_bg_color, alpha=0.6, pad=1.5, edgecolor='none'))

        try:
            buf = io.BytesIO()
            # Save with the figure's facecolor, ensuring it's not transparent if not desired
            # bbox_inches='tight' helps to prevent labels from being cut off.
            plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', pad_inches=0.2)
            buf.seek(0)
            plt.close(fig) # Important to close the figure to free memory
            return buf
        except Exception as e:
            logger.error(f"Error generating profile chart for {user_name}: {e}", exc_info=True)
            plt.close(fig) # Ensure plot is closed even if error
            return None


    @app_commands.command(name="myprofile", description="Shows your or another member's interaction profile.")
    @app_commands.describe(member="The member whose profile you want to see (optional).")
    async def myprofile_command(self, interaction: discord.Interaction, member: Optional[discord.Member] = None):
        await interaction.response.defer(ephemeral=False) 

        target_user = member if member else interaction.user
        target_user_id = target_user.id
        target_user_name = target_user.display_name # Use display_name for server-specific nicknames

        if not self.bot.redis_client_general:
            logger.error(f"ProfileCog: Redis client not available for /myprofile command by {interaction.user.name}.")
            await interaction.followup.send("Sorry, I couldn't access the profile data right now (database connection issue).", ephemeral=True)
            return

        # Get max messages from bot config, default if not present
        max_messages_to_fetch = getattr(self.bot, 'profile_max_scored_messages', 50)
        if not isinstance(max_messages_to_fetch, int) or max_messages_to_fetch <=0:
            max_messages_to_fetch = 50 # Safe default
            logger.warning(f"profile_max_scored_messages not configured properly on bot, defaulting to {max_messages_to_fetch}")


        redis_key = f"user_profile_messages:{target_user_id}"

        try:
            stored_messages_json_list = await discord.utils.asyncio.to_thread(
                self.bot.redis_client_general.lrange, redis_key, 0, max_messages_to_fetch -1 
            )
        except redis.exceptions.RedisError as e_redis:
            logger.error(f"Redis error fetching profile for {target_user_name} (ID: {target_user_id}): {e_redis}", exc_info=True)
            await interaction.followup.send("Sorry, there was an error retrieving profile data.", ephemeral=True)
            return

        if not stored_messages_json_list or len(stored_messages_json_list) < MIN_SCORED_MESSAGES_FOR_PROFILE:
            count_found = len(stored_messages_json_list) if stored_messages_json_list else 0
            logger.info(f"Not enough data for {target_user_name}. Found {count_found} messages, need {MIN_SCORED_MESSAGES_FOR_PROFILE}.")
            embed_no_data = discord.Embed(
                title=f"Profile for {target_user_name}",
                description=f"Not enough interaction data available yet to generate a profile. (Found {count_found} scored interactions, need at least {MIN_SCORED_MESSAGES_FOR_PROFILE}).",
                color=discord.Color.orange()
            )
            await interaction.followup.send(embed=embed_no_data)
            return

        total_scores: Dict[str, float] = {dim: 0.0 for dim in self.score_dimensions}
        valid_score_sets_count = 0

        for msg_json_str in stored_messages_json_list:
            try:
                msg_data = json.loads(msg_json_str)
                scores = msg_data.get("scores")
                if isinstance(scores, dict):
                    current_entry_valid = True
                    for dim in self.score_dimensions:
                        score_val = scores.get(dim)
                        if isinstance(score_val, (int, float)):
                            total_scores[dim] += float(score_val)
                        else:
                            current_entry_valid = False
                            logger.debug(f"Missing or invalid score for dimension '{dim}' in profile entry for user {target_user_id}. Entry: {scores}")
                            break 
                    if current_entry_valid:
                        valid_score_sets_count += 1
            except json.JSONDecodeError:
                logger.warning(f"Could not parse stored message JSON for user {target_user_id}: {msg_json_str[:100]}")
            except Exception as e_parse:
                 logger.error(f"Unexpected error parsing scores for {target_user_id}: {e_parse}", exc_info=True)

        if valid_score_sets_count < MIN_SCORED_MESSAGES_FOR_PROFILE:
            logger.info(f"Not enough *valid* score sets for {target_user_name} after parsing. Found {valid_score_sets_count}, need {MIN_SCORED_MESSAGES_FOR_PROFILE}.")
            embed_no_valid_data = discord.Embed(
                title=f"Profile for {target_user_name}",
                description=f"Not enough fully valid interaction data available to generate a profile. (Found {valid_score_sets_count} complete score sets, need at least {MIN_SCORED_MESSAGES_FOR_PROFILE}).",
                color=discord.Color.orange()
            )
            await interaction.followup.send(embed=embed_no_valid_data)
            return

        average_scores_dict: Dict[str, float] = {
            dim: (total_scores[dim] / valid_score_sets_count) if valid_score_sets_count > 0 else 0.0
            for dim in self.score_dimensions
        }
        logger.info(f"Average scores for {target_user_name} (from {valid_score_sets_count} interactions): {average_scores_dict}")

        chart_image_buffer = self._generate_profile_chart(average_scores_dict, target_user_name)

        if chart_image_buffer:
            discord_file = discord.File(chart_image_buffer, filename="profile_chart.png")
            embed = discord.Embed(
                title=f"Interaction Profile: {target_user_name}",
                description=f"This chart visualizes average interaction scores based on the last {valid_score_sets_count} analyzed messages.",
                color=discord.Color.from_str(self.radar_chart_color) 
            )
            embed.set_image(url="attachment://profile_chart.png") 
            embed.set_footer(text=f"Max score per dimension: {self.max_score_value}")
            
            for dim_index, dim in enumerate(self.score_dimensions):
                embed.add_field(name=dim.capitalize(), value=f"{average_scores_dict.get(dim, 0.0):.2f}", inline=True)
                # Add a blank field for formatting if needed for 3 per row (7 items)
                if (dim_index + 1) % 3 == 0 and (dim_index + 1) < len(self.score_dimensions):
                    pass # Fields will wrap automatically
            
            await interaction.followup.send(embed=embed, file=discord_file)
        else:
            logger.error(f"Failed to generate profile chart for {target_user_name}.")
            await interaction.followup.send("Sorry, I couldn't generate the profile chart at this moment.", ephemeral=True)

async def setup(bot: 'AIBot'):
    if not hasattr(bot, 'redis_client_general') or not bot.redis_client_general:
        logger.error("ProfileCog cannot be loaded: Bot is missing 'redis_client_general'.")
        return
    if not hasattr(bot, 'profile_max_scored_messages'): # From main.py/bot.py
        logger.error("ProfileCog cannot be loaded: Bot is missing 'profile_max_scored_messages' attribute.")
        return
        
    await bot.add_cog(ProfileCog(bot))
    logger.info("ProfileCog loaded successfully.")