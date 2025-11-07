# cogs/message_cog.py
import discord
from discord.ext import commands, tasks
import asyncio
import logging
import time
import redis
from typing import TYPE_CHECKING, Optional
import io
import urllib.parse
import aiohttp

from utils.message_handler import MessageHandler, MessageHandlerResult

if TYPE_CHECKING:
    from bot import AIBot

logger = logging.getLogger(__name__)


class MessageCog(commands.Cog):
    """Handles Discord message events, bot responses, and user restrictions"""
    
    def __init__(self, bot: 'AIBot'):
        self.bot: 'AIBot' = bot
        self.redis_client: Optional[redis.Redis] = self.bot.redis_client
        
        logger.info("MessageCog initialized")
        
        # Start restriction expiry loop if configured
        if (self.bot.restricted_user_role_id and 
            self.bot.restriction_duration_seconds > 0 and 
            self.bot.restriction_check_interval_seconds > 0 and 
            self.redis_client):
            
            self.check_restrictions_loop.change_interval(
                seconds=self.bot.restriction_check_interval_seconds
            )
            self.check_restrictions_loop.start()
            logger.info(
                f"Auto-restriction expiry enabled: "
                f"duration={self.bot.restriction_duration_seconds}s, "
                f"interval={self.bot.restriction_check_interval_seconds}s"
            )
        else:
            logger.info("Auto-restriction expiry disabled")
    
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Handle incoming Discord messages"""
        # Ignore bot messages and DMs
        if message.author == self.bot.user or message.author.bot or not message.guild:
            return
        
        # Ensure author is a Member object
        if not isinstance(message.author, discord.Member):
            try:
                message.author = await message.guild.fetch_member(message.author.id)
            except (discord.NotFound, discord.HTTPException) as e:
                logger.warning(f"Could not resolve member {message.author.id}: {e}")
                return
        
        # Process message through handler
        handler = MessageHandler(self.bot)
        
        async with message.channel.typing():
            try:
                result: MessageHandlerResult = await handler.handle_message(message)
            except Exception as e:
                logger.error(f"Unhandled error in MessageHandler: {e}", exc_info=True)
                try:
                    if message.channel.permissions_for(message.guild.me).send_messages:
                        await message.reply(
                            "Sorry, something went wrong while processing your message.",
                            mention_author=False
                        )
                except Exception:
                    pass
                return
        
        # Log handler result
        if result.get("log_message"):
            logger.info(f"Handler: {result['log_message']}")
        
        # Check if we should respond
        if not result.get("should_respond"):
            if result.get("error"):
                logger.warning(f"Handler error: {result['error']}")
            return
        
        # Check permissions
        if not message.channel.permissions_for(message.guild.me).send_messages:
            logger.warning(f"Missing send_messages permission in {message.channel.name}")
            return
        
        # Handle response based on type
        response_type = result.get("response_type", "text")
        response_text = result.get("response_text", "")
        response_data = result.get("response_data", "")
        
        try:
            if response_type == "text":
                await self._send_text_response(message, response_text)
            
            elif response_type == "url":
                await self._send_url_response(message, response_text, response_data)
            
            elif response_type == "gif":
                await self._send_gif_response(message, response_text, response_data)
            
            elif response_type == "latex":
                await self._send_latex_response(message, response_text, response_data)
            
            elif response_type == "code":
                await self._send_code_response(message, response_text, response_data)
            
            elif response_type == "output":
                await self._send_output_response(message, response_text, response_data)
            
            else:
                logger.warning(f"Unknown response type: {response_type}, sending as text")
                await self._send_text_response(message, response_text)
        
        except Exception as e:
            logger.error(f"Error sending response: {e}", exc_info=True)
    
    async def _send_text_response(self, message: discord.Message, text: str):
        """Send plain text response"""
        if not text:
            return
        
        if len(text) > 2000:
            text = text[:1997] + "..."
        
        await message.reply(text, mention_author=False)
    
    async def _send_url_response(self, message: discord.Message, text: str, url: str):
        """Send response with URL"""
        # Send text first
        if text:
            await self._send_text_response(message, text)
        
        # Send URL in embed
        if url:
            embed = discord.Embed(
                title="Link",
                description=url,
                color=discord.Color.blue()
            )
            
            # If it's an image, embed it
            if any(url.lower().endswith(ext) for ext in ['.gif', '.png', '.jpg', '.jpeg', '.webp']):
                embed.set_image(url=url)
                if not text:  # Only URL was provided
                    embed.description = None
                    embed.title = "Image"
            
            await message.channel.send(embed=embed)
    
    async def _send_gif_response(self, message: discord.Message, text: str, gif_url: str):
        """Send response with GIF"""
        # Send text first
        if text:
            await self._send_text_response(message, text)
        
        # Send GIF
        if gif_url:
            embed = discord.Embed(color=discord.Color.purple())
            embed.set_image(url=gif_url)
            await message.channel.send(embed=embed)
    
    async def _send_latex_response(self, message: discord.Message, text: str, latex: str):
        """Send response with rendered LaTeX"""
        # Send text first
        if text:
            await self._send_text_response(message, text)
        
        # Render and send LaTeX
        if latex:
            try:
                # CodeCogs LaTeX rendering
                fg_color = "FFFFFF"
                bg_color = "40444B"
                dpi = 200
                url_prefix = f"\\dpi{{{dpi}}}\\fg{{{fg_color}}}\\bg{{{bg_color}}}"
                encoded = urllib.parse.quote(f"{url_prefix} {latex}")
                latex_url = f"https://latex.codecogs.com/png.latex?{encoded}"
                
                logger.debug(f"Rendering LaTeX: {latex_url}")
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(latex_url) as resp:
                        if resp.status == 200:
                            image_bytes = await resp.read()
                            
                            with io.BytesIO(image_bytes) as img_buffer:
                                file = discord.File(img_buffer, filename="latex.png")
                                
                                embed = discord.Embed(color=discord.Color.blue())
                                embed.set_image(url="attachment://latex.png")
                                
                                await message.channel.send(embed=embed, file=file)
                        else:
                            logger.error(f"LaTeX render failed: HTTP {resp.status}")
                            # Send raw LaTeX as fallback
                            fallback = f"(Failed to render: `{latex[:100]}`)"
                            await message.channel.send(fallback)
            
            except Exception as e:
                logger.error(f"LaTeX rendering error: {e}")
                await message.channel.send(f"(LaTeX error: `{latex[:50]}...`)")
    
    async def _send_code_response(self, message: discord.Message, text: str, code: str):
        """Send response with code block"""
        # Send text first
        if text:
            await self._send_text_response(message, text)
        
        # Send code
        if code:
            # Try to detect language from first line
            lang = ""
            if '\n' in code:
                first_line = code.split('\n')[0].strip()
                if first_line.startswith('#') or 'python' in first_line.lower():
                    lang = "python"
                elif 'javascript' in first_line.lower() or 'js' in first_line.lower():
                    lang = "javascript"
            
            code_block = f"```{lang}\n{code}\n```"
            
            if len(code_block) <= 2000:
                await message.channel.send(code_block)
            else:
                # Send as file if too long
                logger.info("Code too long, sending as file")
                with io.BytesIO(code.encode('utf-8')) as code_buffer:
                    file = discord.File(code_buffer, filename="code.txt")
                    await message.channel.send("Code (as file):", file=file)
    
    async def _send_output_response(self, message: discord.Message, text: str, output: str):
        """Send response with command output"""
        # Send text first
        if text:
            await self._send_text_response(message, text)
        
        # Send output
        if output:
            output_block = f"```\n{output}\n```"
            
            if len(output_block) <= 2000:
                await message.channel.send(output_block)
            else:
                # Send as file if too long
                logger.info("Output too long, sending as file")
                with io.BytesIO(output.encode('utf-8')) as output_buffer:
                    file = discord.File(output_buffer, filename="output.txt")
                    await message.channel.send("Output (as file):", file=file)
    
    @tasks.loop(seconds=60)  # Default interval, changed in __init__
    async def check_restrictions_loop(self):
        """Periodically check and remove expired restrictions"""
        if not self.redis_client or not self.bot.restricted_user_role_id:
            return
        
        try:
            # Scan for restriction expiry keys
            pattern = "restricted_until:*"
            cursor = 0
            current_time = time.time()
            
            while True:
                cursor, keys = await asyncio.to_thread(
                    self.redis_client.scan,
                    cursor,
                    match=pattern,
                    count=100
                )
                
                for key in keys:
                    try:
                        expiry_str = await asyncio.to_thread(
                            self.redis_client.get,
                            key
                        )
                        
                        if not expiry_str:
                            continue
                        
                        expiry_time = float(expiry_str)
                        
                        # Check if expired
                        if current_time >= expiry_time:
                            # Parse key: restricted_until:guild_id:user_id
                            parts = key.decode('utf-8').split(':')
                            if len(parts) == 3:
                                guild_id = int(parts[1])
                                user_id = int(parts[2])
                                
                                # Remove restriction
                                await self._remove_restriction(guild_id, user_id)
                                
                                # Delete Redis key
                                await asyncio.to_thread(
                                    self.redis_client.delete,
                                    key
                                )
                    
                    except Exception as e:
                        logger.error(f"Error processing restriction key {key}: {e}")
                
                if cursor == 0:
                    break
        
        except Exception as e:
            logger.error(f"Error in restriction check loop: {e}", exc_info=True)
    
    @check_restrictions_loop.before_loop
    async def before_check_restrictions(self):
        """Wait for bot to be ready before starting loop"""
        await self.bot.wait_until_ready()
    
    async def _remove_restriction(self, guild_id: int, user_id: int):
        """Remove restricted role from user"""
        try:
            guild = self.bot.get_guild(guild_id)
            if not guild:
                return
            
            member = guild.get_member(user_id)
            if not member:
                return
            
            role = guild.get_role(self.bot.restricted_user_role_id)
            if not role:
                return
            
            if role in member.roles:
                await member.remove_roles(role, reason="Restriction expired")
                logger.info(f"Removed restriction from {member.name} ({user_id})")
        
        except Exception as e:
            logger.error(f"Error removing restriction: {e}", exc_info=True)


async def setup(bot: 'AIBot'):
    await bot.add_cog(MessageCog(bot))
