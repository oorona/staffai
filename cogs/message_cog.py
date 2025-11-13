# cogs/message_cog.py
import discord
from discord.ext import commands, tasks
import asyncio
import logging
import time
import redis
from typing import TYPE_CHECKING, Optional, Set
import io
import urllib.parse
import aiohttp
import json

from utils.message_handler import MessageHandler, MessageHandlerResult

if TYPE_CHECKING:
    from bot import AIBot

logger = logging.getLogger(__name__)


class MessageCog(commands.Cog):
    """Handles Discord message events, bot responses, and user restrictions"""
    
    # Class-level deduplication set (shared across all instances)
    _processed_messages: Set[str] = set()
    
    def __init__(self, bot: 'AIBot'):
        self.bot: 'AIBot' = bot
        self.redis_client: Optional[redis.Redis] = self.bot.redis_client
        # In-memory cache to track which messages we've already sent channel replies for
        self._channel_sent_keys: Set[str] = set()
        
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

        # Deduplication: Check if we've already processed this message (prevent Discord event duplicates)
        # Use content-based key (exclude message.id since webhook duplicates can have different IDs)
        # Include referenced message id for replies so reply messages don't collide with non-reply messages
        referenced_id = None
        try:
            if getattr(message, 'reference', None) and getattr(message.reference, 'message_id', None):
                referenced_id = str(message.reference.message_id)
        except Exception:
            referenced_id = None

        message_key = f"{message.channel.id}_{message.author.id}_{referenced_id or ''}_{hash(message.content)}"
        
        # Redis deduplication (atomic)
        if self.redis_client:
            dedup_key = f"processed_msg:{message_key}"
            try:
                # SADD returns 1 if added (new), 0 if already exists
                was_added = await asyncio.to_thread(self.redis_client.sadd, dedup_key, "1")
                if not was_added:
                    logger.debug(f"Skipping duplicate message {message.id} from {message.author.name} (Redis)")
                    return
                # Set TTL
                await asyncio.to_thread(self.redis_client.expire, dedup_key, 60)
            except Exception as e:
                logger.warning(f"Redis deduplication failed: {e}")
        
        # In-memory deduplication (always active)
        if message_key in MessageCog._processed_messages:
            logger.debug(f"Skipping duplicate message {message.id} from {message.author.name} (in-memory)")
            return
            
        # Mark as processed
        MessageCog._processed_messages.add(message_key)
        
        logger.debug(f"Processing unique message {message.id} from {message.author.name}: '{message.content}' (key: {message_key})")

        # Ensure author is a Member object
        if not isinstance(message.author, discord.Member):
            try:
                message.author = await message.guild.fetch_member(message.author.id)
            except (discord.NotFound, discord.HTTPException) as e:
                logger.warning(f"Could not resolve member {message.author.id}: {e}")
                return

        # Process message through handler
        handler = MessageHandler(self.bot)

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

        # Handle response based on type (typing indicator already shown during processing)
        response_type = result.get("response_type", "text")
        response_text = result.get("response_text", "")
        response_data = result.get("response_data", "")

        # Pass the deduplication key so send tracing can identify sends
        try:
            if response_type == "text":
                await self._send_text_response(message, response_text, message_key)

            elif response_type == "url":
                await self._send_url_response(message, response_text, response_data, message_key)

            elif response_type == "gif":
                logger.info(f"GIF Response - Text: '{response_text}', URL: '{response_data}'")
                await self._send_gif_response(message, response_text, response_data, message_key)

            elif response_type == "latex":
                await self._send_latex_response(message, response_text, response_data, message_key)

            elif response_type == "code":
                await self._send_code_response(message, response_text, response_data, message_key)

            elif response_type == "output":
                await self._send_output_response(message, response_text, response_data, message_key)

            else:
                logger.warning(f"Unknown response type: {response_type}, sending as text")
                await self._send_text_response(message, response_text, message_key)

            # Send debug context to super users if enabled
            if self.bot.debug_context_super_users and isinstance(message.author, discord.Member):
                # Check if user has super role
                user_role_ids = {role.id for role in message.author.roles}
                is_super_user = not self.bot.super_role_ids_set.isdisjoint(user_role_ids)

                if is_super_user and result.get("context_messages"):
                    await self._send_debug_context(message, result)

        except Exception as e:
            logger.error(f"Error sending response: {e}", exc_info=True)
    
    async def _send_text_response(self, message: discord.Message, text: str, message_key: str = None):
        """Send plain text response"""
        if not text:
            return
        
        if len(text) > 2000:
            text = text[:1997] + "..."
        # Ensure we only send one channel message per message_key
        allowed = await self._ensure_channel_send_allowed(message, message_key)
        if not allowed:
            logger.debug(f"Channel send skipped for text reply (key={message_key})")
            return

        logger.debug(f"SEND TRACE: sending text reply (key={message_key}) to channel {message.channel.id}")
        await message.reply(text, mention_author=False)
    
    async def _send_url_response(self, message: discord.Message, text: str, url: str, message_key: str = None):
        """Send response with URL - text as message, URL/image as embed"""
        allowed = await self._ensure_channel_send_allowed(message, message_key)
        if not allowed:
            logger.debug(f"Channel send skipped for url response (key={message_key})")
            return

        # Send text response first if present
        if text:
            await message.channel.send(text)

        # Send URL/image as embed
        embed = discord.Embed(color=discord.Color.blue())

        # If it's an image, embed it
        if any(url.lower().endswith(ext) for ext in ['.gif', '.png', '.jpg', '.jpeg', '.webp']):
            embed.set_image(url=url)
        else:
            # Regular URL - just show the link
            embed.description = url

        logger.debug(f"SEND TRACE: sending url embed (key={message_key}) to channel {message.channel.id}")
        await message.channel.send(embed=embed)
    
    async def _send_gif_response(self, message: discord.Message, text: str, gif_url: str, message_key: str = None):
        """Send response with GIF - text as message, GIF as embed"""
        # Validate URL format
        if not gif_url or not gif_url.startswith(('http://', 'https://')):
            logger.error(f"Invalid GIF URL format: {gif_url}")
            # Fall back to text response with error message
            error_msg = f"{text}\n\n*(Invalid GIF URL returned)*" if text else "Sorry, couldn't load the GIF (invalid URL)"
            await self._send_text_response(message, error_msg, message_key)
            return

        allowed = await self._ensure_channel_send_allowed(message, message_key)
        if not allowed:
            logger.debug(f"Channel send skipped for gif response (key={message_key})")
            return

        # Send text response first if present
        if text:
            await message.channel.send(text)

        # Send GIF as embed
        embed = discord.Embed(color=discord.Color.purple())
        embed.set_image(url=gif_url)

        logger.debug(f"SEND TRACE: sending gif embed (key={message_key}) to channel {message.channel.id}")
        await message.channel.send(embed=embed)
    
    async def _send_latex_response(self, message: discord.Message, text: str, latex: str, message_key: str = None):
        """Send response with rendered LaTeX - consolidated into single message"""
        if not latex:
            if text:
                await self._send_text_response(message, text)
            return
        
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
                            
                            embed = discord.Embed(
                                title=text or "LaTeX Formula",
                                color=discord.Color.blue()
                            )
                            embed.set_image(url="attachment://latex.png")
                            
                            logger.debug(f"SEND TRACE: sending latex embed (key={message_key}) to channel {message.channel.id}")
                            await message.channel.send(embed=embed, file=file)
                    else:
                        logger.error(f"LaTeX render failed: HTTP {resp.status}")
                        # Send raw LaTeX as fallback
                        fallback = f"{text}\n\n(Failed to render: `{latex[:100]}`)" if text else f"(Failed to render: `{latex[:100]}`)"
                        logger.debug(f"SEND TRACE: sending latex fallback reply (key={message_key}) to channel {message.channel.id}")
                        await message.reply(fallback, mention_author=False)
        
        except Exception as e:
            logger.error(f"LaTeX rendering error: {e}")
            fallback = f"{text}\n\n(LaTeX error: `{latex[:50]}...`)" if text else f"(LaTeX error: `{latex[:50]}...`)"
            logger.debug(f"SEND TRACE: sending latex error reply (key={message_key}) to channel {message.channel.id}")
            await message.reply(fallback, mention_author=False)
    
    async def _send_code_response(self, message: discord.Message, text: str, code: str, message_key: str = None):
        """Send response with code block - text separate from code"""
        if not code:
            if text:
                await self._send_text_response(message, text)
            return

        allowed = await self._ensure_channel_send_allowed(message, message_key)
        if not allowed:
            logger.debug(f"Channel send skipped for code response (key={message_key})")
            return

        # Send text first if present
        if text:
            await message.channel.send(text)

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
            logger.debug(f"SEND TRACE: sending code block (key={message_key}) to channel {message.channel.id}")
            await message.channel.send(code_block)
        else:
            # Send as file if too long
            logger.info("Code too long, sending as file")
            embed = discord.Embed(
                title=text or "Code",
                color=discord.Color.green()
            )
            with io.BytesIO(code.encode('utf-8')) as code_buffer:
                file = discord.File(code_buffer, filename="code.txt")
                embed.set_footer(text="Code too long for message - see attached file")
                logger.debug(f"SEND TRACE: sending code file embed (key={message_key}) to channel {message.channel.id}")
                await message.channel.send(embed=embed, file=file)
    
    async def _send_output_response(self, message: discord.Message, text: str, output: str, message_key: str = None):
        """Send response with command output - text separate from output"""
        if not output:
            if text:
                await self._send_text_response(message, text)
            return

        allowed = await self._ensure_channel_send_allowed(message, message_key)
        if not allowed:
            logger.debug(f"Channel send skipped for output response (key={message_key})")
            return

        # Send text first if present
        if text:
            await message.channel.send(text)

        output_block = f"```\n{output}\n```"

        if len(output_block) <= 2000:
            logger.debug(f"SEND TRACE: sending output block (key={message_key}) to channel {message.channel.id}")
            await message.channel.send(output_block)
        else:
            # Send as file if too long
            logger.info("Output too long, sending as file")
            embed = discord.Embed(
                title="Command Output",
                color=discord.Color.light_grey()
            )
            with io.BytesIO(output.encode('utf-8')) as output_buffer:
                file = discord.File(output_buffer, filename="output.txt")
                embed.set_footer(text="Output too long for message - see attached file")
                logger.debug(f"SEND TRACE: sending output file embed (key={message_key}) to channel {message.channel.id}")
                await message.channel.send(embed=embed, file=file)
    
    async def _send_debug_context(self, message: discord.Message, result: MessageHandlerResult):
        """Send debug context to super users as a DM (ephemeral-like) showing what was sent to the LLM"""
        try:
            context_messages = result.get("context_messages", [])
            if not context_messages:
                return

            # Calculate sizes (handle None values)
            prompt_tokens = result.get("prompt_tokens") or 0
            completion_tokens = result.get("completion_tokens") or 0
            cached_tokens = result.get("cached_tokens") or 0
            total_tokens = result.get("tokens_used") or 0

            # Create formatted JSON
            context_json = json.dumps(context_messages, indent=2, ensure_ascii=False)

            # Try to send as embed first (Discord limit: 4096 chars for description)
            if len(context_json) <= 3800:  # Leave room for embed header
                embed = discord.Embed(
                    title="🔍 Debug: Conversation Context",
                    description=f"```json\n{context_json}\n```",
                    color=discord.Color.orange()
                )
                embed.add_field(
                    name="📊 Token Usage",
                    value=f"**Total:** {total_tokens:,}\n**Input:** {prompt_tokens:,}\n**Output:** {completion_tokens:,}\n**Cached:** {cached_tokens:,}",
                    inline=False
                )
                embed.add_field(
                    name="📝 Context Info",
                    value=f"**Messages:** {len(context_messages)}\n**History:** {result.get('history_length', 0)}",
                    inline=True
                )
                embed.set_footer(text="Debug mode enabled for super users • Sent privately")

                # Send as DM (private) - NEVER send to channel
                try:
                    await message.author.send(embed=embed)
                    logger.info(f"Sent debug context embed to {message.author.name} via DM")
                except discord.Forbidden:
                    # DMs disabled - log only, never send to channel
                    logger.warning(f"Cannot send debug context to {message.author.name}: DMs disabled")
                except Exception as e:
                    # Any other error - log only, never send to channel
                    logger.error(f"Failed to send debug context DM to {message.author.name}: {e}")
            else:
                # Send as file if too large
                logger.info("Debug context too large for embed, sending as file")

                # Create a more detailed JSON with metadata
                debug_data = {
                    "metadata": {
                        "messages_count": len(context_messages),
                        "history_length": result.get("history_length", 0),
                        "tokens": {
                            "total": total_tokens,
                            "prompt": prompt_tokens,
                            "completion": completion_tokens,
                            "cached": cached_tokens
                        }
                    },
                    "context": context_messages,
                    "raw_output": result.get("raw_output", {})
                }

                debug_json = json.dumps(debug_data, indent=2, ensure_ascii=False)

                with io.BytesIO(debug_json.encode('utf-8')) as context_buffer:
                    file = discord.File(context_buffer, filename="debug_context.json")

                    embed = discord.Embed(
                        title="🔍 Debug: Conversation Context",
                        description=f"Context too large for embed. See attached file.\n\n**Messages:** {len(context_messages)}\n**Tokens:** {total_tokens:,}",
                        color=discord.Color.orange()
                    )
                    embed.set_footer(text="Debug mode enabled for super users • Sent privately")

                    # Send as DM (private) - NEVER send to channel
                    try:
                        await message.author.send(embed=embed, file=file)
                        logger.info(f"Sent debug context file to {message.author.name} via DM")
                    except discord.Forbidden:
                        # DMs disabled - log only, never send to channel
                        logger.warning(f"Cannot send debug context file to {message.author.name}: DMs disabled")
                    except Exception as e:
                        # Any other error - log only, never send to channel
                        logger.error(f"Failed to send debug context file DM to {message.author.name}: {e}")

            logger.info(f"Debug context sending attempted for super user {message.author.name}")

        except Exception as e:
            logger.error(f"Error in _send_debug_context: {e}")
            # Never send anything to channel on error

    async def _ensure_channel_send_allowed(self, message: discord.Message, message_key: str = None) -> bool:
        """Return True if sending to the channel is allowed for this message_key.

        Uses Redis SADD with a small TTL to atomically allow only the first sender.
        Falls back to in-memory set if Redis unavailable.
        """
        if not message_key:
            # If no key provided, don't block sending
            return True

        # Try Redis first (atomic)
        if self.redis_client:
            dedup_key = f"channel_sent:{message_key}"
            try:
                was_added = await asyncio.to_thread(self.redis_client.sadd, dedup_key, "1")
                if was_added:
                    # Set a short TTL so keys don't pile up
                    await asyncio.to_thread(self.redis_client.expire, dedup_key, 60)
                    return True
                return False
            except Exception as e:
                logger.debug(f"Redis channel send guard failed: {e}")

        # Fallback to in-memory guard
        if message_key in self._channel_sent_keys:
            return False

        self._channel_sent_keys.add(message_key)
        # Schedule cleanup after a short delay to avoid memory leak
        async def _cleanup(key: str):
            await asyncio.sleep(60)
            self._channel_sent_keys.discard(key)

        asyncio.create_task(_cleanup(message_key))
        return True
    
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
