# utils/message_handler.py
import discord
import asyncio
import time
import random
import re
import logging
import json
import redis
from typing import TYPE_CHECKING, Optional, Dict, List, Tuple, Set, Literal, TypedDict, Any

if TYPE_CHECKING:
    from bot import AIBot
    from utils.litellm_client import LiteLLMClient

logger = logging.getLogger(__name__)

class MessageHandlerResult(TypedDict, total=False):
    """Result object returned by MessageHandler.process()"""
    should_respond: bool
    response_text: Optional[str]
    response_type: Optional[Literal["text", "url", "gif", "latex", "code", "output"]]
    response_data: Optional[str]
    error: Optional[str]
    restriction_applied: bool
    was_random_chance: bool
    rate_limit_type: Optional[Literal["message", "token"]]
    log_message: Optional[str]


class MessageHandler:
    """Handles message processing, rate limiting, and LLM interactions"""
    
    def __init__(self, bot: 'AIBot'):
        self.bot = bot
        
    async def handle_message(self, message: discord.Message) -> MessageHandlerResult:
        """
        Main entry point for processing a Discord message.
        Returns MessageHandlerResult with response details.
        """
        # Initialize result
        result: MessageHandlerResult = {
            "should_respond": False,
            "response_text": None,
            "response_type": None,
            "response_data": None,
            "error": None,
            "restriction_applied": False,
            "was_random_chance": False,
            "rate_limit_type": None,
            "log_message": None
        }
        
        user_id = message.author.id
        channel_id = message.channel.id
        guild_id = message.guild.id if message.guild else 0
        
        # Get member object
        member: Optional[discord.Member] = None
        if message.guild and isinstance(message.author, discord.Member):
            member = message.author
        
        # Check if user has ignored role
        if member and self.bot.ignored_role_ids_set:
            user_role_ids = {role.id for role in member.roles}
            if not self.bot.ignored_role_ids_set.isdisjoint(user_role_ids):
                logger.debug(f"Ignoring message from {message.author.name} (has ignored role)")
                return result
        
        # Determine engagement trigger
        should_engage, interaction_case, was_random = self._determine_engagement(message)
        
        if not should_engage:
            return result
        
        logger.info(f"Engagement triggered: {interaction_case} | User: {message.author.name}")
        
        # Check if user is super user (bypasses rate limits)
        is_super_user = False
        if member and self.bot.super_role_ids_set:
            user_role_ids = {role.id for role in member.roles}
            is_super_user = not self.bot.super_role_ids_set.isdisjoint(user_role_ids)
        
        # Rate limit checks (skip for super users)
        if not is_super_user:
            # Check message count rate limit
            if not await self._check_message_rate_limit(guild_id, user_id):
                logger.warning(f"Message rate limit exceeded for user {user_id}")
                await self._apply_restriction(guild_id, user_id, member)
                result["restriction_applied"] = True
                result["was_random_chance"] = was_random
                result["rate_limit_type"] = "message"
                result["error"] = "Message rate limit exceeded"
                return result
            
            # Check token consumption rate limit (placeholder - will update after response)
            # Token check happens after LLM call
        
        # Build conversation context
        try:
            # Get user's conversation history
            history = await asyncio.to_thread(
                self.bot.litellm_client.get_context_history,
                user_id,
                channel_id
            )
            
            # Handle context injection for replies
            context_to_inject = None
            if message.reference and message.reference.resolved:
                replied_msg = message.reference.resolved
                if isinstance(replied_msg, discord.Message) and replied_msg.author == self.bot.user:
                    # Replying to bot - context should already be in history
                    pass
                else:
                    # Replying to another user - might need their context
                    # For now, just include the replied message
                    context_to_inject = f"[User replied to: {replied_msg.content[:100]}]"
            
            # Fetch recent channel messages for random responses
            channel_context = None
            if was_random:
                channel_context = await self._fetch_channel_context(message.channel)
            
            # CRITICAL: Fetch MCP tools FIRST to determine conversation structure
            # This matches demo approach: fresh conversation when using tools
            logger.info(f"🔍 FETCHING MCP tools...")
            mcp_tools = await self.bot.litellm_client.get_mcp_tools()
            logger.info(f"🔧 MCP tools fetched: {len(mcp_tools) if mcp_tools else 0} tools available")
            
            if not mcp_tools:
                logger.warning(f"⚠️  No MCP tools available - this will skip tool calling entirely!")
                logger.info(f"🔍 MCP tools is: {mcp_tools} (type: {type(mcp_tools)})")
            else:
                logger.info(f"✅ MCP tools ready: {len(mcp_tools)} tools for LLM")
            
            # Build messages for LLM
            messages = []
            
            # Add system prompt
            messages.append({
                "role": "system",
                "content": self.bot.chat_system_prompt
            })
            
            # CRITICAL: When using tools, use FRESH conversation (no history)
            # This matches the demo's approach and prevents LLM confusion
            # The conversation history can make the LLM think it shouldn't use tools
            if mcp_tools:
                logger.info(f"🔧 Using fresh conversation (no history) for tool calling - matches demo approach")
            else:
                # No tools available - include history for context
                messages.extend(history)
            
            # Add channel context if random response
            if channel_context:
                messages.append({
                    "role": "system",
                    "content": f"Recent channel conversation:\n{channel_context}\n\nGenerate a contextually relevant response to join this conversation naturally."
                })
            
            # Add context injection if replying
            if context_to_inject:
                messages.append({
                    "role": "system",
                    "content": context_to_inject
                })
            
            # Add current message
            content = message.content
            # Remove bot mentions
            if self.bot.user:
                bot_mention_strings = [f'<@{self.bot.user.id}>', f'<@!{self.bot.user.id}>']
                for mention_str in bot_mention_strings:
                    content = content.replace(mention_str, '')
            content = re.sub(r'\s+', ' ', content).strip()
            
            messages.append({
                "role": "user",
                "content": content
            })
            
            # Call LLM with structured output
            logger.debug(f"Calling LLM for user {user_id} in channel {channel_id}")
            try:
                # Tools already fetched above - no need to fetch again
                # Pass tools to LLM but ALWAYS enable structured output enforcement
                # The chat_completion method will handle the mutual exclusivity internally
                use_structured = True  # Always want structured output in the end
                logger.info(f"🔧 Tool calling config: tools={bool(mcp_tools)}, structured_output={use_structured}")
                
                # Enable call tracking to capture tool calling passes
                result_tuple = await self.bot.litellm_client.chat_completion(
                    messages=messages,
                    tools=mcp_tools,
                    use_structured_output=use_structured,
                    track_calls=True
                )
                
                # Unpack response and call metadata
                if isinstance(result_tuple, tuple):
                    response, call_metadata = result_tuple
                else:
                    response = result_tuple
                    call_metadata = []
                
                if not response:
                    logger.error("LLM returned None response")
                    logger.error(f"Model {self.bot.litellm_client.model} may not support tool calling + structured output")
                    result["error"] = f"Model does not support this feature (empty response)"
                    return result
                
                # Extract the message content (should be JSON from structured output)
                message_content = response.choices[0].message.content
                
                # Handle empty or None message content
                if not message_content or message_content.strip() == "":
                    logger.error("LLM returned empty message content")
                    result["error"] = "LLM returned empty response"
                    return result
                
                # Strip markdown code fences if present (some LLMs wrap JSON in ```json ... ```)
                cleaned_content = message_content.strip()
                if cleaned_content.startswith("```"):
                    # Remove opening fence (```json or just ```)
                    lines = cleaned_content.split('\n')
                    if lines[0].startswith("```"):
                        lines = lines[1:]  # Remove first line
                    # Remove closing fence
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]  # Remove last line
                    cleaned_content = '\n'.join(lines).strip()
                    logger.debug(f"Stripped markdown code fences from response")
                
                try:
                    response_dict = json.loads(cleaned_content)
                except json.JSONDecodeError as json_err:
                    logger.error(f"Failed to parse LLM response as JSON: {json_err}")
                    logger.error(f"Raw response content: {repr(message_content)}")
                    logger.error(f"Cleaned content: {repr(cleaned_content)}")
                    result["error"] = f"Invalid JSON response from LLM: {json_err}"
                    return result
                
                usage = response.usage
                
                if not response_dict:
                    logger.error("LLM returned empty response dict")
                    result["error"] = "Empty response from LLM"
                    return result
                    
            except Exception as e:
                logger.error(f"Error calling LLM: {e}", exc_info=True)
                result["error"] = str(e)
                return result
            
            # Parse structured response
            response_type = response_dict.get("type", "text")
            response_text = response_dict.get("response", "")
            response_data = response_dict.get("data", "")
            
            # Token rate limit check (now that we have usage)
            if not is_super_user and usage:
                tokens_used = usage.total_tokens if hasattr(usage, 'total_tokens') else 0
                if tokens_used > 0:
                    # Record token usage stats
                    if hasattr(self.bot, 'stats_cog') and self.bot.stats_cog:
                        await self.bot.stats_cog.record_token_usage(
                            user_id=user_id,
                            guild_id=guild_id,
                            tokens=tokens_used,
                            message_type=interaction_case
                        )
                    
                    if not await self._check_token_rate_limit(guild_id, user_id, tokens_used):
                        logger.warning(f"Token rate limit exceeded for user {user_id}")
                        await self._apply_restriction(guild_id, user_id, member)
                        result["restriction_applied"] = True
                        result["was_random_chance"] = was_random
                        result["rate_limit_type"] = "token"
                        result["error"] = "Token rate limit exceeded"
                        return result
            
            # Random response delivery chance (secondary filter)
            if was_random:
                if random.random() > self.bot.random_response_delivery_chance:
                    logger.debug(f"Random response filtered out by delivery chance")
                    return result
            
            # Save conversation history
            new_history = history + [
                {"role": "user", "content": content, "timestamp": time.time()},
                {"role": "assistant", "content": response_text, "timestamp": time.time()}
            ]
            
            await asyncio.to_thread(
                self.bot.litellm_client.save_context_history,
                user_id,
                channel_id,
                new_history
            )
            
            # Prepare result
            result["should_respond"] = True
            result["response_text"] = response_text
            result["response_type"] = response_type
            result["response_data"] = response_data
            result["was_random_chance"] = was_random
            result["log_message"] = f"Generated {response_type} response for {interaction_case}"
            
            # Add token usage and raw output for testing/debugging
            if usage:
                result["tokens_used"] = usage.total_tokens if hasattr(usage, 'total_tokens') else 0
                result["prompt_tokens"] = usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0
                result["completion_tokens"] = usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0
                
                # Add prompt_tokens_details if available (includes cached tokens)
                if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
                    details = usage.prompt_tokens_details
                    if hasattr(details, 'cached_tokens'):
                        result["cached_tokens"] = details.cached_tokens
                
                # Add completion_tokens_details if available (includes reasoning tokens for o1 models)
                if hasattr(usage, 'completion_tokens_details') and usage.completion_tokens_details:
                    details = usage.completion_tokens_details
                    if hasattr(details, 'reasoning_tokens'):
                        result["reasoning_tokens"] = details.reasoning_tokens
            
            # Add conversation context info for debugging
            result["history_length"] = len(history)
            result["messages_sent"] = len(messages)
            
            # Add LLM call metadata (for tool calling tracking)
            if call_metadata:
                result["llm_calls"] = call_metadata
            
            # Add raw LLM response for debugging
            result["raw_output"] = response_dict
            
            logger.info(f"Successfully processed message: {interaction_case} | Type: {response_type}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            result["error"] = f"Processing error: {str(e)}"
            return result
    
    def _determine_engagement(self, message: discord.Message) -> Tuple[bool, str, bool]:
        """
        Determine if bot should respond to this message.
        Returns: (should_respond, interaction_case, was_random_chance)
        """
        is_mention = self.bot.user in message.mentions if self.bot.user else False
        is_reply_to_bot = False
        
        # Check if reply to bot
        if message.reference and message.reference.resolved:
            replied_msg = message.reference.resolved
            if isinstance(replied_msg, discord.Message) and replied_msg.author == self.bot.user:
                is_reply_to_bot = True
        
        # Priority 1: Direct mention
        if is_mention:
            return (True, "Mention", False)
        
        # Priority 2: Reply to bot
        if is_reply_to_bot:
            return (True, "Reply to Bot", False)
        
        # Priority 3: Random chance
        if random.random() < self.bot.response_chance:
            return (True, "Random Chance", True)
        
        return (False, "No Trigger", False)
    
    async def _fetch_channel_context(self, channel: discord.TextChannel, limit: int = 10) -> str:
        """Fetch recent messages from channel for context"""
        try:
            messages = []
            async for msg in channel.history(limit=limit):
                if not msg.author.bot:  # Skip bot messages
                    messages.append(f"{msg.author.display_name}: {msg.content}")
            
            messages.reverse()  # Chronological order
            return "\n".join(messages[-5:])  # Last 5 messages
        except Exception as e:
            logger.error(f"Error fetching channel context: {e}")
            return ""
    
    async def _check_message_rate_limit(self, guild_id: int, user_id: int) -> bool:
        """Check if user is within message rate limit"""
        if not self.bot.redis_client:
            return True
        
        try:
            key = f"msg_rl:{guild_id}:{user_id}"
            current_time = time.time()
            window_start = current_time - self.bot.rate_limit_window_seconds
            
            # Remove old entries
            await asyncio.to_thread(
                self.bot.redis_client.zremrangebyscore,
                key,
                0,
                window_start
            )
            
            # Count messages in window
            count = await asyncio.to_thread(
                self.bot.redis_client.zcard,
                key
            )
            
            if count >= self.bot.rate_limit_count:
                return False
            
            # Add current message
            await asyncio.to_thread(
                self.bot.redis_client.zadd,
                key,
                {str(current_time): current_time}
            )
            
            # Set expiry
            await asyncio.to_thread(
                self.bot.redis_client.expire,
                key,
                self.bot.rate_limit_window_seconds * 2
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking message rate limit: {e}")
            return True  # Allow on error
    
    async def _check_token_rate_limit(self, guild_id: int, user_id: int, tokens: int) -> bool:
        """Check if user is within token rate limit"""
        if not self.bot.redis_client:
            return True
        
        try:
            key = f"token_rl:{guild_id}:{user_id}"
            current_time = time.time()
            window_start = current_time - self.bot.rate_limit_window_seconds
            
            # Remove old entries
            await asyncio.to_thread(
                self.bot.redis_client.zremrangebyscore,
                key,
                0,
                window_start
            )
            
            # Sum tokens in window
            entries = await asyncio.to_thread(
                self.bot.redis_client.zrange,
                key,
                0,
                -1,
                withscores=True
            )
            
            total_tokens = sum(int(score) for _, score in entries)
            
            if total_tokens + tokens > self.bot.token_rate_limit_count:
                return False
            
            # Add current tokens
            await asyncio.to_thread(
                self.bot.redis_client.zadd,
                key,
                {f"{current_time}:{tokens}": tokens}
            )
            
            # Set expiry
            await asyncio.to_thread(
                self.bot.redis_client.expire,
                key,
                self.bot.rate_limit_window_seconds * 2
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking token rate limit: {e}")
            return True  # Allow on error
    
    async def _apply_restriction(self, guild_id: int, user_id: int, member: Optional[discord.Member]):
        """Apply restricted user role and set expiry"""
        if not member or not self.bot.restricted_user_role_id:
            return
        
        try:
            # Add restricted role
            role = member.guild.get_role(self.bot.restricted_user_role_id)
            if role:
                await member.add_roles(role, reason="Rate limit exceeded")
                logger.info(f"Applied restricted role to user {user_id}")
            
            # Set expiry in Redis
            if self.bot.redis_client and self.bot.restriction_duration_seconds > 0:
                key = f"restricted_until:{guild_id}:{user_id}"
                expiry_time = time.time() + self.bot.restriction_duration_seconds
                await asyncio.to_thread(
                    self.bot.redis_client.set,
                    key,
                    str(expiry_time),
                    ex=self.bot.restriction_duration_seconds * 2
                )
                
        except Exception as e:
            logger.error(f"Error applying restriction: {e}", exc_info=True)
