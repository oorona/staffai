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

        # Show typing indicator while processing
        async with message.channel.typing():
            # Build conversation context
            result = await self._process_message_with_context(
                message, user_id, channel_id, guild_id, was_random, is_super_user, member, interaction_case
            )

        return result

    async def _process_message_with_context(
        self,
        message: discord.Message,
        user_id: int,
        channel_id: int,
        guild_id: int,
        was_random: bool,
        is_super_user: bool,
        member: Optional[discord.Member],
        interaction_case: str
    ) -> MessageHandlerResult:
        """Process message with full context gathering and LLM interaction"""
        result: MessageHandlerResult = {
            "should_respond": False,
            "response_text": None,
            "response_type": None,
            "response_data": None,
            "error": None,
            "restriction_applied": False,
            "was_random_chance": was_random,
            "rate_limit_type": None,
            "log_message": None
        }

        try:
            # Get user's conversation history with bot
            history = await asyncio.to_thread(
                self.bot.litellm_client.get_context_history,
                user_id,
                channel_id
            )

            # Determine the interaction scenario and gather appropriate context
            context_to_inject = None
            channel_context = None
            is_mention = self.bot.user in message.mentions if self.bot.user else False
            is_reply_to_bot = False
            replied_to_other_user = False
            other_user_id = None

            # Check if this is a reply (resolve the referenced message even if not cached)
            referenced_msg: Optional[discord.Message] = None
            if message.reference:
                referenced_msg = message.reference.resolved if isinstance(message.reference.resolved, discord.Message) else None
                if not referenced_msg and getattr(message.reference, "message_id", None):
                    try:
                        referenced_msg = await message.channel.fetch_message(message.reference.message_id)
                    except (discord.NotFound, discord.Forbidden, discord.HTTPException) as e:
                        logger.debug(f"Could not fetch referenced message {message.reference.message_id}: {e}")

                if isinstance(referenced_msg, discord.Message):
                    if referenced_msg.author == self.bot.user:
                        # Scenario 2: Reply to bot
                        is_reply_to_bot = True
                    else:
                        # Replying to another user
                        replied_to_other_user = True
                        other_user_id = referenced_msg.author.id
                elif message.reference and not is_reply_to_bot:
                    # Fallback: unable to resolve referenced message, but it's still a reply (likely to another user)
                    replied_to_other_user = True

            # Scenario resolution order (per rules):
            # 1) Reply to bot -> Scenario 2
            # 2) Mention + reply to another user -> Scenario 3 (inject context)
            # 3) Mention (no reply) -> Scenario 1
            # 4) Random -> Scenario 4

            if is_reply_to_bot:
                logger.info(f"Scenario 2: User {user_id} replied to bot (using conversation history only)")
                # context_to_inject remains None - history is sufficient

            elif is_mention and replied_to_other_user:
                interaction_case = "Reply to User + Mention"
                logger.info(f"Scenario 3: User {user_id} tagged bot while replying to user {other_user_id or 'unknown'}")

                if referenced_msg:
                    # Use author ID (not display name) and quote the exact content
                    context_to_inject = f"user @{referenced_msg.author.id} said \"{referenced_msg.content}\""
                else:
                    # Provide at least a placeholder so the LLM knows there is upstream context
                    ref_id = getattr(message.reference, "message_id", "unknown") if message.reference else "unknown"
                    context_to_inject = f"user @unknown said \"<missing content for message id {ref_id}>\""

            elif is_mention:
                logger.info(f"Scenario 1: User {user_id} mentioned bot (using conversation history only)")
                # context_to_inject remains None - history is sufficient

            elif was_random:
                logger.info(f"Scenario 4: Random response to user {user_id} (no conversation history)")

                # Fetch general channel context for awareness
                channel_context = await self._fetch_channel_context(message.channel, limit=10)
                # channel_context will be added separately below, not as context_to_inject

            # After resolving the specific scenario, log the final scenario mapping for clarity
            scenario_map_verbose = {
                "Mention": "Scenario 1: Mention",
                "Reply to Bot": "Scenario 2: Reply to Bot",
                "Reply to User + Mention": "Scenario 3: Reply to User + Mention",
                "Random Chance": "Scenario 4: Random Response"
            }
            scenario_label = scenario_map_verbose.get(interaction_case, interaction_case)

            # Get message preview (first 50 characters)
            message_preview = message.content[:50] if message.content else "(empty)"
            if len(message.content) > 50:
                message_preview += "..."

            logger.info(f"{'='*20} {message.author.name} | {scenario_label} {'='*20}")
            logger.info(f"Engagement resolved: {interaction_case} | User: {message.author.name}")
            logger.info(f"📝 Message: \"{message_preview}\"")

            # Extract and clean message content first (needed for tool filtering)
            content = message.content
            # Remove bot mentions
            if self.bot.user:
                bot_mention_strings = [f'<@{self.bot.user.id}>', f'<@!{self.bot.user.id}>']
                for mention_str in bot_mention_strings:
                    content = content.replace(mention_str, '')
            content = re.sub(r'\s+', ' ', content).strip()

            # Fetch MCP tools for availability
            all_mcp_tools = await self.bot.litellm_client.get_mcp_tools()

            # Always provide tools - let the LLM decide when to use them
            # The personality prompt already instructs appropriate tool usage
            # Keyword filtering is fragile (language-dependent, misses natural requests)
            mcp_tools = all_mcp_tools if all_mcp_tools else None

            if mcp_tools:
                logger.info(f"✅ {len(mcp_tools)} tools available for LLM selection")
            else:
                logger.info(f"⚠️  No MCP tools available")

            # Build messages for LLM
            messages = []

            # Add system prompt
            messages.append({
                "role": "system",
                "content": self.bot.chat_system_prompt
            })

            # Scenario-based context building:
            # - Scenarios 1, 2, 3: Include conversation history
            # - Scenario 4 (random): NO conversation history, only channel context

            if was_random:
                # Scenario 4: Random response - NO conversation history
                # Only add channel context for awareness
                if channel_context:
                    messages.append({
                        "role": "system",
                        "content": f"Recent channel conversation:\n{channel_context}\n\nGenerate a contextually relevant response to join this conversation naturally."
                    })
            else:
                # Scenarios 1, 2, 3: Include conversation history
                # Limit history when tools are available to prevent token overflow
                if mcp_tools and len(mcp_tools) > 0:
                    # When tools are available, include only recent history (last 4 messages)
                    recent_history = history[-4:] if len(history) > 4 else history
                    messages.extend(recent_history)
                    if len(history) > len(recent_history):
                        logger.debug(f"Trimmed history from {len(history)} to {len(recent_history)} messages (tools available)")
                else:
                    # No tools, include full history for better conversation
                    messages.extend(history)

                # Scenario 3: Add context injection for referenced message
                if context_to_inject:
                    messages.append({
                        # Treat injected context as a user message per spec
                        "role": "user",
                        "content": context_to_inject
                    })

            # Add current message (content already extracted and cleaned above)
            messages.append({
                "role": "user",
                "content": content
            })

            # Log context information
            logger.info(f"📊 Context: {len(messages)} messages sent to LLM (system: 1, history: {len(history)}, current: 1)")

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

                # Aggregate token usage across ALL LLM calls (including tool selection passes)
                usage = response.usage
                total_prompt_tokens = 0
                total_completion_tokens = 0
                total_tokens = 0

                if call_metadata:
                    # Sum tokens from all passes
                    for call_info in call_metadata:
                        if 'tokens' in call_info:
                            total_prompt_tokens += call_info['tokens'].get('prompt', 0)
                            total_completion_tokens += call_info['tokens'].get('completion', 0)
                            total_tokens += call_info['tokens'].get('total', 0)

                    logger.debug(f"📊 Aggregated tokens from {len(call_metadata)} LLM passes: {total_tokens:,} total ({total_prompt_tokens:,} prompt + {total_completion_tokens:,} completion)")

                    # Override usage with aggregated totals (use the usage object structure)
                    if usage and total_tokens > 0:
                        # Keep the original usage object but update the token counts
                        usage.prompt_tokens = total_prompt_tokens
                        usage.completion_tokens = total_completion_tokens
                        usage.total_tokens = total_tokens
                elif usage:
                    # Fallback: use final response usage if no metadata available
                    total_prompt_tokens = usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0
                    total_completion_tokens = usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0
                    total_tokens = usage.total_tokens if hasattr(usage, 'total_tokens') else 0
                
                if not response_dict:
                    logger.error("LLM returned empty response dict")
                    result["error"] = "Empty response from LLM"
                    return result
                
                # Validate response structure against expected schema
                if "type" not in response_dict:
                    logger.error(f"LLM response missing 'type' field: {response_dict}")
                    result["error"] = "LLM response missing required 'type' field"
                    return result
                
                if "response" not in response_dict:
                    logger.error(f"LLM response missing 'response' field: {response_dict}")
                    result["error"] = "LLM response missing required 'response' field"
                    return result
                
                allowed_types = ["text", "url", "gif", "latex", "code", "output"]
                if response_dict["type"] not in allowed_types:
                    logger.error(f"LLM response has invalid 'type': {response_dict['type']}")
                    result["error"] = f"LLM response has invalid type: {response_dict['type']}"
                    return result
                    
            except Exception as e:
                logger.error(f"Error calling LLM: {e}", exc_info=True)
                result["error"] = str(e)
                return result
            
            # Parse structured response
            response_type = response_dict.get("type", "text")
            response_text = response_dict.get("response", "")
            response_data = response_dict.get("data", "")
            
            # Token tracking and rate limit check
            if usage:
                tokens_used = usage.total_tokens if hasattr(usage, 'total_tokens') else 0
                prompt_tokens = usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0
                completion_tokens = usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0
                cached_tokens = result.get("cached_tokens", 0)

                if tokens_used > 0:
                    # ALWAYS record token usage and log cost (for all users, including super users)
                    if hasattr(self.bot, 'stats_cog') and self.bot.stats_cog:
                        await self.bot.stats_cog.record_token_usage(
                            user_id=user_id,
                            guild_id=guild_id,
                            tokens=tokens_used,
                            message_type=interaction_case,
                            cached_tokens=cached_tokens,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens
                        )

                        # Calculate and log the cost for this response
                        cost = self.bot.stats_cog.calculate_cost(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            model_name=self.bot.litellm_client.model
                        )
                        # Convert to cents for more readable numbers
                        cost_cents = cost * 100

                        # Enhanced logging with pass breakdown if multiple calls were made
                        if call_metadata and len(call_metadata) > 1:
                            # Show breakdown of each pass
                            pass_details = []
                            for i, call_info in enumerate(call_metadata, 1):
                                if 'tokens' in call_info:
                                    pass_total = call_info['tokens'].get('total', 0)
                                    pass_purpose = call_info.get('purpose', 'unknown')
                                    pass_details.append(f"Pass {i} ({pass_purpose}): {pass_total:,} tokens")

                            logger.info(
                                f"💰 Cost: {cost_cents:.4f}¢ (${cost:.6f}) | "
                                f"Tokens: {tokens_used:,} total ({prompt_tokens:,} in + {completion_tokens:,} out, {cached_tokens:,} cached) | "
                                f"Passes: {len(call_metadata)} | User: {message.author.name}"
                            )
                            for detail in pass_details:
                                logger.info(f"  └─ {detail}")
                        else:
                            # Single pass - simple log
                            logger.info(
                                f"💰 Cost: {cost_cents:.4f}¢ (${cost:.6f}) | "
                                f"Tokens: {tokens_used:,} total ({prompt_tokens:,} in + {completion_tokens:,} out, {cached_tokens:,} cached) | "
                                f"User: {message.author.name}"
                            )

                    # Rate limit check (only for non-super users)
                    if not is_super_user and not await self._check_token_rate_limit(guild_id, user_id, tokens_used):
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

            # Save conversation history ONLY for scenarios 1, 2, 3
            # Scenario 4 (random) does NOT save history - it's not part of an ongoing conversation
            if not was_random:
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
                logger.debug(f"Saved conversation history for {interaction_case}")
            else:
                logger.debug(f"Skipped saving history for random response (not part of ongoing conversation)")
            
            # Prepare result
            result["should_respond"] = True
            result["response_text"] = response_text
            result["response_type"] = response_type
            result["response_data"] = response_data
            result["was_random_chance"] = was_random
            result["log_message"] = f"Generated {response_type} response for {interaction_case}"

            # Log successful processing before delimiter
            logger.info(f"Successfully processed message: {interaction_case} | Type: {response_type}")

            # Log end of interaction with cost/token summary
            if usage:
                tokens_used = usage.total_tokens if hasattr(usage, 'total_tokens') else 0
                prompt_tokens = usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0
                completion_tokens = usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0
                cached_tokens = result.get("cached_tokens", 0)

                # Calculate cost
                cost = 0
                if hasattr(self.bot, 'stats_cog') and self.bot.stats_cog:
                    cost = self.bot.stats_cog.calculate_cost(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        model_name=self.bot.litellm_client.model
                    )
                cost_cents = cost * 100

                # Update scenario label for final log
                scenario_map = {
                    "Mention": "Scenario 1",
                    "Reply to Bot": "Scenario 2",
                    "Reply to User + Mention": "Scenario 3",
                    "Random Chance": "Scenario 4"
                }
                scenario_num = scenario_map.get(interaction_case, interaction_case)

                # Show pass count if multiple calls were made
                pass_info = ""
                if call_metadata and len(call_metadata) > 1:
                    pass_info = f" | {len(call_metadata)} passes"

                logger.info(
                    f"{'='*20} {message.author.name} | Cost: {cost_cents:.4f}¢ (${cost:.6f}) | "
                    f"Tokens: {tokens_used:,} total ({prompt_tokens:,} in + {completion_tokens:,} out, {cached_tokens:,} cached){pass_info} {'='*20}"
                )

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
            result["context_messages"] = messages  # Full context sent to LLM

            # Add LLM call metadata (for tool calling tracking)
            if call_metadata:
                result["llm_calls"] = call_metadata

            # Add raw LLM response for debugging
            result["raw_output"] = response_dict

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
        
        # Priority 1: Reply to bot (takes precedence even if user also mentions)
        if is_reply_to_bot:
            return (True, "Reply to Bot", False)
        
        # Priority 2: Direct mention
        if is_mention:
            return (True, "Mention", False)
        
        # Priority 3: Random chance
        if random.random() < self.bot.response_chance:
            return (True, "Random Chance", True)
        
        return (False, "No Trigger", False)
    
    async def _fetch_channel_context(self, channel: discord.TextChannel, limit: int = 10) -> str:
        """Fetch recent messages from channel for context (respecting time limits)"""
        try:
            messages = []
            current_time = time.time()
            max_age = self.bot.context_message_max_age_seconds

            async for msg in channel.history(limit=limit):
                # Skip bot messages
                if msg.author.bot:
                    continue

                # Check message age
                message_age = current_time - msg.created_at.timestamp()
                if message_age > max_age:
                    continue  # Skip old messages

                messages.append(f"{msg.author.display_name}: {msg.content}")

            messages.reverse()  # Chronological order (oldest to newest)
            return "\n".join(messages)  # All recent messages (already filtered by age)
        except Exception as e:
            logger.error(f"Error fetching channel context: {e}")
            return ""

    async def _fetch_user_recent_messages(
        self,
        channel: discord.TextChannel,
        user_id: int,
        limit: int = 5,
        max_search: int = 50
    ) -> List[str]:
        """
        Fetch recent messages from a specific user in a channel.

        Args:
            channel: Discord channel to search
            user_id: User ID to fetch messages from
            limit: Number of messages to return (default: 5)
            max_search: Maximum messages to search through (default: 50)

        Returns:
            List of message content strings in chronological order (oldest first)
        """
        try:
            user_messages = []
            async for msg in channel.history(limit=max_search):
                if msg.author.id == user_id and not msg.author.bot:
                    user_messages.append(msg.content)
                    if len(user_messages) >= limit:
                        break

            # Reverse to get chronological order (oldest to newest)
            user_messages.reverse()
            logger.debug(f"Fetched {len(user_messages)} recent messages from user {user_id}")
            return user_messages
        except Exception as e:
            logger.error(f"Error fetching user messages: {e}")
            return []
    
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
