# utils/message_handler.py
import discord
import asyncio
import time
import random
import re
import logging
import json
import redis
from typing import TYPE_CHECKING, Optional, Dict, List, Tuple, Literal, TypedDict, Any

from utils.log_formatting import emit_plain_block_marker, format_log_panel

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
    memory_injected: bool
    memory_update_status: Optional[str]


class MessageHandler:
    """Handles message processing, rate limiting, and LLM interactions"""
    _STYLE_PLACEHOLDER = "{DYNAMIC_STYLE_TRAITS}"
    _STYLE_PLACEHOLDER_WITH_FALLBACK_PREFIX = "{DYNAMIC_STYLE_TRAITS|"
    _DEFAULT_STYLE_LINE = "be calm, courteous, uplifting, practical."
    _EXPERTISE_PLACEHOLDER = "{DYNAMIC_EXPERTISE_LEVEL}"
    _EXPERTISE_PLACEHOLDER_WITH_FALLBACK_PREFIX = "{DYNAMIC_EXPERTISE_LEVEL|"
    _DEFAULT_EXPERTISE_LEVEL = "intermediate"
    _HISTORY_PROFILE_BOOTSTRAP_MESSAGE_LIMIT = 6
    _HISTORY_PROFILE_BOOTSTRAP_MAX_SEARCH = 80
    
    def __init__(self, bot: 'AIBot'):
        self.bot = bot

    @classmethod
    def _normalize_style_line(cls, style_line: Optional[str]) -> str:
        line = re.sub(r"\s+", " ", str(style_line or "")).strip()
        if not line:
            line = cls._DEFAULT_STYLE_LINE
        if not line.lower().startswith("be "):
            line = f"be {line}"
        if not line.endswith("."):
            line += "."
        return line

    @classmethod
    def _apply_dynamic_style_to_system_prompt(cls, system_prompt: str, style_line: Optional[str]) -> str:
        prompt = system_prompt or ""
        token_pattern = r"\{DYNAMIC_STYLE_TRAITS(?:\|([^}]+))?\}"
        match = re.search(token_pattern, prompt)
        if match:
            fallback = match.group(1) or cls._DEFAULT_STYLE_LINE
            chosen_line = cls._normalize_style_line(style_line or fallback)
            return re.sub(token_pattern, chosen_line, prompt)

        # If prompt has no style token, keep channel/global persona unchanged when style is unknown.
        if not style_line:
            return prompt

        effective_line = cls._normalize_style_line(style_line)
        return f"{prompt}\n- {effective_line}"

    @classmethod
    def _normalize_expertise_level(cls, expertise_level: Optional[str]) -> str:
        normalized = re.sub(r"\s+", " ", str(expertise_level or "")).strip().lower()
        if normalized in {"beginner", "intermediate", "advanced"}:
            return normalized
        return cls._DEFAULT_EXPERTISE_LEVEL

    @classmethod
    def _apply_dynamic_expertise_to_system_prompt(cls, system_prompt: str, expertise_level: Optional[str]) -> str:
        prompt = system_prompt or ""
        token_pattern = r"\{DYNAMIC_EXPERTISE_LEVEL(?:\|([^}]+))?\}"
        match = re.search(token_pattern, prompt)
        if match:
            fallback = cls._normalize_expertise_level(match.group(1))
            chosen_level = cls._normalize_expertise_level(expertise_level or fallback)
            return re.sub(token_pattern, chosen_level, prompt)

        if not expertise_level:
            return prompt

        effective_level = cls._normalize_expertise_level(expertise_level)
        return f"{prompt}\n- Adapt explanation depth for {effective_level} level."

    @staticmethod
    def _format_user_disk_label(username: Optional[str], display_name: Optional[str]) -> str:
        username_text = re.sub(r"\s+", " ", str(username or "")).strip()
        display_text = re.sub(r"\s+", " ", str(display_name or "")).strip()
        if not username_text and not display_text:
            return ""
        if not username_text:
            username_text = display_text
        if not display_text:
            display_text = username_text
        return f"{username_text}({display_text})"

    @classmethod
    def _resolve_effective_style_line(cls, system_prompt: str, style_line: Optional[str]) -> Optional[str]:
        prompt = system_prompt or ""
        token_pattern = r"\{DYNAMIC_STYLE_TRAITS(?:\|([^}]+))?\}"
        match = re.search(token_pattern, prompt)
        if match:
            fallback = match.group(1) or cls._DEFAULT_STYLE_LINE
            return cls._normalize_style_line(style_line or fallback)
        if style_line:
            return cls._normalize_style_line(style_line)
        return None

    @classmethod
    def _resolve_effective_expertise_level(cls, system_prompt: str, expertise_level: Optional[str]) -> Optional[str]:
        prompt = system_prompt or ""
        token_pattern = r"\{DYNAMIC_EXPERTISE_LEVEL(?:\|([^}]+))?\}"
        match = re.search(token_pattern, prompt)
        if match:
            fallback = cls._normalize_expertise_level(match.group(1))
            return cls._normalize_expertise_level(expertise_level or fallback)
        if expertise_level:
            return cls._normalize_expertise_level(expertise_level)
        return None
        
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
        should_engage, interaction_case, was_random = await self._determine_engagement(
            message=message,
            guild_id=guild_id,
            channel_id=channel_id,
            user_id=user_id,
        )
        
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
            "log_message": None,
            "memory_injected": False,
            "memory_update_status": None
        }
        interaction_started_at = time.time()
        interaction_block_open = False
        interaction_block_closed = False
        history: List[Dict[str, Any]] = []
        messages: List[Dict[str, str]] = []
        response_type = "text"
        response_text = ""
        response_data = ""
        usage: Any = None
        call_metadata: List[Dict[str, Any]] = []
        memory_injected = False
        memory_update_status = "not_started"
        requester_memory_chars = 0
        referenced_users_detected = 0
        referenced_memories_injected = 0
        referenced_memory_chars = 0
        referenced_memory_user_ids: List[int] = []
        total_memory_chars = 0
        is_topic_thread = False
        llm_attempted = False
        llm_audit_written = False

        try:
            is_topic_thread = self._is_daily_topic_thread(message.channel)
            if is_topic_thread:
                history = await self._fetch_thread_history_for_llm(
                    thread=message.channel,  # type: ignore[arg-type]
                    limit=self.bot.daily_topic_thread_context_messages,
                    exclude_message_id=message.id
                )
                logger.info(
                    "Using daily-topic thread context mode: %s messages (ignoring normal TTL/history policy).",
                    len(history)
                )
            else:
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
            referenced_user_candidates: Dict[int, str] = {}

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

            # Collect referenced users from mentions + replied-to author.
            # These memories are injected in addition to the requester memory.
            bot_user_id = self.bot.user.id if self.bot.user else None
            for mentioned_user in message.mentions:
                if bot_user_id is not None and mentioned_user.id == bot_user_id:
                    continue
                if mentioned_user.id == user_id:
                    continue
                mentioned_label = self._format_user_disk_label(
                    getattr(mentioned_user, "name", None) or str(mentioned_user.id),
                    getattr(mentioned_user, "display_name", None),
                )
                referenced_user_candidates[mentioned_user.id] = str(mentioned_label)

            if referenced_msg and referenced_msg.author and referenced_msg.author != self.bot.user:
                referenced_author_id = referenced_msg.author.id
                if referenced_author_id != user_id:
                    referenced_author_label = self._format_user_disk_label(
                        getattr(referenced_msg.author, "name", None) or str(referenced_author_id),
                        getattr(referenced_msg.author, "display_name", None),
                    )
                    referenced_user_candidates[referenced_author_id] = str(referenced_author_label)

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

            elif interaction_case == "Name Trigger":
                logger.info(
                    "Scenario 5: User %s referenced configured bot name within active follow-up window",
                    user_id
                )

            elif was_random and not is_topic_thread:
                logger.info(f"Scenario 4: Random response to user {user_id} (no conversation history)")

                # Fetch general channel context for awareness
                channel_context = await self._fetch_channel_context(message.channel, limit=10)
                # channel_context will be added separately below, not as context_to_inject

            # After resolving the specific scenario, log the final scenario mapping for clarity
            scenario_map_verbose = {
                "Mention": "Scenario 1: Mention",
                "Reply to Bot": "Scenario 2: Reply to Bot",
                "Reply to User + Mention": "Scenario 3: Reply to User + Mention",
                "Random Chance": "Scenario 4: Random Response",
                "Name Trigger": "Scenario 5: Name Trigger in Follow-Up Window",
            }
            scenario_label = scenario_map_verbose.get(interaction_case, interaction_case)

            # Get message preview (first 50 characters)
            message_preview = message.content[:50] if message.content else "(empty)"
            if len(message.content) > 50:
                message_preview += "..."

            # Extract and clean message content first (needed for tool filtering)
            content = message.content
            # Remove bot mentions
            if self.bot.user:
                bot_mention_strings = [f'<@{self.bot.user.id}>', f'<@!{self.bot.user.id}>']
                for mention_str in bot_mention_strings:
                    content = content.replace(mention_str, '')
            content = re.sub(r'\s+', ' ', content).strip()

            memory_manager = getattr(self.bot, "user_memory_manager", None)
            parent_channel_id = message.channel.parent_id if isinstance(message.channel, discord.Thread) else None
            system_prompt = self.bot.get_chat_system_prompt(
                channel_id=channel_id,
                parent_channel_id=parent_channel_id
            )
            style_line: Optional[str] = None
            expertise_level: Optional[str] = None
            if (
                not is_topic_thread
                and self.bot.user_memory_enabled
                and memory_manager
            ):
                try:
                    style_line = await memory_manager.get_user_style_line(user_id)
                except Exception as e:
                    logger.error("Failed loading user style line for %s: %s", user_id, e, exc_info=True)
                try:
                    expertise_level = await memory_manager.get_user_expertise_level(user_id)
                except Exception as e:
                    logger.error("Failed loading user expertise level for %s: %s", user_id, e, exc_info=True)
            effective_style_line = self._resolve_effective_style_line(system_prompt, style_line)
            effective_expertise_level = self._resolve_effective_expertise_level(system_prompt, expertise_level)

            emit_plain_block_marker("INTERACTION START", style="interaction")
            logger.info("[[ INTERACTION START ]]")
            logger.info(
                "\n%s",
                format_log_panel(
                    "INTERACTION HEADER",
                    [
                        ("user", f"{message.author.name} ({user_id})"),
                        ("guild_id", guild_id),
                        ("channel_id", channel_id),
                        ("scenario", scenario_label),
                        ("model", self.bot.litellm_client.model),
                        ("super_user", is_super_user),
                        ("style", effective_style_line or "none"),
                        ("expertise", effective_expertise_level or "none"),
                        ("message_preview", message_preview),
                    ],
                ),
            )
            interaction_block_open = True

            # Fetch MCP tools for availability
            all_mcp_tools = await self.bot.litellm_client.get_mcp_tools()

            # Always provide tools - let the LLM decide when to use them
            # The personality prompt already instructs appropriate tool usage
            # Keyword filtering is fragile (language-dependent, misses natural requests)
            mcp_tools = all_mcp_tools if all_mcp_tools else None

            if mcp_tools:
                logger.info("tools available=%s", len(mcp_tools))
            else:
                logger.info("tools available=0")

            # Build messages for LLM
            messages = []

            # Add system prompt (with optional dynamic communication-style and expertise lines).
            system_prompt = self._apply_dynamic_style_to_system_prompt(system_prompt, style_line)
            system_prompt = self._apply_dynamic_expertise_to_system_prompt(system_prompt, expertise_level)
            messages.append({
                "role": "system",
                "content": f"{system_prompt}\n\nCurrent Date: {time.strftime('%Y-%m-%d')}"
            })

            # Inject persistent user memory before any short-term context.
            # Includes requester memory + any referenced users (mentions/reply target).
            user_memory = ""
            requester_memory_chars = 0
            referenced_users_detected = len(referenced_user_candidates)
            referenced_memories_injected = 0
            referenced_memory_chars = 0
            referenced_memory_user_ids: List[int] = []
            memory_injection_reason = "not_attempted"

            if is_topic_thread:
                memory_injection_reason = "daily_topic_thread_scope"
                logger.debug("Skipping user-memory injection for daily-topic thread context isolation.")
            elif not self.bot.user_memory_enabled:
                memory_injection_reason = "feature_disabled"
            elif not memory_manager:
                memory_injection_reason = "memory_manager_unavailable"
            else:
                # Always load requester memory when feature is enabled (all interaction scenarios).
                memory_context_payload: Dict[str, Any] = {
                    "requesting_user": {
                        "id": user_id,
                        "label": message.author.name,
                        "memory": ""
                    },
                    "referenced_users": []
                }

                try:
                    requester_disk_label = self._format_user_disk_label(
                        getattr(message.author, "name", None),
                        getattr(message.author, "display_name", None),
                    )
                    user_memory = await memory_manager.get_memory(user_id, user_label=requester_disk_label)
                    requester_memory_chars = len(user_memory)
                    if user_memory:
                        memory_context_payload["requesting_user"]["memory"] = user_memory
                        memory_injected = True
                except Exception as e:
                    logger.error("Failed loading requester memory for %s: %s", user_id, e, exc_info=True)

                # Load memory for every referenced user (mentions + replied-to author), excluding bot/self.
                for referenced_user_id, referenced_label in referenced_user_candidates.items():
                    try:
                        referenced_memory = await memory_manager.get_memory(
                            referenced_user_id,
                            user_label=referenced_label,
                        )
                        if not referenced_memory:
                            continue
                        referenced_memories_injected += 1
                        referenced_memory_chars += len(referenced_memory)
                        referenced_memory_user_ids.append(referenced_user_id)
                        memory_injected = True
                        memory_context_payload["referenced_users"].append({
                            "id": referenced_user_id,
                            "label": referenced_label,
                            "memory": referenced_memory
                        })
                    except Exception as e:
                        logger.error(
                            "Failed loading referenced user memory for %s: %s",
                            referenced_user_id,
                            e,
                            exc_info=True
                        )

                if memory_injected:
                    messages.append({
                        "role": "system",
                        "content": (
                            "User memory context for personalization and profile questions (do not expose unless asked).\n"
                            "For user profile facts such as preferences, likes, dislikes, habits, communication style, and background, treat this memory as higher trust than transient conversation context.\n"
                            "Do not let unrelated short-term context overwrite these stored profile facts.\n"
                            "MEMORY_CONTEXT_JSON:\n"
                            + json.dumps(memory_context_payload, ensure_ascii=False)
                        )
                    })

                reason_parts: List[str] = []
                reason_parts.append("requester_injected" if user_memory else "requester_empty")
                if referenced_users_detected > 0:
                    reason_parts.append(
                        f"referenced_injected:{referenced_memories_injected}/{referenced_users_detected}"
                    )
                else:
                    reason_parts.append("referenced_none")
                memory_injection_reason = ",".join(reason_parts)

            total_memory_chars = requester_memory_chars + referenced_memory_chars
            logger.info(
                "[MEMCTX] user=%s(%s) channel=%s case=%s injected=%s reason=%s requester_chars=%s referenced_users_detected=%s referenced_users_injected=%s referenced_chars=%s total_chars=%s",
                message.author.name,
                user_id,
                channel_id,
                interaction_case,
                "yes" if memory_injected else "no",
                memory_injection_reason,
                requester_memory_chars,
                referenced_users_detected,
                referenced_memories_injected,
                referenced_memory_chars,
                total_memory_chars
            )

            # Special daily-topic threads are strictly isolated:
            # always use ONLY thread history as conversation context.
            if is_topic_thread:
                if mcp_tools and len(mcp_tools) > 0:
                    tool_history_limit = max(1, int(getattr(self.bot, "llm_tool_history_limit", 4)))
                    recent_history = history[-tool_history_limit:] if len(history) > tool_history_limit else history
                    messages.extend(recent_history)
                    if len(history) > len(recent_history):
                        logger.debug(
                            "Trimmed thread history from %s to %s messages (tools available)",
                            len(history),
                            len(recent_history)
                        )
                else:
                    messages.extend(history)

            # Scenario-based context building:
            # - Scenarios 1, 2, 3: Include conversation history
            # - Scenario 4 (random): NO conversation history, only channel context

            if was_random and not is_topic_thread:
                # Scenario 4: Random response - NO conversation history
                # Inject channel context as regular user-context text.
                if channel_context:
                    messages.append({
                        "role": "user",
                        "content": channel_context
                    })
            elif not is_topic_thread:
                # Scenarios 1, 2, 3: Include conversation history
                # Limit history when tools are available to prevent token overflow
                if mcp_tools and len(mcp_tools) > 0:
                    # When tools are available, include only recent history.
                    tool_history_limit = max(1, int(getattr(self.bot, "llm_tool_history_limit", 4)))
                    recent_history = history[-tool_history_limit:] if len(history) > tool_history_limit else history
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
            logger.info(
                "context_summary context_messages=%s history_messages=%s memory_injected=%s memory_chars_total=%s memory_chars_requester=%s memory_chars_referenced=%s referenced_users_injected=%s referenced_user_ids=%s style_line=\"%s\" expertise_level=%s",
                len(messages),
                len(history),
                memory_injected,
                total_memory_chars,
                requester_memory_chars,
                referenced_memory_chars,
                referenced_memories_injected,
                referenced_memory_user_ids,
                style_line,
                expertise_level or "none",
            )

            # Call LLM with structured output
            logger.debug(f"Calling LLM for user {user_id} in channel {channel_id}")
            try:
                # Tools already fetched above - no need to fetch again
                # Pass tools to LLM but ALWAYS enable structured output enforcement
                # The chat_completion method will handle the mutual exclusivity internally
                use_structured = True  # Always want structured output in the end
                logger.info("llm_request tools_enabled=%s structured_output=%s", bool(mcp_tools), use_structured)
                llm_attempted = True
                
                # Enable call tracking to capture tool calling passes
                result_tuple = await self.bot.litellm_client.chat_completion(
                    messages=messages,
                    tools=mcp_tools,
                    use_structured_output=use_structured,
                    track_calls=True,
                    call_context={
                        "user_id": user_id,
                        "user_name": getattr(message.author, "display_name", message.author.name),
                        "channel_name": getattr(message.channel, "name", "unknown"),
                        "guild_name": message.guild.name if message.guild else "DM",
                        "source": "message_handler",
                        "interaction_case": interaction_case,
                    }
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

                    logger.debug(
                        "Aggregated tokens from %s LLM passes: %s total (%s prompt + %s completion)",
                        len(call_metadata),
                        f"{total_tokens:,}",
                        f"{total_prompt_tokens:,}",
                        f"{total_completion_tokens:,}",
                    )

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

                    # Rate limit check (only for non-super users)
                    if not is_super_user and not await self._check_token_rate_limit(guild_id, user_id, tokens_used):
                        logger.warning(f"Token rate limit exceeded for user {user_id}")
                        await self._apply_restriction(guild_id, user_id, member)
                        result["restriction_applied"] = True
                        result["was_random_chance"] = was_random
                        result["rate_limit_type"] = "token"
                        result["error"] = "Token rate limit exceeded"
                        return result
            
            # Schedule async user-memory refresh from this message when worthwhile
            memory_update_status = "disabled_or_unavailable"
            if self.bot.user_memory_enabled and memory_manager:
                if is_topic_thread:
                    memory_update_status = "skipped:daily_topic_thread_scope"
                    logger.debug("Skipping memory update for daily-topic thread context isolation.")
                elif interaction_case not in {"Mention", "Reply to Bot", "Reply to User + Mention"}:
                    memory_update_status = f"skipped:interaction_case={interaction_case}"
                    logger.debug(
                        "Skipping memory/profile analysis for non-direct interaction_case=%s user=%s",
                        interaction_case,
                        user_id,
                    )
                else:
                    should_capture, reason = memory_manager.should_capture_message(content)
                    if getattr(self.bot, "user_memory_debug_classification", False):
                        logger.info(
                            "[MEMDBG] pre_gate user=%s(%s) message=\"%s\" selected=%s reason=%s",
                            message.author.name,
                            user_id,
                            (re.sub(r"\s+", " ", content).strip()[:80] + ("..." if len(content) > 80 else "")),
                            "yes" if should_capture else "no",
                            reason
                        )
                    if should_capture:
                        memory_update_status = f"scheduled:{reason}"

                        async def _update_user_memory() -> None:
                            try:
                                ok, update_reason = await memory_manager.update_memory_from_message(
                                    user_id=user_id,
                                    message_content=content,
                                    current_memory=user_memory if user_memory else None,
                                    user_label=self._format_user_disk_label(
                                        getattr(message.author, "name", None),
                                        getattr(message.author, "display_name", None),
                                    ),
                                    guild_id=guild_id,
                                    channel_id=channel_id,
                                    message_id=message.id
                                )
                                logger.info(
                                    "User memory update for %s => ok=%s reason=%s",
                                    user_id,
                                    ok,
                                    update_reason
                                )
                                if (not ok) and str(update_reason or "").startswith("injection_blocked:"):
                                    blocked_notice = None
                                    blocked_confidence = None
                                    blocked_reason = None
                                    try:
                                        _, _, payload_raw = str(update_reason).partition(":")
                                        payload = json.loads(payload_raw) if payload_raw else {}
                                        if isinstance(payload, dict):
                                            parsed_notice = str(payload.get("user_notice", "")).strip()
                                            parsed_confidence = str(payload.get("confidence", "")).strip().lower()
                                            parsed_reason = str(payload.get("reason", "")).strip()
                                            blocked_notice = parsed_notice or None
                                            blocked_confidence = parsed_confidence or None
                                            blocked_reason = parsed_reason or None
                                    except Exception:
                                        blocked_notice = None
                                        blocked_confidence = None
                                        blocked_reason = None
                                    await self._handle_memory_injection_block(
                                        message=message,
                                        interaction_case=interaction_case,
                                        guard_confidence=blocked_confidence,
                                        guard_reason=blocked_reason,
                                        message_content=content,
                                        user_notice=blocked_notice,
                                    )
                                elif (not ok) and str(update_reason or "").startswith("not_worthwhile:"):
                                    should_bootstrap = await memory_manager.should_attempt_history_profile_bootstrap(user_id)
                                    if should_bootstrap:
                                        history_messages = await self._fetch_user_recent_messages(
                                            message.channel,
                                            user_id,
                                            limit=self._HISTORY_PROFILE_BOOTSTRAP_MESSAGE_LIMIT,
                                            max_search=self._HISTORY_PROFILE_BOOTSTRAP_MAX_SEARCH,
                                            exclude_message_id=message.id,
                                        )
                                        bootstrap_ok, bootstrap_reason = await memory_manager.bootstrap_missing_profile_from_history(
                                            user_id=user_id,
                                            user_label=self._format_user_disk_label(
                                                getattr(message.author, "name", None),
                                                getattr(message.author, "display_name", None),
                                            ),
                                            history_messages=history_messages,
                                            guild_id=guild_id,
                                            channel_id=channel_id,
                                            message_id=message.id,
                                        )
                                        logger.info(
                                            "User history profile bootstrap for %s => ok=%s reason=%s",
                                            user_id,
                                            bootstrap_ok,
                                            bootstrap_reason,
                                        )
                            except Exception as e:
                                logger.error(
                                    "User memory async task failed for %s: %s",
                                    user_id,
                                    e,
                                    exc_info=True
                                )

                        asyncio.create_task(_update_user_memory())
                    else:
                        memory_update_status = f"skipped:{reason}"
            elif not self.bot.user_memory_enabled:
                memory_update_status = "feature_disabled"

            # Save conversation history ONLY for scenarios 1, 2, 3
            # Scenario 4 (random) does NOT save history - it's not part of an ongoing conversation
            if not was_random and not is_topic_thread:
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
            elif is_topic_thread:
                logger.debug("Skipped Redis context save for daily-topic thread (using live thread context).")
            else:
                logger.debug(f"Skipped saving history for random response (not part of ongoing conversation)")
            
            # Prepare result
            result["should_respond"] = True
            result["response_text"] = response_text
            result["response_type"] = response_type
            result["response_data"] = response_data
            result["was_random_chance"] = was_random
            result["log_message"] = f"Generated {response_type} response for {interaction_case}"
            result["memory_injected"] = memory_injected
            result["memory_update_status"] = memory_update_status

            logger.info("Interaction processed successfully: scenario=%s response_type=%s", interaction_case, response_type)

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

                logger.info(
                    "\n%s",
                    format_log_panel(
                        "INTERACTION FOOTER",
                        [
                            ("status", "ok"),
                            ("elapsed_ms", f"{(time.time() - interaction_started_at) * 1000:.2f}"),
                            ("tokens_total", f"{tokens_used:,}"),
                            ("tokens_prompt", f"{prompt_tokens:,}"),
                            ("tokens_completion", f"{completion_tokens:,}"),
                            ("tokens_cached", f"{cached_tokens:,}"),
                            ("cost_usd", f"{cost:.6f}"),
                            ("cost_cents", f"{cost_cents:.4f}"),
                            ("llm_calls", len(call_metadata) if call_metadata else 1),
                            ("memory_injected", memory_injected),
                            ("referenced_users_injected", referenced_memories_injected),
                            ("referenced_user_ids", referenced_memory_user_ids),
                            ("memory_update_status", memory_update_status),
                        ],
                    ),
                )
                for idx, call_info in enumerate(call_metadata or [], start=1):
                    tool_calls = call_info.get("tool_calls", [])
                    logger.info(
                        "footer_call[%s] purpose=%s duration_ms=%.2f finish_reason=%s tools=%s",
                        idx,
                        call_info.get("purpose", "unknown"),
                        float(call_info.get("duration", 0.0)) * 1000,
                        call_info.get("finish_reason", "unknown"),
                        len(tool_calls),
                    )
                    for tool_idx, tool in enumerate(tool_calls, start=1):
                        logger.info(
                            "footer_tool[%s.%s] name=%s duration_ms=%s args=%s",
                            idx,
                            tool_idx,
                            tool.get("name", "unknown"),
                            f"{float(tool.get('duration', 0.0)) * 1000:.2f}" if tool.get("duration") is not None else "n/a",
                            json.dumps(tool.get("arguments", {}), ensure_ascii=False)[:400],
                        )
                emit_plain_block_marker("INTERACTION END", style="interaction")
                logger.info("[[ INTERACTION END ]]")
                interaction_block_closed = True
            else:
                logger.info(
                    "\n%s",
                    format_log_panel(
                        "INTERACTION FOOTER",
                        [
                            ("status", "ok"),
                            ("elapsed_ms", f"{(time.time() - interaction_started_at) * 1000:.2f}"),
                            ("tokens_total", "0"),
                            ("tokens_prompt", "0"),
                            ("tokens_completion", "0"),
                            ("tokens_cached", "0"),
                            ("cost_usd", "0.000000"),
                            ("cost_cents", "0.0000"),
                            ("llm_calls", len(call_metadata) if call_metadata else 1),
                            ("memory_injected", memory_injected),
                            ("referenced_users_injected", referenced_memories_injected),
                            ("referenced_user_ids", referenced_memory_user_ids),
                            ("memory_update_status", memory_update_status),
                        ],
                    ),
                )
                emit_plain_block_marker("INTERACTION END", style="interaction")
                logger.info("[[ INTERACTION END ]]")
                interaction_block_closed = True

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

            # Persist rolling LLM interaction audit log in Redis
            await self._record_llm_call_audit(
                guild_id=guild_id,
                user_id=user_id,
                channel_id=channel_id,
                message_id=message.id,
                interaction_case=interaction_case,
                was_random=was_random,
                is_topic_thread=is_topic_thread,
                memory_injected=memory_injected,
                memory_update_status=memory_update_status,
                context_messages=messages,
                response_type=response_type,
                response_text=response_text,
                response_data=response_data,
                usage=usage,
                call_metadata=call_metadata,
                status="ok",
                elapsed_ms=(time.time() - interaction_started_at) * 1000,
            )
            llm_audit_written = True

            return result
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            result["error"] = f"Processing error: {str(e)}"
            return result
        finally:
            if interaction_block_open and not interaction_block_closed:
                prompt_tokens = usage.prompt_tokens if usage and hasattr(usage, "prompt_tokens") else 0
                completion_tokens = usage.completion_tokens if usage and hasattr(usage, "completion_tokens") else 0
                total_tokens = usage.total_tokens if usage and hasattr(usage, "total_tokens") else 0
                logger.info(
                    "\n%s",
                    format_log_panel(
                        "INTERACTION FOOTER",
                        [
                            ("status", "aborted"),
                            ("elapsed_ms", f"{(time.time() - interaction_started_at) * 1000:.2f}"),
                            ("llm_calls", len(call_metadata) if call_metadata else 0),
                            ("tokens_total", total_tokens),
                            ("tokens_prompt", prompt_tokens),
                            ("tokens_completion", completion_tokens),
                            ("referenced_users_injected", referenced_memories_injected),
                            ("referenced_user_ids", referenced_memory_user_ids),
                            ("error", result.get("error", "unknown")),
                        ],
                    ),
                )
                emit_plain_block_marker("INTERACTION END", style="interaction")
                logger.info("[[ INTERACTION END ]]")
            if llm_attempted and not llm_audit_written:
                try:
                    await self._record_llm_call_audit(
                        guild_id=guild_id,
                        user_id=user_id,
                        channel_id=channel_id,
                        message_id=message.id,
                        interaction_case=interaction_case,
                        was_random=was_random,
                        is_topic_thread=is_topic_thread,
                        memory_injected=memory_injected,
                        memory_update_status=memory_update_status,
                        context_messages=messages,
                        response_type=result.get("response_type") or "error",
                        response_text=result.get("response_text") or "",
                        response_data=result.get("response_data") or "",
                        usage=usage,
                        call_metadata=call_metadata,
                        status="error",
                        error=result.get("error") or "unknown",
                        elapsed_ms=(time.time() - interaction_started_at) * 1000,
                    )
                except Exception as audit_err:
                    logger.error("Failed writing fallback LLM audit: %s", audit_err, exc_info=True)

    def _is_daily_topic_thread(self, channel: discord.abc.Messageable) -> bool:
        if not isinstance(channel, discord.Thread):
            return False
        publish_channel_id = getattr(self.bot, "daily_topic_publish_channel_id", None)
        return bool(publish_channel_id and channel.parent_id == publish_channel_id)

    async def _fetch_thread_history_for_llm(
        self,
        thread: discord.Thread,
        limit: int,
        exclude_message_id: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Fetch recent messages from a thread as LLM chat history.

        This intentionally bypasses Redis TTL/max-history rules to preserve thread continuity.
        """
        messages: List[Dict[str, str]] = []
        try:
            async for msg in thread.history(limit=limit, oldest_first=False):
                if exclude_message_id and msg.id == exclude_message_id:
                    continue
                if not msg.content:
                    continue

                if self.bot.user and msg.author.id == self.bot.user.id:
                    messages.append({"role": "assistant", "content": msg.content})
                else:
                    author_label = msg.author.display_name if hasattr(msg.author, "display_name") else msg.author.name
                    messages.append({"role": "user", "content": f"{author_label}: {msg.content}"})
        except Exception as e:
            logger.error("Error fetching thread history for LLM: %s", e)
            return []

        messages.reverse()
        return messages

    def _truncate_context_messages_for_audit(
        self,
        messages: List[Dict[str, str]],
        max_messages: Optional[int] = None,
        max_chars_per_message: Optional[int] = None
    ) -> List[Dict[str, str]]:
        effective_max_messages = max_messages or int(getattr(self.bot, "llm_audit_context_max_messages", 40))
        effective_max_chars = max_chars_per_message or int(getattr(self.bot, "llm_audit_context_max_chars", 1200))
        clipped = messages[-effective_max_messages:] if len(messages) > effective_max_messages else messages
        result: List[Dict[str, str]] = []
        for msg in clipped:
            role = str(msg.get("role", "unknown"))
            content = str(msg.get("content", ""))
            if len(content) > effective_max_chars:
                content = content[:effective_max_chars] + " ...[truncated]"
            result.append({"role": role, "content": content})
        return result

    async def _record_llm_call_audit(
        self,
        guild_id: int,
        user_id: int,
        channel_id: int,
        message_id: int,
        interaction_case: str,
        was_random: bool,
        is_topic_thread: bool,
        memory_injected: bool,
        memory_update_status: str,
        context_messages: List[Dict[str, str]],
        response_type: str,
        response_text: str,
        response_data: str,
        usage: Any,
        call_metadata: List[Dict[str, Any]],
        status: str = "ok",
        error: str = "",
        elapsed_ms: float = 0.0
    ) -> None:
        if not getattr(self.bot, "llm_call_audit_enabled", False):
            return
        if not self.bot.redis_client:
            return

        try:
            prompt_tokens = usage.prompt_tokens if usage and hasattr(usage, "prompt_tokens") else 0
            completion_tokens = usage.completion_tokens if usage and hasattr(usage, "completion_tokens") else 0
            total_tokens = usage.total_tokens if usage and hasattr(usage, "total_tokens") else 0

            payload = {
                "ts": time.time(),
                "guild_id": guild_id,
                "user_id": user_id,
                "channel_id": channel_id,
                "message_id": message_id,
                "model": self.bot.litellm_client.model,
                "interaction_case": interaction_case,
                "was_random": was_random,
                "is_topic_thread": is_topic_thread,
                "status": status,
                "error": error,
                "elapsed_ms": round(float(elapsed_ms), 2),
                "memory_injected": memory_injected,
                "memory_update_status": memory_update_status,
                "tokens": {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                    "total": total_tokens,
                },
                "call_count": len(call_metadata) if call_metadata else 1,
                "llm_calls": call_metadata or [],
                "context_messages": self._truncate_context_messages_for_audit(context_messages),
                "response_type": response_type,
                "response_text": (str(response_text)[:800] + "...") if len(str(response_text)) > 800 else str(response_text),
                "response_data": (str(response_data)[:600] + "...") if len(str(response_data)) > 600 else str(response_data),
            }

            key = f"llm_calls:recent:{guild_id}"
            await asyncio.to_thread(self.bot.redis_client.lpush, key, json.dumps(payload, ensure_ascii=False))
            await asyncio.to_thread(
                self.bot.redis_client.ltrim,
                key,
                0,
                int(getattr(self.bot, "llm_call_audit_max_entries", 100)) - 1
            )
        except Exception as e:
            logger.error("Failed to record LLM call audit: %s", e, exc_info=True)

    async def _notify_user_memory_injection_block(
        self,
        message: discord.Message,
        user_notice: Optional[str] = None,
    ) -> str:
        notice = re.sub(r"\s+", " ", str(user_notice or "")).strip()
        if not notice:
            notice = (
                "I can see you are trying to influence my internal memory/profile pipeline. "
                "I will not store that message."
            )

        try:
            await message.author.send(notice)
            return "dm"
        except Exception:
            pass

        try:
            await message.reply(notice, mention_author=False, delete_after=20)
            return "channel_delete_after"
        except Exception:
            return "notify_failed"

    async def _report_memory_injection_block(
        self,
        *,
        message: discord.Message,
        interaction_case: str,
        guard_confidence: Optional[str],
        guard_reason: Optional[str],
        message_content: str,
        notify_mode: str,
        user_notice: Optional[str] = None,
    ) -> None:
        preview = re.sub(r"\s+", " ", str(message_content or "")).strip()
        if len(preview) > 300:
            preview = preview[:300] + "..."
        notice_preview = re.sub(r"\s+", " ", str(user_notice or "")).strip()
        if len(notice_preview) > 200:
            notice_preview = notice_preview[:200] + "..."

        guard_confidence_value = re.sub(r"\s+", " ", str(guard_confidence or "")).strip().lower() or "unknown"
        guard_reason_value = re.sub(r"\s+", " ", str(guard_reason or "")).strip()
        if len(guard_reason_value) > 300:
            guard_reason_value = guard_reason_value[:300] + "..."

        guild_name = message.guild.name if message.guild else "Direct Message"
        channel_name = getattr(message.channel, "name", None) or str(message.channel)
        channel_label = f"#{channel_name}" if message.guild else channel_name
        message_link = getattr(message, "jump_url", "") or ""
        channel_link = ""
        if message.guild:
            channel_link = f"https://discord.com/channels/{message.guild.id}/{message.channel.id}"
        created_at = discord.utils.format_dt(message.created_at, style="F")

        payload = {
            "ts": time.time(),
            "ts_iso": message.created_at.isoformat(),
            "guild_id": message.guild.id if message.guild else 0,
            "guild_name": guild_name,
            "channel_id": message.channel.id,
            "channel_name": channel_name,
            "channel_link": channel_link,
            "message_id": message.id,
            "message_link": message_link,
            "user_id": message.author.id,
            "user_global_name": getattr(message.author, "global_name", None) or "",
            "user_name": getattr(message.author, "display_name", message.author.name),
            "interaction_case": interaction_case,
            "confidence": guard_confidence_value,
            "reason": guard_reason_value,
            "notify_mode": notify_mode,
            "message_preview": preview,
            "user_notice": notice_preview,
        }

        logger.warning(
            "[MEMSAFE] blocked user=%s channel=%s guild=%s case=%s confidence=%s reason=%s notify=%s notice=\"%s\" preview=\"%s\"",
            payload["user_name"],
            payload["channel_name"],
            payload["guild_name"],
            payload["interaction_case"],
            payload["confidence"],
            payload["reason"],
            payload["notify_mode"],
            payload["user_notice"],
            payload["message_preview"],
        )

        redis_client = self.bot.redis_client
        if redis_client:
            try:
                key = "user_memory_security_alerts:recent"
                await asyncio.to_thread(redis_client.lpush, key, json.dumps(payload, ensure_ascii=False))
                await asyncio.to_thread(
                    redis_client.ltrim,
                    key,
                    0,
                    max(1, int(getattr(self.bot, "user_memory_audit_max_entries", 200))) - 1
                )
            except Exception as e:
                logger.error("Failed to record memory security alert in Redis: %s", e, exc_info=True)

        report_channel_id = getattr(self.bot, "stats_report_channel_id", None)
        if not report_channel_id:
            return

        try:
            report_channel = self.bot.get_channel(report_channel_id)
            if not report_channel or not isinstance(report_channel, discord.TextChannel):
                return

            embed = discord.Embed(
                title="Memory Security Alert",
                description="Memory/profile update was blocked due to suspected injection attempt.",
                color=discord.Color.orange()
            )
            user_title = payload["user_name"]
            if payload["user_global_name"] and payload["user_global_name"] != payload["user_name"]:
                user_title = f"{payload['user_name']} ({payload['user_global_name']})"
            embed.add_field(name="User", value=user_title, inline=False)
            embed.add_field(name="Server", value=payload["guild_name"], inline=False)
            embed.add_field(name="Channel", value=channel_label, inline=True)
            embed.add_field(name="Time", value=created_at, inline=True)
            embed.add_field(name="Interaction", value=payload["interaction_case"], inline=True)
            embed.add_field(name="Confidence", value=payload["confidence"], inline=True)
            embed.add_field(name="Notify Mode", value=payload["notify_mode"], inline=True)
            embed.add_field(name="Analysis Reason", value=payload["reason"] or "(none)", inline=False)
            embed.add_field(name="User Notice", value=payload["user_notice"] or "(none)", inline=False)
            embed.add_field(name="Original Message", value=payload["message_preview"] or "(empty)", inline=False)
            if payload["message_link"]:
                embed.add_field(name="Message Link", value=f"[Open Message]({payload['message_link']})", inline=True)
            if payload["channel_link"]:
                embed.add_field(name="Channel Link", value=f"[Open Channel]({payload['channel_link']})", inline=True)
            await report_channel.send(embed=embed)
        except Exception as e:
            logger.error("Failed to send memory security alert report: %s", e, exc_info=True)

    async def _handle_memory_injection_block(
        self,
        *,
        message: discord.Message,
        interaction_case: str,
        guard_confidence: Optional[str],
        guard_reason: Optional[str],
        message_content: str,
        user_notice: Optional[str],
    ) -> None:
        notify_mode = await self._notify_user_memory_injection_block(message, user_notice=user_notice)
        await self._report_memory_injection_block(
            message=message,
            interaction_case=interaction_case,
            guard_confidence=guard_confidence,
            guard_reason=guard_reason,
            message_content=message_content,
            notify_mode=notify_mode,
            user_notice=user_notice,
        )
    
    def _name_followup_window_key(self, guild_id: int, channel_id: int, user_id: int) -> str:
        return f"name_followup_window:{guild_id}:{channel_id}:{user_id}"

    async def _get_name_followup_window_remaining(self, guild_id: int, channel_id: int, user_id: int) -> int:
        key = self._name_followup_window_key(guild_id, channel_id, user_id)
        redis_client = self.bot.redis_client
        if redis_client:
            try:
                raw = await asyncio.to_thread(redis_client.get, key)
                return int(raw) if raw is not None else 0
            except Exception as e:
                logger.error("Failed reading name follow-up window from Redis: %s", e, exc_info=True)
                return 0
        local_windows = getattr(self.bot, "bot_name_followup_windows", {})
        return int(local_windows.get(key, 0))

    async def _set_name_followup_window_remaining(
        self,
        guild_id: int,
        channel_id: int,
        user_id: int,
        remaining: int
    ) -> None:
        key = self._name_followup_window_key(guild_id, channel_id, user_id)
        bounded_remaining = max(0, int(remaining))
        redis_client = self.bot.redis_client
        if redis_client:
            try:
                if bounded_remaining > 0:
                    await asyncio.to_thread(redis_client.set, key, bounded_remaining)
                else:
                    await asyncio.to_thread(redis_client.delete, key)
            except Exception as e:
                logger.error("Failed writing name follow-up window to Redis: %s", e, exc_info=True)
            return

        local_windows = getattr(self.bot, "bot_name_followup_windows", None)
        if local_windows is None:
            local_windows = {}
            setattr(self.bot, "bot_name_followup_windows", local_windows)

        if bounded_remaining > 0:
            local_windows[key] = bounded_remaining
        else:
            local_windows.pop(key, None)

    def _message_contains_configured_bot_name(self, content: str) -> bool:
        aliases = getattr(self.bot, "bot_name_trigger_aliases", [])
        if not aliases:
            return False

        normalized_content = (content or "").casefold()
        for alias in aliases:
            alias_clean = alias.strip()
            if not alias_clean:
                continue
            pattern = r"(?<!\w)" + re.escape(alias_clean) + r"(?!\w)"
            if re.search(pattern, normalized_content):
                return True
        return False

    async def _determine_engagement(
        self,
        message: discord.Message,
        guild_id: int,
        channel_id: int,
        user_id: int
    ) -> Tuple[bool, str, bool]:
        """
        Determine if bot should respond to this message.
        Returns: (should_respond, interaction_case, was_random_chance)
        """
        name_window_enabled = (
            getattr(self.bot, "bot_name_followup_window_messages", 0) > 0
            and bool(getattr(self.bot, "bot_name_trigger_aliases", []))
        )
        is_mention = self.bot.user in message.mentions if self.bot.user else False
        is_reply_to_bot = False
        window_remaining = 0
        if name_window_enabled:
            window_remaining = await self._get_name_followup_window_remaining(guild_id, channel_id, user_id)
        
        # Check if reply to bot
        if message.reference and message.reference.resolved:
            replied_msg = message.reference.resolved
            if isinstance(replied_msg, discord.Message) and replied_msg.author == self.bot.user:
                is_reply_to_bot = True
        
        # Priority 1: Reply to bot (takes precedence even if user also mentions)
        if is_reply_to_bot:
            if name_window_enabled:
                await self._set_name_followup_window_remaining(
                    guild_id,
                    channel_id,
                    user_id,
                    self.bot.bot_name_followup_window_messages
                )
            return (True, "Reply to Bot", False)
        
        # Priority 2: Direct mention
        if is_mention:
            if name_window_enabled:
                await self._set_name_followup_window_remaining(
                    guild_id,
                    channel_id,
                    user_id,
                    self.bot.bot_name_followup_window_messages
                )
            return (True, "Mention", False)

        # Priority 3: Configured bot-name trigger (only while follow-up window is active)
        if name_window_enabled and window_remaining > 0:
            if self._message_contains_configured_bot_name(message.content):
                await self._set_name_followup_window_remaining(
                    guild_id,
                    channel_id,
                    user_id,
                    self.bot.bot_name_followup_window_messages
                )
                logger.info(
                    "Name follow-up trigger matched: user=%s channel=%s remaining_reset_to=%s",
                    user_id,
                    channel_id,
                    self.bot.bot_name_followup_window_messages
                )
                return (True, "Name Trigger", False)

            await self._set_name_followup_window_remaining(
                guild_id,
                channel_id,
                user_id,
                window_remaining - 1
            )
            logger.debug(
                "Name follow-up window consumed: user=%s channel=%s remaining=%s",
                user_id,
                channel_id,
                max(0, window_remaining - 1)
            )
            return (False, "No Trigger", False)
        
        # Priority 4: Random chance
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
        max_search: int = 50,
        exclude_message_id: Optional[int] = None,
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
                if exclude_message_id and msg.id == exclude_message_id:
                    continue
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
