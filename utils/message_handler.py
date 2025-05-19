# utils/message_handler.py
import discord
import time
import random
import re
import logging
from typing import TYPE_CHECKING, Optional, Dict, List, Tuple, Set, Literal, TypedDict

if TYPE_CHECKING:
    from bot import AIBot 
    from .webui_api import WebUIAPI 

logger = logging.getLogger(__name__)

class MessageHandlerResult(TypedDict, total=False):
    action: Literal["reply", "notify_restricted_channel", "apply_restriction", "error", "do_nothing"]
    content: Optional[str] 
    user_id_to_restrict: Optional[int]
    guild_id_for_restriction: Optional[int]
    restriction_reason: Optional[str]
    log_message: Optional[str] 
    triggering_interaction_case: Optional[str] # NEW FIELD

class MessageHandler:
    def __init__(self, bot_instance: 'AIBot', message: discord.Message):
        self.bot: 'AIBot' = bot_instance
        self.message: discord.Message = message
        self.member: discord.Member = message.author if isinstance(message.author, discord.Member) else None # type: ignore
        self.guild: discord.Guild = message.guild # type: ignore
        self.user_id: int = message.author.id
        self.channel_id: int = message.channel.id
        self.current_time: float = time.time()
        
        self.api_client: 'WebUIAPI' = self.bot.api_client # type: ignore
        self.redis_client = self.bot.redis_client_general

        self.author_role_ids: Set[int] = set()
        if self.member: 
             self.author_role_ids = {role.id for role in self.member.roles}

        self.is_rate_limit_exempt: bool = False
        if self.member: 
            self.is_rate_limit_exempt = self.bot.rate_limit_exempt_role_ids_set and \
                                        not self.bot.rate_limit_exempt_role_ids_set.isdisjoint(self.author_role_ids)

    def _determine_engagement(self) -> Tuple[bool, str, str, bool, bool]:
        content_for_llm = self.message.content
        bot_user_id = self.bot.user.id if self.bot.user else 0 # type: ignore
        bot_mention_strings = [f'<@{bot_user_id}>', f'<@!{bot_user_id}>']
        
        for mention_str in bot_mention_strings:
            content_for_llm = content_for_llm.replace(mention_str, '')
        content_for_llm = re.sub(r'\s+', ' ', content_for_llm).strip()

        should_respond = False
        interaction_case_debug = "No Interaction Triggered"
        is_reply = self.message.reference and self.message.reference.resolved and isinstance(self.message.reference.resolved, discord.Message)
        is_reply_to_bot = False
        is_mention = False
        if self.bot.user: 
            is_mention = self.bot.user in self.message.mentions
        
        if is_reply:
            replied_to_message: discord.Message = self.message.reference.resolved # type: ignore
            if replied_to_message.author == self.bot.user:
                is_reply_to_bot = True
                should_respond = True
                interaction_case_debug = "Reply to Bot" # Simplified case name
            elif is_mention and content_for_llm:
                should_respond = True
                interaction_case_debug = "Reply to User + Mention"
        
        if not should_respond and is_mention:
            if content_for_llm:
                if not is_reply: 
                    should_respond = True
                    interaction_case_debug = "Direct Mention"
            else: 
                interaction_case_debug = "Mention Only (No Content)"

        if not should_respond and (random.random() < self.bot.response_chance):
            should_respond = True
            interaction_case_debug = "Random Chance" # This is the key case name
            if not content_for_llm and self.message.content: 
                 content_for_llm = self.message.content 
        
        author_name = self.member.name if self.member else f"User {self.user_id}"
        logger.debug(f"MessageHandler: Engagement for {author_name}: should_respond={should_respond}, case='{interaction_case_debug}', content='{content_for_llm[:30]}...'")
        return should_respond, content_for_llm, interaction_case_debug, is_reply_to_bot, is_mention

    def _check_channel_restrictions(self) -> Optional[MessageHandlerResult]:
        if not self.member: return None 

        is_currently_restricted_by_role = self.bot.restricted_user_role_id and \
                                          (self.bot.restricted_user_role_id in self.author_role_ids)
        if is_currently_restricted_by_role:
            logger.debug(f"MessageHandler: User {self.member.name} has restricted role.")
            if self.bot.restricted_channel_id and self.channel_id != self.bot.restricted_channel_id:
                logger.info(f"MessageHandler: Restricted user {self.member.name} in disallowed channel {self.message.channel.name}. Notifying.") # type: ignore
                notification_content = self.bot.restricted_channel_message_user_template.replace("<#{channel_id}>", f"<#{self.bot.restricted_channel_id}>")
                return MessageHandlerResult(action="notify_restricted_channel", content=notification_content)
        return None

    async def _check_message_rate_limit(self, interaction_case_debug: str) -> Optional[MessageHandlerResult]:
        if not self.member: return None 
        if self.is_rate_limit_exempt or not self.redis_client or not self.bot.restricted_user_role_id or self.bot.rate_limit_count <= 0:
            return None
        
        if self.bot.restricted_user_role_id in self.author_role_ids: # Already restricted
            return None

        msg_rl_key = f"msg_rl:{self.guild.id}:{self.user_id}"
        try:
            await discord.utils.asyncio.to_thread(self.redis_client.lpush, msg_rl_key, self.current_time) # type: ignore
            min_time_for_window = self.current_time - self.bot.rate_limit_window_seconds
            
            timestamps_in_list_str_bytes = await discord.utils.asyncio.to_thread(self.redis_client.lrange, msg_rl_key, 0, -1) # type: ignore
            timestamps_in_list_str = [ts.decode('utf-8') if isinstance(ts, bytes) else ts for ts in timestamps_in_list_str_bytes]
            
            messages_in_window_count = 0
            valid_timestamps_for_new_list = []
            for ts_str in timestamps_in_list_str:
                try:
                    ts = float(ts_str)
                    if ts > min_time_for_window:
                        messages_in_window_count += 1
                        valid_timestamps_for_new_list.append(str(ts))
                except ValueError:
                    logger.warning(f"MessageHandler: Non-float value '{ts_str}' in Redis list {msg_rl_key}")
            
            def transaction_fn(pipe):
                pipe.delete(msg_rl_key)
                if valid_timestamps_for_new_list:
                    pipe.rpush(msg_rl_key, *valid_timestamps_for_new_list)
                pipe.expire(msg_rl_key, self.bot.rate_limit_window_seconds + 120)
            
            await discord.utils.asyncio.to_thread(self.redis_client.transaction, transaction_fn) # type: ignore

            if messages_in_window_count > self.bot.rate_limit_count:
                logger.info(f"MessageHandler: User {self.member.name} ({self.user_id}) exceeded MESSAGE rate limit ({messages_in_window_count}/{self.bot.rate_limit_count}). Trigger: {interaction_case_debug}")
                return MessageHandlerResult(
                    action="apply_restriction",
                    user_id_to_restrict=self.user_id,
                    guild_id_for_restriction=self.guild.id,
                    restriction_reason="Exceeded message rate limit",
                    triggering_interaction_case=interaction_case_debug # Pass it back
                )
        except Exception as e:
            logger.error(f"MessageHandler: Error in message rate limit for {self.member.name}: {e}", exc_info=True)
        return None

    async def _check_token_rate_limit(self, tokens_used: int, interaction_case_debug: str) -> Optional[MessageHandlerResult]:
        if not self.member: return None
        if self.is_rate_limit_exempt or not self.redis_client or not self.bot.restricted_user_role_id or self.bot.token_rate_limit_count <= 0 or tokens_used <= 0:
            return None
        
        is_already_restricted = self.bot.restricted_user_role_id in self.author_role_ids
        if is_already_restricted:
             logger.debug(f"MessageHandler: User {self.member.name} already restricted. Tokens for this interaction ({tokens_used}) will be recorded, but no new token restriction applied now.")
        
        token_rl_key = f"token_rl:{self.guild.id}:{self.user_id}"
        try:
            await discord.utils.asyncio.to_thread(self.redis_client.lpush, token_rl_key, f"{self.current_time}:{tokens_used}") # type: ignore
            min_time_token_window = self.current_time - self.bot.rate_limit_window_seconds
            
            entries_in_list_bytes = await discord.utils.asyncio.to_thread(self.redis_client.lrange, token_rl_key, 0, -1) # type: ignore
            entries_in_list = [entry.decode('utf-8') if isinstance(entry, bytes) else entry for entry in entries_in_list_bytes]

            total_tokens_in_window = 0
            valid_entries_for_trim = []

            for entry in entries_in_list:
                try:
                    ts_str, tk_str = entry.split(":", 1)
                    ts = float(ts_str)
                    tk = int(tk_str)
                    if ts > min_time_token_window:
                        total_tokens_in_window += tk
                        valid_entries_for_trim.append(entry)
                except (ValueError, IndexError):
                    logger.warning(f"MessageHandler: Malformed entry in {token_rl_key}: {entry}")
            
            def transaction_fn_tokens(pipe):
                pipe.delete(token_rl_key)
                if valid_entries_for_trim:
                    pipe.rpush(token_rl_key, *valid_entries_for_trim)
                pipe.expire(token_rl_key, self.bot.rate_limit_window_seconds + 120)

            await discord.utils.asyncio.to_thread(self.redis_client.transaction, transaction_fn_tokens) # type: ignore

            if total_tokens_in_window > self.bot.token_rate_limit_count and not is_already_restricted:
                logger.info(f"MessageHandler: User {self.member.name} ({self.user_id}) exceeded TOKEN rate limit ({total_tokens_in_window}/{self.bot.token_rate_limit_count}). Trigger: {interaction_case_debug}")
                return MessageHandlerResult(
                    action="apply_restriction",
                    user_id_to_restrict=self.user_id,
                    guild_id_for_restriction=self.guild.id,
                    restriction_reason="Exceeded token usage rate limit",
                    triggering_interaction_case=interaction_case_debug # Pass it back
                )
        except Exception as e:
            logger.error(f"MessageHandler: Error in token rate limit for {self.member.name}: {e}", exc_info=True)
        return None

    async def _handle_llm_interaction(self, content_for_llm: str, interaction_case_debug: str, is_reply_to_bot: bool) -> MessageHandlerResult:
        if not self.member: 
            return MessageHandlerResult(action="error", content="Cannot process request without valid member context.")

        chat_system_prompt = self.bot.chat_system_prompt
        current_history: List[Dict[str, str]] = self.api_client.get_context_history(self.user_id, self.channel_id)
        extra_assistant_context: Optional[str] = None
        inject_context_for_saving = False
        
        # Context Injection Logic (simplified from before, ensure it covers your needs)
        if interaction_case_debug == "Reply to User + Mention" and self.message.reference and self.message.reference.resolved:
            replied_to_msg = self.message.reference.resolved
            if isinstance(replied_to_msg, discord.Message) and replied_to_msg.content:
                context_prefix = f"Context from reply to {replied_to_msg.author.display_name} (User ID: {replied_to_msg.author.id}):"
                extra_assistant_context = f"{context_prefix}\n```\n{replied_to_msg.content}\n```"
                inject_context_for_saving = True
        elif is_reply_to_bot and self.message.reference and self.message.reference.resolved: # Catches "Reply to Bot" case
            replied_to_bot_message = self.message.reference.resolved
            if isinstance(replied_to_bot_message, discord.Message):
                last_bot_msg_content_in_history = None
                # Look for the actual last assistant message in history
                for i in range(len(current_history) - 1, -1, -1):
                    if current_history[i]['role'] == 'assistant':
                        last_bot_msg_content_in_history = current_history[i]['content']
                        break
                
                if replied_to_bot_message.content != last_bot_msg_content_in_history:
                    extra_assistant_context = replied_to_bot_message.content
                    inject_context_for_saving = True
                    # Update interaction_case_debug for logging if it was just "Reply to Bot"
                    if interaction_case_debug == "Reply to Bot":
                        interaction_case_debug += " (Injected external bot context)"
        
        author_name = self.member.name if self.member else f"User {self.user_id}"
        logger.info(f"MessageHandler: Requesting LLM for {author_name} (Case: {interaction_case_debug}, ExtraCtx: {extra_assistant_context is not None})")
        
        response_content: Optional[str] = None
        error_message_from_api: Optional[str] = None
        tokens_used: Optional[int] = 0

        async with self.message.channel.typing():
            response_content, error_message_from_api, tokens_used = await self.api_client.generate_response(
                self.user_id, self.channel_id, content_for_llm, chat_system_prompt,
                history=current_history, extra_assistant_context=extra_assistant_context
            )

        if error_message_from_api or not response_content:
            user_facing_error = response_content or "Sorry, an error occurred while I was trying to respond."
            logger.error(f"MessageHandler: API Error for {author_name}: Err='{error_message_from_api}', No Content='{not response_content}'")
            return MessageHandlerResult(action="error", content=user_facing_error, log_message=f"API Error: {error_message_from_api or 'No content'}")

        next_history = list(current_history)
        if inject_context_for_saving and extra_assistant_context:
            next_history.append({"role": "assistant", "content": extra_assistant_context})
        next_history.append({"role": "user", "content": content_for_llm})
        next_history.append({"role": "assistant", "content": response_content})
        self.api_client.save_context_history(self.user_id, self.channel_id, next_history)
        logger.debug(f"MessageHandler: Saved history for {author_name} (Injected: {inject_context_for_saving and extra_assistant_context is not None})")

        if tokens_used is not None and tokens_used > 0:
            # Pass interaction_case_debug to token limit check
            token_limit_result = await self._check_token_rate_limit(tokens_used, interaction_case_debug)
            if token_limit_result:
                 logger.info(f"MessageHandler: Token limit hit for {author_name} AFTER LLM. Prioritizing restriction.")
                 log_msg = (
                    f"Token rate limit hit for {author_name} after response (Trigger: {interaction_case_debug}). "
                    f"LLM content ({response_content[:50]}...) may not be sent by cog."
                 )
                 token_limit_result["log_message"] = log_msg
                 # Ensure triggering_interaction_case is set in the result by _check_token_rate_limit
                 return token_limit_result
        
        return MessageHandlerResult(action="reply", content=response_content)

    async def process(self) -> MessageHandlerResult:
        author_name = self.message.author.name 
        if not self.member: 
             return MessageHandlerResult(action="do_nothing", log_message=f"Message author {author_name} ({self.user_id}) is not a valid member.")

        if self.bot.ignored_role_ids_set and not self.bot.ignored_role_ids_set.isdisjoint(self.author_role_ids):
            logger.debug(f"MessageHandler: Message from {self.member.name} ignored due to globally ignored role(s).")
            return MessageHandlerResult(action="do_nothing", log_message="User has ignored role.")

        should_respond, content_for_llm, interaction_case_debug, is_reply_to_bot, _ = self._determine_engagement()
        
        if not content_for_llm and should_respond: 
            logger.info(f"MessageHandler: Bot to respond to {self.member.name} but content_for_llm is empty. Case: {interaction_case_debug}.")
            return MessageHandlerResult(action="do_nothing", log_message="No processable text content for LLM, though engagement was triggered.")

        if should_respond:
            channel_restriction_result = self._check_channel_restrictions()
            if channel_restriction_result:
                # This restriction isn't tied to a specific user prompt leading to rate limit,
                # so it should probably always notify. We can add interaction_case_debug if needed.
                # For now, these notifications are direct.
                return channel_restriction_result
        
        if not should_respond:
            logger.debug(f"MessageHandler: No engagement for {self.member.name}. Final case: {interaction_case_debug}.")
            return MessageHandlerResult(action="do_nothing", log_message=f"No engagement: {interaction_case_debug}")

        # Pass interaction_case_debug to message rate limit check
        message_rate_limit_result = await self._check_message_rate_limit(interaction_case_debug)
        if message_rate_limit_result:
            return message_rate_limit_result

        if not self.guild or not self.message.channel.permissions_for(self.guild.me).send_messages: 
             logger.error(f"MessageHandler: Cannot respond in {self.message.channel.name}: Missing 'Send Messages' permission or invalid guild context.") 
             return MessageHandlerResult(action="do_nothing", log_message="Missing send message permission in channel or invalid guild.")

        # Pass interaction_case_debug to LLM handler
        llm_result = await self._handle_llm_interaction(content_for_llm, interaction_case_debug, is_reply_to_bot)
        return llm_result