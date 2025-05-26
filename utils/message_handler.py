# utils/message_handler.py
# ... (imports including langdetect, TypedDicts remain the same as the last complete version) ...
import discord
import time
import random
import re
import logging
import json
import redis
from typing import TYPE_CHECKING, Optional, Dict, List, Tuple, Set, Literal, TypedDict, Any

from langdetect import detect as detect_language_code
from langdetect.lang_detect_exception import LangDetectException

if TYPE_CHECKING:
    from bot import AIBot
    # from .webui_api import WebUIAPI # Already imported via AIBot type hint if needed

logger = logging.getLogger(__name__)

# MODIFIED MessageHandlerResult TypedDict
class MessageHandlerResult(TypedDict, total=False):
    action: Literal[
        "reply_text", "reply_with_url", "reply_with_gif", "reply_with_latex", "reply_with_code",
        "notify_restricted_channel", "apply_restriction", "error", "do_nothing",
        "add_reaction_and_do_nothing"
    ]
    content: Optional[str]
    base_response_text: Optional[str]
    url_data: Optional[str]
    gif_data_url: Optional[str]
    latex_data: Optional[str]
    code_data_language: Optional[str]
    code_data_content: Optional[str]
    # user_id_to_restrict: Optional[int] # Will be part of pending_restriction
    # guild_id_for_restriction: Optional[int] # Will be part of pending_restriction
    # restriction_reason: Optional[str] # Will be part of pending_restriction
    log_message: Optional[str]
    triggering_interaction_case: Optional[str] # We'll still use this for context
    pending_restriction: Optional[Dict[str, Any]] # NEW: e.g., {"reason": str, "user_id": int, "guild_id": int, "trigger_case_for_restriction_log": str}

class MessageHandler:
    # ... (__init__, _determine_engagement, _check_channel_restrictions are mostly the same) ...
    def __init__(self, bot_instance: 'AIBot', message: discord.Message):
        self.bot: 'AIBot' = bot_instance
        self.message: discord.Message = message
        self.member: Optional[discord.Member] = None
        if message.guild:
            if isinstance(message.author, discord.Member):
                self.member = message.author
            else:
                self.member = message.guild.get_member(message.author.id)

        self.guild: Optional[discord.Guild] = message.guild
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
        # This method remains the same as provided in the initial files
        content_for_llm = self.message.content
        bot_user_id = self.bot.user.id if self.bot.user else 0
        bot_mention_strings = [f'<@{bot_user_id}>', f'<@!{bot_user_id}>']

        temp_content_for_llm = content_for_llm
        for mention_str in bot_mention_strings:
            temp_content_for_llm = temp_content_for_llm.replace(mention_str, '')
        temp_content_for_llm = re.sub(r'\s+', ' ', temp_content_for_llm).strip()

        should_respond = False
        interaction_case_debug = "No Interaction Triggered"
        is_reply = bool(self.message.reference and self.message.reference.resolved and isinstance(self.message.reference.resolved, discord.Message))
        is_reply_to_bot = False
        is_mention = False
        if self.bot.user:
            is_mention = self.bot.user in self.message.mentions

        if is_reply:
            replied_to_message: discord.Message = self.message.reference.resolved # type: ignore
            if replied_to_message.author == self.bot.user:
                is_reply_to_bot = True
                should_respond = True
                interaction_case_debug = "Reply to Bot"
                content_for_llm = temp_content_for_llm
            elif is_mention and temp_content_for_llm:
                should_respond = True
                interaction_case_debug = "Reply to User + Mention"
                content_for_llm = temp_content_for_llm

        if not should_respond and is_mention:
            if temp_content_for_llm:
                if not is_reply:
                    should_respond = True
                    interaction_case_debug = "Direct Mention"
                    content_for_llm = temp_content_for_llm
            else:
                interaction_case_debug = "Mention Only (No Content)"

        if not should_respond and (random.random() < self.bot.response_chance): # This is the initial random chance to *consider* responding
            current_content_to_consider = temp_content_for_llm if temp_content_for_llm else self.message.content.strip()
            if current_content_to_consider:
                should_respond = True # Further checks (worthiness, delivery chance) happen in process()
                interaction_case_debug = "Random Chance" # Marks it as potentially random
                content_for_llm = current_content_to_consider
            else:
                interaction_case_debug = "Random Chance Attempt on Empty Message"
        
        author_name_for_log = self.member.name if self.member else (self.message.author.name if self.message.author else f"User {self.user_id}")
        logger.debug(f"MessageHandler: Engagement for {author_name_for_log}: should_respond={should_respond}, case='{interaction_case_debug}', content_for_llm='{content_for_llm[:30]}...'")
        return should_respond, content_for_llm, interaction_case_debug, is_reply_to_bot, is_mention

    def _check_channel_restrictions(self) -> Optional[MessageHandlerResult]:
        # This method remains the same
        if not self.member or not self.bot.restricted_user_role_id: return None
        is_currently_restricted_by_role = self.bot.restricted_user_role_id in self.author_role_ids
        if is_currently_restricted_by_role:
            member_name = self.member.name if self.member else "Restricted User"
            channel_name = self.message.channel.name if hasattr(self.message.channel, 'name') else f"Channel {self.channel_id}"
            logger.debug(f"MessageHandler: User {member_name} has restricted role.")
            if self.bot.restricted_channel_id and self.channel_id != self.bot.restricted_channel_id:
                logger.info(f"MessageHandler: Restricted user {member_name} in disallowed channel {channel_name}. Notifying.")
                notification_content = self.bot.restricted_channel_message_user_template.replace("<#{channel_id}>", f"<#{self.bot.restricted_channel_id}>")
                # This notification will be made ephemeral/DM by ListenerCog if Problem 1 was implemented that way.
                # For now, we stick to public channel message as per user's last clarification.
                return MessageHandlerResult(action="notify_restricted_channel", content=notification_content)
        return None

    # MODIFIED _check_message_rate_limit
    async def _check_message_rate_limit(self, interaction_case_debug: str, is_bot_initiated_random: bool = False) -> Optional[Dict[str, Any]]:
        if not self.member or not self.guild: return None
        # Exemption for specific roles OR if the interaction is bot-initiated random (for restriction purposes)
        if self.is_rate_limit_exempt or not self.redis_client or not self.bot.restricted_user_role_id or self.bot.rate_limit_count <= 0:
            return None # No limit to apply or exempt

        # If user already has the restricted role, we don't need to re-evaluate for *message* rate limiting here.
        # Token limits might still be relevant for logging or further actions if we wanted, but _check_token_rate_limit handles tokens.
        if self.bot.restricted_user_role_id in self.author_role_ids:
            logger.debug(f"MessageHandler: User {self.member.name} already restricted. Message rate limit check skipped for new restriction.")
            return None

        msg_rl_key = f"msg_rl:{self.guild.id}:{self.user_id}"
        member_name = self.member.name if self.member else f"User {self.user_id}"
        try:
            # Always record the message event for tracking purposes
            await discord.utils.asyncio.to_thread(self.redis_client.lpush, msg_rl_key, float(self.current_time)) # type: ignore
            min_time_for_window = self.current_time - self.bot.rate_limit_window_seconds
            timestamps_in_list_raw = await discord.utils.asyncio.to_thread(self.redis_client.lrange, msg_rl_key, 0, -1) # type: ignore
            timestamps_in_list_str = [ts.decode('utf-8') if isinstance(ts, bytes) else str(ts) for ts in timestamps_in_list_raw]
            messages_in_window_count = 0
            valid_timestamps_for_new_list = []
            for ts_str in timestamps_in_list_str:
                try:
                    ts = float(ts_str)
                    if ts > min_time_for_window:
                        messages_in_window_count += 1
                        valid_timestamps_for_new_list.append(ts_str)
                except ValueError:
                    logger.warning(f"MessageHandler: Non-float value '{ts_str}' in Redis list {msg_rl_key}")

            # Transaction to trim the list and set expiry
            def transaction_fn(pipe: redis.client.Pipeline): # type: ignore
                pipe.multi()
                pipe.delete(msg_rl_key)
                if valid_timestamps_for_new_list:
                    pipe.rpush(msg_rl_key, *valid_timestamps_for_new_list)
                pipe.expire(msg_rl_key, self.bot.rate_limit_window_seconds + 120) # Add some buffer
                return pipe.execute()
            await discord.utils.asyncio.to_thread(self.redis_client.transaction, transaction_fn, msg_rl_key, value_from_callable=True) # type: ignore

            # If bot initiated this specific interaction randomly, do not trigger a restriction from it.
            if is_bot_initiated_random:
                logger.debug(f"MessageHandler: Message from {member_name} was bot-initiated random. Message recorded, but restriction check bypassed for this interaction.")
                return None # Don't trigger restriction

            if messages_in_window_count > self.bot.rate_limit_count:
                logger.info(f"MessageHandler: User {member_name} ({self.user_id}) EXCEEDED MESSAGE rate limit ({messages_in_window_count}/{self.bot.rate_limit_count}). Trigger: {interaction_case_debug}. Restriction pending.")
                return { # This is now a dictionary to be part of `pending_restriction`
                    "user_id_to_restrict": self.user_id,
                    "guild_id_for_restriction": self.guild.id,
                    "restriction_reason": "Exceeded message rate limit",
                    "trigger_case_for_restriction_log": interaction_case_debug
                }
        except Exception as e:
            logger.error(f"MessageHandler: Error in message rate limit for {member_name}: {e}", exc_info=True)
        return None

    # MODIFIED _check_token_rate_limit
    async def _check_token_rate_limit(self, tokens_used: int, interaction_case_debug: str, is_bot_initiated_random: bool = False) -> Optional[Dict[str, Any]]:
        if not self.member or not self.guild: return None
        # Exemption for specific roles OR if the interaction is bot-initiated random (for restriction purposes)
        if self.is_rate_limit_exempt or not self.redis_client or not self.bot.restricted_user_role_id or self.bot.token_rate_limit_count <= 0 or tokens_used <= 0:
            return None

        member_name = self.member.name if self.member else f"User {self.user_id}"
        is_already_restricted = self.bot.restricted_user_role_id in self.author_role_ids
        if is_already_restricted:
            logger.debug(f"MessageHandler: User {member_name} already restricted. Token usage ({tokens_used}) recorded. Token rate limit check skipped for new restriction.")
            # Still record token usage for already restricted users if needed for other logic/monitoring
            # but we won't apply a *new* restriction based on it here.
            # Fall through to record, but the `is_bot_initiated_random` or `is_already_restricted` will prevent new restriction signal.


        token_rl_key = f"token_rl:{self.guild.id}:{self.user_id}"
        try:
            # Always record the token usage event
            await discord.utils.asyncio.to_thread(self.redis_client.lpush, token_rl_key, f"{self.current_time}:{tokens_used}") # type: ignore
            min_time_token_window = self.current_time - self.bot.rate_limit_window_seconds
            entries_in_list_raw = await discord.utils.asyncio.to_thread(self.redis_client.lrange, token_rl_key, 0, -1) # type: ignore
            entries_in_list = [entry.decode('utf-8') if isinstance(entry, bytes) else str(entry) for entry in entries_in_list_raw]
            total_tokens_in_window = 0
            valid_entries_for_trim = []
            for entry_str in entries_in_list:
                try:
                    ts_str, tk_str = entry_str.split(":", 1)
                    ts = float(ts_str)
                    tk = int(tk_str)
                    if ts > min_time_token_window:
                        total_tokens_in_window += tk
                        valid_entries_for_trim.append(entry_str)
                except (ValueError, IndexError):
                    logger.warning(f"MessageHandler: Malformed entry in {token_rl_key}: {entry_str}")

            # Transaction to trim the list and set expiry
            def transaction_fn_tokens(pipe: redis.client.Pipeline): # type: ignore
                pipe.multi()
                pipe.delete(token_rl_key)
                if valid_entries_for_trim:
                    pipe.rpush(token_rl_key, *valid_entries_for_trim)
                pipe.expire(token_rl_key, self.bot.rate_limit_window_seconds + 120) # Add some buffer
                return pipe.execute()
            await discord.utils.asyncio.to_thread(self.redis_client.transaction, transaction_fn_tokens, token_rl_key, value_from_callable=True) # type: ignore

            if is_already_restricted: # Already handled above, just re-confirming
                return None # Don't trigger new restriction if already restricted

            # If bot initiated this specific interaction randomly, do not trigger a restriction from it.
            if is_bot_initiated_random:
                logger.debug(f"MessageHandler: Token usage from {member_name} was bot-initiated random. Tokens recorded, but restriction check bypassed for this interaction.")
                return None # Don't trigger restriction

            if total_tokens_in_window > self.bot.token_rate_limit_count:
                logger.info(f"MessageHandler: User {member_name} ({self.user_id}) EXCEEDED TOKEN rate limit ({total_tokens_in_window}/{self.bot.token_rate_limit_count}). Trigger: {interaction_case_debug}. Restriction pending.")
                return { # This is now a dictionary to be part of `pending_restriction`
                    "user_id_to_restrict": self.user_id,
                    "guild_id_for_restriction": self.guild.id,
                    "restriction_reason": "Exceeded token usage rate limit",
                    "trigger_case_for_restriction_log": interaction_case_debug
                }
        except Exception as e:
            logger.error(f"MessageHandler: Error in token rate limit for {member_name}: {e}", exc_info=True)
        return None

    async def _save_user_message_score(self, scores_dict: Dict[str, Any]):
        # This method remains the same
        if not self.redis_client:
            logger.debug("Skipping user message score saving: Redis client (general) unavailable.")
            return
        max_messages = getattr(self.bot, 'profile_max_scored_messages', 0)
        if not isinstance(max_messages, int) or max_messages <= 0:
            logger.debug(f"Skipping user message score saving: User profile storage is disabled or misconfigured (max_messages: {max_messages}).")
            return

        redis_key = f"user_profile_messages:{self.user_id}"
        data_to_store = {
            "message_content": self.message.content, # Storing the original message content for context
            "scores": scores_dict,
            "timestamp": self.current_time
        }
        try:
            json_data = json.dumps(data_to_store)
            def score_transaction(pipe: redis.client.Pipeline): # type: ignore
                pipe.multi()
                pipe.lpush(redis_key, json_data)
                pipe.ltrim(redis_key, 0, max_messages - 1) # Ensure 0-based index for ltrim
                return pipe.execute()
            await discord.utils.asyncio.to_thread(self.redis_client.transaction, score_transaction, redis_key, value_from_callable=True) # type: ignore
            logger.debug(f"Saved scored message for user {self.user_id} to Redis list {redis_key}.")
        except Exception as e:
            logger.error(f"Error saving user message score to Redis for user {self.user_id}, key {redis_key}: {e}", exc_info=True)


    def _is_message_worthy(self, message_content: str) -> bool:
        # This method remains the same
        if not message_content:
            return False
        author_name = self.member.name if self.member else f"User {self.user_id}" # type: ignore
        min_length = getattr(self.bot, 'worthiness_min_length', 10)
        if len(message_content) < min_length:
            logger.debug(f"WorthyCheck: Msg from {author_name} too short ({len(message_content)} < {min_length} chars). Not worthy.")
            return False

        if hasattr(self.bot, 'spacy_models') and self.bot.spacy_models:
            lang_code = None
            try:
                if len(message_content) > 5: # Heuristic: very short strings might not detect well
                    lang_code = detect_language_code(message_content)
                    logger.debug(f"WorthyCheck: Detected language '{lang_code}' for message from {author_name}: '{message_content[:30]}...'")
                else:
                    logger.debug(f"WorthyCheck: Message from {author_name} too short for reliable language detection. SpaCy check depends on default/fallback.")
            except LangDetectException:
                logger.warning(f"WorthyCheck: Could not detect language for msg from {author_name}: '{message_content[:50]}...'. SpaCy check depends on default/fallback.")

            nlp_model = None
            if lang_code and lang_code in self.bot.spacy_models:
                nlp_model = self.bot.spacy_models[lang_code]
            elif not lang_code and "en" in self.bot.spacy_models : # Fallback to English if detection fails/not supported
                 logger.debug(f"WorthyCheck: Language detection failed or lang not supported, attempting fallback to English model for {author_name}.")
                 nlp_model = self.bot.spacy_models["en"]


            if nlp_model:
                doc = nlp_model(message_content)
                significant_pos_tags = {"NOUN", "VERB", "ADJ", "PROPN", "ADV"} # Consider "INTJ" (interjection) too?
                min_significant_words = getattr(self.bot, 'worthiness_min_significant_words', 2)
                significant_word_count = sum(1 for token in doc if token.pos_ in significant_pos_tags and not token.is_stop)

                if significant_word_count < min_significant_words:
                    logger.debug(f"WorthyCheck: Msg from {author_name} has {significant_word_count} significant words (min {min_significant_words}). Lang: {lang_code or 'unknown/default'}. Not worthy.")
                    return False
                logger.debug(f"WorthyCheck: Msg from {author_name} passed SpaCy checks ({significant_word_count} sig words). Lang: {lang_code or 'unknown/default'}.")
            else: # No suitable SpaCy model, rely on length alone if it passed that
                logger.debug(f"WorthyCheck: No suitable SpaCy model for '{message_content[:30]}...' (lang: {lang_code}). Length check passed. Considered worthy based on length alone.")
        else: # SpaCy not configured, rely on length
            logger.debug("WorthyCheck: SpaCy models not loaded/configured. Worthiness relies on length check.")
        logger.info(f"Message from {author_name} deemed 'worthy' for random processing: '{message_content[:50]}...'")
        return True

    # MODIFIED _handle_llm_interaction
    async def _handle_llm_interaction(
        self,
        content_for_llm: str,
        interaction_case_debug: str,
        is_reply_to_bot: bool,
        show_typing_for_llm: bool,
        is_bot_initiated_random_interaction: bool # NEW parameter
    ) -> Tuple[MessageHandlerResult, Optional[Dict[str,Any]]]: # Returns main result and pending_token_restriction_info

        if not self.member:
            return MessageHandlerResult(action="error", content="Cannot process request without valid member context."), None

        chat_system_prompt = self.bot.chat_system_prompt
        current_history: List[Dict[str, str]] = self.api_client.get_context_history(self.user_id, self.channel_id)
        extra_assistant_context: Optional[str] = None
        inject_context_for_saving = False
        author_name = self.member.name if self.member else f"User {self.user_id}"

        # Context injection logic (remains the same as original)
        if interaction_case_debug == "Reply to User + Mention" and self.message.reference and self.message.reference.resolved:
            replied_to_msg = self.message.reference.resolved
            if isinstance(replied_to_msg, discord.Message) and replied_to_msg.content:
                replied_author_name = replied_to_msg.author.display_name if replied_to_msg.author else "User"
                context_prefix = f"Context from reply to {replied_author_name} (User ID: {replied_to_msg.author.id}):"
                extra_assistant_context = f"{context_prefix}\n```\n{replied_to_msg.content}\n```"
                inject_context_for_saving = True
        elif is_reply_to_bot and self.message.reference and self.message.reference.resolved:
            replied_to_bot_message = self.message.reference.resolved
            if isinstance(replied_to_bot_message, discord.Message):
                last_bot_msg_content_in_history = None
                for i in range(len(current_history) - 1, -1, -1):
                    if current_history[i]['role'] == 'assistant':
                        try:
                            hist_entry_json = json.loads(current_history[i]['content'])
                            last_bot_msg_content_in_history = hist_entry_json.get("response")
                        except (json.JSONDecodeError, TypeError):
                            last_bot_msg_content_in_history = current_history[i]['content']
                        break
                if replied_to_bot_message.content != last_bot_msg_content_in_history:
                    extra_assistant_context = replied_to_bot_message.content
                    inject_context_for_saving = True
                    if interaction_case_debug == "Reply to Bot": # type: ignore
                        interaction_case_debug += " (Injected external bot context)" # type: ignore

        logger.info(f"MessageHandler: Requesting LLM for {author_name} (Case: {interaction_case_debug}, Typing: {show_typing_for_llm}, BotRandom: {is_bot_initiated_random_interaction})")

        llm_output_dict: Dict[str, Any]
        error_message_from_api: Optional[str]
        tokens_used: Optional[int] = 0

        if show_typing_for_llm:
            async with self.message.channel.typing(): # type: ignore
                llm_output_dict, error_message_from_api, tokens_used = await self.api_client.generate_response(
                    self.user_id, self.channel_id, content_for_llm, chat_system_prompt,
                    history=current_history, extra_assistant_context=extra_assistant_context
                )
        else:
            llm_output_dict, error_message_from_api, tokens_used = await self.api_client.generate_response(
                self.user_id, self.channel_id, content_for_llm, chat_system_prompt,
                history=current_history, extra_assistant_context=extra_assistant_context
            )

        response_text_for_user = llm_output_dict.get("response", "Sorry, I encountered an issue and couldn't formulate a proper response.")
        scores_from_llm = llm_output_dict.get("scores")

        if error_message_from_api:
            logger.error(f"MessageHandler: LLM interaction for {author_name} had issues. Loggable error: '{error_message_from_api}'. User-facing from LLM dict: '{response_text_for_user}'")
            # ... (error handling for LLM response as before) ...
            if any(err_indicator in response_text_for_user for err_indicator in ["AI service error", "timed out", "Connection error", "Error: The AI service response was not valid JSON."]): # type: ignore
                 return MessageHandlerResult(action="error", content=response_text_for_user, log_message=f"API Error for {author_name}: {error_message_from_api}"), None


        if isinstance(scores_from_llm, dict):
            max_profile_messages = getattr(self.bot, 'profile_max_scored_messages', 0)
            if isinstance(max_profile_messages, int) and max_profile_messages > 0:
                 await self._save_user_message_score(scores_from_llm)
        elif scores_from_llm is None:
             logger.info(f"MessageHandler: No valid scores from LLM for {author_name}. Not saving profile data.")


        next_history = list(current_history)
        if inject_context_for_saving and extra_assistant_context:
            next_history.append({"role": "assistant", "content": extra_assistant_context})
        next_history.append({"role": "user", "content": content_for_llm})
        next_history.append({"role": "assistant", "content": json.dumps(llm_output_dict)})
        self.api_client.save_context_history(self.user_id, self.channel_id, next_history)
        logger.debug(f"MessageHandler: Saved history for {author_name}. Assistant response: {json.dumps(llm_output_dict)[:100]}...")

        # Token rate limit check - now considers if it's bot-initiated
        pending_token_restriction_info: Optional[Dict[str, Any]] = None
        if tokens_used is not None and tokens_used > 0:
            # Pass is_bot_initiated_random directly
            token_limit_hit_details = await self._check_token_rate_limit(tokens_used, interaction_case_debug, is_bot_initiated_random=is_bot_initiated_random_interaction)
            if token_limit_hit_details:
                # This means a user-initiated action hit the token limit.
                logger.info(f"MessageHandler: Token limit would be hit for {author_name} by this interaction ({tokens_used} tokens). Restriction info prepared.")
                pending_token_restriction_info = token_limit_hit_details
                # The restriction itself is applied by ListenerCog after the reply.

        # Construct the primary reply action based on LLM output
        llm_reply_action_result: MessageHandlerResult
        llm_type = llm_output_dict.get("type")
        llm_data = llm_output_dict.get("data")

        if llm_type == "gif" and isinstance(llm_data, str):
            llm_reply_action_result = MessageHandlerResult(action="reply_with_gif", base_response_text=response_text_for_user, gif_data_url=llm_data) # type: ignore
        elif llm_type == "url" and isinstance(llm_data, str):
            llm_reply_action_result = MessageHandlerResult(action="reply_with_url", base_response_text=response_text_for_user, url_data=llm_data) # type: ignore
        elif llm_type == "latex" and isinstance(llm_data, str):
            llm_reply_action_result = MessageHandlerResult(action="reply_with_latex", base_response_text=response_text_for_user, latex_data=llm_data) # type: ignore
        elif llm_type == "code" and isinstance(llm_data, dict):
            llm_reply_action_result = MessageHandlerResult(action="reply_with_code", base_response_text=response_text_for_user, code_data_language=llm_data.get("language"), code_data_content=llm_data.get("content")) # type: ignore
        else:
            if llm_type not in ["text", "url", "latex", "code"]: # type: ignore
                logger.warning(f"Unknown LLM response type '{llm_type}' or malformed 'data'. Defaulting to text. Data: {llm_data}")
            llm_reply_action_result = MessageHandlerResult(action="reply_text", content=response_text_for_user) # type: ignore
        
        return llm_reply_action_result, pending_token_restriction_info


    # MODIFIED process method
    async def process(self) -> MessageHandlerResult:
        if not self.message.guild: # type: ignore
            return MessageHandlerResult(action="do_nothing", log_message="Message not from a guild.")
        if not self.member:
            return MessageHandlerResult(action="do_nothing", log_message=f"Author {self.message.author.name} not resolved to member.") # type: ignore

        author_name_for_log = self.member.name

        if self.bot.ignored_role_ids_set and not self.bot.ignored_role_ids_set.isdisjoint(self.author_role_ids):
            return MessageHandlerResult(action="do_nothing", log_message="User has ignored role.")

        should_respond_initially, content_for_llm, interaction_case_debug, is_reply_to_bot, _ = self._determine_engagement()
        
        # Determine if this specific "Random Chance" interaction should proceed to an actual reply AND if it's bot-initiated for rate limit purposes
        is_bot_initiated_random_for_ratelimit = False
        deliver_this_random_reply_publicly = False # For "Random Chance" that results in a public reply
        show_typing_indicator = False

        if interaction_case_debug == "Random Chance":
            is_bot_initiated_random_for_ratelimit = True # Mark this interaction as bot-initiated for rate limit logic
            text_to_check = content_for_llm if content_for_llm else self.message.content.strip() # type: ignore
            
            if not text_to_check or not self._is_message_worthy(text_to_check):
                logger.debug(f"MessageHandler: Random chance msg from {author_name_for_log} not 'worthy' or no text. Ignoring for reply, but might score if config allows.")
                # If we still wanted to score unworthy randoms silently, this logic would need adjustment.
                # For now, if not worthy, it effectively becomes "do_nothing" for reply purposes.
                return MessageHandlerResult(action="do_nothing", log_message=f"Random chance message not worthy or empty: {interaction_case_debug}")

            logger.info(f"MessageHandler: Random chance msg from {author_name_for_log} is 'worthy'. Proceeding with random logic.")
            should_respond_initially = True # Confirmed worthy for processing

            delivery_chance_config = getattr(self.bot, 'random_response_delivery_chance', 0.3)
            if not isinstance(delivery_chance_config, (float, int)): delivery_chance_config = 0.3
            
            if random.random() < delivery_chance_config:
                deliver_this_random_reply_publicly = True
                show_typing_indicator = True # Will deliver, so show typing
                logger.info(f"Random chance (worthy): Pre-calculated decision to DELIVER LLM response to user {author_name_for_log}.")
            else:
                # Will process for scoring/history (if configured) but not send public reply.
                # Typing indicator remains False for this silent processing.
                logger.info(f"Random chance (worthy): Pre-calculated decision to SUPPRESS LLM response delivery to user {author_name_for_log}. Adding reaction.")
                # The _handle_llm_interaction will still be called, but the final action from process() will be "add_reaction_and_do_nothing"
        
        elif should_respond_initially: # Direct mention, reply to bot, etc. (user-initiated)
            if not content_for_llm:
                return MessageHandlerResult(action="do_nothing", log_message="No processable text for LLM in direct engagement.")
            show_typing_indicator = True # Always show typing for direct interactions
            is_bot_initiated_random_for_ratelimit = False # User initiated
        
        if not should_respond_initially: # If neither direct engagement nor passed random checks
            return MessageHandlerResult(action="do_nothing", log_message=f"No engagement: {interaction_case_debug}")

        # --- If we are here, should_respond_initially is True ---

        # Channel restriction check (remains early as it's about *existing* restriction)
        channel_restriction_result = self._check_channel_restrictions()
        if channel_restriction_result:
            return channel_restriction_result # type: ignore

        # Permissions check (remains early)
        if self.guild and not self.message.channel.permissions_for(self.guild.me).send_messages: # type: ignore
             return MessageHandlerResult(action="do_nothing", log_message="Missing send message permission in channel.")

        # --- Core interaction logic ---
        # Call _handle_llm_interaction first to get the bot's intended reply and any token limit issues
        llm_reply_action_result, pending_token_restriction_info = await self._handle_llm_interaction(
            content_for_llm, # type: ignore
            interaction_case_debug,
            is_reply_to_bot,
            show_typing_for_llm=show_typing_indicator,
            is_bot_initiated_random_interaction=is_bot_initiated_random_for_ratelimit # Pass the flag
        )

        # If LLM interaction itself resulted in an error action, return that.
        if llm_reply_action_result.get("action") == "error":
            return llm_reply_action_result

        # Now, check for message rate limit, passing the bot_initiated_random flag
        pending_message_restriction_info = await self._check_message_rate_limit(
            interaction_case_debug,
            is_bot_initiated_random=is_bot_initiated_random_for_ratelimit
        )
        
        # Determine the final action and if any restriction needs to be applied post-reply
        final_result = llm_reply_action_result
        
        # Special handling for "Random Chance" that was processed but not meant for public delivery
        if interaction_case_debug == "Random Chance" and not deliver_this_random_reply_publicly:
            # Even if limits were "hit" by this random processing, we don't apply restriction
            # because the interaction itself was bot-initiated and not for public reply.
            # We also change the action to "add_reaction_and_do_nothing".
            logger.info(f"Random chance (worthy) for {author_name_for_log}: Suppressing LLM reply, action becomes 'add_reaction_and_do_nothing'. No restriction applied from this.")
            final_result = MessageHandlerResult(action="add_reaction_and_do_nothing", content="✅", triggering_interaction_case=interaction_case_debug) # type: ignore
            final_result["log_message"] = f"LLM processed for random chance (no delivery), reaction added. Original LLM action was: {llm_reply_action_result.get('action')}"
            # Ensure no pending restriction is carried over from this suppressed random reply
            if "pending_restriction" in final_result: # type: ignore
                del final_result["pending_restriction"] # type: ignore
            return final_result


        # Consolidate pending restrictions. Prioritize token restriction message if both hit by the same user-initiated message.
        # If it was a bot-initiated random interaction that *was* delivered, pending_token_restriction_info and pending_message_restriction_info will be None.
        final_pending_restriction: Optional[Dict[str, Any]] = None
        if pending_token_restriction_info: # From a user-initiated interaction
            final_pending_restriction = pending_token_restriction_info
        elif pending_message_restriction_info: # From a user-initiated interaction
            final_pending_restriction = pending_message_restriction_info
        
        if final_pending_restriction:
            final_result["pending_restriction"] = final_pending_restriction # Add to the result going to ListenerCog
            final_result["log_message"] = (final_result.get("log_message", "") + # type: ignore
                                          f" Restriction pending: {final_pending_restriction.get('restriction_reason')}.")
        
        final_result["triggering_interaction_case"] = interaction_case_debug # Ensure this is set

        return final_result