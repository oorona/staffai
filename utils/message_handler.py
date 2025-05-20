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
    from .webui_api import WebUIAPI 

logger = logging.getLogger(__name__)

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
    user_id_to_restrict: Optional[int]
    guild_id_for_restriction: Optional[int]
    restriction_reason: Optional[str]
    log_message: Optional[str] 
    triggering_interaction_case: Optional[str]

class MessageHandler:
    # __init__, _determine_engagement, _check_channel_restrictions, 
    # _check_message_rate_limit, _check_token_rate_limit, 
    # _save_user_message_score, _is_message_worthy
    # remain IDENTICAL to the last complete version I provided (with corrected indentation for _is_message_worthy).
    # For brevity, I'm omitting them again, but they MUST BE PRESENT.

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
        
        self.api_client: 'WebUIAPI' = self.bot.api_client
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
        
        if not should_respond and (random.random() < self.bot.response_chance):
            current_content_to_consider = temp_content_for_llm if temp_content_for_llm else self.message.content.strip()
            if current_content_to_consider: 
                should_respond = True
                interaction_case_debug = "Random Chance"
                content_for_llm = current_content_to_consider
            else: 
                interaction_case_debug = "Random Chance Attempt on Empty Message"
        
        author_name_for_log = self.member.name if self.member else (self.message.author.name if self.message.author else f"User {self.user_id}")
        logger.debug(f"MessageHandler: Engagement for {author_name_for_log}: should_respond={should_respond}, case='{interaction_case_debug}', content_for_llm='{content_for_llm[:30]}...'")
        return should_respond, content_for_llm, interaction_case_debug, is_reply_to_bot, is_mention

    def _check_channel_restrictions(self) -> Optional[MessageHandlerResult]:
        if not self.member or not self.bot.restricted_user_role_id: return None 
        is_currently_restricted_by_role = self.bot.restricted_user_role_id in self.author_role_ids
        if is_currently_restricted_by_role:
            member_name = self.member.name if self.member else "Restricted User"
            channel_name = self.message.channel.name if hasattr(self.message.channel, 'name') else f"Channel {self.channel_id}"
            logger.debug(f"MessageHandler: User {member_name} has restricted role.")
            if self.bot.restricted_channel_id and self.channel_id != self.bot.restricted_channel_id:
                logger.info(f"MessageHandler: Restricted user {member_name} in disallowed channel {channel_name}. Notifying.") 
                notification_content = self.bot.restricted_channel_message_user_template.replace("<#{channel_id}>", f"<#{self.bot.restricted_channel_id}>")
                return MessageHandlerResult(action="notify_restricted_channel", content=notification_content)
        return None

    async def _check_message_rate_limit(self, interaction_case_debug: str) -> Optional[MessageHandlerResult]:
        if not self.member or not self.guild: return None 
        if self.is_rate_limit_exempt or not self.redis_client or not self.bot.restricted_user_role_id or self.bot.rate_limit_count <= 0:
            return None
        if self.bot.restricted_user_role_id in self.author_role_ids: 
            return None
        msg_rl_key = f"msg_rl:{self.guild.id}:{self.user_id}"
        member_name = self.member.name if self.member else f"User {self.user_id}"
        try:
            await discord.utils.asyncio.to_thread(self.redis_client.lpush, msg_rl_key, float(self.current_time))
            min_time_for_window = self.current_time - self.bot.rate_limit_window_seconds
            timestamps_in_list_raw = await discord.utils.asyncio.to_thread(self.redis_client.lrange, msg_rl_key, 0, -1)
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
            
            def transaction_fn(pipe: redis.client.Pipeline): # type: ignore
                pipe.multi()
                pipe.delete(msg_rl_key)
                if valid_timestamps_for_new_list:
                    pipe.rpush(msg_rl_key, *valid_timestamps_for_new_list)
                pipe.expire(msg_rl_key, self.bot.rate_limit_window_seconds + 120)
                return pipe.execute()
            
            await discord.utils.asyncio.to_thread(self.redis_client.transaction, transaction_fn, msg_rl_key, value_from_callable=True) # type: ignore
            if messages_in_window_count > self.bot.rate_limit_count:
                logger.info(f"MessageHandler: User {member_name} ({self.user_id}) exceeded MESSAGE rate limit ({messages_in_window_count}/{self.bot.rate_limit_count}). Trigger: {interaction_case_debug}")
                return MessageHandlerResult(
                    action="apply_restriction", user_id_to_restrict=self.user_id,
                    guild_id_for_restriction=self.guild.id, restriction_reason="Exceeded message rate limit",
                    triggering_interaction_case=interaction_case_debug)
        except Exception as e:
            logger.error(f"MessageHandler: Error in message rate limit for {member_name}: {e}", exc_info=True)
        return None

    async def _check_token_rate_limit(self, tokens_used: int, interaction_case_debug: str) -> Optional[MessageHandlerResult]:
        if not self.member or not self.guild : return None
        if self.is_rate_limit_exempt or not self.redis_client or not self.bot.restricted_user_role_id or self.bot.token_rate_limit_count <= 0 or tokens_used <= 0:
            return None
        member_name = self.member.name if self.member else f"User {self.user_id}"
        is_already_restricted = self.bot.restricted_user_role_id in self.author_role_ids
        if is_already_restricted:
             logger.debug(f"MessageHandler: User {member_name} already restricted. Tokens for this interaction ({tokens_used}) recorded, but no new token restriction.")
        token_rl_key = f"token_rl:{self.guild.id}:{self.user_id}"
        try:
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
            
            def transaction_fn_tokens(pipe: redis.client.Pipeline): # type: ignore
                pipe.multi()
                pipe.delete(token_rl_key)
                if valid_entries_for_trim:
                    pipe.rpush(token_rl_key, *valid_entries_for_trim)
                pipe.expire(token_rl_key, self.bot.rate_limit_window_seconds + 120)
                return pipe.execute()
            await discord.utils.asyncio.to_thread(self.redis_client.transaction, transaction_fn_tokens, token_rl_key, value_from_callable=True) # type: ignore
            if total_tokens_in_window > self.bot.token_rate_limit_count and not is_already_restricted:
                logger.info(f"MessageHandler: User {member_name} ({self.user_id}) exceeded TOKEN rate limit ({total_tokens_in_window}/{self.bot.token_rate_limit_count}). Trigger: {interaction_case_debug}")
                return MessageHandlerResult(
                    action="apply_restriction", user_id_to_restrict=self.user_id,
                    guild_id_for_restriction=self.guild.id, restriction_reason="Exceeded token usage rate limit",
                    triggering_interaction_case=interaction_case_debug)
        except Exception as e:
            logger.error(f"MessageHandler: Error in token rate limit for {member_name}: {e}", exc_info=True)
        return None

    async def _save_user_message_score(self, scores_dict: Dict[str, Any]):
        if not self.redis_client:
            logger.debug("Skipping user message score saving: Redis client (general) unavailable.")
            return
        max_messages = getattr(self.bot, 'profile_max_scored_messages', 0)
        if not isinstance(max_messages, int) or max_messages <= 0:
            logger.debug(f"Skipping user message score saving: User profile storage is disabled or misconfigured (max_messages: {max_messages}).")
            return

        redis_key = f"user_profile_messages:{self.user_id}"
        data_to_store = {
            "message_content": self.message.content,
            "scores": scores_dict,
            "timestamp": self.current_time 
        }
        try:
            json_data = json.dumps(data_to_store)
            def score_transaction(pipe: redis.client.Pipeline): # type: ignore
                pipe.multi()
                pipe.lpush(redis_key, json_data)
                pipe.ltrim(redis_key, 0, max_messages - 1)
                return pipe.execute()
            await discord.utils.asyncio.to_thread(self.redis_client.transaction, score_transaction, redis_key, value_from_callable=True) # type: ignore
            logger.debug(f"Saved scored message for user {self.user_id} to Redis list {redis_key}.")
        except Exception as e:
            logger.error(f"Error saving user message score to Redis for user {self.user_id}, key {redis_key}: {e}", exc_info=True)

    def _is_message_worthy(self, message_content: str) -> bool:
        if not message_content:
            return False
        author_name = self.member.name if self.member else f"User {self.user_id}"
        min_length = getattr(self.bot, 'worthiness_min_length', 10)
        if len(message_content) < min_length:
            logger.debug(f"WorthyCheck: Msg from {author_name} too short ({len(message_content)} < {min_length} chars). Not worthy.")
            return False

        if hasattr(self.bot, 'spacy_models') and self.bot.spacy_models:
            lang_code = None
            try:
                if len(message_content) > 5: 
                    lang_code = detect_language_code(message_content)
                    logger.debug(f"WorthyCheck: Detected language '{lang_code}' for message from {author_name}: '{message_content[:30]}...'")
                else:
                    logger.debug(f"WorthyCheck: Message from {author_name} too short for reliable language detection. SpaCy check depends on default/fallback.")
            except LangDetectException:
                logger.warning(f"WorthyCheck: Could not detect language for msg from {author_name}: '{message_content[:50]}...'. SpaCy check depends on default/fallback.")
            
            nlp_model = None
            if lang_code and lang_code in self.bot.spacy_models:
                nlp_model = self.bot.spacy_models[lang_code]
            elif not lang_code and "en" in self.bot.spacy_models : 
                 logger.debug(f"WorthyCheck: Language detection failed or lang not supported, attempting fallback to English model for {author_name}.")
                 nlp_model = self.bot.spacy_models["en"]

            if nlp_model:
                doc = nlp_model(message_content)
                significant_pos_tags = {"NOUN", "VERB", "ADJ", "PROPN", "ADV"}
                min_significant_words = getattr(self.bot, 'worthiness_min_significant_words', 2)
                significant_word_count = sum(1 for token in doc if token.pos_ in significant_pos_tags and not token.is_stop)
                if significant_word_count < min_significant_words:
                    logger.debug(f"WorthyCheck: Msg from {author_name} has {significant_word_count} significant words (min {min_significant_words}). Lang: {lang_code or 'unknown/default'}. Not worthy.")
                    return False
                logger.debug(f"WorthyCheck: Msg from {author_name} passed SpaCy checks ({significant_word_count} sig words). Lang: {lang_code or 'unknown/default'}.")
            else:
                logger.debug(f"WorthyCheck: No suitable SpaCy model for '{message_content[:30]}...' (lang: {lang_code}). Length check passed. Considered worthy based on length alone.")
        else: 
            logger.debug("WorthyCheck: SpaCy models not loaded/configured. Worthiness relies on length check.")
        logger.info(f"Message from {author_name} deemed 'worthy' for random processing: '{message_content[:50]}...'")
        return True

    # MODIFIED: Added show_typing_for_llm and deliver_reply_if_random parameters
    async def _handle_llm_interaction(
        self, 
        content_for_llm: str, 
        interaction_case_debug: str, 
        is_reply_to_bot: bool,
        show_typing_for_llm: bool, # NEW: Controls typing indicator for this LLM call
        deliver_reply_if_random: bool # NEW: For "Random Chance", determines if actual reply is sent
    ) -> MessageHandlerResult:

        if not self.member: 
            return MessageHandlerResult(action="error", content="Cannot process request without valid member context.")

        chat_system_prompt = self.bot.chat_system_prompt
        current_history: List[Dict[str, str]] = self.api_client.get_context_history(self.user_id, self.channel_id)
        extra_assistant_context: Optional[str] = None
        inject_context_for_saving = False
        author_name = self.member.name if self.member else f"User {self.user_id}"

        if interaction_case_debug == "Reply to User + Mention" and self.message.reference and self.message.reference.resolved:
            # ... (context injection logic as before)
            replied_to_msg = self.message.reference.resolved
            if isinstance(replied_to_msg, discord.Message) and replied_to_msg.content:
                replied_author_name = replied_to_msg.author.display_name if replied_to_msg.author else "User"
                context_prefix = f"Context from reply to {replied_author_name} (User ID: {replied_to_msg.author.id}):"
                extra_assistant_context = f"{context_prefix}\n```\n{replied_to_msg.content}\n```"
                inject_context_for_saving = True
        elif is_reply_to_bot and self.message.reference and self.message.reference.resolved: 
            # ... (context injection logic as before)
            replied_to_bot_message = self.message.reference.resolved
            if isinstance(replied_to_bot_message, discord.Message):
                last_bot_msg_content_in_history = None
                for i in range(len(current_history) - 1, -1, -1): # Corrected loop
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
                    if interaction_case_debug == "Reply to Bot":
                        interaction_case_debug += " (Injected external bot context)"
        
        logger.info(f"MessageHandler: Requesting LLM for {author_name} (Case: {interaction_case_debug}, Typing: {show_typing_for_llm}, DeliverIfRandom: {deliver_reply_if_random if interaction_case_debug == 'Random Chance' else 'N/A'})")
        
        llm_output_dict: Dict[str, Any] 
        error_message_from_api: Optional[str]
        tokens_used: Optional[int] = 0

        # Conditionally show typing indicator AROUND the LLM call
        if show_typing_for_llm:
            async with self.message.channel.typing():
                logger.debug(f"Typing indicator started for LLM call (User: {author_name})")
                llm_output_dict, error_message_from_api, tokens_used = await self.api_client.generate_response(
                    self.user_id, self.channel_id, content_for_llm, chat_system_prompt,
                    history=current_history, extra_assistant_context=extra_assistant_context
                )
                logger.debug(f"Typing indicator ended for LLM call (User: {author_name})")
        else: # No typing indicator (e.g., silent random chance for scoring)
            logger.debug(f"Calling LLM silently (no typing indicator) for {author_name}")
            llm_output_dict, error_message_from_api, tokens_used = await self.api_client.generate_response(
                self.user_id, self.channel_id, content_for_llm, chat_system_prompt,
                history=current_history, extra_assistant_context=extra_assistant_context
            )
        
        response_text_for_user = llm_output_dict.get("response", "Sorry, I encountered an issue and couldn't formulate a proper response.")
        scores_from_llm = llm_output_dict.get("scores") 

        if error_message_from_api:
            # ... (error handling for LLM response as before) ...
            logger.error(f"MessageHandler: LLM interaction for {author_name} had issues. Loggable error: '{error_message_from_api}'. User-facing from LLM dict: '{response_text_for_user}'")
            if any(err_indicator in response_text_for_user for err_indicator in ["AI service error", "timed out", "Connection error", "Error: The AI service response was not valid JSON."]):
                 return MessageHandlerResult(action="error", content=response_text_for_user, log_message=f"API Error for {author_name}: {error_message_from_api}")

        # Always save scores if valid, regardless of delivery decision (Phase 1 logic)
        if isinstance(scores_from_llm, dict):
            # ... (score saving logic as before) ...
            max_profile_messages = getattr(self.bot, 'profile_max_scored_messages', 0)
            if isinstance(max_profile_messages, int) and max_profile_messages > 0:
                await self._save_user_message_score(scores_from_llm)
        elif scores_from_llm is None:
             logger.info(f"MessageHandler: No valid scores from LLM for {author_name}. Not saving profile data.")


        # Always save history (Phase 1 logic)
        next_history = list(current_history)
        # ... (history saving logic as before) ...
        if inject_context_for_saving and extra_assistant_context:
            next_history.append({"role": "assistant", "content": extra_assistant_context})
        next_history.append({"role": "user", "content": content_for_llm})
        next_history.append({"role": "assistant", "content": json.dumps(llm_output_dict)}) 
        self.api_client.save_context_history(self.user_id, self.channel_id, next_history)
        logger.debug(f"MessageHandler: Saved history for {author_name}. Assistant response: {json.dumps(llm_output_dict)[:100]}...")


        # Token rate limit check (Phase 1 logic)
        if tokens_used is not None and tokens_used > 0:
            # ... (token rate limit logic as before) ...
            token_limit_result = await self._check_token_rate_limit(tokens_used, interaction_case_debug)
            if token_limit_result:
                 logger.info(f"MessageHandler: Token limit hit for {author_name} AFTER LLM. Prioritizing restriction.")
                 log_msg = (f"Token rate limit hit for {author_name} ({tokens_used} tokens). Trigger: {interaction_case_debug}.")
                 token_limit_result["log_message"] = log_msg
                 return token_limit_result
        
        # Determine final action
        # For "Random Chance", use the 'deliver_reply_if_random' flag passed into this method
        if interaction_case_debug == "Random Chance":
            if not deliver_reply_if_random:
                logger.info(f"Random chance (worthy): LLM response processed for {author_name}, but suppressing delivery based on pre-calculated chance. Adding reaction.")
                return MessageHandlerResult(action="add_reaction_and_do_nothing", content="✅") 
            # If deliver_reply_if_random is True, proceed to construct the actual reply action below
            logger.info(f"Random chance (worthy): Pre-calculated decision to deliver LLM response to user {author_name}.")

        # Construct the actual reply action (Phase 2 logic)
        llm_type = llm_output_dict.get("type")
        llm_data = llm_output_dict.get("data") 
        
        if llm_type == "gif" and isinstance(llm_data, str): # llm_data is the GIF URL
            return MessageHandlerResult(action="reply_with_gif", base_response_text=response_text_for_user, gif_data_url=llm_data)
        elif llm_type == "url" and isinstance(llm_data, str):
            return MessageHandlerResult(action="reply_with_url", base_response_text=response_text_for_user, url_data=llm_data)
        elif llm_type == "latex" and isinstance(llm_data, str):
            return MessageHandlerResult(action="reply_with_latex", base_response_text=response_text_for_user, latex_data=llm_data)
        elif llm_type == "code" and isinstance(llm_data, dict):
            return MessageHandlerResult(action="reply_with_code", base_response_text=response_text_for_user, code_data_language=llm_data.get("language"), code_data_content=llm_data.get("content"))
        else: 
            if llm_type not in ["text", "url", "latex", "code"]:
                logger.warning(f"Unknown LLM response type '{llm_type}' or malformed 'data'. Defaulting to text. Data: {llm_data}")
            return MessageHandlerResult(action="reply_text", content=response_text_for_user)

    async def process(self) -> MessageHandlerResult:
        if not self.message.guild:
            logger.warning("MessageHandler: Message is not from a guild. Skipping.")
            return MessageHandlerResult(action="do_nothing", log_message="Message not from a guild.")
        if not self.member:
            logger.warning(f"MessageHandler: Author {self.message.author.name} ({self.user_id}) could not be resolved to a guild member. Skipping.")
            return MessageHandlerResult(action="do_nothing", log_message=f"Author {self.message.author.name} not resolved to member.")

        author_name_for_log = self.member.name

        if self.bot.ignored_role_ids_set and not self.bot.ignored_role_ids_set.isdisjoint(self.author_role_ids):
            logger.debug(f"MessageHandler: Message from {author_name_for_log} ignored (ignored role).")
            return MessageHandlerResult(action="do_nothing", log_message="User has ignored role.")

        should_respond_initially, content_for_llm, interaction_case_debug, is_reply_to_bot, _ = self._determine_engagement()
        
        show_typing_for_llm_call = False # Default
        deliver_this_random_reply = False # Only relevant for random chance

        if interaction_case_debug == "Random Chance":
            text_to_check = content_for_llm if content_for_llm else self.message.content.strip()
            if not text_to_check or not self._is_message_worthy(text_to_check):
                logger.debug(f"MessageHandler: Random chance msg from {author_name_for_log} not 'worthy' or no text. Ignoring.")
                return MessageHandlerResult(action="do_nothing", log_message=f"Random chance message not worthy or empty: {interaction_case_debug}")
            
            logger.info(f"MessageHandler: Random chance msg from {author_name_for_log} is 'worthy'. Proceeding.")
            should_respond_initially = True # Confirm response path
            
            # Pre-calculate delivery chance for this worthy random message
            delivery_chance_config = getattr(self.bot, 'random_response_delivery_chance', 0.3)
            if not isinstance(delivery_chance_config, (float, int)): delivery_chance_config = 0.3
            deliver_this_random_reply = random.random() < delivery_chance_config
            
            if deliver_this_random_reply:
                show_typing_for_llm_call = True # Will deliver, so show typing during LLM
            # If not deliver_this_random_reply, show_typing_for_llm_call remains False (silent LLM call for scoring)
        
        elif should_respond_initially: # Direct mention, reply to bot, etc.
            if not content_for_llm:
                logger.info(f"MessageHandler: Direct engagement ({interaction_case_debug}) for {author_name_for_log} but no processable text. Ignoring.")
                return MessageHandlerResult(action="do_nothing", log_message="No processable text for LLM in direct engagement.")
            show_typing_for_llm_call = True # Always show typing for direct interactions that proceed to LLM
            # deliver_this_random_reply is not used for non-random cases, but _handle_llm_interaction needs it.
            # We can pass False, or handle it inside _handle_llm_interaction to ignore it if not Random Chance.
            # For clarity, _handle_llm_interaction will only use deliver_this_random_reply IF interaction_case_debug is "Random Chance".
        
        if not should_respond_initially:
            logger.debug(f"MessageHandler: No engagement for {author_name_for_log}. Final case: {interaction_case_debug}.")
            return MessageHandlerResult(action="do_nothing", log_message=f"No engagement: {interaction_case_debug}")

        # If we are here, should_respond_initially is True.
        channel_restriction_result = self._check_channel_restrictions()
        if channel_restriction_result:
            return channel_restriction_result

        message_rate_limit_result = await self._check_message_rate_limit(interaction_case_debug)
        if message_rate_limit_result:
            return message_rate_limit_result

        if self.guild and not self.message.channel.permissions_for(self.guild.me).send_messages:
             logger.error(f"MessageHandler: Cannot respond in {self.message.channel.name}: Missing 'Send Messages' permission.") 
             return MessageHandlerResult(action="do_nothing", log_message="Missing send message permission in channel.")

        # Pass the typing decision and delivery decision for randoms to _handle_llm_interaction
        return await self._handle_llm_interaction(
            content_for_llm, # Should not be None if should_respond_initially is True and content check passed
            interaction_case_debug, 
            is_reply_to_bot,
            show_typing_for_llm=show_typing_for_llm_call,
            deliver_reply_if_random=deliver_this_random_reply # Only used if interaction_case_debug is "Random Chance"
        )