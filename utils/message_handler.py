# utils/message_handler.py
import discord
import time
import random
import re
import logging
import json
import redis
from typing import TYPE_CHECKING, Optional, Dict, List, Tuple, Set, Literal, TypedDict, Any

from langdetect import detect as detect_language_code # Keep if used by _is_message_worthy
from langdetect.lang_detect_exception import LangDetectException # Keep if used

if TYPE_CHECKING:
    from bot import AIBot
    from utils.webui_api import WebUIAPI # Already here

logger = logging.getLogger(__name__)

class MessageHandlerResult(TypedDict, total=False):
    action: Literal[
        "reply_text", "reply_with_url", "reply_with_gif", "reply_with_latex", "reply_with_code", "reply_with_output",
        "notify_restricted_channel", "apply_restriction", "error", "do_nothing",
        "add_reaction_and_do_nothing"
    ]
    content: Optional[str] # For text replies, error messages, reaction emojis
    base_response_text: Optional[str] # For multi-part replies (e.g. text + gif)
    url_data: Optional[str]
    gif_data_url: Optional[str]
    latex_data: Optional[str]
    code_data_language: Optional[str]
    code_data_content: Optional[str]
    code_data_output: Optional[str] 
    log_message: Optional[str]
    triggering_interaction_case: Optional[str]
    pending_restriction: Optional[Dict[str, Any]]
    # We don't pass scores through MessageHandlerResult directly, they are handled internally

class MessageHandler:
    def __init__(self, bot_instance: 'AIBot', message: discord.Message):
        self.bot: 'AIBot' = bot_instance
        self.message: discord.Message = message
        self.member: Optional[discord.Member] = None
        if message.guild:
            if isinstance(message.author, discord.Member):
                self.member = message.author
            else:
                # Try to get from cache, though fetch_member in process() is more robust
                self.member = message.guild.get_member(message.author.id) 
                if not self.member:
                    logger.warning(f"MessageHandler init: Could not get Member object for {message.author.id} from cache. Fetch will be attempted in process().")

        self.guild: Optional[discord.Guild] = message.guild
        self.user_id: int = message.author.id
        self.channel_id: int = message.channel.id
        self.current_time: float = time.time()

        self.api_client: 'WebUIAPI' = self.bot.api_client
        self.redis_client: Optional[redis.Redis] = self.bot.redis_client_general # type: ignore

        self.author_role_ids: Set[int] = set()
        if self.member:
             self.author_role_ids = {role.id for role in self.member.roles}

        self.is_super_user: bool = False
        if self.member:
            self.is_super_user = self.bot.super_role_ids_set and \
                                 not self.bot.super_role_ids_set.isdisjoint(self.author_role_ids)
            if self.is_super_user:
                 logger.debug(f"MessageHandler: User {self.member.display_name} ({self.user_id}) is a Super User.")
        elif message.guild :
            logger.debug(f"MessageHandler: No Member object for {self.user_id} in guild {message.guild.id} at init. Cannot determine Super User status by roles yet.")

    # ... (_determine_engagement, _check_channel_restrictions, _check_message_rate_limit, _check_token_rate_limit unchanged) ...
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
            elif is_mention and temp_content_for_llm: # Replying to another user BUT also mentioning the bot
                should_respond = True
                interaction_case_debug = "Reply to User + Mention"
                content_for_llm = temp_content_for_llm
        
        if not should_respond and is_mention: # Not a reply, but a direct mention
            if temp_content_for_llm: # Mention with content
                should_respond = True
                interaction_case_debug = "Direct Mention"
                content_for_llm = temp_content_for_llm
            else: # Mention only, no content (e.g. "@Bot")
                interaction_case_debug = "Mention Only (No Content)"
                # Potentially handle this with a specific "how can I help?" type response later,
                # for now, it might fall through or be caught by should_respond_initially logic

        if not should_respond and (random.random() < self.bot.response_chance):
            # Use temp_content_for_llm if available (bot mention removed), otherwise full content
            current_content_to_consider = temp_content_for_llm if temp_content_for_llm else self.message.content.strip()
            if current_content_to_consider: # Ensure there's some text
                should_respond = True 
                interaction_case_debug = "Random Chance"
                content_for_llm = current_content_to_consider # Use the (potentially stripped) content
            else:
                interaction_case_debug = "Random Chance Attempt on Empty Message" # Log this, but should_respond remains False
        
        author_name_for_log = self.member.display_name if self.member else self.message.author.name # Use author.name if member not resolved
        logger.debug(f"MessageHandler: Engagement for {author_name_for_log}: should_respond={should_respond}, case='{interaction_case_debug}', content_for_llm='{content_for_llm[:30]}...'")
        return should_respond, content_for_llm, interaction_case_debug, is_reply_to_bot, is_mention

    def _check_channel_restrictions(self) -> Optional[MessageHandlerResult]:
        if not self.member or not self.bot.restricted_user_role_id: return None 
        is_currently_restricted_by_role = self.bot.restricted_user_role_id in self.author_role_ids
        if is_currently_restricted_by_role:
            member_name = self.member.display_name
            # Ensure message.channel has a 'name' attribute (it should for Guild channels)
            channel_name = self.message.channel.name if hasattr(self.message.channel, 'name') else f"Channel {self.channel_id}" 
            logger.debug(f"MessageHandler: User {member_name} has restricted role.")
            if self.bot.restricted_channel_id and self.channel_id != self.bot.restricted_channel_id:
                logger.info(f"MessageHandler: Restricted user {member_name} in disallowed channel {channel_name}. Notifying.")
                # Use the bot's configured template string for the notification
                notification_content = self.bot.restricted_channel_message_user_template.replace("<#{channel_id}>", f"<#{self.bot.restricted_channel_id}>")
                return MessageHandlerResult(action="notify_restricted_channel", content=notification_content)
        return None

    async def _check_message_rate_limit(self, interaction_case_debug: str, is_bot_initiated_random: bool = False) -> Optional[Dict[str, Any]]:
        if not self.member or not self.guild: return None # Guard against no member/guild
        member_name = self.member.display_name
        
        # Bypass for super users OR if Redis is down OR if rate limiting is disabled (role_id or count not set)
        if self.is_super_user or not self.redis_client or not self.bot.restricted_user_role_id or self.bot.rate_limit_count <= 0:
            if self.is_super_user:
                logger.debug(f"MessageHandler: User {member_name} is a Super User. Message rate limit check bypassed.")
            return None # No restriction to apply

        # If user already has the restricted role, don't try to re-restrict for messages.
        # Still log their message for rate limit purposes, but don't trigger new restriction logic here.
        if self.bot.restricted_user_role_id in self.author_role_ids:
            logger.debug(f"MessageHandler: User {member_name} already restricted. Message rate limit check skipped for new restriction, but message logged for RL purposes.")
            # We still need to record this message event in Redis for the window.
            # The logic below handles adding to Redis and trimming.
            # However, we will NOT return a restriction dictionary from this check.

        msg_rl_key = f"msg_rl:{self.guild.id}:{self.user_id}"
        try:
            # Record current message timestamp
            await discord.utils.asyncio.to_thread(self.redis_client.lpush, msg_rl_key, float(self.current_time)) # type: ignore
            
            # Calculate window start time
            min_time_for_window = self.current_time - self.bot.rate_limit_window_seconds
            
            # Get all timestamps (potentially including very old ones)
            timestamps_in_list_raw = await discord.utils.asyncio.to_thread(self.redis_client.lrange, msg_rl_key, 0, -1) # type: ignore
            timestamps_in_list_str = [ts.decode('utf-8') if isinstance(ts, bytes) else str(ts) for ts in timestamps_in_list_raw]

            messages_in_window_count = 0
            valid_timestamps_for_new_list = [] # To store timestamps we want to keep

            for ts_str in timestamps_in_list_str:
                try:
                    ts = float(ts_str)
                    if ts > min_time_for_window: # Message is within the current window
                        messages_in_window_count += 1
                        valid_timestamps_for_new_list.append(ts_str) # Keep this timestamp
                except ValueError:
                    logger.warning(f"MessageHandler: Non-float value '{ts_str}' in Redis list {msg_rl_key}")

            # Transaction to trim the list to only valid timestamps and set expiry
            # This keeps the list clean and automatically removes old data.
            def transaction_fn(pipe: redis.client.Pipeline): # type: ignore
                pipe.multi()
                pipe.delete(msg_rl_key) # Delete the old list
                if valid_timestamps_for_new_list:
                    pipe.rpush(msg_rl_key, *valid_timestamps_for_new_list) # Re-add only valid ones
                pipe.expire(msg_rl_key, self.bot.rate_limit_window_seconds + 120) # Expire key after window + buffer
                return pipe.execute()
            await discord.utils.asyncio.to_thread(self.redis_client.transaction, transaction_fn, msg_rl_key, value_from_callable=True) # type: ignore

            # If user is already restricted, we don't trigger a *new* restriction here based on messages.
            if self.bot.restricted_user_role_id in self.author_role_ids:
                return None # Already restricted, no new message-based restriction.

            # If it's a bot-initiated random reply, don't restrict for this message.
            if is_bot_initiated_random:
                logger.debug(f"MessageHandler: Message from {member_name} (bot-random). Message rate limit restriction check bypassed.")
                return None

            # Now check if the count exceeds the limit
            if messages_in_window_count > self.bot.rate_limit_count:
                logger.info(f"MessageHandler: User {member_name} ({self.user_id}) EXCEEDED MESSAGE rate limit ({messages_in_window_count}/{self.bot.rate_limit_count}). Trigger: {interaction_case_debug}. Restriction pending.")
                return { # Restriction details
                    "user_id_to_restrict": self.user_id,
                    "guild_id_for_restriction": self.guild.id,
                    "restriction_reason": "Exceeded message rate limit",
                    "trigger_case_for_restriction_log": interaction_case_debug
                }
        except Exception as e:
            logger.error(f"MessageHandler: Error in message rate limit for {member_name}: {e}", exc_info=True)
        return None # No restriction needed or error occurred

    async def _check_token_rate_limit(self, tokens_used: int, interaction_case_debug: str, is_bot_initiated_random: bool = False) -> Optional[Dict[str, Any]]:
        if not self.member or not self.guild: return None
        member_name = self.member.display_name
        
        # Bypass if super user, redis unavailable, token limiting disabled, or no tokens used
        if self.is_super_user or not self.redis_client or not self.bot.restricted_user_role_id or self.bot.token_rate_limit_count <= 0 or tokens_used <= 0:
            if self.is_super_user:
                logger.debug(f"MessageHandler: User {member_name} is a Super User. Token rate limit check bypassed (Tokens used: {tokens_used}).")
            return None

        is_already_restricted = self.bot.restricted_user_role_id in self.author_role_ids
        if is_already_restricted:
            logger.debug(f"MessageHandler: User {member_name} already restricted. Token usage ({tokens_used}) recorded. Token rate limit check for new restriction skipped.")
            # Still record token usage below, but don't return a new restriction dict.

        token_rl_key = f"token_rl:{self.guild.id}:{self.user_id}"
        try:
            # Record current token usage: "timestamp:token_count"
            await discord.utils.asyncio.to_thread(self.redis_client.lpush, token_rl_key, f"{self.current_time}:{tokens_used}") # type: ignore
            
            min_time_token_window = self.current_time - self.bot.rate_limit_window_seconds
            entries_in_list_raw = await discord.utils.asyncio.to_thread(self.redis_client.lrange, token_rl_key, 0, -1) # type: ignore
            entries_in_list = [entry.decode('utf-8') if isinstance(entry, bytes) else str(entry) for entry in entries_in_list_raw]
            
            total_tokens_in_window = 0
            valid_entries_for_trim = [] # Store "ts:count" strings we want to keep

            for entry_str in entries_in_list:
                try:
                    ts_str, tk_str = entry_str.split(":", 1)
                    ts = float(ts_str); tk = int(tk_str)
                    if ts > min_time_token_window: # Entry is within the current window
                        total_tokens_in_window += tk
                        valid_entries_for_trim.append(entry_str) # Keep this entry
                except (ValueError, IndexError): # Handle malformed entries
                    logger.warning(f"MessageHandler: Malformed entry in {token_rl_key}: {entry_str}")

            # Transaction to trim the list to only valid entries and set expiry
            def transaction_fn_tokens(pipe: redis.client.Pipeline): # type: ignore
                pipe.multi(); pipe.delete(token_rl_key) # Delete old list
                if valid_entries_for_trim: pipe.rpush(token_rl_key, *valid_entries_for_trim) # Re-add valid
                pipe.expire(token_rl_key, self.bot.rate_limit_window_seconds + 120); return pipe.execute() # type: ignore
            await discord.utils.asyncio.to_thread(self.redis_client.transaction, transaction_fn_tokens, token_rl_key, value_from_callable=True) # type: ignore

            # If user is already restricted, we don't trigger a *new* restriction here.
            if is_already_restricted: return None 
            
            # If it's a bot-initiated random reply, don't restrict for these tokens.
            if is_bot_initiated_random:
                logger.debug(f"MessageHandler: Token usage from {member_name} (bot-random). Token rate limit restriction check bypassed.")
                return None

            if total_tokens_in_window > self.bot.token_rate_limit_count:
                logger.info(f"MessageHandler: User {member_name} ({self.user_id}) EXCEEDED TOKEN rate limit ({total_tokens_in_window}/{self.bot.token_rate_limit_count}). Trigger: {interaction_case_debug}. Restriction pending.")
                return { # Restriction details
                    "user_id_to_restrict": self.user_id,
                    "guild_id_for_restriction": self.guild.id,
                    "restriction_reason": "Exceeded token usage rate limit",
                    "trigger_case_for_restriction_log": interaction_case_debug
                }
        except Exception as e:
            logger.error(f"MessageHandler: Error in token rate limit for {member_name}: {e}", exc_info=True)
        return None

    async def _save_user_message_score(self, scores_dict: Optional[Dict[str, Any]]): # Scores can be None
        if not self.redis_client or not scores_dict: return # Exit if no redis or no scores
        
        max_messages = getattr(self.bot, 'profile_max_scored_messages', 0)
        if not isinstance(max_messages, int) or max_messages <= 0: return

        redis_key = f"user_profile_messages:{self.user_id}"
        # Ensure scores_dict is actually a dict here, though type hint says Optional
        if not isinstance(scores_dict, dict):
            logger.warning(f"Attempted to save non-dict scores for user {self.user_id}: {scores_dict}")
            return

        data_to_store = { "message_content": self.message.content, "scores": scores_dict, "timestamp": self.current_time }
        try:
            json_data = json.dumps(data_to_store)
            # Transaction: LPUSH the new data, then LTRIM to keep only the latest max_messages
            def score_transaction(pipe: redis.client.Pipeline): # type: ignore
                pipe.multi(); pipe.lpush(redis_key, json_data); pipe.ltrim(redis_key, 0, max_messages - 1); return pipe.execute() # type: ignore
            await discord.utils.asyncio.to_thread(self.redis_client.transaction, score_transaction, redis_key, value_from_callable=True) # type: ignore
            logger.debug(f"Saved scored message for user {self.user_id} to {redis_key}. (Scores: {scores_dict})")
        except Exception as e: logger.error(f"Error saving user score to Redis for {self.user_id}, key {redis_key}: {e}", exc_info=True)

    def _is_message_worthy(self, message_content: str) -> bool:
        if not message_content: return False
        author_name = self.member.display_name if self.member else f"User {self.user_id}"
        min_length = getattr(self.bot, 'worthiness_min_length', 10) # Default 10 if not set
        if len(message_content) < min_length:
            logger.debug(f"WorthyCheck: Msg from {author_name} too short ({len(message_content)} < {min_length}). Not worthy.")
            return False

        # SpaCy based check (if models are loaded)
        if hasattr(self.bot, 'spacy_models') and self.bot.spacy_models:
            lang_code = None
            try:
                # Only detect language if message is reasonably long enough for detection
                if len(message_content) > 5: # Arbitrary short threshold
                    lang_code = detect_language_code(message_content)
            except LangDetectException: # langdetect can fail on very short/ambiguous text
                pass # lang_code remains None

            nlp_model = None
            if lang_code and lang_code in self.bot.spacy_models:
                nlp_model = self.bot.spacy_models[lang_code]
            elif "en" in self.bot.spacy_models: # Fallback to English if detected lang model not found or lang detection failed
                nlp_model = self.bot.spacy_models["en"]
            # If no models loaded at all, nlp_model will remain None

            if nlp_model:
                doc = nlp_model(message_content)
                # POS tags to consider significant (Nouns, Verbs, Adjectives, Proper Nouns, Adverbs)
                significant_pos_tags = {"NOUN", "VERB", "ADJ", "PROPN", "ADV"}
                min_significant_words = getattr(self.bot, 'worthiness_min_significant_words', 2) # Default 2

                significant_word_count = sum(1 for token in doc if token.pos_ in significant_pos_tags and not token.is_stop)
                
                if significant_word_count < min_significant_words:
                    logger.debug(f"WorthyCheck: Msg from {author_name} has {significant_word_count} sig words (min {min_significant_words}). Not worthy. (Lang: {lang_code or 'unknown/default'})")
                    return False
        # If SpaCy checks passed or were skipped (no models), message is considered worthy if length check passed
        logger.info(f"Message from {author_name} deemed 'worthy' for random processing: '{message_content[:50]}...'")
        return True


    async def _handle_llm_interaction(
        self, content_for_llm: str, interaction_case_debug: str, is_reply_to_bot: bool,
        show_typing_for_llm: bool, is_bot_initiated_random_interaction: bool
    ) -> Tuple[MessageHandlerResult, Optional[Dict[str,Any]]]: # Returns (result_for_cog, pending_token_restriction_info)
        
        if not self.member: # Should have been ensured by process()
            logger.error("MessageHandler: _handle_llm_interaction called without a valid member object.")
            return MessageHandlerResult(action="error", content="Cannot process LLM request without member context (internal error)."), None

        author_name = self.member.display_name
        
        # Determine tools for the conversational LLM call
        current_tools_for_llm: List[str]
        if self.is_super_user:
            standard_tools = self.bot.list_tools
            restricted_tools = self.bot.restricted_list_tools # Assuming this attribute exists on bot
            current_tools_for_llm = list(set(standard_tools + restricted_tools)) 
            logger.debug(f"MessageHandler: User {author_name} is Super User. Using combined tools for conversation: {current_tools_for_llm}")
        else:
            current_tools_for_llm = self.bot.list_tools
            logger.debug(f"MessageHandler: User {author_name} is standard user. Using STANDARD tools for conversation: {current_tools_for_llm}")
        
        if not current_tools_for_llm:
             logger.debug(f"MessageHandler: No conversational tools configured/selected for user {author_name} (Super: {self.is_super_user}).")

        # --- 1. Conversational LLM Call ---
        chat_system_prompt = self.bot.chat_system_prompt # The main personality prompt
        current_history: List[Dict[str, str]] = self.api_client.get_context_history(self.user_id, self.channel_id)
        extra_assistant_context: Optional[str] = None; inject_context_for_saving = False

        if interaction_case_debug == "Reply to User + Mention" and self.message.reference and self.message.reference.resolved:
            replied_to_msg = self.message.reference.resolved
            if isinstance(replied_to_msg, discord.Message) and replied_to_msg.content:
                replied_author_name = replied_to_msg.author.display_name if replied_to_msg.author else "User" # Handle potential webhook/deleted user
                extra_assistant_context = f"Context from reply to {replied_author_name} (ID: {replied_to_msg.author.id}):\n```\n{replied_to_msg.content}\n```"
                inject_context_for_saving = True # This context should be part of the assistant's thought process
        elif is_reply_to_bot and self.message.reference and self.message.reference.resolved:
            replied_to_bot_message = self.message.reference.resolved
            if isinstance(replied_to_bot_message, discord.Message):
                # If the bot's message was simple text, use it. If it was a JSON blob (old history format), try to extract "response"
                try:
                    bot_msg_json = json.loads(replied_to_bot_message.content)
                    extra_assistant_context = bot_msg_json.get("response", replied_to_bot_message.content)
                except json.JSONDecodeError:
                    extra_assistant_context = replied_to_bot_message.content
                inject_context_for_saving = True
                if interaction_case_debug == "Reply to Bot": interaction_case_debug += " (Injected bot context)" 


        logger.info(f"MessageHandler: Requesting LLM CONVERSATION for {author_name} (Case: {interaction_case_debug}, Typing: {show_typing_for_llm}, BotRandom: {is_bot_initiated_random_interaction}, Tools: {current_tools_for_llm})")

        llm_conversation_output_dict: Dict[str, Any]
        error_message_from_convo_api: Optional[str]
        tokens_used_convo: Optional[int] = 0
        
        # Conditional typing indicator for the conversational call
        if show_typing_for_llm:
            async with self.message.channel.typing(): 
                llm_conversation_output_dict, error_message_from_convo_api, tokens_used_convo = await self.api_client.generate_response(
                    self.user_id, self.channel_id, content_for_llm, chat_system_prompt,
                    history=current_history, extra_assistant_context=extra_assistant_context,
                    tools_to_use=current_tools_for_llm
                )
        else: # No typing indicator (e.g., for suppressed random chance processing)
            llm_conversation_output_dict, error_message_from_convo_api, tokens_used_convo = await self.api_client.generate_response(
                self.user_id, self.channel_id, content_for_llm, chat_system_prompt,
                history=current_history, extra_assistant_context=extra_assistant_context,
                tools_to_use=current_tools_for_llm
            )

        response_text_for_user = llm_conversation_output_dict.get("response", "Sorry, I encountered an issue processing your request (conversation part).")

        if error_message_from_convo_api: 
            logger.error(f"MessageHandler: LLM CONVERSATION for {author_name} failed. Loggable Error: '{error_message_from_convo_api}'. User-facing: '{response_text_for_user}'")
            # Decide if we should still try to get sentiment or just return error
            # For now, if conversation fails, we return error for the whole interaction.
            return MessageHandlerResult(action="error", content=response_text_for_user, log_message=f"LLM Conversation API Error for {author_name}: {error_message_from_convo_api}"), None

        # --- 2. Sentiment Analysis LLM Call ---
        scores_from_llm: Optional[Dict[str, Any]] = None
        tokens_used_sentiment: Optional[int] = 0
        error_message_from_sentiment_api: Optional[str] = None

        if self.bot.sentiment_system_prompt and self.bot.profile_max_scored_messages > 0 : # Only call if prompt exists and scoring is enabled
            logger.info(f"MessageHandler: Requesting LLM SENTIMENT for {author_name}'s message: '{content_for_llm[:50]}...'")
            scores_from_llm, error_message_from_sentiment_api, tokens_used_sentiment = await self.api_client.generate_sentiment_scores(
                self.user_id, self.channel_id, content_for_llm, # Pass the original user message content
                self.bot.sentiment_system_prompt
            )
            if error_message_from_sentiment_api:
                logger.error(f"MessageHandler: LLM SENTIMENT for {author_name} failed. Loggable Error: '{error_message_from_sentiment_api}'. Scores will be missing.")
                # Continue with the conversation response even if sentiment fails.
            elif scores_from_llm:
                logger.info(f"MessageHandler: Successfully received scores for {author_name}: {scores_from_llm}")
            else: # No error, but no scores (should ideally not happen if no error)
                 logger.warning(f"MessageHandler: LLM Sentiment call for {author_name} returned no error but also no scores.")
        else:
            logger.debug(f"MessageHandler: Sentiment scoring skipped for {author_name} (no prompt or profile_max_scored_messages <= 0).")


        # --- 3. Process Results & History ---
        if scores_from_llm and isinstance(scores_from_llm, dict): # Ensure it's a dict
            # Save scores if profiling is enabled (profile_max_scored_messages > 0 was already checked for calling)
            await self._save_user_message_score(scores_from_llm)
        elif error_message_from_sentiment_api: # Log if sentiment failed but convo succeeded
            logger.warning(f"MessageHandler: Conversational reply for {author_name} succeeded, but sentiment scoring failed: {error_message_from_sentiment_api}")


        # Update conversation history with the *conversational response only*
        next_history = list(current_history) # Make a copy
        if inject_context_for_saving and extra_assistant_context:
            next_history.append({"role": "assistant", "content": extra_assistant_context}) # This was context for the current turn's LLM call
        next_history.append({"role": "user", "content": content_for_llm})
        # The assistant's response for history should be the JSON string of the *conversational output*
        next_history.append({"role": "assistant", "content": json.dumps(llm_conversation_output_dict)}) 
        self.api_client.save_context_history(self.user_id, self.channel_id, next_history)
        logger.debug(f"MessageHandler: History saved for {author_name}. Assistant JSON (conversation part): {json.dumps(llm_conversation_output_dict)[:100]}...")

        # --- 4. Token Rate Limiting ---
        total_tokens_this_interaction = (tokens_used_convo or 0) + (tokens_used_sentiment or 0)
        pending_token_restriction_info: Optional[Dict[str, Any]] = None
        if total_tokens_this_interaction > 0:
            token_limit_hit_details = await self._check_token_rate_limit(
                total_tokens_this_interaction, 
                interaction_case_debug, 
                is_bot_initiated_random=is_bot_initiated_random_interaction
            )
            if token_limit_hit_details:
                pending_token_restriction_info = token_limit_hit_details
        
        # --- 5. Prepare result for Cog ---
        # The llm_conversation_output_dict now only contains conversation data (type, response, data)
        llm_type = llm_conversation_output_dict.get("type")
        llm_data = llm_conversation_output_dict.get("data")
        
        # Use the response_text_for_user which is derived from llm_conversation_output_dict.get("response")
        action_map = {
            "gif": MessageHandlerResult(action="reply_with_gif", base_response_text=response_text_for_user, gif_data_url=llm_data if isinstance(llm_data, str) else None),
            "url": MessageHandlerResult(action="reply_with_url", base_response_text=response_text_for_user, url_data=llm_data if isinstance(llm_data, str) else None),
            "latex": MessageHandlerResult(action="reply_with_latex", base_response_text=response_text_for_user, latex_data=llm_data if isinstance(llm_data, str) else None),
            "code": MessageHandlerResult(action="reply_with_code", base_response_text=response_text_for_user, code_data_language=llm_data.get("language") if isinstance(llm_data, dict) else None, code_data_content=llm_data.get("content") if isinstance(llm_data, dict) else None),
            "output": MessageHandlerResult(action="reply_with_output", base_response_text=response_text_for_user, code_data_language=llm_data.get("language") if isinstance(llm_data, dict) else None, code_data_content=llm_data.get("content") if isinstance(llm_data, dict) else None, code_data_output=llm_data.get("output") if isinstance(llm_data, dict) else None) # Ensure 'output' matches your JSON spec for this type
        }
        # Default to "reply_text" if type is "text" or unknown
        llm_reply_action_result = action_map.get(llm_type, MessageHandlerResult(action="reply_text", content=response_text_for_user)) # type: ignore
        
        if llm_type not in action_map and llm_type != "text": # Log if type is unknown and not 'text'
            logger.warning(f"Unknown LLM type '{llm_type}' from conversation response. Defaulting to text reply.")
        
        return llm_reply_action_result, pending_token_restriction_info


    async def process(self) -> MessageHandlerResult:
        if not self.message.guild: # Bot doesn't operate in DMs or non-guild contexts
            return MessageHandlerResult(action="do_nothing", log_message="Msg not in guild.")
        
        # Ensure self.member is populated.
        if not self.member: 
            try:
                logger.debug(f"MessageHandler.process: Member object for {self.message.author.name} was None at init, attempting fetch.")
                fetched_member = await self.message.guild.fetch_member(self.message.author.id)
                if fetched_member:
                    self.member = fetched_member
                    # Re-populate role IDs and super_user status after successful fetch
                    self.author_role_ids = {role.id for role in self.member.roles}
                    self.is_super_user = self.bot.super_role_ids_set and \
                                         not self.bot.super_role_ids_set.isdisjoint(self.author_role_ids)
                    if self.is_super_user: logger.debug(f"MessageHandler (process fetch): User {self.member.display_name} confirmed Super User.")
                else: # Should not happen if fetch_member doesn't raise NotFound
                    logger.error(f"MessageHandler.process: fetch_member for {self.message.author.id} returned None unexpectedly. Cannot proceed.")
                    return MessageHandlerResult(action="do_nothing", log_message=f"Author {self.message.author.name} could not be resolved to a member object after fetch.")
            except discord.NotFound:
                logger.error(f"MessageHandler.process: fetch_member failed for {self.message.author.id} (NotFound). Cannot proceed with role-dependent logic.")
                return MessageHandlerResult(action="do_nothing", log_message=f"Author {self.message.author.name} not found in guild after fetch attempt.")
            except discord.HTTPException as e:
                logger.error(f"MessageHandler.process: HTTPException fetching member {self.message.author.id}: {e}. Cannot proceed.")
                return MessageHandlerResult(action="do_nothing", log_message=f"HTTP error fetching member {self.message.author.name}.")
        
        # Now self.member should be valid if we haven't returned.
        author_name_for_log = self.member.display_name if self.member else self.message.author.name # Fallback if member somehow still None

        if self.bot.ignored_role_ids_set and not self.bot.ignored_role_ids_set.isdisjoint(self.author_role_ids):
            return MessageHandlerResult(action="do_nothing", log_message=f"User {author_name_for_log} has ignored role.")

        should_respond_initially, content_for_llm, interaction_case_debug, is_reply_to_bot, _ = self._determine_engagement()
        
        is_bot_initiated_random_for_ratelimit = False; deliver_this_random_reply_publicly = False; show_typing_indicator_for_convo = False

        if interaction_case_debug == "Random Chance":
            is_bot_initiated_random_for_ratelimit = True # Mark for rate limit checks
            text_to_check_worthiness = content_for_llm if content_for_llm else self.message.content.strip() 
            if not text_to_check_worthiness or not self._is_message_worthy(text_to_check_worthiness):
                return MessageHandlerResult(action="do_nothing", log_message=f"Random msg from {author_name_for_log} not worthy.")
            # Message is worthy for random processing
            should_respond_initially = True # We will process it
            if random.random() < getattr(self.bot, 'random_response_delivery_chance', 0.3): # Configurable chance to actually send reply
                deliver_this_random_reply_publicly = True; show_typing_indicator_for_convo = True # Show typing if we send
                logger.info(f"Random chance (worthy) for {author_name_for_log}: DELIVERING LLM response publicly.")
            else: # Process for scoring/profiling, but don't send reply, just react
                deliver_this_random_reply_publicly = False; show_typing_indicator_for_convo = False # No typing if not sending
                logger.info(f"Random chance (worthy) for {author_name_for_log}: Suppressing public delivery, LLM will process (e.g. for scoring). Adding reaction.")
        
        elif should_respond_initially: # Direct mention, reply to bot, etc.
            if not content_for_llm: # e.g. just "@Bot"
                 # Potentially handle with a specific "how can I help?" or similar fixed response.
                 # For now, if no content for LLM, treat as no engagement for LLM call.
                 # Could add a specific MessageHandlerResult action for this.
                 logger.debug(f"Engagement for {author_name_for_log} triggered ({interaction_case_debug}), but no actual content for LLM.")
                 # Let's add a simple reaction if it's just a mention with no content
                 if interaction_case_debug == "Mention Only (No Content)":
                     return MessageHandlerResult(action="add_reaction_and_do_nothing", content="👋", log_message="Bot mentioned with no content, added wave reaction.")
                 return MessageHandlerResult(action="do_nothing", log_message="No text for LLM despite initial engagement trigger.")
            show_typing_indicator_for_convo = True # Show typing for direct interactions
            is_bot_initiated_random_for_ratelimit = False # Not a random interaction for rate limit purposes
        
        if not should_respond_initially: # If no engagement trigger, do nothing
            return MessageHandlerResult(action="do_nothing", log_message=f"No engagement for {author_name_for_log}: {interaction_case_debug}")

        # Channel restriction check (requires self.member to be valid for self.author_role_ids)
        channel_restriction_result = self._check_channel_restrictions()
        if channel_restriction_result: return channel_restriction_result # e.g. notify user to use correct channel

        # Check bot's permissions to send messages in the current channel
        if self.guild and not self.message.channel.permissions_for(self.guild.me).send_messages: 
             return MessageHandlerResult(action="do_nothing", log_message=f"No send_messages permission in channel {self.message.channel.name}.")

        # --- Core LLM Interaction ---
        # _handle_llm_interaction now returns (conversational_reply_result, pending_token_restriction_info)
        llm_based_reply_action_result, pending_token_restriction_info = await self._handle_llm_interaction(
            content_for_llm, 
            interaction_case_debug, 
            is_reply_to_bot, 
            show_typing_indicator_for_convo, # Pass whether to show typing for the convo part
            is_bot_initiated_random_interaction=is_bot_initiated_random_for_ratelimit # Pass if it was bot-random
        )

        # If LLM conversation part itself had a critical error (e.g., API down, bad auth)
        if llm_based_reply_action_result.get("action") == "error":
            return llm_based_reply_action_result # Propagate the error result

        # --- Message Rate Limiting (after LLM calls, as it's about user's message frequency) ---
        # This check is independent of token usage for the current reply.
        pending_message_restriction_info = await self._check_message_rate_limit(
            interaction_case_debug, 
            is_bot_initiated_random=is_bot_initiated_random_for_ratelimit # Pass if it was bot-random
        )
        
        # Determine the final action based on delivery suppression for random chance
        final_result_for_cog = llm_based_reply_action_result
        if interaction_case_debug == "Random Chance" and not deliver_this_random_reply_publicly:
            # We processed LLM (convo + sentiment if applicable), but won't send the convo reply.
            # Instead, we'll send a reaction.
            final_result_for_cog = MessageHandlerResult(
                action="add_reaction_and_do_nothing", 
                content="✅", # Example reaction
                triggering_interaction_case=interaction_case_debug
            ) 
            final_result_for_cog["log_message"] = (f"LLM processed for random chance (scores saved if applicable), "
                                                  f"public reply suppressed. Original intended action: {llm_based_reply_action_result.get('action')}")
            # IMPORTANT: If reply is suppressed, we should NOT apply restrictions triggered by THIS interaction.
            # Clear any pending restrictions if this random reply is suppressed.
            # This prevents restricting a user for an interaction they didn't "see" a reply to.
            pending_message_restriction_info = None
            pending_token_restriction_info = None
            logger.info(f"Suppressed random reply for {author_name_for_log}. Any pending restrictions from this interaction are cleared.")
        

        # Combine pending restrictions (token limit takes precedence if both hit by this single interaction)
        final_pending_restriction: Optional[Dict[str, Any]] = pending_token_restriction_info or pending_message_restriction_info
        
        if final_pending_restriction:
            # Add to the result that will be returned to the cog
            final_result_for_cog["pending_restriction"] = final_pending_restriction
            final_result_for_cog["log_message"] = ( (final_result_for_cog.get("log_message", "") or "") + 
                                                  f" Restriction pending: {final_pending_restriction.get('restriction_reason')}.")
        
        # Ensure triggering_interaction_case is part of the final result for the cog
        if "triggering_interaction_case" not in final_result_for_cog:
            final_result_for_cog["triggering_interaction_case"] = interaction_case_debug
            
        return final_result_for_cog