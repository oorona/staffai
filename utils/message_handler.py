# utils/message_handler.py
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
    from utils.webui_api import WebUIAPI 

logger = logging.getLogger(__name__)

class MessageHandlerResult(TypedDict, total=False):
    action: Literal[
        "reply_text", "reply_with_url", "reply_with_gif", "reply_with_latex", "reply_with_code", "reply_with_output",
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
    code_data_output: Optional[str] 
    log_message: Optional[str]
    triggering_interaction_case: Optional[str]
    pending_restriction: Optional[Dict[str, Any]]

class MessageHandler:
    def __init__(self, bot_instance: 'AIBot', message: discord.Message):
        self.bot: 'AIBot' = bot_instance
        self.message: discord.Message = message
        self.member: Optional[discord.Member] = None
        if message.guild:
            if isinstance(message.author, discord.Member):
                self.member = message.author
            else:
                # Relies on cache; ListenerCog should ensure member is fetched if not available
                self.member = message.guild.get_member(message.author.id) 
                if not self.member:
                    logger.warning(f"MessageHandler init: Could not get Member object for {message.author.id} from cache. Some features (like role checks) might be affected until member is fetched.")


        self.guild: Optional[discord.Guild] = message.guild
        self.user_id: int = message.author.id
        self.channel_id: int = message.channel.id
        self.current_time: float = time.time()

        self.api_client: 'WebUIAPI' = self.bot.api_client
        self.redis_client: Optional[redis.Redis] = self.bot.redis_client_general # type: ignore

        self.author_role_ids: Set[int] = set()
        if self.member: # Only try to get roles if member object exists
             self.author_role_ids = {role.id for role in self.member.roles}

        self.is_super_user: bool = False
        if self.member: # Check member again before accessing roles
            self.is_super_user = self.bot.super_role_ids_set and \
                                 not self.bot.super_role_ids_set.isdisjoint(self.author_role_ids)
            if self.is_super_user:
                 logger.debug(f"MessageHandler: User {self.member.display_name} ({self.user_id}) is a Super User.")
        elif message.guild : # If no member object, cannot determine super_user status based on roles
            logger.debug(f"MessageHandler: No Member object for {self.user_id} in guild {message.guild.id}. Cannot determine Super User status by roles.")


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
        
        author_name = self.member.display_name if self.member else self.message.author.name
        logger.debug(f"MessageHandler: Engagement for {author_name}: should_respond={should_respond}, case='{interaction_case_debug}', content_for_llm='{content_for_llm[:30]}...'")
        return should_respond, content_for_llm, interaction_case_debug, is_reply_to_bot, is_mention

    def _check_channel_restrictions(self) -> Optional[MessageHandlerResult]:
        if not self.member or not self.bot.restricted_user_role_id: return None # Requires member for role check
        is_currently_restricted_by_role = self.bot.restricted_user_role_id in self.author_role_ids
        if is_currently_restricted_by_role:
            member_name = self.member.display_name
            channel_name = self.message.channel.name if hasattr(self.message.channel, 'name') else f"Channel {self.channel_id}" 
            logger.debug(f"MessageHandler: User {member_name} has restricted role.")
            if self.bot.restricted_channel_id and self.channel_id != self.bot.restricted_channel_id:
                logger.info(f"MessageHandler: Restricted user {member_name} in disallowed channel {channel_name}. Notifying.")
                notification_content = self.bot.restricted_channel_message_user_template.replace("<#{channel_id}>", f"<#{self.bot.restricted_channel_id}>")
                return MessageHandlerResult(action="notify_restricted_channel", content=notification_content)
        return None

    async def _check_message_rate_limit(self, interaction_case_debug: str, is_bot_initiated_random: bool = False) -> Optional[Dict[str, Any]]:
        if not self.member or not self.guild: return None 
        member_name = self.member.display_name
        
        if self.is_super_user or not self.redis_client or not self.bot.restricted_user_role_id or self.bot.rate_limit_count <= 0:
            if self.is_super_user:
                logger.debug(f"MessageHandler: User {member_name} is a Super User. Message rate limit check bypassed.")
            return None 

        if self.bot.restricted_user_role_id in self.author_role_ids:
            logger.debug(f"MessageHandler: User {member_name} already restricted. Message rate limit check skipped.")
            return None

        msg_rl_key = f"msg_rl:{self.guild.id}:{self.user_id}"
        try:
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

            def transaction_fn(pipe: redis.client.Pipeline): 
                pipe.multi()
                pipe.delete(msg_rl_key)
                if valid_timestamps_for_new_list:
                    pipe.rpush(msg_rl_key, *valid_timestamps_for_new_list)
                pipe.expire(msg_rl_key, self.bot.rate_limit_window_seconds + 120)
                return pipe.execute()
            await discord.utils.asyncio.to_thread(self.redis_client.transaction, transaction_fn, msg_rl_key, value_from_callable=True) # type: ignore

            if is_bot_initiated_random:
                logger.debug(f"MessageHandler: Message from {member_name} (bot-random). Message rate limit restriction check bypassed.")
                return None

            if messages_in_window_count > self.bot.rate_limit_count:
                logger.info(f"MessageHandler: User {member_name} ({self.user_id}) EXCEEDED MESSAGE rate limit ({messages_in_window_count}/{self.bot.rate_limit_count}). Trigger: {interaction_case_debug}. Restriction pending.")
                return { "user_id_to_restrict": self.user_id, "guild_id_for_restriction": self.guild.id, "restriction_reason": "Exceeded message rate limit", "trigger_case_for_restriction_log": interaction_case_debug }
        except Exception as e:
            logger.error(f"MessageHandler: Error in message rate limit for {member_name}: {e}", exc_info=True)
        return None

    async def _check_token_rate_limit(self, tokens_used: int, interaction_case_debug: str, is_bot_initiated_random: bool = False) -> Optional[Dict[str, Any]]:
        if not self.member or not self.guild: return None
        member_name = self.member.display_name
        
        if self.is_super_user or not self.redis_client or not self.bot.restricted_user_role_id or self.bot.token_rate_limit_count <= 0 or tokens_used <= 0:
            if self.is_super_user:
                logger.debug(f"MessageHandler: User {member_name} is a Super User. Token rate limit check bypassed (Tokens used: {tokens_used}).")
            return None

        is_already_restricted = self.bot.restricted_user_role_id in self.author_role_ids
        if is_already_restricted:
            logger.debug(f"MessageHandler: User {member_name} already restricted. Token usage ({tokens_used}) recorded. Token rate limit check for new restriction skipped.")

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
                    ts = float(ts_str); tk = int(tk_str)
                    if ts > min_time_token_window:
                        total_tokens_in_window += tk
                        valid_entries_for_trim.append(entry_str)
                except (ValueError, IndexError): logger.warning(f"MessageHandler: Malformed entry in {token_rl_key}: {entry_str}")

            def transaction_fn_tokens(pipe: redis.client.Pipeline): 
                pipe.multi(); pipe.delete(token_rl_key)
                if valid_entries_for_trim: pipe.rpush(token_rl_key, *valid_entries_for_trim)
                pipe.expire(token_rl_key, self.bot.rate_limit_window_seconds + 120); return pipe.execute()
            await discord.utils.asyncio.to_thread(self.redis_client.transaction, transaction_fn_tokens, token_rl_key, value_from_callable=True) # type: ignore

            if is_already_restricted: return None 
            if is_bot_initiated_random:
                logger.debug(f"MessageHandler: Token usage from {member_name} (bot-random). Token rate limit restriction check bypassed.")
                return None

            if total_tokens_in_window > self.bot.token_rate_limit_count:
                logger.info(f"MessageHandler: User {member_name} ({self.user_id}) EXCEEDED TOKEN rate limit ({total_tokens_in_window}/{self.bot.token_rate_limit_count}). Trigger: {interaction_case_debug}. Restriction pending.")
                return { "user_id_to_restrict": self.user_id, "guild_id_for_restriction": self.guild.id, "restriction_reason": "Exceeded token usage rate limit", "trigger_case_for_restriction_log": interaction_case_debug }
        except Exception as e:
            logger.error(f"MessageHandler: Error in token rate limit for {member_name}: {e}", exc_info=True)
        return None

    async def _save_user_message_score(self, scores_dict: Dict[str, Any]):
        if not self.redis_client: return
        max_messages = getattr(self.bot, 'profile_max_scored_messages', 0)
        if not isinstance(max_messages, int) or max_messages <= 0: return

        redis_key = f"user_profile_messages:{self.user_id}"
        data_to_store = { "message_content": self.message.content, "scores": scores_dict, "timestamp": self.current_time }
        try:
            json_data = json.dumps(data_to_store)
            def score_transaction(pipe: redis.client.Pipeline): 
                pipe.multi(); pipe.lpush(redis_key, json_data); pipe.ltrim(redis_key, 0, max_messages - 1); return pipe.execute()
            await discord.utils.asyncio.to_thread(self.redis_client.transaction, score_transaction, redis_key, value_from_callable=True) # type: ignore
            logger.debug(f"Saved scored message for user {self.user_id} to {redis_key}.")
        except Exception as e: logger.error(f"Error saving user score to Redis for {self.user_id}, key {redis_key}: {e}", exc_info=True)

    def _is_message_worthy(self, message_content: str) -> bool:
        if not message_content: return False
        author_name = self.member.display_name if self.member else f"User {self.user_id}"
        min_length = getattr(self.bot, 'worthiness_min_length', 10)
        if len(message_content) < min_length:
            logger.debug(f"WorthyCheck: Msg from {author_name} too short ({len(message_content)} < {min_length}). Not worthy.")
            return False

        if hasattr(self.bot, 'spacy_models') and self.bot.spacy_models:
            lang_code = None
            try:
                if len(message_content) > 5: lang_code = detect_language_code(message_content)
            except LangDetectException: pass

            nlp_model = None
            if lang_code and lang_code in self.bot.spacy_models: nlp_model = self.bot.spacy_models[lang_code]
            elif "en" in self.bot.spacy_models: nlp_model = self.bot.spacy_models["en"]

            if nlp_model:
                doc = nlp_model(message_content)
                significant_pos_tags = {"NOUN", "VERB", "ADJ", "PROPN", "ADV"}
                min_significant_words = getattr(self.bot, 'worthiness_min_significant_words', 2)
                significant_word_count = sum(1 for token in doc if token.pos_ in significant_pos_tags and not token.is_stop)
                if significant_word_count < min_significant_words:
                    logger.debug(f"WorthyCheck: Msg from {author_name} has {significant_word_count} sig words (min {min_significant_words}). Not worthy.")
                    return False
        logger.info(f"Message from {author_name} deemed 'worthy' for random processing: '{message_content[:50]}...'")
        return True

    async def _handle_llm_interaction(
        self, content_for_llm: str, interaction_case_debug: str, is_reply_to_bot: bool,
        show_typing_for_llm: bool, is_bot_initiated_random_interaction: bool
    ) -> Tuple[MessageHandlerResult, Optional[Dict[str,Any]]]:
        if not self.member: # Should be guaranteed by process() if fetch is successful
            return MessageHandlerResult(action="error", content="Cannot process LLM request without member context."), None

        author_name = self.member.display_name
        
        current_tools_for_llm: List[str]
        if self.is_super_user:
            # Combine standard tools and restricted tools for super users, avoiding duplicates
            standard_tools = self.bot.list_tools
            restricted_tools = self.bot.restricted_list_tools
            current_tools_for_llm = list(set(standard_tools + restricted_tools)) # Use set to remove duplicates
            logger.debug(f"MessageHandler: User {author_name} is Super User. Using combined tools: {current_tools_for_llm}")
        else:
            current_tools_for_llm = self.bot.list_tools
            logger.debug(f"MessageHandler: User {author_name} is standard user. Using STANDARD tools: {current_tools_for_llm}")
        
        if not current_tools_for_llm: # Handles if both lists were empty.
             logger.debug(f"MessageHandler: No tools configured/selected for user {author_name} (Super: {self.is_super_user}).")

        chat_system_prompt = self.bot.chat_system_prompt
        current_history: List[Dict[str, str]] = self.api_client.get_context_history(self.user_id, self.channel_id)
        extra_assistant_context: Optional[str] = None; inject_context_for_saving = False

        if interaction_case_debug == "Reply to User + Mention" and self.message.reference and self.message.reference.resolved:
            replied_to_msg = self.message.reference.resolved
            if isinstance(replied_to_msg, discord.Message) and replied_to_msg.content:
                replied_author_name = replied_to_msg.author.display_name if replied_to_msg.author else "User"
                extra_assistant_context = f"Context from reply to {replied_author_name} (ID: {replied_to_msg.author.id}):\n```\n{replied_to_msg.content}\n```"
                inject_context_for_saving = True
        elif is_reply_to_bot and self.message.reference and self.message.reference.resolved:
            replied_to_bot_message = self.message.reference.resolved
            if isinstance(replied_to_bot_message, discord.Message):
                extra_assistant_context = replied_to_bot_message.content
                inject_context_for_saving = True
                if interaction_case_debug == "Reply to Bot": interaction_case_debug += " (Injected bot context)" 

        logger.info(f"MessageHandler: Requesting LLM for {author_name} (Case: {interaction_case_debug}, Typing: {show_typing_for_llm}, BotRandom: {is_bot_initiated_random_interaction}, Tools: {current_tools_for_llm})")

        llm_output_dict: Dict[str, Any]; error_message_from_api: Optional[str]; tokens_used: Optional[int] = 0
        pending_token_restriction_info: Optional[Dict[str, Any]] = None

        if show_typing_for_llm:
            async with self.message.channel.typing(): 
                llm_output_dict, error_message_from_api, tokens_used = await self.api_client.generate_response(
                    self.user_id, self.channel_id, content_for_llm, chat_system_prompt,
                    history=current_history, extra_assistant_context=extra_assistant_context,
                    tools_to_use=current_tools_for_llm
                )
        else:
            llm_output_dict, error_message_from_api, tokens_used = await self.api_client.generate_response(
                self.user_id, self.channel_id, content_for_llm, chat_system_prompt,
                history=current_history, extra_assistant_context=extra_assistant_context,
                tools_to_use=current_tools_for_llm
            )

        response_text_for_user = llm_output_dict.get("response", "Sorry, I encountered an issue processing your request.")
        scores_from_llm = llm_output_dict.get("scores")

        if error_message_from_api: 
            logger.error(f"MessageHandler: LLM for {author_name} failed. Loggable Error: '{error_message_from_api}'. User-facing: '{response_text_for_user}'")
            return MessageHandlerResult(action="error", content=response_text_for_user, log_message=f"LLM API Error for {author_name}: {error_message_from_api}"), None

        if isinstance(scores_from_llm, dict):
            if getattr(self.bot, 'profile_max_scored_messages', 0) > 0:
                 await self._save_user_message_score(scores_from_llm)
        elif scores_from_llm is None:
             logger.info(f"MessageHandler: No valid 'scores' from LLM for {author_name}.")

        next_history = list(current_history)
        if inject_context_for_saving and extra_assistant_context: next_history.append({"role": "assistant", "content": extra_assistant_context})
        next_history.append({"role": "user", "content": content_for_llm})
        next_history.append({"role": "assistant", "content": json.dumps(llm_output_dict)})
        self.api_client.save_context_history(self.user_id, self.channel_id, next_history)
        logger.debug(f"MessageHandler: History saved for {author_name}. Assistant JSON: {json.dumps(llm_output_dict)[:100]}...")

        if tokens_used is not None and tokens_used > 0:
            token_limit_hit_details = await self._check_token_rate_limit(tokens_used, interaction_case_debug, is_bot_initiated_random=is_bot_initiated_random_interaction)
            if token_limit_hit_details: pending_token_restriction_info = token_limit_hit_details

        llm_type = llm_output_dict.get("type"); llm_data = llm_output_dict.get("data")
        action_map = {
            "gif": MessageHandlerResult(action="reply_with_gif", base_response_text=response_text_for_user, gif_data_url=llm_data if isinstance(llm_data, str) else None),
            "url": MessageHandlerResult(action="reply_with_url", base_response_text=response_text_for_user, url_data=llm_data if isinstance(llm_data, str) else None),
            "latex": MessageHandlerResult(action="reply_with_latex", base_response_text=response_text_for_user, latex_data=llm_data if isinstance(llm_data, str) else None),
            "code": MessageHandlerResult(action="reply_with_code", base_response_text=response_text_for_user, code_data_language=llm_data.get("language") if isinstance(llm_data, dict) else None, code_data_content=llm_data.get("content") if isinstance(llm_data, dict) else None),
            "output": MessageHandlerResult(action="reply_with_output", base_response_text=response_text_for_user, code_data_language=llm_data.get("language") if isinstance(llm_data, dict) else None, code_data_content=llm_data.get("content") if isinstance(llm_data, dict) else None, code_data_output=llm_data.get("output") if isinstance(llm_data, dict) else None)
        }
        llm_reply_action_result = action_map.get(llm_type, MessageHandlerResult(action="reply_text", content=response_text_for_user)) 
        if llm_type not in action_map and llm_type != "text": logger.warning(f"Unknown LLM type '{llm_type}'. Defaulting to text.")
        
        return llm_reply_action_result, pending_token_restriction_info

    async def process(self) -> MessageHandlerResult:
        if not self.message.guild: return MessageHandlerResult(action="do_nothing", log_message="Msg not in guild.")
        
        if not self.member: 
            try:
                logger.debug(f"MessageHandler.process: Member object for {self.message.author.name} was None (likely cache miss), attempting fetch.")
                self.member = await self.message.guild.fetch_member(self.message.author.id)
                self.author_role_ids = {role.id for role in self.member.roles}
                # Re-evaluate is_super_user after fetching member
                self.is_super_user = self.bot.super_role_ids_set and \
                                     not self.bot.super_role_ids_set.isdisjoint(self.author_role_ids)
                if self.is_super_user: logger.debug(f"MessageHandler (process fetch): User {self.member.display_name} confirmed Super User.")
            except discord.NotFound:
                logger.error(f"MessageHandler.process: fetch_member failed for {self.message.author.id}. Cannot proceed with role-dependent logic.")
                return MessageHandlerResult(action="do_nothing", log_message=f"Author {self.message.author.name} not found in guild after fetch attempt.")
            except discord.HTTPException as e:
                logger.error(f"MessageHandler.process: HTTPException fetching member {self.message.author.id}: {e}. Cannot proceed.")
                return MessageHandlerResult(action="do_nothing", log_message=f"HTTP error fetching member {self.message.author.name}.")
        
        author_name_for_log = self.member.display_name

        if self.bot.ignored_role_ids_set and not self.bot.ignored_role_ids_set.isdisjoint(self.author_role_ids):
            return MessageHandlerResult(action="do_nothing", log_message=f"User {author_name_for_log} has ignored role.")

        should_respond_initially, content_for_llm, interaction_case_debug, is_reply_to_bot, _ = self._determine_engagement()
        
        is_bot_initiated_random_for_ratelimit = False; deliver_this_random_reply_publicly = False; show_typing_indicator = False

        if interaction_case_debug == "Random Chance":
            is_bot_initiated_random_for_ratelimit = True
            text_to_check = content_for_llm if content_for_llm else self.message.content.strip() 
            if not text_to_check or not self._is_message_worthy(text_to_check):
                return MessageHandlerResult(action="do_nothing", log_message=f"Random msg from {author_name_for_log} not worthy.")
            should_respond_initially = True
            if random.random() < getattr(self.bot, 'random_response_delivery_chance', 0.3):
                deliver_this_random_reply_publicly = True; show_typing_indicator = True
                logger.info(f"Random chance (worthy) for {author_name_for_log}: DELIVERING LLM response.")
            else: logger.info(f"Random chance (worthy) for {author_name_for_log}: Suppressing delivery, adding reaction.")
        
        elif should_respond_initially:
            if not content_for_llm: return MessageHandlerResult(action="do_nothing", log_message="No text for LLM.")
            show_typing_indicator = True; is_bot_initiated_random_for_ratelimit = False
        
        if not should_respond_initially: return MessageHandlerResult(action="do_nothing", log_message=f"No engagement for {author_name_for_log}: {interaction_case_debug}")

        # Channel restriction check requires self.member to be valid for self.author_role_ids
        channel_restriction_result = self._check_channel_restrictions()
        if channel_restriction_result: return channel_restriction_result

        if self.guild and not self.message.channel.permissions_for(self.guild.me).send_messages: 
             return MessageHandlerResult(action="do_nothing", log_message="No send perms.")

        llm_reply_action_result, pending_token_restriction_info = await self._handle_llm_interaction(
            content_for_llm, interaction_case_debug, is_reply_to_bot, 
            show_typing_indicator, is_bot_initiated_random_for_ratelimit
        )

        if llm_reply_action_result.get("action") == "error": return llm_reply_action_result

        # Message rate limit check also requires self.member
        pending_message_restriction_info = await self._check_message_rate_limit(interaction_case_debug, is_bot_initiated_random_for_ratelimit)
        
        final_result = llm_reply_action_result
        if interaction_case_debug == "Random Chance" and not deliver_this_random_reply_publicly:
            final_result = MessageHandlerResult(action="add_reaction_and_do_nothing", content="✅", triggering_interaction_case=interaction_case_debug) 
            final_result["log_message"] = f"LLM processed (no delivery), reaction added. Original: {llm_reply_action_result.get('action')}"
            # Ensure no pending restriction from a suppressed random reply causes issues
            if "pending_restriction" in final_result: del final_result["pending_restriction"] 
            return final_result

        final_pending_restriction: Optional[Dict[str, Any]] = pending_token_restriction_info or pending_message_restriction_info
        if final_pending_restriction:
            final_result["pending_restriction"] = final_pending_restriction
            final_result["log_message"] = (final_result.get("log_message", "") + 
                                          f" Restriction pending: {final_pending_restriction.get('restriction_reason')}.")
        final_result["triggering_interaction_case"] = interaction_case_debug
        return final_result