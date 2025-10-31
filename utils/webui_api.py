## utils/webui_api.py
import aiohttp
import json
import time
import os
import asyncio
from typing import List, Dict, Optional, Tuple, Any, Union # Added Union
import logging
import discord # Keep for type hints if member objects passed, though not directly used now
import redis
import tiktoken
# import unittest.mock # Not needed for runtime

logger = logging.getLogger(__name__)

# _load_prompt_from_file_for_test is a test utility, not directly used by core logic here
# but can remain for testing WebUIAPI independently if needed.

class WebUIAPI:
    def __init__(self, base_url: str, model: str, api_key: Optional[str],
                 max_history_per_user: int = 10,
                 knowledge_id: Optional[str] = None, 
                 list_tools_default: Optional[List[str]] = None,
                 redis_config: Optional[Dict[str, Any]] = None,
                 context_history_ttl_seconds: int = 1800,
                 context_message_max_age_seconds: int = 1800,
                 llm_response_validation_retries: int = 0):
        self.base_url = base_url.rstrip('/')
        self.chat_endpoint = f"{self.base_url}/api/chat/completions"
        self.model = model
        self.max_history_per_context = max_history_per_user
        self.list_tools_default: List[str] = list_tools_default if list_tools_default is not None else []
        self.knowledge_id = knowledge_id
        self.headers = {"Content-Type": "application/json"}
        if api_key: self.headers["Authorization"] = f"Bearer {api_key}"
        
        self.llm_response_validation_retries = llm_response_validation_retries

        # Cache stores {'history': [...], 'saved_at': epoch_seconds}
        self.conversation_histories_cache: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
        self.redis_client_history: Optional[redis.Redis] = None # type: ignore
        # How long to keep per-user/channel conversation history in Redis before it expires (seconds).
        # If <= 0, no expiry will be set.
        self.context_history_ttl_seconds = int(context_history_ttl_seconds) if context_history_ttl_seconds is not None else 0
        # How old (seconds) individual messages can be before being purged from context
        self.context_message_max_age_seconds = int(context_message_max_age_seconds) if context_message_max_age_seconds is not None else 0

        if redis_config:
            try:
                self.redis_client_history = redis.Redis(**redis_config, decode_responses=True, socket_connect_timeout=3) # type: ignore
                self.redis_client_history.ping() # type: ignore
                logger.info(f"WebUIAPI History: Successfully connected to Redis at {redis_config.get('host')}:{redis_config.get('port')}, DB {redis_config.get('db')}")
            except redis.exceptions.ConnectionError as e: # type: ignore
                logger.error(f"WebUIAPI History: Failed to connect to Redis: {e}. History context will be in-memory only.", exc_info=True)
                self.redis_client_history = None
            except Exception as e:
                logger.error(f"WebUIAPI History: An unexpected error occurred initializing Redis client: {e}", exc_info=True)
                self.redis_client_history = None
        else:
            logger.warning("WebUIAPI History: Redis configuration not provided. History context will be in-memory only.")

        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.info("WebUIAPI: Tiktoken encoder 'cl100k_base' initialized.")
        except Exception as e:
            logger.error(f"WebUIAPI: Failed to initialize tiktoken: {e}. Token counting will be disabled.", exc_info=True)
            self.tokenizer = None
        logger.info(f"WebUIAPI Initialized: URL='{self.chat_endpoint}', Model='{self.model}', Max History: {self.max_history_per_context}, Default Tools: {self.list_tools_default}, Validation Retries: {self.llm_response_validation_retries}")

    def _filter_messages_by_age(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out messages older than context_message_max_age_seconds.
        
        Args:
            history: List of message dicts, each may have a 'timestamp' field
            
        Returns:
            Filtered history with only recent messages
        """
        if self.context_message_max_age_seconds <= 0:
            return history  # No age filtering if disabled
            
        current_time = time.time()
        cutoff_time = current_time - self.context_message_max_age_seconds
        
        filtered_history = []
        for message in history:
            # Check if message has timestamp and if it's recent enough
            msg_timestamp = message.get('timestamp', current_time)  # Default to current time if no timestamp
            if msg_timestamp >= cutoff_time:
                filtered_history.append(message)
            else:
                logger.debug(f"Filtered out message older than {self.context_message_max_age_seconds}s: {message.get('role', 'unknown')} - age: {current_time - msg_timestamp:.1f}s")
        
        return filtered_history

    def _add_timestamps_to_history(self, history: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Add timestamps to history messages that don't have them.
        
        Args:
            history: List of message dicts (role, content)
            
        Returns:
            History with timestamp added to each message
        """
        current_time = time.time()
        timestamped_history = []
        
        for message in history:
            timestamped_message = dict(message)  # Copy the message
            if 'timestamp' not in timestamped_message:
                timestamped_message['timestamp'] = current_time
            timestamped_history.append(timestamped_message)
            
        return timestamped_history

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer and text:
            try: return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.error(f"Error counting tokens for text '{text[:30]}...': {e}", exc_info=True)
                return 0
        return 0

    def _estimate_input_tokens(self, messages: List[Dict[str, str]]) -> int:
        if not self.tokenizer: return 0
        num_tokens = 0
        for message in messages:
            num_tokens += 4 
            for key, value in message.items():
                if value: num_tokens += self._count_tokens(str(value))
                if key == "name": num_tokens -= 1 
        num_tokens += 2 
        return num_tokens

    def _get_context_redis_key(self, user_id: Any, channel_id: Any) -> str:
        return f"discord_context:{str(user_id)}:{str(channel_id)}"

    def _load_history_from_redis(self, user_id: Any, channel_id: Any) -> Optional[List[Dict[str, str]]]:
        if not self.redis_client_history: return None
        redis_key = self._get_context_redis_key(user_id, channel_id)
        try:
            history_json = self.redis_client_history.get(redis_key) # type: ignore
            if history_json:
                history = json.loads(history_json)
                logger.debug(f"Loaded history for {redis_key} from Redis. Length: {len(history)}")
                # Update in-memory cache with a fresh timestamp
                self.conversation_histories_cache[(user_id, channel_id)] = {"history": history, "saved_at": time.time()}
                return history
            return None
        except Exception as e:
            logger.error(f"Error loading/decoding history from Redis for {redis_key}: {e}", exc_info=True)
            return None

    def _save_history_to_redis(self, user_id: Any, channel_id: Any, history: List[Dict[str, str]]) -> bool:
        if not self.redis_client_history: return False
        redis_key = self._get_context_redis_key(user_id, channel_id)
        try:
            history_json = json.dumps(history)
            # Save the JSON blob and set an expiry if configured so inactive contexts decay automatically.
            self.redis_client_history.set(redis_key, history_json) # type: ignore
            try:
                if self.context_history_ttl_seconds and int(self.context_history_ttl_seconds) > 0:
                    # Use expire to update TTL on every save
                    self.redis_client_history.expire(redis_key, int(self.context_history_ttl_seconds)) # type: ignore
            except Exception:
                # Non-fatal: if expire fails, we still saved the history
                logger.debug(f"WebUIAPI: Failed to set TTL on history key {redis_key}. Continuing without TTL.")
            logger.debug(f"Saved history for {redis_key} to Redis. Length: {len(history)}")
            return True
        except Exception as e:
            logger.error(f"Error saving history to Redis for {redis_key}: {e}", exc_info=True)
            return False

    def get_context_history(self, user_id: Any, channel_id: Any) -> List[Dict[str, str]]:
        context_key_tuple = (user_id, channel_id)
        # If cache exists, validate TTL (if configured)
        cached = self.conversation_histories_cache.get(context_key_tuple)
        if cached:
            saved_at = cached.get("saved_at", 0)
            if self.context_history_ttl_seconds and int(self.context_history_ttl_seconds) > 0:
                if time.time() - float(saved_at) > float(self.context_history_ttl_seconds):
                    # Cache expired — remove and also remove Redis key to be safe
                    try:
                        if self.redis_client_history:
                            redis_key = self._get_context_redis_key(user_id, channel_id)
                            self.redis_client_history.delete(redis_key) # type: ignore
                    except Exception:
                        logger.debug(f"WebUIAPI: Failed to delete expired Redis key for {context_key_tuple}")
                    del self.conversation_histories_cache[context_key_tuple]
                    return []
            
            # Apply age filtering to cached history
            history_with_timestamps = self._add_timestamps_to_history(cached.get("history", []))
            filtered_history = self._filter_messages_by_age(history_with_timestamps)
            # Convert back to simple format for API compatibility
            return [{"role": msg["role"], "content": msg["content"]} for msg in filtered_history]

        history_from_redis = self._load_history_from_redis(user_id, channel_id)
        if history_from_redis is not None:
            # Apply age filtering to Redis history as well
            history_with_timestamps = self._add_timestamps_to_history(history_from_redis)
            filtered_history = self._filter_messages_by_age(history_with_timestamps)
            # Convert back to simple format for API compatibility
            return [{"role": msg["role"], "content": msg["content"]} for msg in filtered_history]
        return []

    def save_context_history(self, user_id: Any, channel_id: Any, history_list: List[Dict[str, str]]):
        context_key_tuple = (user_id, channel_id)
        
        # Add timestamps to new messages and filter by age
        history_with_timestamps = self._add_timestamps_to_history(history_list)
        filtered_history = self._filter_messages_by_age(history_with_timestamps)
        
        # Apply count-based truncation after age filtering
        truncated_history = filtered_history
        if len(filtered_history) > self.max_history_per_context:
            truncated_history = filtered_history[-self.max_history_per_context:]
            logger.debug(f"History for {context_key_tuple} truncated to {self.max_history_per_context} entries after age filtering.")
        
        # Log if age filtering removed messages
        if len(history_with_timestamps) > len(filtered_history):
            removed_count = len(history_with_timestamps) - len(filtered_history)
            logger.debug(f"Age filtering removed {removed_count} old messages for {context_key_tuple}")
        
        # Convert back to simple format for storage (timestamps are implicit in save time)
        simple_history = [{"role": msg["role"], "content": msg["content"]} for msg in truncated_history]
        
        # Update cache with timestamp so TTL checks can evict later
        self.conversation_histories_cache[context_key_tuple] = {"history": simple_history, "saved_at": time.time()}
        self._save_history_to_redis(user_id, channel_id, simple_history)
        
    async def generate_response(
        self,
        user_id: Any, # Retain for logging context, though not used in API payload directly unless for history key
        channel_id: Any, # Retain for logging context
        prompt: str,
        system_message: Optional[str] = None,
        history: Optional[List[Dict[str,str]]] = None, 
        extra_assistant_context: Optional[str] = None,
        tools_to_use: Optional[List[str]] = None
        ) -> Tuple[Dict[str, Any], Optional[str], Optional[int]]: # Return structure no longer includes scores explicitly here

        # History is already fetched by MessageHandler and passed in
        # if history is None:
        # history = self.get_context_history(user_id, channel_id) # This line might be redundant if MH always passes it

        context_identifier = f"user {user_id}, channel {channel_id} (conversation)" # Clarify context
        
        messages_payload = []
        if system_message: messages_payload.append({"role": "system", "content": system_message})
        if history: messages_payload.extend(history) # Ensure history is not None before extending
        if extra_assistant_context:
            messages_payload.append({"role": "assistant", "content": extra_assistant_context})
        messages_payload.append({"role": "user", "content": prompt})

        estimated_input_tokens = self._estimate_input_tokens(messages_payload)
        final_tools_for_api = tools_to_use if tools_to_use is not None else self.list_tools_default
        
        # Sort tool_ids for consistency and to potentially reduce race conditions
        sorted_tools_for_api = sorted(final_tools_for_api) if final_tools_for_api else []
        
        api_payload = {
            "model": self.model, "messages": messages_payload,
            "tool_ids": sorted_tools_for_api, 
            "files": [{"type": "collection", "id": self.knowledge_id}] if self.knowledge_id else [],
            "stream": True  # Use streaming to avoid OpenWebUI MCP TaskGroup bug with multiple servers
        }

        max_attempts = 1 + self.llm_response_validation_retries
        last_error_for_logging: Optional[str] = "Max retries reached for LLM conversation call after validation failures."
        # Default response if all retries fail for the conversation part
        last_final_json_to_return: Dict[str, Any] = {
            "type": "text", "response": "Sorry, the AI's response was not in the expected format after multiple attempts.", 
            "data": None 
            # No "scores" field here anymore
        }
        last_total_tokens_used: Optional[int] = 0

        for attempt in range(max_attempts):
            logger.info(f"[generate_response attempt {attempt + 1}/{max_attempts}] Context: {context_identifier}, Sending payload. Tokens_est: {estimated_input_tokens}, Tools: {final_tools_for_api}")
            if logger.isEnabledFor(logging.DEBUG) and attempt == 0:
                 logger.debug(f"Payload for {context_identifier} (Attempt {attempt+1}):\n{json.dumps(api_payload, indent=2, ensure_ascii=False)}")
            
            current_attempt_tokens_used: Optional[int] = 0
            current_attempt_error_for_logging: Optional[str] = None
            current_attempt_final_json_to_return: Dict[str, Any] = {}

            try:
                async with aiohttp.ClientSession(headers=self.headers) as session:
                    async with session.post(self.chat_endpoint, json=api_payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                        logger.info(f"[generate_response attempt {attempt + 1}] Context: {context_identifier}, Received status: {response.status}")
                        
                        if response.status == 200:
                            # Handle streaming response (SSE format)
                            llm_content_string = ""
                            api_usage_data = None
                            
                            try:
                                async for line in response.content:
                                    line_text = line.decode('utf-8').strip()
                                    if not line_text or line_text.startswith(':'):
                                        continue
                                    
                                    if line_text.startswith('data: '):
                                        data_part = line_text[6:]  # Remove 'data: ' prefix
                                        
                                        if data_part == '[DONE]':
                                            break
                                        
                                        try:
                                            chunk = json.loads(data_part)
                                            
                                            # Extract content from delta
                                            if chunk.get("choices") and chunk["choices"]:
                                                delta = chunk["choices"][0].get("delta", {})
                                                if "content" in delta and delta["content"]:
                                                    llm_content_string += delta["content"]
                                            
                                            # Extract usage if present (usually in final chunk)
                                            if "usage" in chunk and chunk["usage"]:
                                                api_usage_data = chunk["usage"]
                                        
                                        except json.JSONDecodeError:
                                            logger.debug(f"Skipping non-JSON chunk: {data_part[:100]}")
                                            continue
                                
                                logger.debug(f"[generate_response attempt {attempt + 1}] Full streamed LLM content: {llm_content_string}")
                            
                            except Exception as e:
                                logger.error(f"[generate_response attempt {attempt + 1}] Error reading stream: {e}", exc_info=True)
                                current_attempt_error_for_logging = f"Failed to read streaming response: {e}"
                                current_attempt_final_json_to_return = {"type": "text", "response": "Error: Failed to read AI response stream.", "data": None}
                                if attempt < max_attempts - 1: await asyncio.sleep(1 + attempt); continue
                                else: break

                            llm_response_field_content_for_token_counting = ""
                            validation_passed = False

                            # Clean up JSON wrapper if present
                            if llm_content_string and llm_content_string.startswith('```json') and llm_content_string.endswith('```'):
                                llm_content_string = llm_content_string[len('```json'):-3].strip()
                                logger.debug(f"[generate_response attempt {attempt + 1}] LLM content string fixed: {llm_content_string}")
                            
                            if llm_content_string and llm_content_string.strip():
                                try:
                                    parsed_llm_json = json.loads(llm_content_string)
                                    # Validation: "type" and "response" must be strings
                                    if not isinstance(parsed_llm_json.get("type"), str) or \
                                       not isinstance(parsed_llm_json.get("response"), str):
                                        logger.warning(f"[generate_response attempt {attempt + 1}] LLM JSON missing 'type' or 'response' string. Content: {llm_content_string[:300]}")
                                        current_attempt_final_json_to_return = {"type": "text", "response": f"Error: AI response format was incomplete. Raw: {llm_content_string}", "data": parsed_llm_json.get("data")}
                                        current_attempt_error_for_logging = "LLM JSON malformed (missing/invalid type or response strings)."
                                    else:
                                        # Remove "scores" if it accidentally comes back
                                        if "scores" in parsed_llm_json:
                                            del parsed_llm_json["scores"]
                                            logger.debug("[generate_response] Removed unexpected 'scores' field from conversation response.")

                                        current_attempt_final_json_to_return = parsed_llm_json
                                        llm_response_field_content_for_token_counting = current_attempt_final_json_to_return.get("response", "")
                                        validation_passed = True
                                        current_attempt_error_for_logging = None
                                        logger.info(f"[generate_response attempt {attempt + 1}] Successfully parsed and validated LLM JSON for conversation.")
                                except json.JSONDecodeError as e:
                                    logger.error(f"[generate_response attempt {attempt + 1}] Failed to parse LLM content string as JSON: {e}. Content: {llm_content_string[:500]}", exc_info=True)
                                    current_attempt_final_json_to_return = {"type": "text", "response": "The llm provided invalid json for conversation, please try again.", "data": None}
                                    llm_response_field_content_for_token_counting = llm_content_string # Count raw if parse fails
                                    current_attempt_error_for_logging = "LLM conversation content was not valid JSON."
                            else: # llm_content_string is None or empty
                                user_facing_text = "AI returned no content for conversation." if not llm_content_string else "AI returned an empty response for conversation."
                                current_attempt_error_for_logging = "LLM message content was null or empty."
                                logger.warning(f"[generate_response attempt {attempt+1}] {current_attempt_error_for_logging} Content: '{llm_content_string}'")
                                current_attempt_final_json_to_return = {"type": "text", "response": user_facing_text, "data": None}
                                llm_response_field_content_for_token_counting = user_facing_text # Minimal count for this case
                            
                            # Process usage data
                            if isinstance(api_usage_data, dict):
                                if api_usage_data.get("total_tokens") is not None: 
                                    current_attempt_tokens_used = int(api_usage_data["total_tokens"])
                                elif api_usage_data.get("prompt_tokens") is not None and api_usage_data.get("completion_tokens") is not None:
                                    current_attempt_tokens_used = int(api_usage_data["prompt_tokens"]) + int(api_usage_data["completion_tokens"])
                            
                            if current_attempt_tokens_used == 0 and self.tokenizer: # Fallback token estimation
                                output_tokens_estimated = self._count_tokens(llm_response_field_content_for_token_counting)
                                current_attempt_tokens_used = estimated_input_tokens + output_tokens_estimated
                            
                            if validation_passed: # If good, return immediately
                                return current_attempt_final_json_to_return, current_attempt_error_for_logging, current_attempt_tokens_used
                            # Else, validation failed, loop will continue if attempts remain
                        
                        else: # Non-200 status for conversation call
                            response_text = await response.text()
                            logger.error(f"[generate_response attempt {attempt + 1}] API request failed. Status: {response.status}. Body: {response_text[:500]}")
                            error_detail_msg = f"API Error Status {response.status}"
                            user_facing_error_text = "AI service error for conversation."
                            try:
                                error_data_non_200 = json.loads(response_text)
                                if isinstance(error_data_non_200.get("error"), dict): error_detail_msg = error_data_non_200["error"].get("message", error_detail_msg)
                                elif isinstance(error_data_non_200.get("detail"), str): error_detail_msg = error_data_non_200["detail"]
                            except json.JSONDecodeError: pass # Keep generic if error response isn't JSON
                            if response.status == 401: user_facing_error_text = "AI service authentication failed."
                            elif response.status == 404: user_facing_error_text = "AI model/endpoint not found."
                            # No retry for HTTP errors like 4xx/5xx for this call, return immediately
                            return {"type": "text", "response": user_facing_error_text, "data": None}, error_detail_msg, 0 

            except aiohttp.ClientConnectorError as e_conn:
                logger.error(f"[generate_response attempt {attempt + 1}] Connection Error: {e_conn}", exc_info=True)
                return {"type": "text", "response": "Connection error with AI service for conversation.", "data": None}, f"Could not connect to API: {e_conn}", 0
            except asyncio.TimeoutError:
                logger.error(f"[generate_response attempt {attempt + 1}] Timeout Error.")
                current_attempt_error_for_logging = "API conversation request timed out."
                current_attempt_final_json_to_return = {"type": "text", "response": "AI service conversation request timed out.", "data": None}
            except Exception as e_unexp: # Catch-all for unexpected issues during this attempt
                logger.error(f"[generate_response attempt {attempt + 1}] Unexpected Error: {e_unexp}", exc_info=True)
                # For truly unexpected errors, probably best to bail out and not retry.
                return {"type": "text", "response": "An unexpected error occurred while contacting AI for conversation.", "data": None}, f"Unexpected error: {str(e_unexp)}", 0
            
            # Update last known states before potential retry
            last_final_json_to_return = current_attempt_final_json_to_return
            last_error_for_logging = current_attempt_error_for_logging
            last_total_tokens_used = current_attempt_tokens_used

            if attempt < max_attempts - 1: # If not the last attempt
                logger.info(f"Attempt {attempt + 1} for conversation failed validation or timed out. Retrying after a delay...")
                await asyncio.sleep(1 + attempt) # Basic exponential backoff
            else: # Max attempts reached
                logger.error(f"All {max_attempts} attempts for conversation failed for {context_identifier}. Last error: {last_error_for_logging}")
                break # Exit loop

        # Return the last known state after all attempts for conversation call
        return last_final_json_to_return, last_error_for_logging, last_total_tokens_used

    async def generate_sentiment_scores(
        self,
        user_id: Any, # For logging
        channel_id: Any, # For logging
        user_message_content: str,
        sentiment_system_prompt: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[int]]:
        """
        Generates sentiment scores for a given user message using a specific prompt.
        This call is context-free (no history).
        Returns: (scores_dict, error_message, tokens_used)
                 scores_dict will be None if an error occurs or parsing fails.
                 error_message contains a loggable error string if issues arise.
                 tokens_used is the count from the API or estimated.
        """
        context_identifier = f"user {user_id}, channel {channel_id} (sentiment_scores)"
        
        messages_payload = [
            {"role": "system", "content": sentiment_system_prompt},
            {"role": "user", "content": user_message_content}
        ]

        estimated_input_tokens = self._estimate_input_tokens(messages_payload)
        
        api_payload = {
            "model": self.model, # Use the same model as the main chat for consistency, unless specified otherwise
            "messages": messages_payload,
            "stream": False,
            # Tools and knowledge base are likely not needed for a simple sentiment score
        }

        max_attempts = 1 + self.llm_response_validation_retries # Can use the same retry logic
        last_error_for_logging: Optional[str] = "Max retries reached for LLM sentiment call after validation failures."
        last_scores_to_return: Optional[Dict[str, Any]] = None
        last_total_tokens_used: Optional[int] = 0

        for attempt in range(max_attempts):
            logger.info(f"[generate_sentiment_scores attempt {attempt + 1}/{max_attempts}] Context: {context_identifier}, Sending payload. Tokens_est: {estimated_input_tokens}")
            if logger.isEnabledFor(logging.DEBUG) and attempt == 0:
                 logger.debug(f"Payload for {context_identifier} (Attempt {attempt+1}):\n{json.dumps(api_payload, indent=2, ensure_ascii=False)}")

            current_attempt_tokens_used: Optional[int] = 0
            current_attempt_error_for_logging: Optional[str] = None
            current_attempt_scores_to_return: Optional[Dict[str, Any]] = None

            try:
                async with aiohttp.ClientSession(headers=self.headers) as session:
                    async with session.post(self.chat_endpoint, json=api_payload, timeout=aiohttp.ClientTimeout(total=60)) as response: # Shorter timeout for scores?
                        logger.info(f"[generate_sentiment_scores attempt {attempt + 1}] Context: {context_identifier}, Received status: {response.status}")
                        response_text = await response.text()

                        if response.status == 200:
                            try:
                                api_data = json.loads(response_text)
                            except json.JSONDecodeError as e:
                                logger.error(f"[generate_sentiment_scores attempt {attempt + 1}] API response JSON Decode Error: {e}. Body: {response_text[:500]}", exc_info=True)
                                current_attempt_error_for_logging = "Failed to decode sentiment API response as JSON."
                                if attempt < max_attempts - 1: await asyncio.sleep(1 + attempt); continue
                                else: break

                            llm_response_content_for_token_counting = "" # For token estimation if API fails
                            validation_passed = False

                            if api_data.get("choices") and api_data["choices"]:
                                message_obj = api_data["choices"][0].get("message")
                                if message_obj and isinstance(message_obj, dict):
                                    llm_content_string = message_obj.get("content")
                                    logger.debug(f"[generate_sentiment_scores attempt {attempt + 1}] LLM content string: {llm_content_string}")

                                    if llm_content_string and llm_content_string.startswith('```json') and llm_content_string.endswith('```'):
                                        llm_content_string = llm_content_string[len('```json'):-3].strip()
                                    
                                    if llm_content_string and llm_content_string.strip():
                                        try:
                                            parsed_llm_json = json.loads(llm_content_string)
                                            # Validate structure: must be a dict containing a "scores" dict
                                            if isinstance(parsed_llm_json.get("scores"), dict):
                                                # Further validation: check if all expected score keys are numbers (optional, but good)
                                                # For now, just ensure "scores" itself is a dictionary.
                                                current_attempt_scores_to_return = parsed_llm_json.get("scores")
                                                llm_response_content_for_token_counting = json.dumps(current_attempt_scores_to_return) # Stringify for token counting
                                                validation_passed = True
                                                current_attempt_error_for_logging = None
                                                logger.info(f"[generate_sentiment_scores attempt {attempt + 1}] Successfully parsed LLM JSON for scores.")
                                            else:
                                                logger.warning(f"[generate_sentiment_scores attempt {attempt + 1}] LLM JSON missing 'scores' dictionary. Content: {llm_content_string[:300]}")
                                                current_attempt_error_for_logging = "LLM sentiment JSON malformed (missing 'scores' dict)."
                                                llm_response_content_for_token_counting = llm_content_string # Raw string if format is wrong
                                        except json.JSONDecodeError as e:
                                            logger.error(f"[generate_sentiment_scores attempt {attempt + 1}] Failed to parse LLM content for scores as JSON: {e}. Content: {llm_content_string[:500]}", exc_info=True)
                                            current_attempt_error_for_logging = "LLM sentiment content was not valid JSON."
                                            llm_response_content_for_token_counting = llm_content_string # Raw string for token count
                                    else: # llm_content_string is None or empty
                                        current_attempt_error_for_logging = "LLM message content for scores was null or empty."
                                        logger.warning(f"[generate_sentiment_scores attempt {attempt+1}] {current_attempt_error_for_logging}")
                                        llm_response_content_for_token_counting = "" # Empty
                                    
                                    api_usage = api_data.get("usage")
                                    if isinstance(api_usage, dict):
                                        if api_usage.get("total_tokens") is not None: current_attempt_tokens_used = int(api_usage["total_tokens"])
                                        elif api_usage.get("prompt_tokens") is not None and api_usage.get("completion_tokens") is not None:
                                            current_attempt_tokens_used = int(api_usage["prompt_tokens"]) + int(api_usage["completion_tokens"])
                                    if current_attempt_tokens_used == 0 and self.tokenizer:
                                        output_tokens_estimated = self._count_tokens(llm_response_content_for_token_counting)
                                        current_attempt_tokens_used = estimated_input_tokens + output_tokens_estimated
                                    
                                    if validation_passed:
                                        return current_attempt_scores_to_return, current_attempt_error_for_logging, current_attempt_tokens_used
                                else: # No message_obj or not a dict
                                    logger.warning(f"[generate_sentiment_scores attempt {attempt + 1}] API response structure unexpected (no message obj). Data: {api_data}")
                                    current_attempt_error_for_logging = "API sentiment response format error (no message obj)."
                            else: # No choices
                                logger.warning(f"[generate_sentiment_scores attempt {attempt + 1}] API response structure unexpected (no choices). Data: {api_data}")
                                current_attempt_error_for_logging = "API sentiment response format error (no choices)."
                        else: # Non-200 status
                            logger.error(f"[generate_sentiment_scores attempt {attempt + 1}] API request failed. Status: {response.status}. Body: {response_text[:500]}")
                            error_detail_msg = f"Sentiment API Error Status {response.status}"
                            try: # Try to get more specific error from response body
                                error_data_non_200 = json.loads(response_text)
                                if isinstance(error_data_non_200.get("error"), dict): error_detail_msg = error_data_non_200["error"].get("message", error_detail_msg)
                                elif isinstance(error_data_non_200.get("detail"), str): error_detail_msg = error_data_non_200["detail"]
                            except json.JSONDecodeError: pass
                            return None, error_detail_msg, 0 # No retry for HTTP errors

            except aiohttp.ClientConnectorError as e_conn:
                logger.error(f"[generate_sentiment_scores attempt {attempt + 1}] Connection Error: {e_conn}", exc_info=True)
                return None, f"Could not connect to sentiment API: {e_conn}", 0
            except asyncio.TimeoutError:
                logger.error(f"[generate_sentiment_scores attempt {attempt + 1}] Timeout Error.")
                current_attempt_error_for_logging = "API sentiment request timed out."
            except Exception as e_unexp:
                logger.error(f"[generate_sentiment_scores attempt {attempt + 1}] Unexpected Error: {e_unexp}", exc_info=True)
                return None, f"Unexpected error during sentiment scoring: {str(e_unexp)}", 0
            
            last_scores_to_return = current_attempt_scores_to_return # This is None if validation failed
            last_error_for_logging = current_attempt_error_for_logging
            last_total_tokens_used = current_attempt_tokens_used

            if attempt < max_attempts - 1:
                logger.info(f"Attempt {attempt + 1} for sentiment scores failed validation or timed out. Retrying after a delay...")
                await asyncio.sleep(1 + attempt)
            else:
                logger.error(f"All {max_attempts} attempts for sentiment scores failed for {context_identifier}. Last error: {last_error_for_logging}")
                break
        
        return last_scores_to_return, last_error_for_logging, last_total_tokens_used

    
async def run_tests():
    logger.info("--- Starting WebUIAPI Tests (basic execution) ---")
    # Test setup would need to be more elaborate to truly test retries,
    # potentially mocking session.post to return bad JSON initially, then good JSON.
    # For brevity, this test run remains a placeholder for such advanced testing.
    api = WebUIAPI(
        base_url="[http://mock-llm-api.com](http://mock-llm-api.com)", model="test-model", api_key=None,
        llm_response_validation_retries=1 
    )
    # Example: Test successful case (mock would return good JSON on first try)
    # Example: Test retry case (mock returns bad JSON then good JSON)
    logger.info("--- WebUIAPI Tests Finished (basic execution, retry logic needs specific mocks) ---")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')
    asyncio.run(run_tests())