# utils/webui_api.py
import aiohttp
import json
import os
import asyncio
from typing import List, Dict, Optional, Tuple, Any, Union
import logging
import discord
import redis
import tiktoken
import unittest.mock

logger = logging.getLogger(__name__)

def _load_prompt_from_file_for_test(file_path: str, prompt_name: str) -> str:
    default_prompt_content = f"Default test content for {prompt_name}"
    try:
        abs_file_path = os.path.join(os.path.dirname(__file__), "prompts", file_path)
        if not os.path.exists(abs_file_path):
            logger.warning(f"[_load_prompt_from_file_for_test] Prompt file not found: {abs_file_path}. Using default for {prompt_name}.")
            return default_prompt_content
        with open(abs_file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            logger.info(f"[_load_prompt_from_file_for_test] Loaded prompt '{prompt_name}' from: {abs_file_path}")
            return content
    except Exception as e:
        logger.error(f"[_load_prompt_from_file_for_test] Error reading prompt file {abs_file_path} for {prompt_name}: {e}. Using default.", exc_info=True)
        return default_prompt_content

class WebUIAPI:
    def __init__(self, base_url: str, model: str, api_key: Optional[str],
                 welcome_system: str, welcome_prompt: str, max_history_per_user: int = 10,
                 knowledge_id: Optional[str] = None, 
                 list_tools_default: Optional[List[str]] = None,
                 redis_config: Optional[Dict[str, Any]] = None,
                 llm_response_validation_retries: int = 0):
        self.base_url = base_url.rstrip('/')
        self.chat_endpoint = f"{self.base_url}/api/chat/completions"
        self.model = model
        self.max_history_per_context = max_history_per_user
        self.list_tools_default: List[str] = list_tools_default if list_tools_default is not None else []
        self.knowledge_id = knowledge_id
        self.welcome_system = welcome_system
        self.welcome_prompt = welcome_prompt
        self.headers = {"Content-Type": "application/json"}
        if api_key: self.headers["Authorization"] = f"Bearer {api_key}"
        
        self.llm_response_validation_retries = llm_response_validation_retries
        
        self.conversation_histories_cache: Dict[Tuple[Any, Any], List[Dict[str, str]]] = {}
        self.redis_client_history: Optional[redis.Redis] = None # type: ignore
        
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
            self.redis_client_history.set(redis_key, history_json) # type: ignore
            logger.debug(f"Saved history for {redis_key} to Redis. Length: {len(history)}")
            return True
        except Exception as e:
            logger.error(f"Error saving history to Redis for {redis_key}: {e}", exc_info=True)
            return False

    def get_context_history(self, user_id: Any, channel_id: Any) -> List[Dict[str, str]]:
        context_key_tuple = (user_id, channel_id)
        if context_key_tuple in self.conversation_histories_cache:
            return self.conversation_histories_cache[context_key_tuple]
        history_from_redis = self._load_history_from_redis(user_id, channel_id)
        if history_from_redis is not None:
            self.conversation_histories_cache[context_key_tuple] = history_from_redis
            return history_from_redis
        return []

    def save_context_history(self, user_id: Any, channel_id: Any, history_list: List[Dict[str, str]]):
        context_key_tuple = (user_id, channel_id)
        truncated_history = history_list
        if len(history_list) > self.max_history_per_context:
            truncated_history = history_list[-self.max_history_per_context:]
            logger.debug(f"History for {context_key_tuple} truncated to {self.max_history_per_context} entries before saving.")
        self.conversation_histories_cache[context_key_tuple] = truncated_history
        self._save_history_to_redis(user_id, channel_id, truncated_history)
        
    async def generate_response(
        self,
        user_id: Any,
        channel_id: Any,
        prompt: str,
        system_message: Optional[str] = None,
        history: Optional[List[Dict[str,str]]] = None, 
        extra_assistant_context: Optional[str] = None,
        tools_to_use: Optional[List[str]] = None
        ) -> Tuple[Dict[str, Any], Optional[str], Optional[int]]:

        if history is None:
            history = self.get_context_history(user_id, channel_id)

        context_identifier = f"user {user_id}, channel {channel_id}"
        
        messages_payload = []
        if system_message: messages_payload.append({"role": "system", "content": system_message})
        messages_payload.extend(history)
        if extra_assistant_context:
            messages_payload.append({"role": "assistant", "content": extra_assistant_context})
        messages_payload.append({"role": "user", "content": prompt})

        estimated_input_tokens = self._estimate_input_tokens(messages_payload)
        final_tools_for_api = tools_to_use if tools_to_use is not None else self.list_tools_default
        
        api_payload = {
            "model": self.model, "messages": messages_payload,
            "tool_ids": final_tools_for_api or [], 
            "files": [{"type": "collection", "id": self.knowledge_id}] if self.knowledge_id else [],
            "stream": False
        }

        max_attempts = 1 + self.llm_response_validation_retries
        last_error_for_logging: Optional[str] = "Max retries reached for LLM call after validation failures."
        last_final_json_to_return: Dict[str, Any] = {
            "type": "text", "response": "Sorry, the AI's response was not in the expected format after multiple attempts.", 
            "data": None, "scores": None
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
                        response_text = await response.text()

                        if response.status == 200:
                            try:
                                api_data = json.loads(response_text)
                            except json.JSONDecodeError as e:
                                logger.error(f"[generate_response attempt {attempt + 1}] API response JSON Decode Error: {e}. Body: {response_text[:500]}", exc_info=True)
                                current_attempt_error_for_logging = "Failed to decode API response as JSON."
                                current_attempt_final_json_to_return = {"type": "text", "response": "Error: AI service response was not valid JSON.", "data": None, "scores": None}
                                if attempt < max_attempts - 1: await asyncio.sleep(1 + attempt); continue
                                else: break # Break from loop if max attempts reached

                            llm_response_field_content_for_token_counting = ""
                            validation_passed = False

                            if api_data.get("choices") and api_data["choices"]:
                                message_obj = api_data["choices"][0].get("message")
                                if message_obj and isinstance(message_obj, dict):
                                    llm_content_string = message_obj.get("content")
                                    logger.debug(f"[generate_response attempt {attempt + 1}] LLM content string: {llm_content_string}")

                                    if llm_content_string and llm_content_string.startswith('```json') and llm_content_string.endswith('```'):
                                        llm_content_string = llm_content_string[len('```json'):-3].strip()
                                        logger.debug(f"[generate_response attempt {attempt + 1}] LLM content string fixed: {llm_content_string}")
                                    
                                    if llm_content_string and llm_content_string.strip():
                                        try:
                                            parsed_llm_json = json.loads(llm_content_string)
                                            if not isinstance(parsed_llm_json.get("type"), str) or \
                                               not isinstance(parsed_llm_json.get("response"), str):
                                                logger.warning(f"[generate_response attempt {attempt + 1}] LLM JSON missing 'type' or 'response'. Content: {llm_content_string[:300]}")
                                                current_attempt_final_json_to_return = {"type": "text", "response": f"Error: AI response format was incomplete. Raw: {llm_content_string}", "data": parsed_llm_json.get("data"), "scores": None}
                                                current_attempt_error_for_logging = "LLM JSON malformed (missing/invalid type or response)."
                                            else:
                                                if "scores" in parsed_llm_json and not isinstance(parsed_llm_json.get("scores"), dict):
                                                    parsed_llm_json["scores"] = None
                                                elif "scores" not in parsed_llm_json:
                                                     parsed_llm_json["scores"] = None
                                                current_attempt_final_json_to_return = parsed_llm_json
                                                llm_response_field_content_for_token_counting = current_attempt_final_json_to_return.get("response", "")
                                                validation_passed = True
                                                current_attempt_error_for_logging = None
                                                logger.info(f"[generate_response attempt {attempt + 1}] Successfully parsed and validated LLM JSON.")
                                        except json.JSONDecodeError as e:
                                            logger.error(f"[generate_response attempt {attempt + 1}] Failed to parse LLM content string as JSON: {e}. Content: {llm_content_string[:500]}", exc_info=True)
                                            current_attempt_final_json_to_return = {"type": "text", "response": "The llm provided invalid json, please try again.", "data": None, "scores": None}
                                            llm_response_field_content_for_token_counting = llm_content_string
                                            current_attempt_error_for_logging = "LLM content was not valid JSON."
                                    else:
                                        user_facing_text = "AI returned no content." if llm_content_string is None else "AI returned an empty response."
                                        current_attempt_error_for_logging = "LLM message content was null." if llm_content_string is None else "LLM message content was empty."
                                        logger.warning(f"[generate_response attempt {attempt+1}] {current_attempt_error_for_logging} Content: '{llm_content_string}'")
                                        current_attempt_final_json_to_return = {"type": "text", "response": user_facing_text, "data": None, "scores": None}
                                        llm_response_field_content_for_token_counting = user_facing_text
                                    
                                    api_usage = api_data.get("usage") # Calculate tokens
                                    if isinstance(api_usage, dict):
                                        if api_usage.get("total_tokens") is not None: current_attempt_tokens_used = int(api_usage["total_tokens"])
                                        elif api_usage.get("prompt_tokens") is not None and api_usage.get("completion_tokens") is not None:
                                            current_attempt_tokens_used = int(api_usage["prompt_tokens"]) + int(api_usage["completion_tokens"])
                                    if current_attempt_tokens_used == 0 and self.tokenizer:
                                        output_tokens_estimated = self._count_tokens(llm_response_field_content_for_token_counting)
                                        current_attempt_tokens_used = estimated_input_tokens + output_tokens_estimated
                                    
                                    if validation_passed:
                                        return current_attempt_final_json_to_return, current_attempt_error_for_logging, current_attempt_tokens_used
                                    # Else, validation failed, loop will continue if attempts remain

                                else: # No message_obj or not a dict
                                    logger.warning(f"[generate_response attempt {attempt + 1}] API response structure unexpected (no message obj). Data: {api_data}")
                                    current_attempt_error_for_logging = "API response format error (no message obj)."
                                    current_attempt_final_json_to_return = {"type": "text", "response": "AI service response format error.", "data": None, "scores": None}
                            else: # No choices
                                logger.warning(f"[generate_response attempt {attempt + 1}] API response structure unexpected (no choices). Data: {api_data}")
                                current_attempt_error_for_logging = "API response format error (no choices)."
                                current_attempt_final_json_to_return = {"type": "text", "response": "AI service response format error.", "data": None, "scores": None}
                        
                        else: # Non-200 status
                            logger.error(f"[generate_response attempt {attempt + 1}] API request failed. Status: {response.status}. Body: {response_text[:500]}")
                            error_detail_msg = f"API Error Status {response.status}"
                            user_facing_error_text = "AI service error."
                            try:
                                error_data_non_200 = json.loads(response_text)
                                if isinstance(error_data_non_200.get("error"), dict): error_detail_msg = error_data_non_200["error"].get("message", error_detail_msg)
                                elif isinstance(error_data_non_200.get("detail"), str): error_detail_msg = error_data_non_200["detail"]
                            except json.JSONDecodeError: pass
                            if response.status == 401: user_facing_error_text = "AI service authentication failed."
                            elif response.status == 404: user_facing_error_text = "AI model/endpoint not found."
                            return {"type": "text", "response": user_facing_error_text, "data": None, "scores": None}, error_detail_msg, 0 # No retry for HTTP errors

            except aiohttp.ClientConnectorError as e_conn:
                logger.error(f"[generate_response attempt {attempt + 1}] Connection Error: {e_conn}", exc_info=True)
                return {"type": "text", "response": "Connection error with AI service.", "data": None, "scores": None}, f"Could not connect to API: {e_conn}", 0
            except asyncio.TimeoutError:
                logger.error(f"[generate_response attempt {attempt + 1}] Timeout Error.")
                current_attempt_error_for_logging = "API request timed out."
                current_attempt_final_json_to_return = {"type": "text", "response": "AI service request timed out.", "data": None, "scores": None}
            except Exception as e_unexp:
                logger.error(f"[generate_response attempt {attempt + 1}] Unexpected Error: {e_unexp}", exc_info=True)
                return {"type": "text", "response": "An unexpected error occurred while contacting AI.", "data": None, "scores": None}, f"Unexpected error: {str(e_unexp)}", 0
            
            # Update last known states before retry
            last_final_json_to_return = current_attempt_final_json_to_return
            last_error_for_logging = current_attempt_error_for_logging
            last_total_tokens_used = current_attempt_tokens_used

            if attempt < max_attempts - 1:
                logger.info(f"Attempt {attempt + 1} failed validation or timed out. Retrying after a delay...")
                await asyncio.sleep(1 + attempt) # Basic exponential backoff
            else: # Max attempts reached
                logger.error(f"All {max_attempts} attempts failed for {context_identifier}. Last error: {last_error_for_logging}")
                break # Exit loop

        return last_final_json_to_return, last_error_for_logging, last_total_tokens_used

    async def generate_welcome_message(self, member: discord.Member) -> Tuple[Optional[str], Optional[str]]:
        member_name = member.display_name; guild_name = member.guild.name; member_id_str = str(member.id)
        context_identifier = f"welcome for {member_name}"
        try:
            if not self.welcome_system or not self.welcome_prompt: raise AttributeError("Welcome prompt(s) not configured.")
            system_message = self.welcome_system.format(user_name=member_name, guild_name=guild_name, member_id=member_id_str)
            prompt_content = self.welcome_prompt.format(user_name=member_name, guild_name=guild_name)
        except Exception as e: 
            logger.error(f"[{context_identifier}] Error formatting welcome prompt: {e}. Using fallback.", exc_info=True)
            system_message = f"Welcome {member_name} to {guild_name}!"
            prompt_content = f"Please give a warm welcome to <@{member_id_str}>."

        payload = {"model": self.model, "messages": [{"role": "system", "content": system_message}, {"role": "user", "content": prompt_content}], "stream": False, "max_tokens": 300}
        # Welcome messages typically don't need retries for JSON validation as they are simpler.
        # If they did, this call would also need to be wrapped or use a similar retry logic.
        logger.info(f"[generate_welcome_message] Context: {context_identifier}, Sending payload.")
        if logger.isEnabledFor(logging.DEBUG): logger.debug(f"Payload for {context_identifier}:\n{json.dumps(payload, indent=2, ensure_ascii=False)}")

        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.post(self.chat_endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=45)) as response:
                    logger.info(f"[generate_welcome_message] Context: {context_identifier}, Received status: {response.status}")
                    response_text = await response.text()
                    if response.status == 200:
                        try: data = json.loads(response_text)
                        except json.JSONDecodeError as e:
                            logger.error(f"[generate_welcome_message] JSON Decode Error: {e}. Body: {response_text[:500]}", exc_info=True)
                            return None, "Failed to decode welcome API response."
                        if data.get("choices") and data["choices"]:
                            msg_obj = data["choices"][0].get("message")
                            if msg_obj and isinstance(msg_obj, dict):
                                content = msg_obj.get("content")
                                if content is not None: return content.strip(), None
                        logger.warning(f"[generate_welcome_message] API response structure for welcome unexpected or content missing. Data: {data}")
                        return None, "Welcome API response format error or no content."
                    else:
                        logger.error(f"[generate_welcome_message] API request failed. Status: {response.status}. Body: {response_text[:500]}")
                        return None, f"Welcome API Error Status {response.status}"
        except Exception as e:
            logger.error(f"[generate_welcome_message] Error for {context_identifier}: {e}", exc_info=True)
            return None, f"Unexpected welcome error: {str(e)}"
        return None, "Unknown error during welcome message generation."

async def run_tests():
    logger.info("--- Starting WebUIAPI Tests (basic execution) ---")
    # Test setup would need to be more elaborate to truly test retries,
    # potentially mocking session.post to return bad JSON initially, then good JSON.
    # For brevity, this test run remains a placeholder for such advanced testing.
    api = WebUIAPI(
        base_url="[http://mock-llm-api.com](http://mock-llm-api.com)", model="test-model", api_key=None,
        welcome_system="", welcome_prompt="", llm_response_validation_retries=1 
    )
    # Example: Test successful case (mock would return good JSON on first try)
    # Example: Test retry case (mock returns bad JSON then good JSON)
    logger.info("--- WebUIAPI Tests Finished (basic execution, retry logic needs specific mocks) ---")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')
    asyncio.run(run_tests())