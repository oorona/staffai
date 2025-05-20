# utils/webui_api.py
import aiohttp
import json
import os
import sys
import asyncio
from typing import List, Dict, Optional, Tuple, Any, Union
import logging
import discord # Remains for generate_welcome_message type hint
import redis
import tiktoken

# Standard library for mocking
import unittest.mock # ADDED for testing

logger = logging.getLogger(__name__)

# Helper function used only by run_tests below (remains as is)
def _load_prompt_from_file_for_test(file_path: str, prompt_name: str) -> str:
    default_prompt_content = f"Default test content for {prompt_name}"
    # ... (rest of the function remains the same)
    try:
        abs_file_path = os.path.join(os.path.dirname(__file__), "prompts", file_path)
        if not os.path.exists(abs_file_path):
            logger.warning(f"[run_tests] Prompt file not found at: {abs_file_path}. Using default for {prompt_name}.")
            return default_prompt_content
        with open(abs_file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            logger.info(f"[run_tests] Successfully loaded prompt '{prompt_name}' from: {abs_file_path}")
            return content
    except Exception as e:
        logger.error(f"[run_tests] Error reading prompt file {abs_file_path} for {prompt_name}: {e}. Using default.", exc_info=True)
        return default_prompt_content

class WebUIAPI:
    # ... (__init__, _count_tokens, _estimate_input_tokens, Redis helpers, Public History methods remain unchanged) ...
    def __init__(self, base_url: str, model: str, api_key: Optional[str],
                 welcome_system: str, welcome_prompt: str, max_history_per_user: int = 10,
                 knowledge_id: Optional[str] = None, list_tools: Optional[List[str]] = None,
                 redis_config: Optional[Dict[str, Any]] = None):
        self.base_url = base_url.rstrip('/')
        self.chat_endpoint = f"{self.base_url}/api/chat/completions"
        self.model = model
        self.max_history_per_context = max_history_per_user
        self.list_tools = list_tools
        self.knowledge_id = knowledge_id
        self.welcome_system = welcome_system
        self.welcome_prompt = welcome_prompt
        self.headers = {"Content-Type": "application/json"}
        if api_key: self.headers["Authorization"] = f"Bearer {api_key}"
        self.conversation_histories_cache: Dict[Tuple[Any, Any], List[Dict[str, str]]] = {}
        self.redis_client_history: Optional[redis.Redis] = None
        if redis_config:
            try:
                self.redis_client_history = redis.Redis(**redis_config, decode_responses=True, socket_connect_timeout=3)
                self.redis_client_history.ping()
                logger.info(f"WebUIAPI History: Successfully connected to Redis at {redis_config.get('host')}:{redis_config.get('port')}, DB {redis_config.get('db')}")
            except redis.exceptions.ConnectionError as e:
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
        logger.info(f"WebUIAPI Initialized: URL='{self.chat_endpoint}', Model='{self.model}', Max History: {self.max_history_per_context}")

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer and text:
            try: return len(self.tokenizer.encode(text))
            except Exception: return 0
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
            history_json = self.redis_client_history.get(redis_key)
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
            self.redis_client_history.set(redis_key, history_json)
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
        extra_assistant_context: Optional[str] = None
        ) -> Tuple[Dict[str, Any], Optional[str], Optional[int]]:
        """
        Generates LLM response, expecting JSON output from LLM.
        Returns: (parsed_llm_json_or_fallback, error_message_for_logging, tokens_used)
        """
        if history is None:
            history = self.get_context_history(user_id, channel_id)

        context_identifier = f"user {user_id}, channel {channel_id}"
        
        messages_payload = []
        if system_message: messages_payload.append({"role": "system", "content": system_message})
        messages_payload.extend(history)
        if extra_assistant_context:
            logger.debug(f"Injecting extra assistant context for {context_identifier}")
            messages_payload.append({"role": "assistant", "content": extra_assistant_context})
        messages_payload.append({"role": "user", "content": prompt})

        estimated_input_tokens = self._estimate_input_tokens(messages_payload)
        logger.debug(f"[generate_response] Context: {context_identifier}, Estimated input tokens: {estimated_input_tokens}")

        payload = {
            "model": self.model, "messages": messages_payload,
            "tool_ids": self.list_tools or [], "files": [{"type": "collection", "id": self.knowledge_id}] if self.knowledge_id else [],
            "stream": False
        }
        logger.info(f"[generate_response] Context: {context_identifier}, Sending payload to {self.chat_endpoint}")
        if logger.isEnabledFor(logging.DEBUG): logger.debug(f"Payload for {context_identifier}:\n{json.dumps(payload, indent=2, ensure_ascii=False)}")
        
        total_tokens_used: int = 0 
        error_for_logging: Optional[str] = None

        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.post(self.chat_endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                    logger.info(f"[generate_response] Context: {context_identifier}, Received status: {response.status}")
                    response_text = await response.text()

                    if response.status == 200:
                        try:
                            api_data = json.loads(response_text)
                        except json.JSONDecodeError as e:
                            logger.error(f"[generate_response] Context: {context_identifier}, JSON Decode Error for overall API response: {e}. Body: {response_text[:500]}", exc_info=True)
                            return {
                                "type": "text", "response": "Error: The AI service response was not valid JSON.", "data": None, "scores": None
                            }, "Failed to decode API response.", 0
                        
                        if logger.isEnabledFor(logging.DEBUG): logger.debug(f"[generate_response] Parsed initial API JSON data: {json.dumps(api_data, indent=2, ensure_ascii=False)}")

                        api_usage = api_data.get("usage")
                        if isinstance(api_usage, dict):
                            if api_usage.get("total_tokens") is not None: total_tokens_used = int(api_usage["total_tokens"])
                            elif api_usage.get("prompt_tokens") is not None and api_usage.get("completion_tokens") is not None:
                                total_tokens_used = int(api_usage["prompt_tokens"]) + int(api_usage["completion_tokens"])
                            if total_tokens_used > 0: logger.debug(f"[generate_response] Tokens from API usage: Total={total_tokens_used}")
                        
                        final_json_to_return: Dict[str, Any]
                        llm_response_field_content_for_token_counting = ""

                        # ... inside generate_response method in WebUIAPI class ...
                        if api_data.get("choices") and api_data["choices"]:
                            message_obj = api_data["choices"][0].get("message")
                            if message_obj and isinstance(message_obj, dict):
                                llm_content_string = message_obj.get("content") # This is the string we expect to be JSON
                                #llm_content_string=raw_content_string.replace('\\', '\\\\') # Escape backslashes for JSON parsing
                                logger.debug(f"[generate_response] Context: {context_identifier}, LLM content string: {llm_content_string[:300]}") # Log first 300 chars

                                # Ensure llm_content_string is not None and not empty before trying to parse
                                if llm_content_string and llm_content_string.strip(): # ADDED .strip() to check for non-whitespace
                                    try:
                                        parsed_llm_json = json.loads(llm_content_string)
                                        
                                        # Validate presence of 'type' and 'response'.
                                        if not isinstance(parsed_llm_json.get("type"), str) or \
                                           not isinstance(parsed_llm_json.get("response"), str):
                                            logger.warning(f"[generate_response] LLM JSON content missing mandatory 'type' or 'response' (or not strings). Content: {llm_content_string[:300]}")
                                            final_json_to_return = {
                                                "type": "text",
                                                "response": f"Error: AI response format was incomplete or malformed. Raw: {llm_content_string}",
                                                "data": parsed_llm_json.get("data"), 
                                                "scores": None
                                            }
                                            error_for_logging = "LLM JSON malformed (missing/invalid type or response)."
                                        else:
                                            # Ensure 'scores' is a dict or None
                                            if "scores" in parsed_llm_json and not isinstance(parsed_llm_json.get("scores"), dict):
                                                logger.warning(f"[generate_response] LLM JSON 'scores' field is not an object. Scores set to None. Content: {llm_content_string[:300]}")
                                                parsed_llm_json["scores"] = None
                                            elif "scores" not in parsed_llm_json:
                                                logger.info(f"[generate_response] LLM JSON 'scores' field is missing. Scores set to None. Content: {llm_content_string[:300]}")
                                                parsed_llm_json["scores"] = None
                                            
                                            final_json_to_return = parsed_llm_json
                                            logger.info(f"[generate_response] Successfully generated and parsed LLM JSON response for {context_identifier}.")
                                            # error_for_logging remains None or is set by previous issues

                                        llm_response_field_content_for_token_counting = final_json_to_return.get("response", "")

                                    except json.JSONDecodeError as e:
                                        logger.error(f"[generate_response] Context: {context_identifier}, Failed to parse LLM content string as JSON: {e}. Content: {llm_content_string[:500]}", exc_info=True)
                                        final_json_to_return = {
                                            "type": "text", "response": "The llm is dumb and returned a invalid json. Try again", "data": None, "scores": None
                                        }
                                        llm_response_field_content_for_token_counting = llm_content_string # Use the raw string for token counting here
                                        error_for_logging = "LLM content was not valid JSON."
                                else: # llm_content_string is None or empty/whitespace only
                                    if llm_content_string is None:
                                        logger.warning(f"[generate_response] Context: {context_identifier}, LLM message content is null.")
                                        error_for_logging = "LLM message content was null."
                                        user_facing_response = "AI returned no content."
                                    else: # Empty or whitespace string
                                        logger.warning(f"[generate_response] Context: {context_identifier}, LLM message content is empty or whitespace only. Content: '{llm_content_string}'")
                                        error_for_logging = "LLM message content was empty or whitespace."
                                        user_facing_response = "AI returned an empty response."
                                    
                                    final_json_to_return = {"type": "text", "response": user_facing_response, "data": None, "scores": None}
                                    llm_response_field_content_for_token_counting = user_facing_response # Use this for token counting
                                
                                # Estimate completion tokens if not provided by API and tokenizer exists
                                if total_tokens_used == 0 and self.tokenizer: # total_tokens_used is from API usage field
                                    output_tokens_estimated = self._count_tokens(llm_response_field_content_for_token_counting)
                                    total_tokens_used = estimated_input_tokens + output_tokens_estimated # This might overwrite total_tokens_used if it was 0
                                    logger.debug(f"[generate_response] Tokens estimated: Input={estimated_input_tokens}, Output_Response_Field='{llm_response_field_content_for_token_counting[:30]}...' ({output_tokens_estimated}), Total={total_tokens_used}")
                                
                                return final_json_to_return, error_for_logging, total_tokens_used
                        
                        # Fallback if "choices" or "message" structure is not as expected from the API provider
                        logger.warning(f"[generate_response] Context: {context_identifier}, API response structure unexpected (choices/message). Data: {api_data}")
                        return {"type": "text", "response": "AI service response format error.", "data": None, "scores": None}, "API response format error or no content.", total_tokens_used

                    else: 
                        logger.error(f"[generate_response] API request failed. Status: {response.status}. Body: {response_text[:500]}")
                        error_detail_msg = f"API Error Status {response.status}"
                        user_facing_error_text = "AI service error."
                        try:
                            error_data_non_200 = json.loads(response_text)
                            if isinstance(error_data_non_200.get("error"), dict): error_detail_msg = error_data_non_200["error"].get("message", error_detail_msg)
                            elif isinstance(error_data_non_200.get("detail"), str): error_detail_msg = error_data_non_200["detail"]
                        except json.JSONDecodeError: pass
                        
                        if response.status == 401: user_facing_error_text = "AI service authentication failed."
                        elif response.status == 404: user_facing_error_text = "AI model/endpoint not found."
                        
                        return {"type": "text", "response": user_facing_error_text, "data": None, "scores": None}, error_detail_msg, 0
        
        except aiohttp.ClientConnectorError as e_conn:
            logger.error(f"[generate_response] Connection Error for {context_identifier}: {e_conn}", exc_info=True)
            return {"type": "text", "response": "Connection error with AI service.", "data": None, "scores": None}, f"Could not connect to API: {e_conn}", 0
        except asyncio.TimeoutError:
            logger.error(f"[generate_response] Timeout Error for {context_identifier}.")
            return {"type": "text", "response": "AI service request timed out.", "data": None, "scores": None}, "API request timed out.", 0
        except Exception as e_unexp:
            logger.error(f"[generate_response] Unexpected Error for {context_identifier}: {e_unexp}", exc_info=True)
            return {"type": "text", "response": "An unexpected error occurred while contacting AI.", "data": None, "scores": None}, f"Unexpected error: {str(e_unexp)}", 0
        
        logger.critical(f"[generate_response] Reached supposedly unreachable fallback for {context_identifier}.")
        return {"type": "text", "response": "Unknown error processing AI request.", "data": None, "scores": None}, "Unknown processing error.", 0

    async def generate_welcome_message(self, member: discord.Member) -> Tuple[Optional[str], Optional[str]]:
        # ... (Implementation remains unchanged) ...
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
        return None, "Unknown error during welcome message generation." # Fallback

async def run_tests():
    """
    Basic tests for WebUIAPI.generate_response, focusing on JSON parsing and fallbacks.
    """
    # Minimal config for WebUIAPI instantiation
    # Prompts can be simple strings for these tests as we're mocking the response
    api = WebUIAPI(
        base_url="http://mock-llm-api.com",
        model="test-model",
        api_key=None,
        welcome_system="Test Welcome System",
        welcome_prompt="Test Welcome Prompt",
        max_history_per_user=5,
        redis_config=None # Test without Redis for simplicity here
    )

    # --- Test Helper to Create Mock aiohttp Response ---
    def create_mock_aiohttp_response(status_code: int, response_body: str):
        mock_response = unittest.mock.AsyncMock(spec=aiohttp.ClientResponse)
        mock_response.status = status_code
        # mock_response.text = unittest.mock.AsyncMock(return_value=response_body)
        # The text() method itself is a coroutine, so it should return an awaitable
        async def text_coroutine():
            return response_body
        mock_response.text = text_coroutine 
        
        # For context manager (__aenter__ and __aexit__)
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None
        return mock_response

    logger.info("--- Starting WebUIAPI Tests ---")

    test_cases = [
        {
            "name": "Successful: Valid JSON (type: text)",
            "api_status": 200,
            "llm_content_string": json.dumps({
                "type": "text",
                "response": "Hello from LLM!",
                "scores": {"warmth": 5, "humor": 3, "helpful": 4, "civility": 5, "engagement": 4, "creativity": 2, "insightfulness": 3}
            }),
            "expected_type": "text",
            "expected_response_text": "Hello from LLM!",
            "expected_scores_exist": True,
            "expected_error_msg_for_log": None
        },
        {
            "name": "Successful: Valid JSON (type: code)",
            "api_status": 200,
            "llm_content_string": json.dumps({
                "type": "code",
                "response": "Here's some code:",
                "data": {"language": "python", "content": "print('Hello')"},
                "scores": {"warmth": 3, "humor": 1, "helpful": 5, "civility": 4, "engagement": 3, "creativity": 4, "insightfulness": 2}
            }),
            "expected_type": "code",
            "expected_response_text": "Here's some code:",
            "expected_data": {"language": "python", "content": "print('Hello')"},
            "expected_scores_exist": True,
            "expected_error_msg_for_log": None
        },
        {
            "name": "Fallback: LLM content is not JSON",
            "api_status": 200,
            "llm_content_string": "This is just plain text, not JSON.",
            "expected_type": "text",
            "expected_response_text": "This is just plain text, not JSON.", # Raw string becomes response
            "expected_scores_exist": False, # Scores should be None
            "expected_error_msg_for_log": "LLM content was not valid JSON."
        },
        {
            "name": "Fallback: LLM JSON missing 'type' field",
            "api_status": 200,
            "llm_content_string": json.dumps({
                # "type": "text", # Missing type
                "response": "Response without type.",
                "scores": {"warmth": 1, "humor": 1, "helpful": 1, "civility": 1, "engagement": 1, "creativity": 1, "insightfulness": 1}
            }),
            "expected_type": "text", # Fallback type
            "expected_response_text_contains": "Error: AI response format was incomplete or malformed. Raw:",
            "expected_scores_exist": False, # Scores should be None
            "expected_error_msg_for_log": "LLM JSON malformed (missing/invalid type or response)."
        },
        {
            "name": "Fallback: LLM JSON 'scores' is not an object",
            "api_status": 200,
            "llm_content_string": json.dumps({
                "type": "text",
                "response": "Scores are bad.",
                "scores": "not_an_object"
            }),
            "expected_type": "text",
            "expected_response_text": "Scores are bad.",
            "expected_scores_exist": False, # Scores should be None
            "expected_error_msg_for_log": None # No specific log error for this case, but scores are corrected
        },
        {
            "name": "Successful: LLM JSON 'scores' field is missing",
            "api_status": 200,
            "llm_content_string": json.dumps({
                "type": "text",
                "response": "No scores provided by LLM.",
                # "scores": {} # Missing scores field
            }),
            "expected_type": "text",
            "expected_response_text": "No scores provided by LLM.",
            "expected_scores_exist": False, # Scores should be None
            "expected_error_msg_for_log": None
        },
        {
            "name": "API Error: 401 Unauthorized",
            "api_status": 401,
            "llm_content_string": json.dumps({"error": {"message": "Auth failed"}}), # Body for 401
            "expected_type": "text",
            "expected_response_text": "AI service authentication failed.",
            "expected_scores_exist": False,
            "expected_error_msg_for_log": "Auth failed" # Or the more generic "API Error Status 401" if parsing fails
        },
        {
            "name": "API Error: 500 Internal Server Error",
            "api_status": 500,
            "llm_content_string": "Internal Server Error", # Plain text error from server
            "expected_type": "text",
            "expected_response_text": "AI service error.",
            "expected_scores_exist": False,
            "expected_error_msg_for_log": "API Error Status 500"
        }
    ]

    for i, tc in enumerate(test_cases):
        logger.info(f"\n--- Test Case {i+1}: {tc['name']} ---")
        
        # Mock the API response details
        # The outer response from the LLM API provider wraps the LLM's content string
        # For successful (200) cases, the llm_content_string is what's inside choices[0].message.content
        # For non-200 cases, llm_content_string is the direct body of the HTTP error response
        if tc["api_status"] == 200:
            mock_api_body = json.dumps({
                "choices": [{"message": {"content": tc["llm_content_string"]}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60} # Example usage
            })
        else: # Non-200, the llm_content_string is the direct body
            mock_api_body = tc["llm_content_string"]

        mock_http_response = create_mock_aiohttp_response(tc["api_status"], mock_api_body)

        # Patch aiohttp.ClientSession.post
        # We need to patch where it's looked up, which is 'aiohttp.ClientSession'
        # or more specifically where it's used inside the method, often just 'aiohttp.ClientSession' is enough.
        with unittest.mock.patch('aiohttp.ClientSession.post', return_value=mock_http_response) as mock_post:
            # Call the method under test
            result_json, error_log, tokens_used = await api.generate_response(
                user_id="test_user",
                channel_id="test_channel",
                prompt="Test prompt text",
                system_message="Test system message"
            )

            logger.info(f"Returned JSON: {result_json}")
            logger.info(f"Error for log: {error_log}")
            logger.info(f"Tokens used: {tokens_used}")

            # Assertions
            assert isinstance(result_json, dict), f"{tc['name']}: result_json is not a dict"
            assert result_json.get("type") == tc["expected_type"], f"{tc['name']}: Type mismatch. Got {result_json.get('type')}, expected {tc['expected_type']}"
            
            if "expected_response_text_contains" in tc:
                 assert tc["expected_response_text_contains"] in result_json.get("response", ""), f"{tc['name']}: Response text does not contain expected substring."
            else:
                assert result_json.get("response") == tc["expected_response_text"], f"{tc['name']}: Response text mismatch. Got '{result_json.get('response')}', expected '{tc['expected_response_text']}'"

            if tc["expected_scores_exist"]:
                assert isinstance(result_json.get("scores"), dict), f"{tc['name']}: Scores should be a dict but are not."
            else:
                assert result_json.get("scores") is None, f"{tc['name']}: Scores should be None but are {result_json.get('scores')}."
            
            if "expected_data" in tc:
                 assert result_json.get("data") == tc["expected_data"], f"{tc['name']}: Data mismatch."

            if tc["expected_error_msg_for_log"]:
                assert error_log is not None and tc["expected_error_msg_for_log"] in error_log, f"{tc['name']}: Expected log message '{tc['expected_error_msg_for_log']}' not found or mismatch in '{error_log}'"
            else:
                assert error_log is None, f"{tc['name']}: Expected no error_for_log, but got '{error_log}'"
            
            if tc["api_status"] == 200 and "Fallback: LLM content is not JSON" not in tc["name"] and "malformed" not in tc["name"]: # Crude check for non-error token counts
                assert tokens_used is not None and tokens_used > 0, f"{tc['name']}: Expected tokens_used > 0"
            elif tc["api_status"] != 200 : # For API errors
                 assert tokens_used == 0, f"{tc['name']}: Expected tokens_used == 0 for API error status {tc['api_status']}"


    logger.info("--- WebUIAPI Tests Finished ---")


if __name__ == "__main__":
    # Setup basic logging for tests if run directly
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')
    asyncio.run(run_tests())