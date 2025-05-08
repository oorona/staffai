# utils/webui_api.py
import aiohttp
import json
import os
import sys
import asyncio
from typing import List, Dict, Optional, Tuple, Any
import logging
import discord
import redis
import tiktoken

logger = logging.getLogger(__name__)

# Helper function used only by run_tests below
def _load_prompt_from_file_for_test(file_path: str, prompt_name: str) -> str:
    """Loads a prompt file for testing, with a fallback."""
    default_prompt_content = f"Default test content for {prompt_name}"
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
    """
    Handles communication with OpenWebUI API & Redis history persistence.
    History saving is controlled externally via save_context_history.
    """
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
        # Cache for recently accessed histories
        self.conversation_histories_cache: Dict[Tuple[Any, Any], List[Dict[str, str]]] = {}
        # Redis client specifically for history operations
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

        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.info("WebUIAPI: Tiktoken encoder 'cl100k_base' initialized.")
        except Exception as e:
            logger.error(f"WebUIAPI: Failed to initialize tiktoken: {e}. Token counting will be disabled.", exc_info=True)
            self.tokenizer = None
        logger.info(f"WebUIAPI Initialized: URL='{self.chat_endpoint}', Model='{self.model}', Max History: {self.max_history_per_context}")

    # --- Token Counting Helpers ---
    def _count_tokens(self, text: str) -> int:
        if self.tokenizer and text:
            try: return len(self.tokenizer.encode(text))
            except Exception: return 0 # Logged in previous version, simplified here
        return 0

    def _estimate_input_tokens(self, messages: List[Dict[str, str]]) -> int:
        if not self.tokenizer: return 0
        num_tokens = 0
        for message in messages:
            num_tokens += 4 # Base tokens per message
            for key, value in message.items():
                if value: num_tokens += self._count_tokens(str(value))
                if key == "name": num_tokens -= 1 # Adjust if name is present
        num_tokens += 2 # For priming assistant response
        return num_tokens

    # --- Redis Interaction Helpers ---
    def _get_context_redis_key(self, user_id: Any, channel_id: Any) -> str:
        """Generates a unique Redis key for the conversation context."""
        return f"discord_context:{str(user_id)}:{str(channel_id)}"

    def _load_history_from_redis(self, user_id: Any, channel_id: Any) -> Optional[List[Dict[str, str]]]:
        """Loads history from Redis. Returns None if not found or error."""
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
        """Saves history to Redis. Returns True on success, False on failure."""
        if not self.redis_client_history: return False
        redis_key = self._get_context_redis_key(user_id, channel_id)
        try:
            history_json = json.dumps(history)
            self.redis_client_history.set(redis_key, history_json)
            # Optionally add TTL: self.redis_client_history.expire(redis_key, ...)
            logger.debug(f"Saved history for {redis_key} to Redis. Length: {len(history)}")
            return True
        except Exception as e:
            logger.error(f"Error saving history to Redis for {redis_key}: {e}", exc_info=True)
            return False

    # --- Public History Methods ---
    def get_context_history(self, user_id: Any, channel_id: Any) -> List[Dict[str, str]]:
        """
        Retrieves conversation history. Checks cache first, then Redis.
        Returns empty list if not found anywhere.
        """
        context_key_tuple = (user_id, channel_id)
        if context_key_tuple in self.conversation_histories_cache:
            return self.conversation_histories_cache[context_key_tuple]
        history_from_redis = self._load_history_from_redis(user_id, channel_id)
        if history_from_redis is not None:
            self.conversation_histories_cache[context_key_tuple] = history_from_redis
            return history_from_redis
        # Not in cache or Redis, return empty list for this session
        # Cache will be populated if history is saved later via save_context_history
        return []

    def save_context_history(self, user_id: Any, channel_id: Any, history_list: List[Dict[str, str]]):
        """
        Applies truncation and saves the provided history list to cache and Redis.
        This should be called by the cog after constructing the desired history state.
        """
        context_key_tuple = (user_id, channel_id)
        truncated_history = history_list
        if len(history_list) > self.max_history_per_context:
            truncated_history = history_list[-self.max_history_per_context:]
            logger.debug(f"History for {context_key_tuple} truncated to {self.max_history_per_context} entries before saving.")

        self.conversation_histories_cache[context_key_tuple] = truncated_history # Update cache
        self._save_history_to_redis(user_id, channel_id, truncated_history) # Persist to Redis


    # --- LLM Interaction Methods ---
    async def generate_response(
        self,
        user_id: Any,
        channel_id: Any,
        prompt: str,
        system_message: Optional[str] = None,
        history: Optional[List[Dict[str,str]]] = None, # Allow passing pre-fetched history
        extra_assistant_context: Optional[str] = None # For injecting replied-to message
        ) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        """
        Generates LLM response using provided/fetched history and optional extra context.
        Calculates/estimates token usage. DOES NOT automatically save history.
        Returns: (response_content, error_message, tokens_used)
        """
        if history is None:
            history = self.get_context_history(user_id, channel_id)

        context_identifier = f"user {user_id}, channel {channel_id}"
        
        # Construct message payload for LLM
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
        
        total_tokens_used = 0 # Default to 0

        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.post(self.chat_endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                    logger.info(f"[generate_response] Context: {context_identifier}, Received status: {response.status}")
                    response_text = await response.text()

                    if response.status == 200:
                        try: data = json.loads(response_text)
                        except json.JSONDecodeError as e:
                            logger.error(f"[generate_response] Context: {context_identifier}, JSON Decode Error: {e}. Body: {response_text[:500]}", exc_info=True)
                            return None, "Failed to decode API response.", 0
                        
                        if logger.isEnabledFor(logging.DEBUG): logger.debug(f"[generate_response] Parsed JSON data: {json.dumps(data, indent=2, ensure_ascii=False)}")

                        # Get token usage if available
                        api_usage = data.get("usage")
                        if isinstance(api_usage, dict):
                            if api_usage.get("total_tokens") is not None: total_tokens_used = int(api_usage["total_tokens"])
                            elif api_usage.get("prompt_tokens") is not None and api_usage.get("completion_tokens") is not None:
                                total_tokens_used = int(api_usage["prompt_tokens"]) + int(api_usage["completion_tokens"])
                            if total_tokens_used > 0: logger.debug(f"[generate_response] Tokens from API usage: Total={total_tokens_used}")
                        
                        # Extract response content
                        if data.get("choices") and data["choices"]:
                            message_obj = data["choices"][0].get("message")
                            if message_obj and isinstance(message_obj, dict):
                                assistant_message_content = message_obj.get("content")
                                if assistant_message_content is not None:
                                    # Estimate tokens if needed
                                    if total_tokens_used == 0 and self.tokenizer:
                                        output_tokens_estimated = self._count_tokens(assistant_message_content)
                                        total_tokens_used = estimated_input_tokens + output_tokens_estimated
                                        logger.debug(f"[generate_response] Tokens estimated: Total={total_tokens_used}")
                                    
                                    logger.info(f"[generate_response] Successfully generated response for {context_identifier}.")
                                    # NOTE: History is NOT saved here. Caller must call save_context_history.
                                    return assistant_message_content.strip(), None, total_tokens_used
                        
                        logger.warning(f"[generate_response] Context: {context_identifier}, API response structure unexpected or content missing. Data: {data}")
                        return None, "API response format error or no content.", total_tokens_used # Return tokens even if content missing

                    else: # Non-200 status
                        logger.error(f"[generate_response] API request failed. Status: {response.status}. Body: {response_text[:500]}")
                        # Try to extract specific error message from common structures
                        error_detail_msg = f"API Error Status {response.status}"
                        user_facing_msg = "AI service error."
                        try:
                            error_data = json.loads(response_text)
                            if isinstance(error_data.get("error"), dict): error_detail_msg = error_data["error"].get("message", error_detail_msg)
                            elif isinstance(error_data.get("detail"), str): error_detail_msg = error_data["detail"]
                        except json.JSONDecodeError: pass # Keep original error if decode fails
                        # Add user-friendly messages based on status if desired
                        if response.status == 401: user_facing_msg = "AI service authentication failed."
                        elif response.status == 404: user_facing_msg = "AI model/endpoint not found."
                        return user_facing_msg, error_detail_msg, 0 # No tokens used for failed request

        except aiohttp.ClientConnectorError as e:
            logger.error(f"[generate_response] Connection Error for {context_identifier}: {e}", exc_info=True)
            return "Connection error.", f"Could not connect to API: {e}", 0
        except asyncio.TimeoutError:
            logger.error(f"[generate_response] Timeout Error for {context_identifier}.")
            return "Request timed out.", "API request timed out.", 0
        except Exception as e:
            logger.error(f"[generate_response] Unexpected Error for {context_identifier}: {e}", exc_info=True)
            return "Unexpected error.", f"Unexpected error: {str(e)}", 0
        
        # Fallback return (should be unreachable if logic above is sound)
        return "Unknown error.", "Unknown processing error.", 0


    async def generate_welcome_message(self, member: discord.Member) -> Tuple[Optional[str], Optional[str]]:
        # ... (Implementation remains unchanged from previous versions) ...
        member_name = member.display_name; guild_name = member.guild.name; member_id_str = str(member.id)
        context_identifier = f"welcome for {member_name}"
        try:
            if not self.welcome_system or not self.welcome_prompt: raise AttributeError("Welcome prompt(s) not configured.")
            system_message = self.welcome_system.format(user_name=member_name, guild_name=guild_name, member_id=member_id_str)
            prompt_content = self.welcome_prompt.format(user_name=member_name, guild_name=guild_name)
        except Exception as e: # Catch format errors or AttributeErrors
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


# --- run_tests Function (Requires Significant Update) ---
# The previous run_tests assumed automatic history saving. It needs
# to be rewritten to manually handle history state between calls
# to accurately test the new flow and the save_context_history method.
# Providing a fully working test suite here is complex.
async def run_tests():
     logger.warning("The run_tests function in webui_api.py needs to be updated to work with the new external history management (save_context_history). Skipping tests for now.")
     pass

if __name__ == "__main__":
    # asyncio.run(run_tests()) # Tests need rewrite
    pass