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

def _load_prompt_from_file_for_test(file_path: str, prompt_name: str) -> str:
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
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.conversation_histories_cache: Dict[Tuple[Any, Any], List[Dict[str, str]]] = {} # Made keys Any
        self.redis_client_history: Optional[redis.Redis] = None # Specific client for history
        if redis_config:
            try:
                # Create a separate client instance for history to potentially use different DB or settings
                # For now, using the same config.
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
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.error(f"Error during token encoding: {e}", exc_info=True)
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

    def _get_context_redis_key(self, user_id: Any, channel_id: Any) -> str: # Made IDs Any
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
        self.conversation_histories_cache[context_key_tuple] = []
        return []

    def add_to_context_history(self, user_id: Any, channel_id: Any, role: str, content: str):
        context_key_tuple = (user_id, channel_id)
        current_history_list = list(self.get_context_history(user_id, channel_id))
        current_history_list.append({"role": role, "content": content})
        if len(current_history_list) > self.max_history_per_context:
            current_history_list = current_history_list[-self.max_history_per_context:]
        self.conversation_histories_cache[context_key_tuple] = current_history_list
        self._save_history_to_redis(user_id, channel_id, current_history_list)

    async def generate_response(self, user_id: Any, channel_id: Any, prompt: str, system_message: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        history = self.get_context_history(user_id, channel_id)
        context_identifier = f"user {user_id}, channel {channel_id}"
        messages_payload = []
        if system_message: messages_payload.append({"role": "system", "content": system_message})
        messages_payload.extend(history)
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
        
        total_tokens_used = 0 # Default to 0 if unknown

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

                        api_usage = data.get("usage")
                        if isinstance(api_usage, dict) and api_usage.get("total_tokens") is not None:
                            total_tokens_used = int(api_usage["total_tokens"])
                            logger.debug(f"[generate_response] Tokens from API usage: Total={total_tokens_used}")
                        
                        if data.get("choices") and data["choices"]:
                            message_obj = data["choices"][0].get("message")
                            if message_obj and isinstance(message_obj, dict):
                                assistant_message_content = message_obj.get("content")
                                if assistant_message_content is not None:
                                    if total_tokens_used == 0 and self.tokenizer: # Estimate if API didn't provide and we have tokenizer
                                        output_tokens_estimated = self._count_tokens(assistant_message_content)
                                        total_tokens_used = estimated_input_tokens + output_tokens_estimated
                                        logger.debug(f"[generate_response] Tokens estimated: Input={estimated_input_tokens}, Output={output_tokens_estimated}, Total={total_tokens_used}")
                                    
                                    self.add_to_context_history(user_id, channel_id, "user", prompt)
                                    self.add_to_context_history(user_id, channel_id, "assistant", assistant_message_content)
                                    return assistant_message_content.strip(), None, total_tokens_used
                        # Fallthrough for various missing data scenarios
                        logger.warning(f"[generate_response] Context: {context_identifier}, API response structure unexpected or content missing. Data: {data}")
                        return None, "API response format error or no content.", total_tokens_used
                    else: # Non-200
                        logger.error(f"[generate_response] API request failed. Status: {response.status}. Body: {response_text[:500]}")
                        return "AI service error.", f"API Error Status {response.status}", 0
        except Exception as e:
            logger.error(f"[generate_response] Error during API call for {context_identifier}: {e}", exc_info=True)
            return "Connection or unexpected error.", str(e), 0
        # Fallback
        return "Unknown error.", "Unknown error during processing.", 0


    async def generate_welcome_message(self, member: discord.Member) -> Tuple[Optional[str], Optional[str]]:
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
            return None, str(e)
        return None, "Unknown error during welcome message generation." # Fallback

# (run_tests function from previous response, ensure it unpacks 3 values from generate_response)
async def run_tests():
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(name)s: %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
    from dotenv import load_dotenv
    script_dir = os.path.dirname(__file__)
    dotenv_path_standalone = os.path.join(script_dir, '..', '.env')
    if not load_dotenv(dotenv_path=dotenv_path_standalone):
        if not load_dotenv(): logger.warning("Standalone run_tests: .env not found.")

    test_welcome_system = _load_prompt_from_file_for_test("welcome_system.txt", "Welcome System")
    test_welcome_prompt = _load_prompt_from_file_for_test("welcome_prompt.txt", "Welcome Prompt")
    
    # Simplified test config
    test_api_url = os.getenv("OPENWEBUI_API_URL", "http://localhost:3000")
    test_model = os.getenv("OPENWEBUI_MODEL", "test-model") # Fallback model name for test
    test_api_key = os.getenv("OPENWEBUI_API_KEY")
    redis_cfg_test = {"host": os.getenv("REDIS_HOST","localhost"), "port": int(os.getenv("REDIS_PORT",6379)), "db": int(os.getenv("REDIS_DB_TEST",9))}
    if os.getenv("REDIS_PASSWORD"): redis_cfg_test["password"] = os.getenv("REDIS_PASSWORD")

    logger.info(f"--- Starting WebUI API Test (Prompts from files, Redis DB {redis_cfg_test['db']}) ---")
    if not test_model or test_model == "test-model": logger.warning("OPENWEBUI_MODEL not set for test, using fallback.")

    api_client = WebUIAPI(
        base_url=test_api_url, model=test_model, api_key=test_api_key,
        welcome_system=test_welcome_system, welcome_prompt=test_welcome_prompt,
        max_history_per_user=3, redis_config=redis_cfg_test
    )
    
    # Test user/channel IDs (ensure they are int for cache key if needed, though Redis key converts to str)
    user_id1, chan_idA, chan_idB = 123, 789, 456

    # Clear test keys if Redis client initialized
    if api_client.redis_client_history:
        logger.info(f"Clearing test keys from Redis DB {redis_cfg_test['db']} for user {user_id1}...")
        keys_to_delete = [
            api_client._get_context_redis_key(user_id1, chan_idA),
            api_client._get_context_redis_key(user_id1, chan_idB)
        ]
        api_client.redis_client_history.delete(*keys_to_delete)


    logger.info(f"\n--- Test Case 1: User {user_id1}, Channel {chan_idA} ---")
    content, err, tokens = await api_client.generate_response(user_id1, chan_idA, "Hello, world!", "System: You are a test bot.")
    if err: logger.error(f"TC1 Error: {err}")
    else: logger.info(f"TC1 Response: '{content}' (Tokens: {tokens})")

    logger.info(f"\n--- Test Case 2: User {user_id1}, Channel {chan_idA} (Follow-up) ---")
    content, err, tokens = await api_client.generate_response(user_id1, chan_idA, "How are you?")
    if err: logger.error(f"TC2 Error: {err}")
    else: logger.info(f"TC2 Response: '{content}' (Tokens: {tokens})")
    # logger.info(f"TC2 History A: {api_client.get_context_history(user_id1, chan_idA)}")


    logger.info(f"\n--- Test Case 3: User {user_id1}, Channel {chan_idB} (New Context) ---")
    content, err, tokens = await api_client.generate_response(user_id1, chan_idB, "Tell me a joke.", "System: You are a comedian.")
    if err: logger.error(f"TC3 Error: {err}")
    else: logger.info(f"TC3 Response: '{content}' (Tokens: {tokens})")
    # logger.info(f"TC3 History B: {api_client.get_context_history(user_id1, chan_idB)}")
    # logger.info(f"TC3 History A (should be separate): {api_client.get_context_history(user_id1, chan_idA)}")


    logger.info("\n--- Test Case 4: Welcome Message ---")
    class MockMember:
        def __init__(self, id_val, display_name, guild_name): self.id, self.display_name, self.guild = id_val, display_name, MockGuild(guild_name)
    class MockGuild:
        def __init__(self, name): self.name = name
    member = MockMember(999, "TestWelcomeUser", "Test Guild Space")
    content, err = await api_client.generate_welcome_message(member)
    if err: logger.error(f"TC4 Error: {err}")
    else: logger.info(f"TC4 Welcome: '{content}'")
    
    logger.info("\n--- WebUI API Test Finished ---")

if __name__ == "__main__":
    # asyncio.run(run_tests()) # Uncomment to run directly
    pass