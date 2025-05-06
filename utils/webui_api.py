import aiohttp
import json
import os
import asyncio
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from collections import deque
import logging
import discord
from discord.ext import commands



# --- Constants ---
#MAX_HISTORY_LENGTH = 10 # Store last 5 user messages and 5 bot responses

logger = logging.getLogger(__name__)


# --- API Interaction Class ---
class WebUIAPI:
    """Handles communication with an Open WebUI compatible API using an OpenAI-like endpoint."""

    def __init__(self, base_url: str, model: str, api_key: str, 
                 welcome_system: str, welcome_prompt: str,max_history_per_user: int = 10,
                 knowledge_id: Optional[str] = None,list_tools: Optional[List[str]] = None):
        """
        Initializes the API client.

        Args:
            base_url (str): The base URL of the Open WebUI API (e.g., http://localhost:3000).
            model (str): The name of the model to use (e.g., granite3.1-dense:8b).
            api_key (Optional[str]): API key required by the endpoint (passed as Bearer token).
        """
        self.base_url = base_url.rstrip('/')
        self.chat_endpoint = f"{self.base_url}/api/chat/completions"
        self.model = model
        self.max_history_per_user= max_history_per_user
        self.user_histories = {}
        self.list_tools = list_tools
        self.knowledge_id = knowledge_id
        self.welcome_system = welcome_system 
        self.welcome_prompt = welcome_prompt 

        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

        self.conversation_histories: Dict[int, List[Dict[str, str]]] = {}
        print(f"WebUIAPI Initialized (OpenAI-Compatible Endpoint): URL='{self.chat_endpoint}', Model='{self.model}', API Key Set: {'Yes' if api_key else 'No'}")


    def get_user_history(self, user_id: int) -> List[Dict[str, str]]:
        """Retrieves the conversation history for a user."""
        return self.conversation_histories.get(user_id, [])

    def add_to_user_history(self, user_id: int, role: str, content: str):
        """Adds a message to the user's history and truncates if necessary."""
        if user_id not in self.conversation_histories:
            self.conversation_histories[user_id] = []
        self.conversation_histories[user_id].append({"role": role, "content": content})
        if len(self.conversation_histories[user_id]) > self.max_history_per_user:
            self.conversation_histories[user_id] = self.conversation_histories[user_id][-self.max_history_per_user:]
    



    async def generate_response(self, user_id: int, prompt: str, system_message: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Generates a response using an OpenAI-compatible API endpoint, maintaining conversation context.
        """
        history = self.get_user_history(user_id)

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        if self.list_tools: 
            tools= self.list_tools
        else:
            tools = []

        if self.knowledge_id:
            knowledge = [{"type": "collection", "id": self.knowledge_id}]
        else:
            knowledge = []
            
        payload = {
            "model": self.model,
            "messages": messages,
            "tool_ids": tools,
            "files": knowledge,
            "stream": False,
            # "temperature": 0.7,
            # "max_tokens": 150,
        }

        print(f"\n[generate_response] Sending payload to {self.chat_endpoint}:")
        print(json.dumps(payload, indent=2, ensure_ascii=False))

        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.post(self.chat_endpoint, json=payload) as response:
                    print(f"[generate_response] Received status: {response.status}")
                    response_text = await response.text()

                    if response.status == 200:
                        try:
                            data = json.loads(response_text)
                        except json.JSONDecodeError as json_err:
                            print(f"[generate_response] Error decoding JSON: {json_err}")
                            return None, f"Failed to decode JSON response from API. Status: {response.status}. Body: {response_text[:500]}..."

                        print(f"[generate_response] Parsed JSON data: {json.dumps(data, indent=2)}")

                        if data.get("choices") and len(data["choices"]) > 0:
                            message_obj = data["choices"][0].get("message")
                            if message_obj and isinstance(message_obj, dict):
                                assistant_message = message_obj.get("content")
                                if assistant_message:
                                    self.add_to_user_history(user_id, "user", prompt)
                                    self.add_to_user_history(user_id, "assistant", assistant_message)
                                    print(f"[generate_response] Successfully generated response for user {user_id}.")
                                    return assistant_message.strip(), None
                                else:
                                    print("[generate_response] API response had choice but no message content.")
                                    return None, "API response had choice but no message content."
                            else:
                                print("[generate_response] API response format unexpected ('message' object missing or not dict in choice).")
                                return None, f"API response format unexpected in choices[0]. Data: {data}"
                        else:
                            error_detail = data.get("error", {}).get("message", "API returned status 200 but no choices or known error structure.")
                            print(f"[generate_response] API returned status 200 but no choices. Error: {error_detail}")
                            return None, f"API returned status 200 but no choices. Details: {error_detail}"
                    else:
                        print(f"[generate_response] API request failed. Status: {response.status}. Body: {response_text[:500]}...")
                        error_details = response_text
                        try:
                            error_data = json.loads(response_text)
                            if 'detail' in error_data: error_details = error_data['detail']
                            elif 'error' in error_data:
                                if isinstance(error_data['error'], dict) and 'message' in error_data['error']: error_details = error_data['error']['message']
                                else: error_details = str(error_data['error'])
                        except json.JSONDecodeError: pass
                        return "El proxy esta caido eso pasa por hacer selfhosting.", f"API request failed with status {response.status}. Details: {error_details[:500]}..."

        except aiohttp.ClientConnectorError as e:
            print(f"[generate_response] Error connecting to API endpoint: {e}")
            return "Estoy durmiendo por el momento. La electricidad es muy cara", f"Could not connect to the API at {self.chat_endpoint}. Is it running and accessible?"
        except Exception as e:
            print(f"[generate_response] An unexpected error occurred during API call: {e}")
            import traceback
            traceback.print_exc()
            return "Estoy durmiendo por el momento. La electricidad es muy cara", f"An unexpected error occurred: {str(e)}"

    async def generate_welcome_message(self, member: discord.Member) -> Tuple[Optional[str], Optional[str]]:
        """
        Generates a standalone welcome message using the OpenAI-compatible endpoint.
        """
        
        member_name=member.display_name,
        guild_name=member.guild.name
        member_id=member.id

        system_message = self.welcome_system.format(user_name=member_name,guild_name=guild_name,member_id=member_id)
        prompt = self.welcome_prompt.format(user_name=member_name,guild_name=guild_name)

      
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "max_tokens": 100,
        }

        print(f"\n[generate_welcome_message] Sending payload to {self.chat_endpoint}:")
        print(system_message)
        print(json.dumps(payload, indent=2, ensure_ascii=False))

        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.post(self.chat_endpoint, json=payload) as response:
                    print(f"[generate_welcome_message] Received status: {response.status}")
                    response_text = await response.text()

                    if response.status == 200:
                        try:
                            data = json.loads(response_text)
                        except json.JSONDecodeError as json_err:
                            print(f"[generate_welcome_message] Error decoding JSON: {json_err}")
                            return None, f"Failed to decode JSON response from API. Status: {response.status}. Body: {response_text[:500]}..."

                        print(f"[generate_welcome_message] Parsed JSON data: {json.dumps(data, indent=2)}")

                        if data.get("choices") and len(data["choices"]) > 0:
                            message_obj = data["choices"][0].get("message")
                            if message_obj and isinstance(message_obj, dict):
                                assistant_message = message_obj.get("content")
                                if assistant_message:
                                    print(f"[generate_welcome_message] Successfully generated welcome for '{member_name}'.")
                                    return assistant_message.strip(), None
                                else:
                                     print("[generate_welcome_message] API response had choice but no message content.")
                                     return None, "API response had choice but no message content for welcome."
                            else:
                                print("[generate_welcome_message] API response format unexpected ('message' object missing/not dict in choice).")
                                return None, f"API response format unexpected in choices[0] for welcome. Data: {data}"
                        else:
                            error_detail = data.get("error", {}).get("message", "API returned status 200 but no choices or known error structure.")
                            print(f"[generate_welcome_message] API returned 200 but no choices. Error: {error_detail}")
                            return None, f"API returned status 200 but no choices for welcome. Details: {error_detail}"
                    else:
                        print(f"[generate_welcome_message] API request failed. Status: {response.status}. Body: {response_text[:500]}...")
                        error_details = response_text
                        try:
                            error_data = json.loads(response_text)
                            if 'detail' in error_data: error_details = error_data['detail']
                            elif 'error' in error_data:
                                if isinstance(error_data['error'], dict) and 'message' in error_data['error']: error_details = error_data['error']['message']
                                else: error_details = str(error_data['error'])
                        except json.JSONDecodeError: pass
                        return None, f"API request failed for welcome message with status {response.status}. Details: {error_details[:500]}..."

        except aiohttp.ClientConnectorError as e:
             print(f"[generate_welcome_message] Error connecting to API endpoint: {e}")
             return None, f"Could not connect to the API at {self.chat_endpoint} for welcome message."
        except Exception as e:
            print(f"[generate_welcome_message] An unexpected error occurred during welcome API call: {e}")
            import traceback
            traceback.print_exc()
            return None, f"An unexpected error occurred generating welcome: {str(e)}"


# --- Main execution block for testing ---
async def run_tests():
    """Runs test cases for the WebUIAPI class using the OpenAI-compatible endpoint."""

    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path=dotenv_path)

    test_api_url = os.getenv("OPENWEBUI_API_URL", "http://localhost:3000")
    test_model = os.getenv("OPENWEBUI_MODEL")
    test_api_key = os.getenv("OPENWEBUI_API_KEY")

    if not test_api_url:
         print("ERROR: OPENWEBUI_API_URL not set in .env file or default. Cannot run tests.")
         return
    if not test_model:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ERROR: OPENWEBUI_MODEL not found in .env file.       !!!")
        print("!!!          Please set it in .env to run tests.          !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    print("\n--- Starting WebUI API Test (Endpoint: /api/chat/completions) ---")
    print(f"Using API URL: {test_api_url}")
    print(f"Using Model: {test_model}")
    print(f"API Key Provided: {'Yes' if test_api_key else 'No'}")

    api_client = WebUIAPI(base_url=test_api_url, model=test_model, api_key=test_api_key)

    # Test cases remain the same...
    print("\n--- Test Case 1: Simple Question (User 123) ---")
    user_id_1 = 123
    prompt_1 = "Why is the sky blue?"
    system_prompt_1 = "You are a helpful science explainer."
    response_1, error_1 = await api_client.generate_response(user_id_1, prompt_1, system_prompt_1)
    if error_1: print(f"Test Case 1 FAILED: {error_1}")
    else: print(f"Test Case 1 Prompt: '{prompt_1}'\nTest Case 1 Response: '{response_1}'")

    print("\n--- Test Case 2: Follow-up Question (User 123) ---")
    prompt_2 = "Does the same reason apply on Mars?"
    response_2, error_2 = await api_client.generate_response(user_id_1, prompt_2)
    if error_2: print(f"Test Case 2 FAILED: {error_2}")
    else: print(f"Test Case 2 Prompt: '{prompt_2}'\nTest Case 2 Response: '{response_2}'")
    print(f"History for user {user_id_1} after test 2: {api_client.get_user_history(user_id_1)}")

    print("\n--- Test Case 3: Different User (User 456) ---")
    user_id_2 = 456
    prompt_3 = "What is Python used for?"
    system_prompt_3 = "You are a software development assistant."
    response_3, error_3 = await api_client.generate_response(user_id_2, prompt_3, system_prompt_3)
    if error_3: print(f"Test Case 3 FAILED: {error_3}")
    else: print(f"Test Case 3 Prompt: '{prompt_3}'\nTest Case 3 Response: {response_3}")
    print(f"History for user {user_id_2} after test 3: {api_client.get_user_history(user_id_2)}")

    print("\n--- Test Case 4: Welcome Message ---")
    new_user_name = "CodeWizard"
    welcome_msg, error_4 = await api_client.generate_welcome_message(new_user_name)
    if error_4: print(f"Test Case 4 FAILED: {error_4}")
    else: print(f"Test Case 4 User: '{new_user_name}'\nTest Case 4 Welcome Message: '{welcome_msg}'")

    print("\n--- WebUI API Test Finished ---")


if __name__ == "__main__":
    try:
        print("Starting test execution...")
        asyncio.run(run_tests())
        print("Test execution finished.")
    except Exception as e:
         print(f"An unexpected error occurred at the top level: {e}")
         import traceback
         traceback.print_exc()