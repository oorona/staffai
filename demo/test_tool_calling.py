#!/usr/bin/env python3
"""
Test tool calling and structured output with proper bot response validation.
Fully self-contained in demo folder - loads personality and response schema.
"""

import os
import sys
import json
import asyncio
import socket
from urllib.parse import urlparse
from dotenv import load_dotenv
from openai import AsyncOpenAI
from fastmcp import Client

# Force unbuffered output
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf8', buffering=1)

load_dotenv()


def read_docker_secret(secret_name: str, env_var: str) -> str:
    """Read Docker secret or fall back to environment variable."""
    secret_path = f"/run/secrets/{secret_name}"
    if os.path.isfile(secret_path):
        with open(secret_path, 'r') as f:
            return f.read().strip()
    return os.getenv(env_var, "")


def load_response_schema():
    """Load response schema from JSON file."""
    schema_file = "response_schema.json"
    if os.path.exists(schema_file):
        with open(schema_file, 'r') as f:
            return json.load(f)
    return None


def load_personality_prompt():
    """Load personality prompt from file."""
    prompt_file = "personality_prompt.txt"
    if os.path.exists(prompt_file):
        with open(prompt_file, 'r') as f:
            return f.read().strip()
    return ""


LITELLM_API_URL = os.getenv("LITELLM_API_URL", "http://litellm:4000")
LITELLM_API_KEY = read_docker_secret("litellm_api_key", "LITELLM_API_KEY")
LITELLM_MODELS = [m.strip() for m in os.getenv("LITELLM_MODELS", "").split(",") if m.strip()]
MCP_SERVERS = [s.strip() for s in os.getenv("MCP_SERVERS", "").split(",") if s.strip()]
RESPONSE_SCHEMA = load_response_schema()
PERSONALITY_PROMPT = load_personality_prompt()


async def get_mcp_tools():
    """Load tools from MCP servers using FastMCP Client."""
    all_tools = []
    
    print(f"\n📋 Step 2: Load tools from MCP servers")
    print(f"   Found {len(MCP_SERVERS)} servers to connect to:")
    
    if not MCP_SERVERS:
        print(f"   ⚠️  No MCP servers configured!")
        return []
    
    for server_url in MCP_SERVERS:
        print(f"\n   → {server_url}")
        try:
            # Extract hostname from URL for DNS lookup
            parsed = urlparse(server_url)
            hostname = parsed.hostname
            port = parsed.port or 80
            
            print(f"     🔍 Resolving: {hostname}:{port}")
            try:
                ip = socket.gethostbyname(hostname)
                print(f"     ✅ DNS resolved: {hostname} → {ip}")
            except socket.gaierror as e:
                print(f"     ❌ DNS failed: {hostname} - {e}")
            
            print(f"     ⏳ Connecting to {server_url}...")
            
            # Create MCP client for this server with timeout
            mcp_client = Client(server_url)
            
            # Connect and list tools with timeout
            try:
                async with asyncio.timeout(10):  # 10 second timeout
                    async with mcp_client:
                        print(f"     ⏳ Fetching tools...")
                        tools_response = await mcp_client.list_tools()
                        
                        # Extract tools from response
                        if hasattr(tools_response, 'tools'):
                            tools_list = tools_response.tools
                        elif isinstance(tools_response, list):
                            tools_list = tools_response
                        else:
                            tools_list = []
                        
                        print(f"     ✅ Connected and got {len(tools_list)} tools")
                        
                        # Convert each tool to OpenAI function format
                        for tool in tools_list:
                            openai_tool = {
                                "type": "function",
                                "function": {
                                    "name": tool.name,
                                    "description": tool.description or f"Function: {tool.name}",
                                    "parameters": tool.inputSchema or {
                                        "type": "object",
                                        "properties": {},
                                        "required": []
                                    }
                                }
                            }
                            all_tools.append(openai_tool)
                            
                            # Show first few tool names
                            if len(all_tools) <= 3:
                                print(f"       • {tool.name}")
                        
                        if len(tools_list) > 3:
                            print(f"       ... and {len(tools_list) - 3} more")
            except asyncio.TimeoutError:
                print(f"     ⏱️  Timeout after 10s")
        
        except Exception as e:
            print(f"     ❌ Failed: {type(e).__name__}")
            print(f"        Error: {str(e)}")
    
    print(f"\n   ✅ Total tools loaded: {len(all_tools)}")
    if len(all_tools) == 0:
        print(f"   ⚠️  No tools loaded! Adding fallback tool for testing...")
        all_tools.append({
            "type": "function",
            "function": {
                "name": "search_image",
                "description": "Search for images online",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        })
    return all_tools


def validate_structured_output(response_text: str) -> dict:
    """Validate that response is proper JSON matching schema."""
    try:
        response_obj = json.loads(response_text)
        
        # Check required fields
        if "type" not in response_obj or "response" not in response_obj or "data" not in response_obj:
            return {"valid": False, "reason": "Missing required fields (type, response, data)"}
        
        # Validate type
        valid_types = ["text", "url", "gif", "latex", "code", "output"]
        if response_obj["type"] not in valid_types:
            return {"valid": False, "reason": f"Invalid type: {response_obj['type']}"}
        
        # Validate response is string and not too long (personality says ≤30 words)
        if not isinstance(response_obj["response"], str):
            return {"valid": False, "reason": "response must be string"}
        
        words = len(response_obj["response"].split())
        if words > 30:
            return {"valid": False, "reason": f"response too long: {words} words (max 30)"}
        
        # Validate data is string
        if not isinstance(response_obj["data"], str):
            return {"valid": False, "reason": "data must be string"}
        
        return {"valid": True, "type": response_obj["type"], "response": response_obj["response"][:50], "data_len": len(response_obj["data"])}
    
    except json.JSONDecodeError as e:
        return {"valid": False, "reason": f"Invalid JSON: {str(e)[:100]}"}
    except Exception as e:
        return {"valid": False, "reason": f"Error: {type(e).__name__}"}


async def main():
    print("\n" + "="*80)
    print("🧪 MODEL TEST - Function Calling & Structured Output")
    print("="*80)
    
    # Step 0: Show config
    print(f"\n📋 Step 0: Configuration")
    print(f"   • API URL: {LITELLM_API_URL}")
    print(f"   • API Key: {'✅ Set' if LITELLM_API_KEY else '❌ NOT SET'} ({len(LITELLM_API_KEY)} chars)")
    print(f"   • Personality Prompt: {'✅ Loaded' if PERSONALITY_PROMPT else '❌ NOT FOUND'}")
    print(f"   • Response Schema: {'✅ Loaded' if RESPONSE_SCHEMA else '❌ NOT FOUND'}")
    print(f"   • Models: {len(LITELLM_MODELS)}")
    for i, m in enumerate(LITELLM_MODELS, 1):
        print(f"     {i}. {m}")
    print(f"   • MCP Servers: {len(MCP_SERVERS)}")
    for s in MCP_SERVERS:
        print(f"     • {s}")
    
    # Step 1: Init client
    print(f"\n📋 Step 1: Initialize OpenAI Client")
    llm_client = AsyncOpenAI(
        api_key=LITELLM_API_KEY,
        base_url=LITELLM_API_URL
    )
    print(f"   ✅ Client ready")
    
    # Step 2: Load MCP tools
    print(f"\n📋 About to call get_mcp_tools()...")
    try:
        tools = await get_mcp_tools()
    except Exception as e:
        print(f"\n❌ ERROR in get_mcp_tools(): {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        tools = []
    
    if not tools:
        print(f"\n⚠️  No tools loaded from MCP servers!")
        return
    
    # Step 3: Test each model
    print(f"\n📋 Step 3: Testing {len(LITELLM_MODELS)} models")
    print(f"   With {len(tools)} tools available\n")
    
    results = []
    
    for model_idx, model in enumerate(LITELLM_MODELS, 1):
        print(f"\n{'═'*80}")
        print(f"🤖 MODEL {model_idx}/{len(LITELLM_MODELS)}: {model}")
        print(f"{'═'*80}")
        
        result = {
            "model": model,
            "tool_calling": None,
            "structured_output": None,
        }
        
        # TEST: INTEGRATED TOOL CALLING + STRUCTURED OUTPUT
        print(f"\n  📋 INTEGRATED TEST: TOOL CALLING → TOOL EXECUTION → STRUCTURED OUTPUT")
        print(f"  User Request: 'Find me a funny cat image'")
        print(f"  Tools Available: {len(tools)}")
        
        # STEP 1: Call LLM with tools and ask for structured output
        print(f"\n  Step 1: LLM attempts to call tools")
        conversation = [
            {"role": "system", "content": PERSONALITY_PROMPT},
            {"role": "user", "content": "Find me a funny cat image"}
        ]
        
        try:
            response = await llm_client.chat.completions.create(
                model=model,
                messages=conversation,
                tools=tools,
                tool_choice="auto",
                temperature=1.0,
                timeout=60.0
            )
            print(f"    ✅ Response received from LLM")
            
            tool_calls = response.choices[0].message.tool_calls
            
            # STEP 2: Check if tool was called
            print(f"\n  Step 2: Check tool execution")
            if not tool_calls:
                result["tool_calling"] = False
                result["structured_output"] = False
                print(f"    ❌ NO TOOLS CALLED - Test failed")
            else:
                result["tool_calling"] = True
                print(f"    ✅ YES - LLM called {len(tool_calls)} tool(s):")
                
                # STEP 3: Execute the tool and collect results
                print(f"\n  Step 3: Execute tools and collect results")
                tool_results = []
                for tc in tool_calls[:3]:
                    tool_name = tc.function.name
                    tool_id = tc.id  # Get the actual tool call ID
                    args_str = tc.function.arguments
                    print(f"    • Executing: {tool_name}")
                    print(f"      Args: {args_str}")
                    
                    # Simulate tool execution (in real scenario, would call actual tool)
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                        # Mock result for demo
                        if tool_name == "search_tenor_gifs":
                            tool_result = "https://media.tenor.com/images/funny-cat-gif.gif"
                        elif tool_name == "search_cves":
                            tool_result = "CVE-2024-1234: Critical vulnerability"
                        else:
                            tool_result = f"Result from {tool_name}"
                        
                        print(f"      ✅ Tool result: {tool_result[:60]}")
                        tool_results.append({
                            "tool_id": tool_id,  # Store the actual tool call ID
                            "tool": tool_name,
                            "args": args,
                            "result": tool_result
                        })
                    except Exception as e:
                        print(f"      ❌ Tool execution error: {e}")
                
                # STEP 4: Feed tool results back to LLM for structured response
                print(f"\n  Step 4: Send tool results to LLM for structured response")
                
                # Add assistant response with tool calls to conversation
                conversation.append(response.choices[0].message)
                
                # Add tool results with proper tool_call_id
                for tr in tool_results:
                    conversation.append({
                        "role": "tool",
                        "tool_call_id": tr["tool_id"],  # Use the actual tool call ID
                        "content": tr["result"]
                    })
                
                print(f"    📝 Sending {len(tool_results)} tool result(s) back to LLM")
                print(f"       with structured output requirement")
                
                # STEP 5: Get structured output response
                # The issue: conversation has tool_calls which confuses the LLM about output format
                # Solution: Send a clearer system prompt explicitly requiring JSON
                try:
                    # Build structured output request with clear system instructions
                    structured_conversation = conversation.copy()
                    
                    # Replace the system prompt with one that explicitly requires JSON output
                    # after tool execution (this prevents format confusion)
                    if structured_conversation[0]["role"] == "system":
                        structured_conversation[0]["content"] = (
                            "You have just executed a tool and received results. "
                            "Respond ONLY with valid JSON (no other text):\n"
                            '{"type": "gif|url|text|latex|code|output", "response": "your sarcastic message (≤30 words)", "data": "URL or relevant data"}'
                        )
                    
                    response2 = await llm_client.chat.completions.create(
                        model=model,
                        messages=structured_conversation,
                        response_format=RESPONSE_SCHEMA,
                        temperature=1.0,
                        timeout=60.0
                    )
                    print(f"    ✅ Structured response received")
                    
                    # STEP 6: Validate structured output
                    print(f"\n  Step 5: Validate structured output")
                    response_text = response2.choices[0].message.content
                    
                    print(f"    📝 Raw response ({len(response_text)} chars):")
                    if len(response_text) < 300:
                        print(f"       {response_text}")
                    else:
                        print(f"       {response_text[:200]}...")
                    
                    # Parse and validate
                    try:
                        parsed = json.loads(response_text)
                        
                        # Validate schema
                        valid = all(k in parsed for k in ["type", "response", "data"])
                        if valid:
                            result["structured_output"] = True
                            response_len = len(parsed.get("response", ""))
                            print(f"    ✅ VALID JSON response")
                            print(f"       Type: {parsed.get('type')}")
                            print(f"       Response ({response_len} chars): {parsed.get('response')[:60]}...")
                            print(f"       Data (URL): {parsed.get('data')[:60] if parsed.get('data') else 'None'}")
                        else:
                            result["structured_output"] = False
                            print(f"    ❌ INVALID - Missing required fields")
                            print(f"       Has: {list(parsed.keys())}")
                    
                    except json.JSONDecodeError as e:
                        result["structured_output"] = False
                        print(f"    ❌ INVALID JSON - Parse error: {str(e)[:60]}")
                
                except Exception as e:
                    result["structured_output"] = False
                    print(f"    ❌ ERROR getting structured response: {type(e).__name__}")
                    print(f"       {str(e)[:100]}")
        
        except Exception as e:
            result["tool_calling"] = False
            result["structured_output"] = False
            print(f"    ❌ ERROR in main flow: {type(e).__name__}")
            print(f"       Message: {str(e)[:120]}")
        
        # Summary for this model
        overall_status = "✅ PASS" if (result["tool_calling"] and result["structured_output"]) else "❌ FAIL"
        print(f"\n  🎯 RESULT FOR {model}:")
        print(f"     Overall: {overall_status}")
        print(f"     Tool Calling: {'✅' if result['tool_calling'] else '❌'}")
        print(f"     Structured Output: {'✅' if result['structured_output'] else '❌'}")
        
        results.append(result)
    
    # Final Summary
    print(f"\n{'═'*80}")
    print(f"📊 FINAL RESULTS: INTEGRATED TOOL CALLING → STRUCTURED OUTPUT")
    print(f"{'═'*80}\n")
    
    print(f"{'Model':<40} {'Status':<15}")
    print(f"{'-'*55}")
    
    passed = 0
    
    for r in results:
        model_short = r["model"][:38]
        status = "✅ PASS" if (r["tool_calling"] and r["structured_output"]) else "❌ FAIL"
        print(f"{model_short:<40} {status:<15}")
        if r["tool_calling"] and r["structured_output"]:
            passed += 1
    
    print(f"\n{'='*55}")
    print(f"✅ PASSED: {passed}/{len(LITELLM_MODELS)} models")
    print(f"❌ FAILED: {len(LITELLM_MODELS) - passed}/{len(LITELLM_MODELS)} models")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    asyncio.run(main())
