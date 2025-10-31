#!/usr/bin/env python3
"""
Standalone test client for OpenWebUI API with multiple MCP servers.
Tests both streaming and non-streaming modes to identify MCP-related issues.

This tool reads configuration from .env and the personality prompt from file,
matching the behavior of the main bot.

Usage:
    python test_mcp_client.py "your test message here"
    python test_mcp_client.py "dame un trending gif" --stream
    python test_mcp_client.py "hello" --no-stream
"""

import asyncio
import aiohttp
import json
import sys
import os
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv

# Get script directory for relative path resolution
SCRIPT_DIR = Path(__file__).parent.resolve()

# Load environment variables from .env in the same directory as this script
# override=True ensures .env values take precedence over existing environment variables
env_path = SCRIPT_DIR / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
else:
    print(f"⚠️  Warning: .env file not found at {env_path}")
    print(f"   Using default values or environment variables")
    load_dotenv(override=True)  # Try to load from current directory or parent directories

# Configuration from environment (matching main.py pattern)
OPENWEBUI_API_URL = os.getenv("OPENWEBUI_API_URL", "http://localhost:8080")
OPENWEBUI_MODEL = os.getenv("OPENWEBUI_MODEL")
OPENWEBUI_API_KEY = os.getenv("OPENWEBUI_API_KEY")

# Read personality prompt from file (matching main.py pattern)
def load_prompt_from_file(file_path: str) -> Optional[str]:
    """Load prompt from file, matching main.py's load_prompt_from_file function."""
    try:
        # Try relative to script directory first
        abs_file_path = SCRIPT_DIR / file_path
        if not abs_file_path.exists():
            # Try absolute path
            abs_file_path = Path(file_path)
        
        with open(abs_file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            print(f"✅ Loaded prompt from: {abs_file_path}")
            return content
    except FileNotFoundError:
        print(f"❌ FATAL: Prompt file not found: {file_path}")
        print(f"   Tried: {abs_file_path}")
        return None
    except Exception as e:
        print(f"❌ FATAL: Error reading prompt file {file_path}: {e}")
        return None

# Load personality prompt
PERSONALITY_PROMPT_PATH = os.getenv("PERSONALITY_PROMPT_PATH", "../utils/prompts/personality_prompt.txt")
PERSONALITY_PROMPT = load_prompt_from_file(PERSONALITY_PROMPT_PATH)

# Parse MCP servers from LIST_TOOLS (matching main.py pattern)
LIST_TOOLS_STR = os.getenv("LIST_TOOLS", "")
MCP_SERVERS: List[str] = [tool.strip() for tool in LIST_TOOLS_STR.split(',') if tool.strip()] if LIST_TOOLS_STR else []


async def test_streaming_request(user_message: str, mcp_servers: List[str]) -> None:
    """Test OpenWebUI API with streaming enabled (works with multiple MCP servers)."""
    print("\n" + "="*80)
    print("TESTING STREAMING MODE (stream=True)")
    print("="*80)
    
    chat_endpoint = f"{OPENWEBUI_API_URL.rstrip('/')}/api/chat/completions"
    
    headers = {"Content-Type": "application/json"}
    if OPENWEBUI_API_KEY:
        headers["Authorization"] = f"Bearer {OPENWEBUI_API_KEY}"
    
    # Sort tool IDs for consistency (matching webui_api.py)
    sorted_mcp_servers = sorted(mcp_servers)
    
    payload = {
        "model": OPENWEBUI_MODEL,
        "messages": [
            {"role": "system", "content": PERSONALITY_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "tool_ids": sorted_mcp_servers,
        "stream": True
    }
    
    print(f"\n📤 Request URL: {chat_endpoint}")
    print(f"📤 Model: {OPENWEBUI_MODEL}")
    print(f"📤 MCP Servers: {sorted_mcp_servers}")
    print(f"📤 Message: {user_message}")
    print(f"\n📦 Full Payload:")
    print(json.dumps(payload, indent=2))
    
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(chat_endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                print(f"\n📥 Response Status: {response.status}")
                
                if response.status == 200:
                    print("\n✅ SUCCESS - Reading streaming response...")
                    print("-" * 80)
                    
                    full_content = ""
                    usage_data = None
                    chunk_count = 0
                    
                    async for line in response.content:
                        line_text = line.decode('utf-8').strip()
                        
                        if not line_text or line_text.startswith(':'):
                            continue
                        
                        if line_text.startswith('data: '):
                            data_part = line_text[6:]
                            
                            if data_part == '[DONE]':
                                print("\n[Stream ended]")
                                break
                            
                            try:
                                chunk = json.loads(data_part)
                                chunk_count += 1
                                
                                # Extract content from delta
                                if chunk.get("choices") and chunk["choices"]:
                                    delta = chunk["choices"][0].get("delta", {})
                                    if "content" in delta and delta["content"]:
                                        content_piece = delta["content"]
                                        full_content += content_piece
                                        print(content_piece, end='', flush=True)
                                
                                # Extract usage if present
                                if "usage" in chunk:
                                    usage_data = chunk["usage"]
                            
                            except json.JSONDecodeError as e:
                                print(f"\n⚠️  Could not parse chunk: {data_part[:100]}")
                    
                    print("\n" + "-" * 80)
                    print(f"\n📊 Statistics:")
                    print(f"   - Chunks received: {chunk_count}")
                    print(f"   - Total content length: {len(full_content)}")
                    if usage_data:
                        print(f"   - Usage data: {usage_data}")
                    
                    print(f"\n📝 Complete Response:")
                    print(full_content)
                    
                    # Try to parse as JSON
                    try:
                        parsed_json = json.loads(full_content)
                        print(f"\n✅ Valid JSON parsed:")
                        print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
                    except json.JSONDecodeError:
                        print(f"\n⚠️  Response is not valid JSON")
                
                else:
                    error_text = await response.text()
                    print(f"\n❌ FAILED - Status {response.status}")
                    print(f"Error response: {error_text[:500]}")
    
    except aiohttp.ClientConnectorError as e:
        print(f"\n❌ CONNECTION ERROR: {e}")
    except asyncio.TimeoutError:
        print(f"\n❌ TIMEOUT ERROR")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()


async def test_non_streaming_request(user_message: str, mcp_servers: List[str]) -> None:
    """Test OpenWebUI API with streaming disabled (may fail with multiple MCP servers)."""
    print("\n" + "="*80)
    print("TESTING NON-STREAMING MODE (stream=False)")
    print("="*80)
    
    chat_endpoint = f"{OPENWEBUI_API_URL.rstrip('/')}/api/chat/completions"
    
    headers = {"Content-Type": "application/json"}
    if OPENWEBUI_API_KEY:
        headers["Authorization"] = f"Bearer {OPENWEBUI_API_KEY}"
    
    # Sort tool IDs for consistency (matching webui_api.py)
    sorted_mcp_servers = sorted(mcp_servers)
    
    payload = {
        "model": OPENWEBUI_MODEL,
        "messages": [
            {"role": "system", "content": PERSONALITY_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "tool_ids": sorted_mcp_servers,
        "stream": False
    }
    
    print(f"\n📤 Request URL: {chat_endpoint}")
    print(f"📤 Model: {OPENWEBUI_MODEL}")
    print(f"📤 MCP Servers: {sorted_mcp_servers}")
    print(f"📤 Message: {user_message}")
    print(f"\n📦 Full Payload:")
    print(json.dumps(payload, indent=2))
    
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(chat_endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                print(f"\n📥 Response Status: {response.status}")
                
                response_text = await response.text()
                
                if response.status == 200:
                    print("\n✅ SUCCESS")
                    print("-" * 80)
                    
                    try:
                        data = json.loads(response_text)
                        print(f"📝 Full API Response:")
                        print(json.dumps(data, indent=2, ensure_ascii=False))
                        
                        # Extract the actual content
                        if data.get("choices") and data["choices"]:
                            message = data["choices"][0].get("message", {})
                            content = message.get("content", "")
                            
                            print(f"\n📝 Message Content:")
                            print(content)
                            
                            # Try to parse as JSON
                            try:
                                parsed_json = json.loads(content)
                                print(f"\n✅ Valid JSON parsed:")
                                print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
                            except json.JSONDecodeError:
                                print(f"\n⚠️  Content is not valid JSON")
                            
                            # Show usage
                            if data.get("usage"):
                                print(f"\n📊 Usage: {data['usage']}")
                    
                    except json.JSONDecodeError as e:
                        print(f"\n⚠️  Response is not valid JSON: {e}")
                        print(f"Raw response: {response_text[:500]}")
                
                else:
                    print(f"\n❌ FAILED - Status {response.status}")
                    print(f"Error response: {response_text[:500]}")
    
    except aiohttp.ClientConnectorError as e:
        print(f"\n❌ CONNECTION ERROR: {e}")
    except asyncio.TimeoutError:
        print(f"\n❌ TIMEOUT ERROR")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main test function."""
    # Validate configuration
    if not PERSONALITY_PROMPT:
        print("\n❌ FATAL: Could not load personality prompt. Check PERSONALITY_PROMPT_PATH in .env")
        sys.exit(1)
    
    if not OPENWEBUI_MODEL:
        print("\n❌ FATAL: OPENWEBUI_MODEL not set in .env")
        sys.exit(1)
    
    if not MCP_SERVERS:
        print("\n⚠️  WARNING: No MCP servers configured in LIST_TOOLS")
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python test_mcp_client.py <message> [--stream|--no-stream|--both|--test-all]")
        print("\nExamples:")
        print('  python test_mcp_client.py "dame un trending gif"')
        print('  python test_mcp_client.py "hello" --stream')
        print('  python test_mcp_client.py "hello" --no-stream')
        print('  python test_mcp_client.py "hello" --both')
        print('  python test_mcp_client.py "hello" --test-all  # Test with 0, 1, 2, and all MCP servers')
        sys.exit(1)
    
    user_message = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "--both"
    
    print("\n" + "="*80)
    print("OpenWebUI MCP Multi-Server Test Client")
    print("="*80)
    print(f"\n⚙️  Configuration:")
    print(f"   - Script directory: {SCRIPT_DIR}")
    print(f"   - .env file: {env_path}")
    print(f"   - API URL: {OPENWEBUI_API_URL}")
    print(f"   - Model: {OPENWEBUI_MODEL}")
    print(f"   - API Key: {'***' + OPENWEBUI_API_KEY[-4:] if OPENWEBUI_API_KEY else 'Not set'}")
    print(f"   - MCP Servers: {MCP_SERVERS}")
    print(f"   - Prompt file: {PERSONALITY_PROMPT_PATH}")
    print(f"   - Test message: {user_message}")
    print(f"   - Mode: {mode}")
    
    # Run tests based on mode
    if mode == "--test-all":
        # Test with different MCP server configurations
        print("\n" + "="*80)
        print("RUNNING COMPREHENSIVE MCP SERVER TESTS")
        print("="*80)
        
        # Test 1: No MCP servers
        print("\n\n" + "🧪 " + "="*77)
        print("TEST 1: NO MCP SERVERS (stream=True)")
        print("="*80)
        await asyncio.sleep(1)
        await test_streaming_request(user_message, [])
        
        await asyncio.sleep(2)
        
        # Test 2: First MCP server only
        if len(MCP_SERVERS) >= 1:
            print("\n\n" + "🧪 " + "="*77)
            print(f"TEST 2: SINGLE MCP SERVER ({MCP_SERVERS[0]}) (stream=True)")
            print("="*80)
            await asyncio.sleep(1)
            await test_streaming_request(user_message, [MCP_SERVERS[0]])
            
            await asyncio.sleep(2)
        
        # Test 3: Second MCP server only
        if len(MCP_SERVERS) >= 2:
            print("\n\n" + "🧪 " + "="*77)
            print(f"TEST 3: SINGLE MCP SERVER ({MCP_SERVERS[1]}) (stream=True)")
            print("="*80)
            await asyncio.sleep(1)
            await test_streaming_request(user_message, [MCP_SERVERS[1]])
            
            await asyncio.sleep(2)
        
        # Test 4: Both MCP servers with stream=False (expected to fail)
        if len(MCP_SERVERS) >= 2:
            print("\n\n" + "🧪 " + "="*77)
            print(f"TEST 4: MULTIPLE MCP SERVERS ({MCP_SERVERS}) (stream=False)")
            print("="*80)
            await asyncio.sleep(1)
            await test_non_streaming_request(user_message, MCP_SERVERS)
            
            await asyncio.sleep(2)
        
        # Test 5: Both MCP servers with stream=True (should work)
        if len(MCP_SERVERS) >= 2:
            print("\n\n" + "🧪 " + "="*77)
            print(f"TEST 5: MULTIPLE MCP SERVERS ({MCP_SERVERS}) (stream=True)")
            print("="*80)
            await asyncio.sleep(1)
            await test_streaming_request(user_message, MCP_SERVERS)
    
    elif mode == "--stream":
        await test_streaming_request(user_message, MCP_SERVERS)
    elif mode == "--no-stream":
        await test_non_streaming_request(user_message, MCP_SERVERS)
    else:  # --both or default
        await test_non_streaming_request(user_message, MCP_SERVERS)
        await asyncio.sleep(2)  # Brief pause between tests
        await test_streaming_request(user_message, MCP_SERVERS)
    
    print("\n" + "="*80)
    print("Test completed")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
