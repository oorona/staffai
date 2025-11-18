"""
LiteLLM Client for Discord Bot - Conversation with Context & Structured Output

Based on toolcallingdemo/litellm_client.py but simplified for Discord bot needs:
- Conversation context management with Redis
- Structured output (JSON schema for type/response/data)
- MCP tool calling support
- Context decay (time-based and message count)

⚠️ CRITICAL: MCP Tool Calling Requirements (READ BEFORE MODIFYING)
==================================================================

This implementation MUST follow these 5 rules to work correctly across all models:

1. tool_choice="auto" - ALWAYS set explicitly when passing tools
2. Fresh conversations - NO history when tools available (2 messages only)
3. Raw message objects - Use response.choices[0].message, NOT manual dicts
4. No max_tokens - Let models use their defaults (avoid truncation)
5. timeout=60.0 - Always set explicit timeout

Breaking ANY of these causes:
- Tools not being called (missing tool_choice or history confusion)
- Empty responses (manual dicts or max_tokens truncation)
- finish_reason: length (max_tokens too low)

Reference implementation: demo/test_tool_calling.py (ALWAYS WORKS)
Full documentation: TOOL_CALLING_CHECKLIST.md

If tool calling breaks: Run demo first. If demo works, YOU broke one of the 5 rules.
"""

import logging
import json
import time
import asyncio
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
import redis
from pathlib import Path

logger = logging.getLogger(__name__)

class LiteLLMClient:
    """Client for LiteLLM proxy with conversation context and structured output"""
    
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        redis_client: redis.Redis,
        response_schema_path: Optional[str] = None,
        context_history_ttl_seconds: int = 1800,  # 30 minutes default
        context_message_max_age_seconds: int = 1800,  # 30 minutes default
        max_history_messages: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        mcp_servers: Optional[List[str]] = None  # NEW: Optional MCP server override
    ):
        """
        Initialize LiteLLM client for Discord bot conversations.
        
        Args:
            model: LLM model name (e.g., "gpt-4", "gemini/gemini-2.5-flash")
            base_url: LiteLLM proxy URL
            api_key: API key for LiteLLM proxy
            redis_client: Redis client for context storage
            response_schema_path: Path to response JSON schema file (default: ./response_schema.json)
            context_history_ttl_seconds: How long to keep conversation history in Redis (0 = no expiry)
            context_message_max_age_seconds: Max age of individual messages (0 = no age limit)
            max_history_messages: Max number of messages to keep per user/channel
            temperature: LLM temperature (will be overridden to 1.0 for GPT-5)
            max_tokens: Max tokens for completion
            mcp_servers: Optional list of MCP server URLs to use (overrides bot default)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.redis_client = redis_client
        self.context_history_ttl_seconds = context_history_ttl_seconds
        self.context_message_max_age_seconds = context_message_max_age_seconds
        self.max_history_messages = max_history_messages
        self.mcp_servers_override = mcp_servers  # Store MCP override
        
        # Ensure URL ends with /v1
        proxy_url = base_url.rstrip('/')
        if not proxy_url.endswith('/v1'):
            proxy_url = f"{proxy_url}/v1"
        
        logger.info(f"LiteLLM base_url: {base_url} -> proxy_url: {proxy_url}")
        logger.info(f"API key configured: {api_key[:10]}..." if api_key else "No API key")
        
        self.client = AsyncOpenAI(
            base_url=proxy_url,
            api_key=api_key
        )
        
        # Load response schema for structured output
        if response_schema_path is None:
            # Default to response_schema.json in project root
            response_schema_path = Path(__file__).parent.parent / "response_schema.json"
        
        with open(response_schema_path, 'r', encoding='utf-8') as f:
            self.response_schema = json.load(f)
        
        logger.info(f"Initialized LiteLLM client with model: {model}")
        if mcp_servers is not None:
            logger.info(f"MCP servers override: {len(mcp_servers)} servers")
        logger.info(f"Context TTL: {context_history_ttl_seconds}s, Message max age: {context_message_max_age_seconds}s")
        
        # Cache for MCP tools (cached indefinitely - use manual reload only)
        self._mcp_tools_cache: Optional[List[Dict[str, Any]]] = None
        self._mcp_tools_cache_time: float = 0
        self._mcp_failed_servers: set = set()  # Track servers that failed to load
        self._tool_to_server_map: Dict[str, str] = {}  # Map tool names to server URLs
        
        # Control tool execution detail output
        self.show_tool_details: bool = False  # Set to True to show detailed tool execution
    
    def _display_tools_by_server(self, tools_by_server: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Display tools grouped by MCP server in a pretty format.
        
        Args:
            tools_by_server: Dictionary mapping server URLs to their tools
        """
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich import box
            
            console = Console()
            
            # Create main panel
            main_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
            main_table.add_column("Server", style="cyan bold")
            main_table.add_column("Tools", style="white")
            
            total_tools = 0
            for server_url, tools in tools_by_server.items():
                # Get server name (remove protocol and /mcp)
                server_name = server_url.replace("https://", "").replace("http://", "").replace("/mcp", "")
                
                # Create tools list
                tool_names = []
                for tool in tools:
                    name = tool.get("function", {}).get("name", "unknown")
                    desc = tool.get("function", {}).get("description", "")
                    if desc:
                        # Truncate long descriptions
                        desc = desc[:50] + "..." if len(desc) > 50 else desc
                        tool_names.append(f"• {name} - {desc}")
                    else:
                        tool_names.append(f"• {name}")
                
                # Add to main table
                tools_display = "\n".join(tool_names) if tool_names else "No tools"
                main_table.add_row(f"{server_name}\n({len(tools)} tools)", tools_display)
                total_tools += len(tools)
            
            # Display the panel
            panel = Panel(
                main_table,
                title=f"🔧 MCP Tools Summary ({total_tools} total)",
                border_style="green",
                box=box.ROUNDED
            )
            console.print(panel)
            
        except ImportError:
            # Fallback to simple text display if rich not available
            print(f"\n🔧 MCP Tools Summary ({sum(len(tools) for tools in tools_by_server.values())} total):")
            for server_url, tools in tools_by_server.items():
                server_name = server_url.replace("https://", "").replace("http://", "").replace("/mcp", "")
                print(f"\n  📡 {server_name} ({len(tools)} tools):")
                for tool in tools:
                    name = tool.get("function", {}).get("name", "unknown")
                    desc = tool.get("function", {}).get("description", "")
                    if desc:
                        desc = desc[:60] + "..." if len(desc) > 60 else desc
                        print(f"    • {name} - {desc}")
                    else:
                        print(f"    • {name}")
        except Exception as e:
            logger.debug(f"Error displaying tools: {e}")
    
    def clear_failed_servers(self) -> None:
        """
        Manually clear the failed servers cache to force retry on next tool fetch.
        Useful for debugging or when you know servers are back online.
        """
        if self._mcp_failed_servers:
            logger.info(f"🔄 Manually clearing {len(self._mcp_failed_servers)} failed servers from cache")
            self._mcp_failed_servers.clear()
        else:
            logger.info("No failed servers to clear")
    
    async def get_mcp_tools(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch available tools from configured MCP servers using FastMCP Client.
        Returns list of tool definitions in OpenAI function calling format.
        Results are cached indefinitely - use reload_mcp_tools() to manually refresh.
        """
        logger.debug(f"🔍 get_mcp_tools() called - checking cache status...")
        logger.debug(f"🔍 mcp_servers_override: {self.mcp_servers_override}")
        logger.debug(f"🔍 _mcp_tools_cache exists: {bool(self._mcp_tools_cache)}")
        if self._mcp_tools_cache:
            logger.debug(f"🔍 _mcp_tools_cache length: {len(self._mcp_tools_cache)}")
        
        if not self.mcp_servers_override:
            # If no servers configured but we have cached tools, return them
            if self._mcp_tools_cache:
                logger.info(f"✅ Using injected cached MCP tools ({len(self._mcp_tools_cache)} tools)")
                return self._mcp_tools_cache
            logger.warning(f"❌ No MCP servers configured and no cached tools")
            return []

        # Check cache - return cached tools if available (no auto-expiry)
        if self._mcp_tools_cache:
            logger.info(f"✅ Using cached MCP tools ({len(self._mcp_tools_cache)} tools)")
            return self._mcp_tools_cache

        # No cache - fetch tools for first time
        logger.info(f"🔧 No cache found - fetching MCP tools for first time")
        
        # Fetch tools from all MCP servers using FastMCP Client
        all_tools = []
        tools_by_server = {}  # Track which tools come from which server
        failed_servers = set()
        
        try:
            from fastmcp import Client

            # Attempt to load from all configured servers
            logger.info(f"🔧 Attempting to load tools from {len(self.mcp_servers_override)} servers...")

            for server_url in self.mcp_servers_override:
                server_start_time = time.time()
                try:
                    logger.info(f"🔧 Fetching tools from MCP server: {server_url}")
                    
                    # Create FastMCP client - back to working simple approach
                    from fastmcp import Client
                    client = Client(server_url)
                    
                    # Shorter timeout to fail fast (10s instead of 60s)
                    async with asyncio.timeout(10):  # 10 second timeout
                        async with client:
                            logger.debug(f"Connected to {server_url}, calling list_tools()")
                            
                            # List available tools
                            tools_response = await client.list_tools()
                            
                            elapsed = time.time() - server_start_time
                            logger.debug(f"Received tools_response in {elapsed:.2f}s, type: {type(tools_response)}")
                            
                            # Handle both response formats (server-dependent)
                            if isinstance(tools_response, dict) and "tools" in tools_response:
                                tools_list = tools_response["tools"]
                            elif isinstance(tools_response, list):
                                tools_list = tools_response
                            elif hasattr(tools_response, 'tools'):
                                tools_list = tools_response.tools
                            else:
                                logger.warning(f"Unexpected tools response format from {server_url}: {type(tools_response)}")
                                continue
                            
                            logger.debug(f"Extracted {len(tools_list)} tools from response")
                            
                            # Track tools from this server
                            server_tools = []
                            
                            # Convert FastMCP tool format to OpenAI function calling format
                            for tool in tools_list:
                                tool_name = getattr(tool, "name", str(tool))
                                tool_def = {
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "description": getattr(tool, "description", "") or "",
                                        "parameters": getattr(tool, "inputSchema", None) or {
                                            "type": "object",
                                            "properties": {},
                                            "required": []
                                        }
                                    }
                                }
                                all_tools.append(tool_def)
                                server_tools.append(tool_def)

                                # Map this tool to its server
                                self._tool_to_server_map[tool_name] = server_url
                            
                            # Store tools by server for pretty display
                            tools_by_server[server_url] = server_tools
                            
                            elapsed = time.time() - server_start_time
                            logger.info(f"✅ Loaded {len(tools_list)} tools from {server_url} ({elapsed:.2f}s)")
                            print(f"  ✅ {server_url}: {len(tools_list)} tools ({elapsed:.1f}s)")
                        
                except asyncio.TimeoutError:
                    elapsed = time.time() - server_start_time
                    error_msg = f"Connection timeout after {elapsed:.1f}s"
                    logger.debug(f"❌ {server_url}: {error_msg}")  # Debug level, not error
                    print(f"  ❌ {server_url}: {error_msg}")
                    failed_servers.add(server_url)
                    continue
                    
                except Exception as e:
                    elapsed = time.time() - server_start_time
                    error_type = type(e).__name__
                    error_msg = str(e)
                    
                    # Completely suppress all HTTP/connection stack traces - just log the essential info
                    logger.debug(f"❌ Network error for {server_url}: {error_type}")
                    
                    # Simplify common errors for display
                    if "peer closed connection" in error_msg:
                        simplified_error = "Connection closed unexpectedly"
                    elif "Connection refused" in error_msg:
                        simplified_error = "Connection refused"
                    elif "timeout" in error_msg.lower():
                        simplified_error = "Timeout"
                    elif "SSE stream" in error_msg:
                        simplified_error = "SSE stream error"
                    elif "RemoteProtocolError" in error_type:
                        simplified_error = "Connection closed unexpectedly"
                    elif "httpx" in error_type.lower() or "httpcore" in error_type.lower():
                        simplified_error = "Connection error"
                    else:
                        simplified_error = f"{error_type}: {error_msg[:30]}..."
                    
                    print(f"  ❌ {server_url}: {simplified_error} ({elapsed:.1f}s)")
                    failed_servers.add(server_url)
                    continue
            
            # Track failed servers
            if failed_servers:
                self._mcp_failed_servers.update(failed_servers)
                logger.debug(f"Failed servers: {len(failed_servers)} servers failed to load")

            # Cache the results
            if all_tools:
                self._mcp_tools_cache = all_tools
                self._mcp_tools_cache_time = time.time()
                successful_servers = len(self.mcp_servers_override) - len(failed_servers)
                total_servers = len(self.mcp_servers_override)

                status_parts = []
                if successful_servers > 0:
                    status_parts.append(f"{successful_servers} successful")
                if len(failed_servers) > 0:
                    status_parts.append(f"{len(failed_servers)} failed")

                status_summary = " + ".join(status_parts) + f" = {total_servers} total"
                logger.info(f"Cached {len(all_tools)} total MCP tools from {successful_servers}/{total_servers} servers ({status_summary})")

                # Pretty display of tools grouped by server
                self._display_tools_by_server(tools_by_server)
                return all_tools
            else:
                logger.warning(f"No MCP tools loaded from {len(self.mcp_servers_override)} servers")
                logger.warning(f"⚠️  Tool calling will be skipped - using direct LLM response only")
                
                # Return empty list instead of None so tool calling logic still runs
                # but with no tools (which will go straight to structured output)
                return []
                
        except ImportError:
            logger.error("fastmcp not installed - cannot fetch MCP tools")
            return []  # Return empty list instead of None
        except Exception as e:
            logger.error(f"Error fetching MCP tools: {e}")  # Removed exc_info=True
            return []  # Return empty list instead of None
    
    async def execute_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute an MCP tool call by finding the right server and calling it.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments as dict

        Returns:
            Tool result as JSON string
        """
        if not self.mcp_servers_override:
            logger.error("No MCP servers configured")
            return json.dumps({"error": "No MCP servers configured"})

        try:
            from fastmcp import Client

            # Check if we have a mapping for this tool
            if tool_name in self._tool_to_server_map:
                server_url = self._tool_to_server_map[tool_name]
                logger.debug(f"Using mapped server for {tool_name}: {server_url}")

                # Check if the server is in failed list
                if server_url in self._mcp_failed_servers:
                    logger.warning(f"Tool {tool_name} is mapped to failed server {server_url}")
                    return json.dumps({"error": f"Server for {tool_name} is currently unavailable"})

                try:
                    logger.debug(f"Calling {tool_name} on mapped server {server_url}")
                    client = Client(server_url)

                    # Add timeout for tool execution (30 seconds)
                    async with asyncio.timeout(30):
                        async with client:
                            # Call the tool
                            result = await client.call_tool(tool_name, arguments)
                        
                        # Extract result content (same pattern as working client)
                        if hasattr(result, 'content'):
                            if isinstance(result.content, list) and len(result.content) > 0:
                                content = result.content[0]
                                if hasattr(content, 'text'):
                                    result_text = content.text
                                else:
                                    result_text = str(content)
                            else:
                                result_text = str(result.content)
                        else:
                            result_text = str(result)
                        
                        logger.info(f"✅ Tool {tool_name} executed successfully on {server_url}")
                        return result_text

                except asyncio.TimeoutError:
                    logger.warning(f"⏱️ Tool {tool_name} timeout on {server_url} after 30s")
                    return json.dumps({"error": f"Tool execution timeout after 30 seconds"})

                except Exception as e:
                    # Error calling tool on mapped server
                    logger.error(f"Tool {tool_name} failed on mapped server {server_url}: {e}")
                    return json.dumps({"error": f"Tool execution failed: {str(e)}"})
            else:
                # No mapping found - tool not in cache (shouldn't happen if cache is working)
                logger.warning(f"No server mapping found for tool {tool_name} - tool may not be available")
                error_msg = f"Tool {tool_name} not found in server mappings"
                return json.dumps({"error": error_msg})
            
        except Exception as e:
            logger.error(f"Failed to execute MCP tool {tool_name}: {e}")  # Removed exc_info=True
            return json.dumps({"error": str(e)})
    
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        use_structured_output: bool = True,
        enable_caching: bool = True,
        track_calls: bool = False
    ) -> Any:
        """
        Call LiteLLM proxy for chat completion with optional tools and structured output.
        
        Args:
            messages: List of chat messages (system, user, assistant)
            tools: Optional list of MCP tools in OpenAI function calling format
            use_structured_output: Whether to use structured output (response_format)
            enable_caching: Whether to enable prompt caching (for supported models)
            track_calls: Whether to track and return metadata about LLM calls
        
        Returns:
            If track_calls=False: OpenAI ChatCompletion response object
            If track_calls=True: Tuple of (response, call_metadata_list)
        """
        call_metadata = []
        
        try:
            # Apply caching markers for supported models
            if enable_caching:
                messages = self._add_cache_control(messages)
            
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": 1.0 if "gpt-5" in self.model.lower() else self.temperature,
                "timeout": 60.0  # Match demo's explicit timeout
                # NOTE: max_tokens removed to match demo - let model use its default
            }
            
            # Add structured output (ONLY if not using tools - can't use both)
            if use_structured_output and not tools:
                kwargs["response_format"] = self.response_schema
                logger.debug("Using structured output (response_format)")
            
            # Add tools if provided (structured output will be disabled)
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"  # CRITICAL: Must explicitly set tool_choice like the demo
                logger.info(f"Calling {self.model} with {len(tools)} tools available")
                if use_structured_output:
                    logger.debug("Using tools for first call (structured output will be enforced later if needed)")
                
                # DEBUG: Show what we're actually sending
                logger.debug(f"🔍 API Call Debug:")
                logger.debug(f"   - model: {kwargs['model']}")
                logger.debug(f"   - temperature: {kwargs['temperature']}")
                logger.debug(f"   - max_tokens: {kwargs.get('max_tokens', 'NOT SET (using model default)')}")
                logger.debug(f"   - has 'tools' key: {'tools' in kwargs}")
                logger.debug(f"   - has 'tool_choice' key: {'tool_choice' in kwargs}")
                logger.debug(f"   - has 'response_format' key: {'response_format' in kwargs}")
                logger.debug(f"   - messages count: {len(kwargs['messages'])}")
                logger.debug(f"   - First tool name: {kwargs['tools'][0]['function']['name'] if kwargs['tools'] else 'NONE'}")
            
            # Log the LLM call details
            if tools:
                logger.info(f"🤖 Calling {self.model} with {len(tools)} tools for tool selection")
                logger.info(f"🔧 Conversation context: {len(messages)} messages")
                if self.show_tool_details:
                    tool_names = [t['function']['name'] for t in tools]
                    logger.info(f"🔧 Available tools: {', '.join(tool_names)}")
                    # Show what the user asked for (last user message)
                    user_messages = [msg for msg in messages if msg.get('role') == 'user']
                    if user_messages:
                        last_user_msg = user_messages[-1].get('content', '')
                        if len(last_user_msg) > 100:
                            last_user_msg = last_user_msg[:100] + "..."
                        logger.info(f"🗣️  User request: \"{last_user_msg}\"")
            else:
                logger.info(f"🤖 Calling {self.model} for direct response (no tools)")
            
            # CRITICAL DEBUG: Dump the exact kwargs being sent to LiteLLM
            logger.debug(f"🔍 FULL KWARGS DUMP:")
            logger.debug(f"   - Keys: {list(kwargs.keys())}")
            logger.debug(f"   - model: {kwargs.get('model')}")
            logger.debug(f"   - temperature: {kwargs.get('temperature')}")
            logger.debug(f"   - max_tokens: {kwargs.get('max_tokens', 'NOT SET (using model default)')}")
            if 'tools' in kwargs:
                logger.debug(f"   - tools: {len(kwargs['tools'])} tools present")
                logger.debug(f"     - First tool: {kwargs['tools'][0]['function']['name']}")
                logger.debug(f"     - Tool format: {kwargs['tools'][0].keys()}")
            if 'response_format' in kwargs:
                logger.debug(f"   - response_format: {list(kwargs['response_format'].keys())}")
            logger.debug(f"   - messages: {len(kwargs['messages'])} messages")
            logger.debug(f"     - System prompt length: {len(kwargs['messages'][0]['content']) if kwargs['messages'][0]['role'] == 'system' else 'N/A'}")
            
            # Make initial API call
            call_start = time.time()
            response = await self.client.chat.completions.create(**kwargs)
            call_duration = time.time() - call_start
            
            # DEBUG: Log raw response details
            message = response.choices[0].message
            logger.debug(f"🔍 Raw LLM Response Debug:")
            logger.debug(f"   - finish_reason: {response.choices[0].finish_reason}")
            logger.debug(f"   - message type: {type(message)}")
            logger.debug(f"   - message.__dict__ keys: {list(message.__dict__.keys()) if hasattr(message, '__dict__') else 'N/A'}")
            logger.debug(f"   - has tool_calls: {hasattr(message, 'tool_calls')}")
            if hasattr(message, 'tool_calls'):
                logger.debug(f"   - tool_calls value: {message.tool_calls}")
                logger.debug(f"   - tool_calls type: {type(message.tool_calls)}")
                logger.debug(f"   - tool_calls bool: {bool(message.tool_calls)}")
                if message.tool_calls:
                    logger.debug(f"   - tool_calls[0]: {message.tool_calls[0]}")
            logger.debug(f"   - content: {getattr(message, 'content', 'NO_CONTENT')[:120] if getattr(message, 'content', None) else 'NONE'}")
            logger.debug(f"   - response dict: {message.model_dump() if hasattr(message, 'model_dump') else ('N/A' if not hasattr(message, 'dict') else message.dict())}")
            
            # Track this call if requested
            if track_calls:
                message = response.choices[0].message
                call_info = {
                    'pass_number': 1,
                    'purpose': 'tool_selection' if (tools and hasattr(message, 'tool_calls') and message.tool_calls) else 'final_response',
                    'duration': call_duration,
                    'finish_reason': response.choices[0].finish_reason,
                    'tool_calls': []
                }
                
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tc in message.tool_calls:
                        call_info['tool_calls'].append({
                            'name': tc.function.name,
                            'arguments': tc.function.arguments
                        })
                
                if response.usage:
                    call_info['tokens'] = {
                        'prompt': response.usage.prompt_tokens if hasattr(response.usage, 'prompt_tokens') else 0,
                        'completion': response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else 0,
                        'total': response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else 0
                    }
                
                call_metadata.append(call_info)
            
            # Log response
            message = response.choices[0].message
            logger.info(f"🤖 LLM response received in {call_duration:.2f}s (finish_reason: {response.choices[0].finish_reason})")
            
            # Debug tool calling logic
            has_tool_calls = hasattr(message, 'tool_calls') and message.tool_calls
            if tools and has_tool_calls:
                logger.info(f"🎯 LLM selected {len(message.tool_calls)} tool(s) for execution:")
                for tc in message.tool_calls:
                    logger.info(f"   • {tc.function.name}")
            elif tools and not has_tool_calls:
                logger.debug(f"🤔 LLM chose not to use any of the {len(tools)} available tools")
                logger.debug(f"🔍 DEBUGGING: LLM had {len(tools)} tools but didn't call any")
                logger.debug(f"🔍 First 3 available tools were: {[t['function']['name'] for t in tools[:3]]}")
                logger.debug(f"🔍 LLM finish_reason: {response.choices[0].finish_reason}")
                # Safe content logging - check if content exists before slicing
                content = getattr(message, 'content', None)
                logger.debug(f"🔍 LLM response content preview: {str(content)[:100] if content else 'NONE'}...")
                
                # This indicates the LLM chose not to use tools, so we'll need structured output fallback
            
            logger.debug(f"🔧 Post-response analysis: tools={bool(tools)}, use_structured={use_structured_output}, tool_calls_made={has_tool_calls}")
            
            # Handle tool calls if present
            if tools and has_tool_calls:
                logger.info(f"🔧 LLM requested {len(message.tool_calls)} tool call(s), executing...")
                
                # Show tool details if enabled
                if self.show_tool_details:
                    from rich.console import Console
                    from rich.panel import Panel
                    from rich.syntax import Syntax
                    from rich import box
                    
                    console_output = Console()
                    
                    # Show LLM call details instead of redundant tool selection
                    llm_call_content = []
                    llm_call_content.append(f"[cyan]Model:[/cyan] {self.model}")
                    llm_call_content.append(f"[cyan]Tools Available:[/cyan] {len(tools)}")
                    llm_call_content.append(f"[cyan]Messages Sent:[/cyan] {len(messages)}")
                    
                    # Show the last few messages sent to LLM (context)
                    llm_call_content.append(f"\n[yellow]Recent Conversation Context:[/yellow]")
                    for i, msg in enumerate(messages[-3:], 1):  # Show last 3 messages
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')
                        # Truncate long content
                        if len(content) > 100:
                            content = content[:100] + "..."
                        llm_call_content.append(f"  {i}. [{role}]: {content}")
                    
                    # Show tool selection result
                    llm_call_content.append(f"\n[green]LLM Decision:[/green] Selected {len(message.tool_calls)} tool(s)")
                    for tc in message.tool_calls:
                        llm_call_content.append(f"  • {tc.function.name}")
                    
                    llm_panel = Panel(
                        "\n".join(llm_call_content),
                        title="🤖 LLM Call for Tool Selection",
                        border_style="blue",
                        box=box.ROUNDED
                    )
                    console_output.print(llm_panel)
                
                # Add assistant message with tool calls to conversation
                # CRITICAL: Use the raw message object like the demo does, not a manually constructed dict
                # The message object has proper serialization for tool calls
                messages_with_tools = messages + [response.choices[0].message]
                
                # Execute each tool call
                for idx, tc in enumerate(message.tool_calls, 1):
                    try:
                        tool_name = tc.function.name
                        tool_args = json.loads(tc.function.arguments)
                        logger.info(f"🔧 Executing tool: {tool_name} with args: {tool_args}")
                        
                        # Show tool execution details if enabled
                        if self.show_tool_details:
                            # Format arguments as pretty JSON
                            args_formatted = json.dumps(tool_args, indent=2)
                            
                            execution_content = (
                                f"[yellow]Tool:[/yellow] [bold]{tool_name}[/bold]\n\n"
                                f"[yellow]Arguments:[/yellow]\n"
                                f"[cyan]{args_formatted}[/cyan]\n\n"
                                f"[dim]Executing...[/dim]"
                            )
                            
                            execution_panel = Panel(
                                execution_content,
                                title=f"🔧 Tool Execution #{idx}",
                                border_style="yellow",
                                box=box.ROUNDED
                            )
                            console_output.print(execution_panel)
                        
                        # Execute the MCP tool
                        tool_result = await self.execute_mcp_tool(tool_name, tool_args)
                        
                        # Show result if enabled
                        if self.show_tool_details:
                            # Try to parse and beautify JSON results
                            try:
                                result_json = json.loads(tool_result)
                                result_display = json.dumps(result_json, indent=2)
                                # Truncate if too long
                                if len(result_display) > 1000:
                                    result_display = result_display[:1000] + '\n... (truncated)'
                                result_content = f"[green]{result_display}[/green]"
                            except (json.JSONDecodeError, TypeError):
                                # Not JSON, display as-is
                                result_display = tool_result[:500] + ('...' if len(tool_result) > 500 else '')
                                result_content = f"[green]{result_display}[/green]"
                            
                            result_panel = Panel(
                                result_content,
                                title=f"✅ Tool Result #{idx}",
                                border_style="green",
                                box=box.ROUNDED
                            )
                            console_output.print(result_panel)
                        
                        # Add tool result to conversation
                        messages_with_tools.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_result
                        })
                        
                        logger.info(f"✅ Tool {tool_name} result added to conversation")
                        
                    except Exception as e:
                        logger.error(f"❌ Tool execution failed for {tc.function.name}: {e}")
                        
                        if self.show_tool_details:
                            error_panel = Panel(
                                f"[red]{str(e)}[/red]",
                                title=f"❌ Tool Error #{idx}",
                                border_style="red",
                                box=box.ROUNDED
                            )
                            console_output.print(error_panel)
                        
                        # Add error as tool result
                        messages_with_tools.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps({"error": str(e)})
                        })
                
                # Make second LLM call with tool results
                logger.info("🔄 Calling LLM again with tool results for final response...")
                
                # CRITICAL FIX: Update system prompt to explicitly require JSON
                # The conversation has tool_calls which can confuse the LLM about output format
                # Solution: Replace system prompt with explicit JSON requirement
                structured_messages = messages_with_tools.copy()
                if use_structured_output and structured_messages and structured_messages[0].get("role") == "system":
                    structured_messages[0]["content"] = (
                        "You have just executed a tool and received results. "
                        "Respond ONLY with valid JSON (no other text):\n"
                        '{"type": "gif|url|text|latex|code|output", "response": "your sarcastic message (≤30 words)", "data": "URL or relevant data"}'
                    )
                    logger.debug("Updated system prompt to explicitly require JSON after tool execution")
                
                kwargs_final = {
                    "model": self.model,
                    "messages": structured_messages,
                    "temperature": 1.0 if "gpt-5" in self.model.lower() else self.temperature,
                    "timeout": 60.0  # Match demo's explicit timeout
                    # NOTE: max_tokens removed to match demo - let model use its default
                }
                
                # Use structured output for final response if it was requested originally
                if use_structured_output:
                    kwargs_final["response_format"] = self.response_schema
                    logger.debug("Using structured output for final response")
                
                call_start = time.time()
                response = await self.client.chat.completions.create(**kwargs_final)
                call_duration = time.time() - call_start
                
                # Track second call
                if track_calls:
                    final_message = response.choices[0].message
                    final_call_info = {
                        'pass_number': 2,
                        'purpose': 'final_response',
                        'duration': call_duration,
                        'finish_reason': response.choices[0].finish_reason,
                        'tool_calls': []
                    }
                    
                    if response.usage:
                        final_call_info['tokens'] = {
                            'prompt': response.usage.prompt_tokens if hasattr(response.usage, 'prompt_tokens') else 0,
                            'completion': response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else 0,
                            'total': response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else 0
                        }
                    
                    call_metadata.append(final_call_info)
                
                logger.info("✅ Final response received after tool execution")
                
                # DEBUG: Check if final response has content
                final_message = response.choices[0].message
                final_content = getattr(final_message, 'content', None)
                if not final_content:
                    logger.error(f"❌ CRITICAL: Final response after tool execution has NO CONTENT!")
                    logger.error(f"   - finish_reason: {response.choices[0].finish_reason}")
                    logger.error(f"   - message type: {type(final_message)}")
                    logger.error(f"   - message dict: {final_message.dict() if hasattr(final_message, 'dict') else 'N/A'}")
                    logger.error(f"   - This indicates the model failed to generate structured output after tool execution")
                else:
                    logger.debug(f"   Final content preview: {final_content[:100] if len(final_content) > 100 else final_content}")
            
            # Handle case where tools were available but LLM chose not to use them
            # In this case, we need to enforce structured output for consistency
            elif tools and use_structured_output and not (hasattr(message, 'tool_calls') and message.tool_calls):
                logger.debug(f"⚠️ FALLBACK TRIGGERED: Tools available but not used!")
                logger.debug(f"   - tools count: {len(tools)}")
                logger.debug(f"   - has tool_calls attr: {hasattr(message, 'tool_calls')}")
                if hasattr(message, 'tool_calls'):
                    logger.debug(f"   - tool_calls value: {message.tool_calls}")
                    logger.debug(f"   - tool_calls is None: {message.tool_calls is None}")
                    logger.debug(f"   - tool_calls bool: {bool(message.tool_calls)}")
                logger.debug(f"   - finish_reason: {response.choices[0].finish_reason}")
                # Safe content logging - check if content exists before slicing
                content = getattr(message, 'content', None)
                logger.debug(f"   - LLM response content: {content[:100] if content else 'NONE'}")
                logger.info("🔄 Tools were available but not used - enforcing structured output...")
                
                # Make a second call with structured output enabled and no tools
                kwargs_structured = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 1.0 if "gpt-5" in self.model.lower() else self.temperature,
                    "response_format": self.response_schema,
                    "timeout": 60.0  # Match demo's explicit timeout
                    # NOTE: max_tokens removed to match demo - let model use its default
                }
                
                call_start = time.time()
                response = await self.client.chat.completions.create(**kwargs_structured)
                call_duration = time.time() - call_start
                
                # Track this structured call
                if track_calls:
                    structured_message = response.choices[0].message
                    structured_call_info = {
                        'pass_number': 2,
                        'purpose': 'structured_output_fallback',
                        'duration': call_duration,
                        'finish_reason': response.choices[0].finish_reason,
                        'tool_calls': []
                    }
                    
                    if response.usage:
                        structured_call_info['tokens'] = {
                            'prompt': response.usage.prompt_tokens if hasattr(response.usage, 'prompt_tokens') else 0,
                            'completion': response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else 0,
                            'total': response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else 0
                        }
                    
                    call_metadata.append(structured_call_info)
                
                logger.info("✅ Structured output enforced after tools declined")
            
            # Final validation: Check if response has content
            final_message = response.choices[0].message
            final_content = getattr(final_message, 'content', None)
            if not final_content or final_content.strip() == "":
                logger.error(f"❌ CRITICAL: Final LLM response has empty content!")
                logger.error(f"   - Model: {self.model}")
                logger.error(f"   - Finish reason: {response.choices[0].finish_reason}")
                logger.error(f"   - Message type: {type(final_message)}")
                if hasattr(final_message, 'dict'):
                    logger.error(f"   - Message dict: {final_message.dict()}")
                logger.error(f"   - This may indicate the model doesn't support structured output properly")
                # Return None to signal error to caller
                if track_calls:
                    return None, call_metadata
                return None
            
            if track_calls:
                return response, call_metadata
            return response
            
        except Exception as e:
            logger.error(f"LiteLLM API call failed: {e}", exc_info=True)
            raise
    
    def _add_cache_control(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add cache control markers to messages for supported models.
        
        For Anthropic Claude models: Adds cache_control breakpoints
        For OpenAI models: System messages are automatically cached (no changes needed)
        """
        model_lower = self.model.lower()
        
        # Claude models need explicit cache control
        if 'claude' in model_lower:
            # Clone messages to avoid modifying original
            cached_messages = []
            for i, msg in enumerate(messages):
                new_msg = msg.copy()
                
                # Add cache_control to system message and last few user messages
                # This caches the system prompt and conversation history
                if msg['role'] == 'system':
                    # Cache system message (personality prompt)
                    if isinstance(new_msg.get('content'), str):
                        new_msg['content'] = [
                            {
                                "type": "text",
                                "text": new_msg['content'],
                                "cache_control": {"type": "ephemeral"}
                            }
                        ]
                    logger.debug("Added cache_control to system message")
                
                # Cache the last assistant message (conversation context)
                elif msg['role'] == 'assistant' and i == len(messages) - 2:
                    if isinstance(new_msg.get('content'), str):
                        new_msg['content'] = [
                            {
                                "type": "text",
                                "text": new_msg['content'],
                                "cache_control": {"type": "ephemeral"}
                            }
                        ]
                    logger.debug("Added cache_control to last assistant message")
                
                cached_messages.append(new_msg)
            
            return cached_messages
        
        # For OpenAI and Gemini, caching is automatic - return as-is
        return messages
    
    # ========== Context Management ==========
    
    def get_context_history(self, user_id: int, channel_id: int) -> List[Dict[str, str]]:
        """
        Get conversation history for user in channel from Redis.
        Filters out messages older than context_message_max_age_seconds.
        
        Returns:
            List of message dicts: [{"role": "user"|"assistant", "content": "..."}]
        """
        # Return empty list if Redis is not available
        if not self.redis_client:
            logger.debug("Redis not available, returning empty context history")
            return []
        
        redis_key = f"discord_context:{user_id}:{channel_id}"
        
        try:
            raw_history = self.redis_client.get(redis_key)
            if not raw_history:
                logger.debug(f"No context history found for {redis_key}")
                return []
            
            history = json.loads(raw_history)
            
            # Filter by age if configured
            if self.context_message_max_age_seconds > 0:
                current_time = time.time()
                filtered_history = []
                
                for msg in history:
                    # Messages should have timestamp, but handle old format without it
                    msg_time = msg.get('timestamp', current_time)  # Assume recent if no timestamp
                    age = current_time - msg_time
                    
                    if age <= self.context_message_max_age_seconds:
                        # Remove timestamp before returning (not needed in LLM context)
                        filtered_msg = {"role": msg["role"], "content": msg["content"]}
                        filtered_history.append(filtered_msg)
                    else:
                        logger.debug(f"Filtered out message (age: {age:.0f}s > {self.context_message_max_age_seconds}s)")
                
                logger.debug(f"Loaded {len(filtered_history)} messages for {redis_key} (filtered from {len(history)})")
                return filtered_history
            else:
                # No age filtering, just remove timestamps
                clean_history = [{"role": msg["role"], "content": msg["content"]} for msg in history]
                logger.debug(f"Loaded {len(clean_history)} messages for {redis_key}")
                return clean_history
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode context history for {redis_key}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading context history for {redis_key}: {e}", exc_info=True)
            return []
    
    def save_context_history(self, user_id: int, channel_id: int, history: List[Dict[str, str]]) -> None:
        """
        Save conversation history to Redis with timestamp and TTL.
        
        Args:
            user_id: Discord user ID
            channel_id: Discord channel ID
            history: List of messages to save
        """
        redis_key = f"discord_context:{user_id}:{channel_id}"
        
        try:
            # Add timestamp to each message for age-based filtering
            current_time = time.time()
            timestamped_history = []
            
            for msg in history:
                timestamped_msg = {
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": current_time
                }
                timestamped_history.append(timestamped_msg)
            
            # Limit to max_history_messages
            if len(timestamped_history) > self.max_history_messages:
                timestamped_history = timestamped_history[-self.max_history_messages:]
                logger.debug(f"Trimmed history to {self.max_history_messages} messages")
            
            # Skip saving if Redis is not available
            if not self.redis_client:
                logger.debug("Redis not available, skipping context history save")
                return
            
            # Save to Redis
            self.redis_client.set(redis_key, json.dumps(timestamped_history))
            
            # Set TTL if configured
            if self.context_history_ttl_seconds > 0:
                self.redis_client.expire(redis_key, self.context_history_ttl_seconds)
                logger.debug(f"Saved {len(timestamped_history)} messages with TTL {self.context_history_ttl_seconds}s")
            else:
                logger.debug(f"Saved {len(timestamped_history)} messages (no TTL)")
            
        except Exception as e:
            logger.error(f"Error saving context history for {redis_key}: {e}", exc_info=True)
    
    def get_usage_stats(self, response: Any) -> Dict[str, Any]:
        """Extract basic usage statistics from response"""
        try:
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                return {
                    "prompt_tokens": getattr(usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(usage, 'completion_tokens', 0),
                    "total_tokens": getattr(usage, 'total_tokens', 0),
                }
            else:
                return {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
        except Exception as e:
            logger.error(f"Failed to extract usage stats: {e}", exc_info=True)
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
