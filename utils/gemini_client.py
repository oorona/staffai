"""
Gemini Client for Discord Bot - Direct Google Gemini API Integration

Calls Google Gemini models directly using the google-genai SDK, without a
LiteLLM proxy.

Architecture:
  - Extends LiteLLMClient to inherit all Redis context management, MCP tool
    calling, and built-in tool logic unchanged.
  - Overrides __init__ (creates a google.genai.Client instead of AsyncOpenAI),
    chat_completion (Gemini-native multi-round call), _add_cache_control (no-op;
    Gemini caches automatically), and get_usage_stats (different response shape).
  - Returns OpenAI-compatible response dataclasses so all callers
    (cogs, message_handler, user_memory_manager) work without modification.

For future OpenAI provider support, see utils/litellm_client.py.
"""

import json
import logging
import time
import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import redis
from google import genai
from google.genai import types as genai_types

from utils.litellm_client import LiteLLMClient
from utils.log_formatting import emit_plain_block_marker, format_log_panel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight OpenAI-compatible response dataclasses
# ---------------------------------------------------------------------------

@dataclass
class _ToolCallFunction:
    name: str
    arguments: str  # JSON string


@dataclass
class _ToolCall:
    id: str
    type: str
    function: _ToolCallFunction


@dataclass
class _Message:
    content: Optional[str]
    role: str = "assistant"
    tool_calls: Optional[List[_ToolCall]] = None

    def model_dump(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}

    def dict(self) -> Dict[str, Any]:
        return self.model_dump()


@dataclass
class _Choice:
    message: _Message
    finish_reason: str = "stop"
    index: int = 0


@dataclass
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class _Response:
    choices: List[_Choice]
    usage: _Usage


# ---------------------------------------------------------------------------
# GeminiClient
# ---------------------------------------------------------------------------

class GeminiClient(LiteLLMClient):
    """
    Discord bot LLM client using Google Gemini API directly.

    Inherits Redis context management, MCP tool calling, and built-in tool
    logic from LiteLLMClient.  Overrides the core LLM API calls to use
    google.genai instead of the OpenAI-compatible proxy.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        redis_client: Optional[redis.Redis],
        response_schema_path: Optional[str] = None,
        context_history_ttl_seconds: int = 1800,
        context_message_max_age_seconds: int = 1800,
        max_history_messages: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        mcp_servers: Optional[List[str]] = None,
    ):
        """
        Initialize Gemini client.

        Args:
            model: Gemini model name (e.g. "gemini-3.1-pro-preview")
            api_key: Google AI API key (GEMINI_API_KEY)
            redis_client: Redis client for context storage (may be None initially)
            response_schema_path: Path to chat response JSON schema
            context_history_ttl_seconds: Conversation history TTL in Redis
            context_message_max_age_seconds: Max age for individual messages
            max_history_messages: Max messages to keep per user/channel
            temperature: LLM sampling temperature
            max_tokens: Reserved; not sent to Gemini (models use their own default)
            mcp_servers: Optional list of MCP server URLs
        """
        # NOTE: intentionally NOT calling super().__init__() to avoid creating
        # an unnecessary AsyncOpenAI client.  We set all attributes that the
        # inherited methods depend on manually below.

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.redis_client = redis_client
        self.context_history_ttl_seconds = context_history_ttl_seconds
        self.context_message_max_age_seconds = context_message_max_age_seconds
        self.max_history_messages = max_history_messages
        self.mcp_servers_override = mcp_servers

        # Google Gemini client
        self._gemini = genai.Client(api_key=api_key)

        # Load response schema for structured output
        if response_schema_path is None:
            response_schema_path = (
                Path(__file__).parent / "prompts" / "chat_response" / "schema.json"
            )
        with open(response_schema_path, "r", encoding="utf-8") as fh:
            self.response_schema = json.load(fh)

        # MCP tool cache (same shape as LiteLLMClient)
        self._mcp_tools_cache: Optional[List[Dict[str, Any]]] = None
        self._mcp_tools_cache_time: float = 0
        self._mcp_failed_servers: set = set()
        self._tool_to_server_map: Dict[str, str] = {}

        # Set by bot after UserMemoryManager is created
        self.user_memory_manager = None
        self.show_tool_details: bool = False

        logger.info("Initialized GeminiClient with model: %s", model)
        if mcp_servers:
            logger.info("MCP servers: %d configured", len(mcp_servers))
        logger.info(
            "Context TTL: %ds, Message max age: %ds",
            context_history_ttl_seconds,
            context_message_max_age_seconds,
        )

    # ------------------------------------------------------------------
    # Provider-specific helpers
    # ------------------------------------------------------------------

    def _add_cache_control(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Gemini caches automatically – no explicit markers needed."""
        return messages

    def _extract_json_schema(self, response_format: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract the raw JSON Schema from an OpenAI-style response_format dict.

        Expected input:
            {"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}}
        Returns the inner schema with Gemini-incompatible keys removed.
        """
        if not response_format:
            return None
        if response_format.get("type") == "json_schema":
            schema = dict(response_format.get("json_schema", {}).get("schema", {}))
            self._strip_unsupported_schema_keys(schema)
            return schema if schema else None
        return None

    def _strip_unsupported_schema_keys(self, schema: Dict[str, Any]) -> None:
        """Recursively remove JSON Schema keys not supported by Gemini."""
        for key in ("additionalProperties",):
            schema.pop(key, None)
        for prop in schema.get("properties", {}).values():
            if isinstance(prop, dict):
                self._strip_unsupported_schema_keys(prop)
        if "items" in schema and isinstance(schema["items"], dict):
            self._strip_unsupported_schema_keys(schema["items"])

    def _openai_tools_to_gemini(
        self, tools: List[Dict[str, Any]]
    ) -> List[genai_types.Tool]:
        """Convert OpenAI-format tool definitions to a Gemini Tool list."""
        decls = []
        for tool in tools:
            fn = tool.get("function", {})
            params = dict(fn.get("parameters", {}))
            self._strip_unsupported_schema_keys(params)
            decls.append(
                genai_types.FunctionDeclaration(
                    name=fn.get("name", ""),
                    description=fn.get("description", ""),
                    parameters=params,
                )
            )
        return [genai_types.Tool(function_declarations=decls)]

    @staticmethod
    def _content_as_text(content: Any) -> str:
        """Return content as a plain string regardless of its original type."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # OpenAI cache_control format: [{"type": "text", "text": "..."}]
            return " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return str(content)

    def _openai_messages_to_gemini(
        self,
        messages: List[Dict[str, Any]],
        tool_call_id_to_name: Optional[Dict[str, str]] = None,
    ):
        """
        Convert an OpenAI-format messages list to (system_instruction, contents).

        Handles:
          - role="system"    → extracted as system_instruction (string)
          - role="user"      → Content(role="user", parts=[Part(text=...)])
          - role="assistant" → Content(role="model", ...) with text and/or
                               FunctionCall parts when tool_calls is present
          - role="tool"      → Content(role="user", parts=[Part(function_response=...)])
        """
        system_instruction: Optional[str] = None
        contents: List[genai_types.Content] = []
        id_to_name = tool_call_id_to_name or {}

        for msg in messages:
            # Support both plain dicts and OpenAI SDK message objects
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content")
                raw_tool_calls = msg.get("tool_calls")
                tool_call_id = msg.get("tool_call_id", "")
            else:
                role = str(getattr(msg, "role", ""))
                content = getattr(msg, "content", None)
                raw_tool_calls = getattr(msg, "tool_calls", None)
                tool_call_id = getattr(msg, "tool_call_id", "")

            if role == "system":
                system_instruction = self._content_as_text(content)
                continue

            if role == "assistant" or role == "model":
                parts: List[genai_types.Part] = []

                # Convert tool_calls to FunctionCall parts
                if raw_tool_calls:
                    for tc in raw_tool_calls:
                        if isinstance(tc, dict):
                            fn_name = tc.get("function", {}).get("name", "")
                            try:
                                fn_args = json.loads(
                                    tc.get("function", {}).get("arguments", "{}")
                                )
                            except Exception:
                                fn_args = {}
                        else:
                            fn_name = getattr(
                                getattr(tc, "function", None), "name", ""
                            )
                            try:
                                fn_args = json.loads(
                                    getattr(
                                        getattr(tc, "function", None), "arguments", "{}"
                                    )
                                )
                            except Exception:
                                fn_args = {}
                        parts.append(
                            genai_types.Part(
                                function_call=genai_types.FunctionCall(
                                    name=fn_name,
                                    args=fn_args,
                                )
                            )
                        )

                text = self._content_as_text(content)
                if text:
                    parts.append(genai_types.Part(text=text))

                if parts:
                    contents.append(
                        genai_types.Content(role="model", parts=parts)
                    )
                continue

            if role == "user":
                text = self._content_as_text(content)
                if text:
                    contents.append(
                        genai_types.Content(
                            role="user",
                            parts=[genai_types.Part(text=text)],
                        )
                    )
                continue

            if role == "tool":
                # Map the tool_call_id back to the function name
                tool_name = id_to_name.get(tool_call_id, "unknown_tool")
                raw = self._content_as_text(content)
                try:
                    result_data: Dict[str, Any] = json.loads(raw)
                    if not isinstance(result_data, dict):
                        result_data = {"output": raw}
                except Exception:
                    result_data = {"output": raw}

                contents.append(
                    genai_types.Content(
                        role="user",
                        parts=[
                            genai_types.Part(
                                function_response=genai_types.FunctionResponse(
                                    name=tool_name,
                                    response=result_data,
                                )
                            )
                        ],
                    )
                )
                continue

        return system_instruction, contents

    def _gemini_response_to_openai(self, gemini_response: Any) -> _Response:
        """Wrap a google.genai response in OpenAI-compatible dataclasses."""
        content: Optional[str] = None
        tool_calls: Optional[List[_ToolCall]] = None
        finish_reason = "stop"

        if gemini_response.candidates:
            candidate = gemini_response.candidates[0]
            parts = (candidate.content.parts if candidate.content else []) or []

            text_parts: List[str] = []
            function_calls = []
            for part in parts:
                if getattr(part, "text", None):
                    text_parts.append(part.text)
                fc = getattr(part, "function_call", None)
                if fc and getattr(fc, "name", None):
                    function_calls.append(fc)

            content = "".join(text_parts) or None

            if function_calls:
                tool_calls = [
                    _ToolCall(
                        id=f"call_{i}_{fc.name}",
                        type="function",
                        function=_ToolCallFunction(
                            name=fc.name,
                            arguments=json.dumps(dict(fc.args)),
                        ),
                    )
                    for i, fc in enumerate(function_calls)
                ]
                finish_reason = "tool_calls"

            fr_raw = str(getattr(candidate, "finish_reason", "")).lower()
            if "stop" in fr_raw or "end_turn" in fr_raw or fr_raw == "1":
                finish_reason = "stop" if not tool_calls else "tool_calls"
            elif "max" in fr_raw:
                finish_reason = "length"

        usage = _Usage()
        um = getattr(gemini_response, "usage_metadata", None)
        if um:
            usage.prompt_tokens = getattr(um, "prompt_token_count", 0) or 0
            usage.completion_tokens = getattr(um, "candidates_token_count", 0) or 0
            usage.total_tokens = getattr(um, "total_token_count", 0) or 0

        return _Response(
            choices=[
                _Choice(
                    message=_Message(content=content, tool_calls=tool_calls),
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
        )

    async def _gemini_call(
        self,
        system_instruction: Optional[str],
        contents: List[genai_types.Content],
        tools: Optional[List[Dict[str, Any]]] = None,
        use_structured_output: bool = True,
        response_schema_override: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Low-level Gemini API call with pre-converted contents.

        Used directly for pass-2 tool-result calls so that the original
        model Content (including thought_signatures) can be passed back
        without being lost in an OpenAI round-trip conversion.
        """
        target_schema = response_schema_override or self.response_schema
        config_kwargs: Dict[str, Any] = {
            "temperature": self.temperature,
            "thinking_config": genai_types.ThinkingConfig(thinking_budget=0),
        }

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        if tools:
            config_kwargs["tools"] = self._openai_tools_to_gemini(tools)
            config_kwargs["tool_config"] = genai_types.ToolConfig(
                function_calling_config=genai_types.FunctionCallingConfig(
                    mode="AUTO"
                )
            )
        elif use_structured_output:
            json_schema = self._extract_json_schema(target_schema)
            if json_schema:
                config_kwargs["response_mime_type"] = "application/json"
                config_kwargs["response_schema"] = json_schema

        # Inject current date as the first content turn so system_instruction
        # stays static and eligible for Gemini implicit caching.
        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        effective_contents = [
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=f"[Context: Today is {current_date} UTC]")],
            )
        ] + list(contents)

        config = genai_types.GenerateContentConfig(**config_kwargs)
        return await self._gemini.aio.models.generate_content(
            model=self.model,
            contents=effective_contents,
            config=config,
        )

    async def _gemini_generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        use_structured_output: bool = True,
        response_schema_override: Optional[Dict[str, Any]] = None,
        tool_call_id_to_name: Optional[Dict[str, str]] = None,
    ) -> Tuple[Any, Optional[str], List[genai_types.Content]]:
        """
        Convert OpenAI-format messages to Gemini and call the API.

        Returns (raw_response, system_instruction, contents) so callers
        can build subsequent turns natively without losing thought_signatures.
        """
        system_instruction, contents = self._openai_messages_to_gemini(
            messages, tool_call_id_to_name=tool_call_id_to_name
        )
        raw = await self._gemini_call(
            system_instruction,
            contents,
            tools=tools,
            use_structured_output=use_structured_output,
            response_schema_override=response_schema_override,
        )
        return raw, system_instruction, contents

    # ------------------------------------------------------------------
    # Core public method: chat_completion
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        use_structured_output: bool = True,
        enable_caching: bool = True,
        track_calls: bool = False,
        response_schema_override: Optional[Dict[str, Any]] = None,
        call_context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Chat completion via Google Gemini API, compatible with LiteLLMClient.

        Args:
            messages: OpenAI-format message list
            tools: Optional MCP tools in OpenAI function-calling format
            use_structured_output: Enforce JSON schema on the response
            enable_caching: No-op for Gemini (caching is automatic)
            track_calls: Return (response, call_metadata) tuple when True
            response_schema_override: Override default chat response schema
            call_context: Logging context dict (user/channel/guild/source)

        Returns:
            _Response object, or (response, call_metadata) when track_calls=True.
            Returns None (or (None, metadata)) on empty final response.
        """
        call_metadata: List[Dict[str, Any]] = []
        llm_call_started_at = time.time()
        llm_block_open = False
        llm_status = "error"
        llm_error = ""
        api_calls_made = 0
        tools_executed = 0

        try:
            tools = self._merge_builtin_tools(tools, call_context=call_context)
            emit_plain_block_marker("LLM CALL START", style="llm")
            logger.info("## LLM CALL START ##")

            context = call_context or {}
            logger.info(
                "\n%s",
                format_log_panel(
                    "LLM CALL HEADER",
                    [
                        ("user", str(context.get("user_name", "system"))),
                        ("channel", str(context.get("channel_name", "n/a"))),
                        ("guild", str(context.get("guild_name", "n/a"))),
                        ("source", str(context.get("source", "unknown"))),
                        ("interaction", str(context.get("interaction_case", "n/a"))),
                        ("model", self.model),
                        ("messages", len(messages)),
                        ("tools_requested", len(tools) if tools else 0),
                        ("structured_output", use_structured_output),
                        ("caching", enable_caching),
                        ("track_calls", track_calls),
                    ],
                ),
            )
            llm_block_open = True

            target_schema = response_schema_override or self.response_schema

            # ── PASS 1: initial call ────────────────────────────────────
            if tools:
                logger.info(
                    "llm_pass_request pass=1 purpose=tool_selection tools_available=%d context_messages=%d",
                    len(tools),
                    len(messages),
                )
            else:
                logger.info(
                    "llm_pass_request pass=1 purpose=direct_response tools_available=0 context_messages=%d",
                    len(messages),
                )

            call_start = time.time()
            raw, sys_instr, base_contents = await asyncio.wait_for(
                self._gemini_generate(
                    messages,
                    tools=tools or None,
                    use_structured_output=use_structured_output,
                    response_schema_override=target_schema,
                ),
                timeout=60.0,
            )
            call_duration = time.time() - call_start
            api_calls_made += 1

            response = self._gemini_response_to_openai(raw)
            message = response.choices[0].message

            logger.info(
                "llm_pass_result pass=1 duration_ms=%.2f finish_reason=%s has_tool_calls=%s",
                call_duration * 1000,
                response.choices[0].finish_reason,
                bool(message.tool_calls),
            )

            if track_calls:
                call_info: Dict[str, Any] = {
                    "pass_number": 1,
                    "purpose": "tool_selection" if message.tool_calls else "final_response",
                    "duration": call_duration,
                    "finish_reason": response.choices[0].finish_reason,
                    "tool_calls": [],
                    "request_messages": self._truncate_for_call_audit(messages),
                    "tokens": {
                        "prompt": response.usage.prompt_tokens,
                        "completion": response.usage.completion_tokens,
                        "total": response.usage.total_tokens,
                    },
                    "response_preview": self._preview_for_call_audit(message.content),
                }
                if message.tool_calls:
                    for tc in message.tool_calls:
                        try:
                            parsed = json.loads(tc.function.arguments)
                        except Exception:
                            parsed = {"raw": tc.function.arguments}
                        call_info["tool_calls"].append(
                            {"name": tc.function.name, "arguments": parsed}
                        )
                call_metadata.append(call_info)

            # ── Handle tool calls ───────────────────────────────────────
            if tools and message.tool_calls:
                logger.info(
                    "LLM requested %d tool call(s), executing", len(message.tool_calls)
                )

                # Build mapping: tool_call_id → tool_name (needed by Gemini pass 2)
                id_to_name: Dict[str, str] = {
                    tc.id: tc.function.name for tc in message.tool_calls
                }

                # Extend messages list with assistant + tool results (OpenAI format)
                messages_with_tools: List[Dict[str, Any]] = list(messages)
                messages_with_tools.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in message.tool_calls
                        ],
                    }
                )

                executed_tools_meta: List[Dict[str, Any]] = []
                gemini_fn_responses: List[genai_types.Content] = []

                for idx, tc in enumerate(message.tool_calls, 1):
                    tools_executed += 1
                    tool_started_at = time.time()
                    tool_args: Dict[str, Any] = {}
                    try:
                        tool_name = tc.function.name
                        try:
                            tool_args = json.loads(tc.function.arguments)
                        except Exception:
                            tool_args = {"raw": tc.function.arguments}

                        logger.info(
                            "tool_start idx=%d name=%s args=%s",
                            idx,
                            tool_name,
                            json.dumps(tool_args, ensure_ascii=False),
                        )

                        tool_result = await self.execute_mcp_tool(
                            tool_name, tool_args, call_context=call_context
                        )
                        tool_duration = time.time() - tool_started_at
                        executed_tools_meta.append(
                            {
                                "name": tool_name,
                                "arguments": tool_args,
                                "duration": tool_duration,
                                "status": "ok",
                                "result_preview": self._preview_for_call_audit(
                                    tool_result, max_chars=500
                                ),
                            }
                        )

                        messages_with_tools.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": tool_result,
                            }
                        )

                        try:
                            fn_result_data: Dict[str, Any] = json.loads(tool_result)
                            if not isinstance(fn_result_data, dict):
                                fn_result_data = {"output": tool_result}
                        except Exception:
                            fn_result_data = {"output": tool_result}
                        gemini_fn_responses.append(
                            genai_types.Content(
                                role="user",
                                parts=[
                                    genai_types.Part(
                                        function_response=genai_types.FunctionResponse(
                                            name=tool_name,
                                            response=fn_result_data,
                                        )
                                    )
                                ],
                            )
                        )

                        logger.info(
                            "tool_end idx=%d name=%s status=ok duration_ms=%.2f result_chars=%d",
                            idx,
                            tool_name,
                            tool_duration * 1000,
                            len(str(tool_result)),
                        )

                    except Exception as exc:
                        tool_duration = time.time() - tool_started_at
                        err_name = getattr(
                            getattr(tc, "function", None), "name", "?"
                        )
                        logger.error(
                            "tool_end idx=%d name=%s status=error duration_ms=%.2f error=%s",
                            idx,
                            err_name,
                            tool_duration * 1000,
                            exc,
                        )
                        executed_tools_meta.append(
                            {
                                "name": err_name,
                                "arguments": tool_args,
                                "duration": tool_duration,
                                "status": "error",
                                "error": str(exc),
                            }
                        )
                        messages_with_tools.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": json.dumps({"error": str(exc)}),
                            }
                        )
                        gemini_fn_responses.append(
                            genai_types.Content(
                                role="user",
                                parts=[
                                    genai_types.Part(
                                        function_response=genai_types.FunctionResponse(
                                            name=err_name,
                                            response={"error": str(exc)},
                                        )
                                    )
                                ],
                            )
                        )

                if track_calls and call_metadata:
                    call_metadata[0]["tool_calls"] = executed_tools_meta

                # ── PASS 2: final response with tool results ────────────
                # Build contents natively: base_contents + original model Content
                # (with thought_signature intact) + FunctionResponse parts.
                # This avoids the thought_signature being lost via OpenAI conversion.
                logger.info("Calling Gemini again with tool results for final response")

                contents_pass2 = list(base_contents)
                if raw.candidates and raw.candidates[0].content:
                    contents_pass2.append(raw.candidates[0].content)
                contents_pass2.extend(gemini_fn_responses)

                call_start = time.time()
                raw_final = await asyncio.wait_for(
                    self._gemini_call(
                        sys_instr,
                        contents_pass2,
                        tools=None,
                        use_structured_output=use_structured_output,
                        response_schema_override=target_schema,
                    ),
                    timeout=60.0,
                )
                call_duration = time.time() - call_start
                api_calls_made += 1
                response = self._gemini_response_to_openai(raw_final)

                logger.info(
                    "llm_pass_result pass=2 purpose=final_response duration_ms=%.2f finish_reason=%s",
                    call_duration * 1000,
                    response.choices[0].finish_reason,
                )

                if track_calls:
                    call_metadata.append(
                        {
                            "pass_number": 2,
                            "purpose": "final_response",
                            "duration": call_duration,
                            "finish_reason": response.choices[0].finish_reason,
                            "tool_calls": [],
                            "request_messages": self._truncate_for_call_audit(
                                messages_with_tools
                            ),
                            "tokens": {
                                "prompt": response.usage.prompt_tokens,
                                "completion": response.usage.completion_tokens,
                                "total": response.usage.total_tokens,
                            },
                            "response_preview": self._preview_for_call_audit(
                                response.choices[0].message.content
                            ),
                        }
                    )

            # ── Structured-output fallback (tools available but not called) ──
            elif tools and use_structured_output and not message.tool_calls:
                logger.info("llm_structured_fallback_enforcing")

                call_start = time.time()
                raw_struct, *_ = await asyncio.wait_for(
                    self._gemini_generate(
                        messages,
                        tools=None,
                        use_structured_output=True,
                        response_schema_override=target_schema,
                    ),
                    timeout=60.0,
                )
                call_duration = time.time() - call_start
                api_calls_made += 1
                response = self._gemini_response_to_openai(raw_struct)

                logger.info(
                    "llm_pass_result pass=2 purpose=structured_output_fallback duration_ms=%.2f finish_reason=%s",
                    call_duration * 1000,
                    response.choices[0].finish_reason,
                )

                if track_calls:
                    call_metadata.append(
                        {
                            "pass_number": 2,
                            "purpose": "structured_output_fallback",
                            "duration": call_duration,
                            "finish_reason": response.choices[0].finish_reason,
                            "tool_calls": [],
                            "request_messages": self._truncate_for_call_audit(messages),
                            "tokens": {
                                "prompt": response.usage.prompt_tokens,
                                "completion": response.usage.completion_tokens,
                                "total": response.usage.total_tokens,
                            },
                            "response_preview": self._preview_for_call_audit(
                                response.choices[0].message.content
                            ),
                        }
                    )

            # ── Final validation ────────────────────────────────────────
            final_content = response.choices[0].message.content
            if not final_content or not final_content.strip():
                llm_error = "final_response_empty"
                logger.error(
                    "llm_final_validation_failed model=%s finish_reason=%s",
                    self.model,
                    response.choices[0].finish_reason,
                )
                if track_calls:
                    return None, call_metadata
                return None

            llm_status = "ok"
            if track_calls:
                return response, call_metadata
            return response

        except Exception as exc:
            llm_error = str(exc)
            logger.error("Gemini API call failed: %s", exc, exc_info=True)
            raise

        finally:
            if llm_block_open:
                elapsed_ms = (time.time() - llm_call_started_at) * 1000
                footer_fields = [
                    ("status", llm_status),
                    ("elapsed_ms", f"{elapsed_ms:.2f}"),
                    ("api_calls", api_calls_made),
                    ("tools_executed", tools_executed),
                    ("tracked_passes", len(call_metadata)),
                ]
                if llm_status != "ok":
                    footer_fields.append(("error", llm_error or "unknown"))
                logger.info(
                    "\n%s",
                    format_log_panel("LLM CALL FOOTER", footer_fields),
                )
                emit_plain_block_marker("LLM CALL END", style="llm")
                logger.info("## LLM CALL END ##")

    # ------------------------------------------------------------------
    # Usage stats
    # ------------------------------------------------------------------

    def get_usage_stats(self, response: Any) -> Dict[str, Any]:
        """Extract token counts from a _Response (or fall back to zeros)."""
        try:
            if isinstance(response, _Response) and response.usage:
                return {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
        except Exception as exc:
            logger.error("Failed to extract usage stats: %s", exc)
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
