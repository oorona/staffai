# Architecture Overview

This document provides a technical deep-dive into the StaffAI codebase structure, data flow, and design decisions, specifically focusing on the AI inference pipeline, contextual memory systems, and MCP tool networking.

---

## Project Structure

```
staffai/
├── main.py                          # Entry point, config validation
├── bot.py                           # AIBot class, client initialization
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Container build instructions
├── docker-compose.yaml              # Multi-container orchestration
│
├── cogs/
│   ├── message_cog.py              # Discord event handling, response rendering
│   ├── activity_cog.py             # Dynamic presence generation
│   ├── stats_cog.py                # Token tracking and reporting
│   ├── daily_topic_cog.py          # Daily topic proposal/approval/publishing flow
│   └── user_memory_cog.py          # User-facing memory command
│
├── utils/
│   ├── litellm_client.py           # LiteLLM proxy client, MCP tools
│   ├── message_handler.py          # Message processing, rate limiting
│   ├── prompts/
│   │   ├── chat_response/          # Core chat prompt pack (system/user/schema)
│   │   ├── activity_status/        # Activity generation prompt pack
│   │   ├── daily_topic_topic_generation/    # Daily-topic title proposal pack
│   │   ├── daily_topic_body_generation/     # Daily-topic body generation pack
│   │   ├── user_memory_* /         # User-memory prompt packs
│   │   └── interaction_* /         # Interaction helper prompt packs
│   └── user_memory_manager.py      # User-memory pipeline and persistence
│
├── docs/
│   ├── INSTALLATION.md             # Setup instructions
│   ├── CONFIGURATION.md            # Environment variables
│   └── ARCHITECTURE.md             # This document
│
├── specs/
│   └── specs.txt                   # Complete technical specifications
│
└── secrets/
    ├── discord_bot_token.txt       # Docker secret
    └── litellm_api_key.txt         # Docker secret
```

---

## Core Components

### Entry Point: main.py

**Responsibilities:**
- Load environment variables from `.env` file
- Read Docker secrets (if available)
- Validate all configuration parameters
- Load consolidated prompt packs from `utils/prompts/<purpose>/`
- Instantiate and run `AIBot`

**Key Functions:**
- `load_prompt_from_file()` — Safe prompt file loading with error handling
- `_read_docker_secret()` — Docker secrets integration

**Design Decision:** Fail-fast validation. Invalid configuration causes immediate exit with descriptive error messages.

### Bot Core: bot.py

**Class: AIBot (extends commands.Bot)**

**Responsibilities:**
- Store all configuration parameters
- Initialize Redis client
- Initialize LiteLLM client
- Register cogs in `setup_hook()`
- Preload MCP tools at startup

**Initialization Flow:**
```python
AIBot.__init__()
    ├── Store configuration
    ├── Initialize Redis client
    ├── Initialize LiteLLMClient
    └── (setup_hook called by discord.py)
        ├── Load message_cog
        ├── Load activity_cog
        ├── Load stats_cog
        └── Preload MCP tools
```

---

## Cog Architecture

### MessageCog (cogs/message_cog.py)

**Purpose:** Handle Discord message events and render bot responses

**Event Handlers:**
- `on_message()` — Process incoming messages

**Key Features:**
- Message deduplication (Redis + in-memory)
- Response type rendering (text, gif, latex, code, etc.)
- Restriction notification system
- Background task for restriction expiry

**Response Rendering Logic:**
```python
match response_type:
    "text"   → Simple message send
    "gif"    → Embed with image URL
    "latex"  → Fetch PNG from latex2image API, send as attachment
    "code"   → Format with syntax highlighting
    "url"    → Send with contextual message
    "output" → Code block formatting
```

### ActivityCog (cogs/activity_cog.py)

**Purpose:** Generate and update bot presence/activity

**Background Task:**
- `update_bot_activity_loop` — Periodic activity updates

**Features:**
- LLM-generated activity text
- Activity type rotation (Playing/Listening/Watching/Custom)
- Time-based scheduling (hours and days)
- Status changes (online/idle based on schedule)

### StatsCog (cogs/stats_cog.py)

**Purpose:** Token consumption tracking and reporting

**Commands:**
- `/tokenstats @user` — View user token consumption (super users only)

**Background Task:**
- `send_token_report_loop` — Scheduled usage reports

**Redis Keys:**
```
token_stats:total:{guild_id}:{user_id}        # Cumulative total
token_stats:daily:{guild_id}:{date}:{user_id} # Daily usage (7d TTL)
token_stats:log:{guild_id}:{user_id}          # Sorted set of interactions
```

---

## Utility Modules

### LiteLLMClient (utils/litellm_client.py)

**Purpose:** Interface with the LiteLLM proxy for highly concurrent LLM inference routing

**Key Methods:**
- `chat_completion()` — Send normalized messages to LLM with optional MCP tools
- `get_mcp_tools()` — Fetch and aggressively cache MCP tool definitions at startup
- `get_context_history()` — Retrieve serialized conversation history and vector metadata from Redis
- `save_context_history()` — Persist conversation history applying Time-Based Context Decay (TTL)

**MCP Tool Caching:**
```python
get_mcp_tools()
    ├── Check cache age
    ├── If stale → Fetch from all MCP_SERVERS
    │   ├── Use FastMCP Client with StreamableHttpTransport
    │   ├── Convert to OpenAI function calling format
    │   └── Update cache
    └── Return cached tools
```

**Three-Path Response Handling:**
```python
chat_completion()
    ├── Path A: Tools called
    │   ├── Execute tool via MCP server
    │   ├── Add result to messages
    │   └── Make final LLM call with structured output
    │
    ├── Path B: Tools available but not used
    │   └── Make second call forcing structured output
    │
    └── Path C: No tools
        └── Direct structured output call
```

### MessageHandler (utils/message_handler.py)

**Purpose:** Central message processing logic

**Key Methods:**
- `handle_message()` — Main entry point for message processing
- `_determine_engagement()` — Decide if bot should respond
- `_check_message_rate_limit()` — Sliding window rate check
- `_apply_restriction()` — Assign restricted role
- `_process_message_with_context()` — Build context and call LLM

**Engagement Decision Tree:**
```
Message received
    ├── Is bot author? → Ignore
    ├── Is DM? → Ignore
    ├── Has ignored role? → Ignore
    ├── Is @mention? → Respond
    ├── Is reply to bot? → Respond
    ├── Random chance hit? → Respond
    └── Otherwise → Ignore
```

**Rate Limiting Flow:**
```
handle_message()
    ├── Check super user status
    ├── If not super user:
    │   ├── Check message count limit
    │   ├── Check token consumption limit
    │   └── If exceeded → Apply restriction
    └── Process message with LLM
```

---

## Data Flow

### Message Processing Pipeline

```
┌─────────────────┐
│ Discord Message │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MessageCog     │  ← Event capture, deduplication
│  on_message()   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ MessageHandler  │  ← Rate limit check, engagement decision
│ handle_message()│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LiteLLMClient   │  ← Context building, MCP tools
│ chat_completion()│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LiteLLM Proxy   │  ← Model routing
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LLM Provider    │  ← Inference
│ (OpenAI, etc.)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Structured JSON │  ← Response parsing
│ Response        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MessageCog     │  ← Type-specific rendering
│  (render)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Discord Reply   │
└─────────────────┘
```

---

## Redis Data Structures

### Conversation History

**Key:** `discord_context:{user_id}:{channel_id}`
**Type:** JSON String
**Value:**
```json
[
  {"role": "user", "content": "Hello", "timestamp": 1699123456.789},
  {"role": "assistant", "content": "Hi there!", "timestamp": 1699123457.123}
]
```
**TTL:** `CONTEXT_HISTORY_TTL_SECONDS` (refreshed on each save)

### Context Scope, Latency & Multi-User Challenges

The bot does not keep one global "channel conversation state". Context is assembled from multiple sources at runtime, each with different scope and failure modes. These decisions explicitly manage token limits and inference latency:

- **Short-term dialog history** is stored per `user_id + channel_id`, not per full channel thread of all participants. This keeps context focused on the requester and bounds the token usage. Group conversations are reconstructed indirectly.
- **Referenced-user memory (Vector Context)** is global per user and injected only when the current message explicitly references other users (mentions or reply target). This limits context payload sizes while maintaining personalization.
- **Random-response channel context** uses a recent channel snapshot instead of requester history. This provides situational awareness without permanently polluting the conversational vector.
- **Daily-topic threads** bypass normal Redis history and use live thread history instead. 

Keeping context coherent across multiple users while maintaining low latency inference is difficult:

- **Attribution ambiguity:** In a busy channel, the bot may see multiple humans discussing the same topic.
- **Scope mismatch:** Conversation history is scoped by requester, while memory/style/expertise are global per user. 
- **Staleness risk:** Redis history expires by TTL and age filters, while user profile memory persists. Short-term context can disappear while long-term profile data remains.
- **Token pressure & Latency:** Adding requester history, referenced-user memories, channel context, and tool messages can grow prompts exponentially, increasing inference latency. The runtime trims history more aggressively when tools are available to stay within budget.
- **Partial observability:** If a message was deleted, uncached, or outside the current TTL window, the bot may know a reply chain exists without having the full upstream text.

Current mitigations in the optimization pipeline:

- Redis conversation history is strictly bound by `MAX_HISTORY_PER_USER`, message age, and TTL to reduce stale carry-over and control prompt size.
- Tool-enabled calls use `LLM_TOOL_HISTORY_LIMIT` to dynamically trim short-term history and preserve payload capacity for schema arrays and JSON structures.
- Referenced-user memory is injected only for explicitly referenced users to avoid broad speculative vector loading.
- One-time history bootstrap for missing `style`/`expertise` runs only after a direct interaction, only in the same channel, and only when the current message was not worthwhile.

### Rate Limiting

**Message Count:**
- **Key:** `msg_rl:{guild_id}:{user_id}`
- **Type:** List
- **Value:** Timestamps of recent messages

**Token Consumption:**
- **Key:** `token_rl:{guild_id}:{user_id}`
- **Type:** List
- **Value:** Entries in format `timestamp:tokens`

### Restrictions

**Key:** `restricted_until:{guild_id}:{user_id}`
**Type:** String
**Value:** Unix timestamp of restriction expiry

### Message Deduplication

**Key:** `processed_msg:{channel}_{author}_{reference}_{content_hash}`
**Type:** Set
**Value:** `"1"`
**TTL:** 60 seconds

### User Profiles

**Memory Key:**
- **Key:** `user_memory:{user_id}`
- **Type:** String

**Style Key:**
- **Key:** `user_style:{user_id}`
- **Type:** JSON String (`{"traits":[...]}`)

**Expertise Key:**
- **Key:** `user_expertise:{user_id}`
- **Type:** String (`beginner|intermediate|advanced`)

**Memory Security Alerts:**
- **Key:** `user_memory_security_alerts:recent`
- **Type:** List (JSON payloads for blocked high-confidence injection attempts)

**One-Time History Bootstrap Flag:**
- **Key:** `user_profile_history_bootstrap_attempted:{user_id}`
- **Type:** String (`"1"`) to prevent repeated same-channel history backfill attempts for missing style/expertise

**Disk backup layout:**
- `data/user_memory/<user_id>/memory.json`
- `data/user_memory/<user_id>/style.json`
- `data/user_memory/<user_id>/expertise.json`
- Startup hydration loads memory, style, and expertise into Redis when missing.

---

## Design Decisions

### Why LiteLLM Proxy?

- **Provider Agnostic:** Switch between OpenAI, Anthropic, Google, local models without code changes
- **Unified API:** Single interface for 100+ providers
- **Cost Optimization:** Easy A/B testing and fallback support
- **Centralized Configuration:** Model routing handled externally

### Why Structured Output?

- **Zero Parsing Errors:** JSON schema enforcement eliminates malformed responses
- **Type Safety:** Guaranteed response format enables reliable rendering
- **No Retry Logic:** First response is always valid
- **Multi-Modal Support:** Single format handles text/url/code/latex/gif

### Why MCP for Tools?

- **Extensibility:** Add tools without modifying bot code
- **Standardization:** Open protocol for function calling
- **Separation of Concerns:** Tools run in separate services
- **Dynamic Discovery:** Bot auto-detects available tools at runtime

### Why Redis?

- **Performance:** In-memory storage for sub-millisecond lookups
- **Persistence:** Optional disk snapshots for durability
- **Atomic Operations:** Thread-safe LPUSH, EXPIRE operations
- **TTL Support:** Native key expiration for context decay
- **Production Proven:** Widely deployed, well-documented

### Why Cog Architecture?

- **Separation of Concerns:** Each cog handles one responsibility
- **Hot Reloading:** Cogs can be reloaded without bot restart (dev mode)
- **Testability:** Individual cogs can be tested in isolation
- **Scalability:** New features added as new cogs

---

## MCP Integration Details

### Tool Loading Flow

```
Bot Startup
    │
    ▼
setup_hook()
    │
    ▼
get_mcp_tools()
    │
    ├── For each server in MCP_SERVERS:
    │   ├── Create FastMCP Client with StreamableHttpTransport
    │   ├── Connect (10s timeout)
    │   ├── Call list_tools()
    │   ├── Handle response format (ListToolsResult or list)
    │   └── Convert to OpenAI function format
    │
    ├── Cache combined tools list
    └── Return tools
```

### Tool Execution Flow

```
LLM Response with tool_calls
    │
    ▼
For each tool call:
    │
    ├── Get server URL from tool-to-server map
    ├── Create FastMCP Client
    ├── Execute tool with arguments
    ├── Collect result
    └── Add to messages as tool_result
    │
    ▼
Make final LLM call with results
    │
    ▼
Structured JSON response
```

---

## Error Handling Strategy

### Fail-Fast (Startup)
- Invalid configuration → Exit with error message
- Missing prompt files → Exit with error message
- Invalid numeric values → Exit with error message

### Graceful Degradation (Runtime)
- Redis errors → Log and continue (may lose context)
- MCP server timeout → Skip server, use remaining tools
- LLM errors → Return error to user via MessageHandlerResult

### Logging
- All errors logged with context
- Configurable log level (DEBUG → CRITICAL)
- File + console output
- MCP errors suppressed (no stack traces to logs)

---

## See Also

- [Installation Guide](INSTALLATION.md) — Setup instructions
- [Configuration Reference](CONFIGURATION.md) — Environment variables
- [Project Specifications](../specs/specs.txt) — Complete specifications
