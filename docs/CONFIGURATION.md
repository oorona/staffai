# Configuration Reference

This document describes all environment variables and configuration options for StaffAI.

---

## Environment File

Create a `.env` file in the project root with your configuration:

```bash
cp .env.example .env
```

---

## Required Configuration

### Discord Settings

| Variable | Description | Example |
|----------|-------------|---------|
| `DISCORD_BOT_TOKEN` | Bot token from Discord Developer Portal | `MTIzNDU2Nzg5...` |

> **Note**: In Docker, this can be provided via Docker secret at `/run/secrets/discord_bot_token`

### Discord Permissions and Intents

Required bot permissions (minimum for current features):
- `View Channels`
- `Send Messages`
- `Read Message History`
- `Embed Links` (approval embed / reports)
- `Create Public Threads` (daily topic thread publishing)
- `Send Messages in Threads` (posting thread content and participating)
- `Use Application Commands` (slash commands)

Required privileged intents in Discord Developer Portal:
- `MESSAGE CONTENT INTENT` (the bot reads message text for normal conversation logic)
- `SERVER MEMBERS INTENT` (role-based checks and restrictions)

Admin-gated commands:
- `/refresh_status` requires administrator permission.
- `/refresh_topic` requires administrator permission and triggers on-demand daily-topic proposal.
- `/approve_topic` requires administrator permission.
- `/my_memory` is user-scoped and always ephemeral (shows only caller memory).

### LiteLLM Settings

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `LITELLM_API_URL` | LiteLLM proxy URL | — | `http://localhost:4000` |
| `LITELLM_MODEL` | Model identifier | — | `gpt-4o-mini` |
| `LITELLM_API_KEY` | API key for proxy | — | `sk-1234` |

> **Note**: In Docker, API key can be provided via Docker secret at `/run/secrets/litellm_api_key`

### Redis Settings

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `REDIS_HOST` | Redis server hostname | `localhost` | `redis` (Docker) |
| `REDIS_PORT` | Redis server port | `6379` | `6379` |
| `REDIS_DB` | Redis database number | `0` | `0` |
| `REDIS_PASSWORD` | Redis password (if required) | — | `secret` |

---

## MCP Tool Calling

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `MCP_SERVERS` | Comma-separated MCP server URLs | — | `https://tenor.example.com/mcp,https://cve.example.com/mcp` |

---

## Bot Behavior

### Response Settings

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `RESPONSE_CHANCE` | Probability to randomly respond (0.0-1.0) | `0.05` | `0.10` |
| `RANDOM_RESPONSE_DELIVERY_CHANCE` | Secondary filter for random responses | `0.3` | `0.5` |
| `MAX_HISTORY_PER_USER` | Max messages in context per user/channel | `20` | `30` |

### Context Decay

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `CONTEXT_HISTORY_TTL_SECONDS` | Entire conversation history expiry | `1800` | `3600` |
| `CONTEXT_MESSAGE_MAX_AGE_SECONDS` | Individual message age limit | `1800` | `3600` |
| `DEFAULT_CONTEXT_MESSAGES` | Messages to fetch per user for context | `5` | `10` |

---

## Rate Limiting

### Message Limits

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `RATE_LIMIT_COUNT` | Max messages per user per window | `15` | `20` |
| `RATE_LIMIT_WINDOW_SECONDS` | Rate limit window size | `60` | `120` |

### Token Limits

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `TOKEN_RATE_LIMIT_COUNT` | Max tokens per user per window | `20000` | `50000` |

---

## Restriction System

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `RESTRICTED_USER_ROLE_ID` | Role ID for restricted users | — | `123456789012345678` |
| `RESTRICTED_CHANNEL_ID` | Channel for restricted users | — | `123456789012345678` |
| `RESTRICTION_DURATION_SECONDS` | How long restrictions last | `86400` | `43200` |
| `RESTRICTION_CHECK_INTERVAL_SECONDS` | How often to check expirations | `300` | `600` |

### Notification Templates

| Variable | Description | Default |
|----------|-------------|---------|
| `RATE_LIMIT_MESSAGE_USER` | Message when rate limited | `You've sent messages too frequently. Please use <#{channel_id}> for bot interactions.` |
| `RESTRICTED_CHANNEL_MESSAGE_USER` | Message when restricted | `As a restricted user, please use <#{channel_id}> for bot interactions.` |

---

## Role-Based Access

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `SUPER_ROLE_IDS` | Comma-separated role IDs that bypass limits | — | `123,456,789` |
| `IGNORED_ROLE_IDS` | Comma-separated role IDs the bot ignores | — | `111,222,333` |

---

## Activity/Presence

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `ACTIVITY_UPDATE_INTERVAL_SECONDS` | How often to update status | `300` | `600` |
| `ACTIVITY_SCHEDULE_ENABLED` | Enable time-based scheduling | `False` | `True` |
| `ACTIVITY_ACTIVE_START_HOUR_UTC` | Start hour (UTC) | `0` | `9` |
| `ACTIVITY_ACTIVE_END_HOUR_UTC` | End hour (UTC) | `23` | `17` |
| `ACTIVITY_ACTIVE_DAYS_UTC` | Active days (0=Mon, 6=Sun) | `0,1,2,3,4,5,6` | `0,1,2,3,4` |

---

## Daily Topic Workflow

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `DAILY_TOPIC_ENABLED` | Enable admin-approved educational topic workflow | `False` | `True` |
| `DAILY_TOPIC_APPROVAL_CHANNEL_ID` | Channel where approval embed is posted | — | `123456789012345678` |
| `DAILY_TOPIC_PUBLISH_CHANNEL_ID` | Forum/Media channel where approved post is published | — | `123456789012345678` |
| `DAILY_TOPIC_INTERVAL_SECONDS` | `0` = daily mode, `>0` = interval mode | `0` | `86400` |
| `DAILY_TOPIC_APPROVAL_HOUR_UTC` | Proposal hour in daily mode (UTC) | `8` | `8` |
| `DAILY_TOPIC_APPROVAL_TIMEOUT_SECONDS` | Auto-publish timeout if no admin response | `14400` | `14400` |
| `DAILY_TOPIC_CHECK_INTERVAL_SECONDS` | Scheduler tick frequency | `60` | `30` |
| `DAILY_TOPIC_THREAD_AUTO_ARCHIVE_MINUTES` | Thread auto-archive duration | `1440` | `1440` |
| `DAILY_TOPIC_THREAD_CONTEXT_MESSAGES` | Thread messages sent as LLM context for topic threads | `40` | `60` |

Daily topic prompt templates (editable at runtime, no code changes needed):
- `utils/prompts/daily_topic/topic_generation_system_prompt.txt`
- `utils/prompts/daily_topic/topic_generation_user_prompt.txt`
- `utils/prompts/daily_topic/body_generation_system_prompt.txt`
- `utils/prompts/daily_topic/body_generation_user_prompt.txt`
- `utils/prompts/daily_topic/topic_generation_response_schema.json`
- `utils/prompts/daily_topic/body_generation_response_schema.json`

Daily topic category/tag behavior:
- The bot selects one category using a balanced-random strategy (least-used categories are favored).
- The selected category is shown in the approval embed.
- Publishing requires a matching forum tag in `DAILY_TOPIC_PUBLISH_CHANNEL_ID`; post creation fails if tag is missing.

---

## User Memory and LLM Call Audit

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `USER_MEMORY_ENABLED` | Enable per-user memory extraction/injection | `True` | `True` |
| `USER_MEMORY_ROOT_PATH` | On-disk folder for user memory files | `data/user_memory` | `data/user_memory` |
| `USER_MEMORY_UPDATE_CHANCE` | Sampling chance after worthwhile-message filter | `0.25` | `0.15` |
| `USER_MEMORY_MIN_MESSAGE_CHARS` | Minimum chars to consider message worthwhile | `50` | `70` |
| `USER_MEMORY_MIN_MESSAGE_WORDS` | Minimum words to consider message worthwhile | `10` | `12` |
| `USER_MEMORY_MAX_CHARS` | Maximum stored memory size per user | `420` | `350` |
| `USER_MEMORY_PIPELINE_MODE` | Memory pipeline mode (`tiny_extract`, `frontier_pipeline`, `tiny_gate_frontier_core`, `disabled`) | `tiny_gate_frontier_core` | `frontier_pipeline` |
| `USER_MEMORY_OLLAMA_BASE_URL` | Ollama server URL for tiny-model memory pipeline | `http://localhost:11434` | `http://ollama:11434` |
| `USER_MEMORY_OLLAMA_API_KEY` | API key used for Ollama OpenAI-compatible endpoint | `ollama` | `ollama` |
| `USER_MEMORY_OLLAMA_TIMEOUT_S` | Timeout in seconds for tiny-model calls | `30` | `20` |
| `USER_MEMORY_TINY_MODEL` | Default tiny model name in Ollama | `SmolLM2-1.7B-Instruct` | `qwen2.5:3b-instruct` |
| `USER_MEMORY_TINY_MODEL_EXTRACT` | Optional model override for direct tiny extraction stage | — | `SmolLM2-1.7B-Instruct` |
| `USER_MEMORY_TINY_MODEL_CLASSIFIER` | Optional model override for tiny worthwhile classifier stage | — | `SmolLM2-1.7B-Instruct` |
| `USER_MEMORY_TINY_ACCUMULATE_MAX_TOKENS` | In `tiny_extract` mode, compact accumulated memory once this token estimate is reached | `4000` | `3000` |
| `USER_MEMORY_AUDIT_MAX_ENTRIES` | Number of recent memory-pipeline traces kept in Redis | `200` | `500` |
| `USER_MEMORY_DEBUG_CLASSIFICATION` | Enable detailed memory-classification debug logs (`[MEMDBG]`) | `False` | `True` |
| `LLM_CALL_AUDIT_ENABLED` | Save recent LLM calls to Redis | `True` | `True` |
| `LLM_CALL_AUDIT_MAX_ENTRIES` | Number of recent calls kept per guild | `100` | `200` |

User memory prompt files (runtime editable):
- `utils/prompts/user_memory/memory_update_system_prompt.txt`
- `utils/prompts/user_memory/memory_update_user_prompt.txt`
- `utils/prompts/user_memory/memory_update_response_schema.json`
- `utils/prompts/user_memory/tiny_worthwhile_system_prompt.txt`
- `utils/prompts/user_memory/tiny_worthwhile_user_prompt.txt`
- `utils/prompts/user_memory/tiny_worthwhile_response_schema.json`
- `utils/prompts/user_memory/tiny_extract_system_prompt.txt`
- `utils/prompts/user_memory/tiny_extract_user_prompt.txt`
- `utils/prompts/user_memory/tiny_extract_response_schema.json`
- `utils/prompts/user_memory/tiny_compact_system_prompt.txt`
- `utils/prompts/user_memory/tiny_compact_user_prompt.txt`
- `utils/prompts/user_memory/tiny_compact_response_schema.json`
- `utils/prompts/user_memory/frontier_core_extract_system_prompt.txt`
- `utils/prompts/user_memory/frontier_core_extract_user_prompt.txt`
- `utils/prompts/user_memory/frontier_core_extract_response_schema.json`

Redis audit key format:
- `llm_calls:recent:<guild_id>` (LPUSH/LTRIM rolling list)
- `user_memory_pipeline:recent` (LPUSH/LTRIM rolling list, includes `pipeline_mode`)

---

## Token Statistics & Reporting

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `STATS_REPORT_CHANNEL_ID` | Channel for automated reports | — | `123456789012345678` |
| `STATS_REPORT_INTERVAL_SECONDS` | Report frequency | `86400` | `43200` |
| `STATS_REPORT_TOP_USERS` | Number of top users to show | `10` | `20` |

---

## Logging

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `LOG_LEVEL` | Logging verbosity | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |

---

## Debug Options

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `DEBUG_CONTEXT_SUPER_USERS` | Enable context debugging for super users | `False` | `True` |

---

## Complete Example

```env
# =============================================================================
# DISCORD
# =============================================================================
DISCORD_BOT_TOKEN=your_bot_token_here

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL=INFO

# =============================================================================
# LITELLM
# =============================================================================
LITELLM_API_URL=http://localhost:4000
LITELLM_MODEL=gpt-4o-mini
LITELLM_API_KEY=sk-1234

# =============================================================================
# MCP SERVERS
# =============================================================================
MCP_SERVERS=https://tenormcp.example.com/mcp,https://cvemcp.example.com/mcp

# =============================================================================
# BOT BEHAVIOR
# =============================================================================
RESPONSE_CHANCE=0.05
RANDOM_RESPONSE_DELIVERY_CHANCE=0.3
MAX_HISTORY_PER_USER=20

# =============================================================================
# CONTEXT DECAY
# =============================================================================
CONTEXT_HISTORY_TTL_SECONDS=1800
CONTEXT_MESSAGE_MAX_AGE_SECONDS=1800
DEFAULT_CONTEXT_MESSAGES=5

# =============================================================================
# REDIS
# =============================================================================
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
# REDIS_PASSWORD=your_password

# =============================================================================
# RATE LIMITING
# =============================================================================
RATE_LIMIT_COUNT=15
RATE_LIMIT_WINDOW_SECONDS=60
TOKEN_RATE_LIMIT_COUNT=20000

# =============================================================================
# RESTRICTION SYSTEM
# =============================================================================
RESTRICTED_USER_ROLE_ID=123456789012345678
RESTRICTED_CHANNEL_ID=123456789012345678
RESTRICTION_DURATION_SECONDS=86400
RESTRICTION_CHECK_INTERVAL_SECONDS=300

# =============================================================================
# NOTIFICATION TEMPLATES
# =============================================================================
RATE_LIMIT_MESSAGE_USER=You've sent messages too frequently. Please use <#{channel_id}> for bot interactions.
RESTRICTED_CHANNEL_MESSAGE_USER=As a restricted user, please use <#{channel_id}> for bot interactions.

# =============================================================================
# ROLE-BASED ACCESS
# =============================================================================
SUPER_ROLE_IDS=
IGNORED_ROLE_IDS=

# =============================================================================
# ACTIVITY/PRESENCE
# =============================================================================
ACTIVITY_UPDATE_INTERVAL_SECONDS=300
ACTIVITY_SCHEDULE_ENABLED=False
ACTIVITY_ACTIVE_START_HOUR_UTC=0
ACTIVITY_ACTIVE_END_HOUR_UTC=23
ACTIVITY_ACTIVE_DAYS_UTC=0,1,2,3,4,5,6

# =============================================================================
# DAILY TOPIC WORKFLOW
# =============================================================================
DAILY_TOPIC_ENABLED=False
DAILY_TOPIC_APPROVAL_CHANNEL_ID=
DAILY_TOPIC_PUBLISH_CHANNEL_ID=
DAILY_TOPIC_INTERVAL_SECONDS=0
DAILY_TOPIC_APPROVAL_HOUR_UTC=8
DAILY_TOPIC_APPROVAL_TIMEOUT_SECONDS=14400
DAILY_TOPIC_CHECK_INTERVAL_SECONDS=60
DAILY_TOPIC_THREAD_AUTO_ARCHIVE_MINUTES=1440
DAILY_TOPIC_THREAD_CONTEXT_MESSAGES=40

# =============================================================================
# USER MEMORY + LLM CALL AUDIT
# =============================================================================
USER_MEMORY_ENABLED=True
USER_MEMORY_ROOT_PATH=data/user_memory
USER_MEMORY_UPDATE_CHANCE=0.25
USER_MEMORY_MIN_MESSAGE_CHARS=50
USER_MEMORY_MIN_MESSAGE_WORDS=10
USER_MEMORY_MAX_CHARS=420
USER_MEMORY_PIPELINE_MODE=tiny_gate_frontier_core
USER_MEMORY_OLLAMA_BASE_URL=http://localhost:11434
USER_MEMORY_OLLAMA_API_KEY=ollama
USER_MEMORY_OLLAMA_TIMEOUT_S=30
USER_MEMORY_TINY_MODEL=SmolLM2-1.7B-Instruct
USER_MEMORY_TINY_MODEL_EXTRACT=
USER_MEMORY_TINY_MODEL_CLASSIFIER=
USER_MEMORY_TINY_ACCUMULATE_MAX_TOKENS=4000
USER_MEMORY_AUDIT_MAX_ENTRIES=200
USER_MEMORY_DEBUG_CLASSIFICATION=False
LLM_CALL_AUDIT_ENABLED=True
LLM_CALL_AUDIT_MAX_ENTRIES=100

# =============================================================================
# TOKEN STATISTICS
# =============================================================================
STATS_REPORT_CHANNEL_ID=
STATS_REPORT_INTERVAL_SECONDS=86400
STATS_REPORT_TOP_USERS=10
```

---

## Docker Secrets

For production deployments, sensitive values can be provided via Docker secrets:

| Secret Name | Mounted Path | Overrides |
|-------------|--------------|-----------|
| `discord_bot_token` | `/run/secrets/discord_bot_token` | `DISCORD_BOT_TOKEN` |
| `litellm_api_key` | `/run/secrets/litellm_api_key` | `LITELLM_API_KEY` |
| `user_memory_ollama_api_key` | `/run/secrets/user_memory_ollama_api_key` | `USER_MEMORY_OLLAMA_API_KEY` |

The bot automatically detects and uses Docker secrets when available.

---

## See Also

- [Installation Guide](INSTALLATION.md) — Setup instructions
- [Architecture Overview](ARCHITECTURE.md) — Technical documentation
