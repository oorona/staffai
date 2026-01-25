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

The bot automatically detects and uses Docker secrets when available.

---

## See Also

- [Installation Guide](INSTALLATION.md) — Setup instructions
- [Architecture Overview](ARCHITECTURE.md) — Technical documentation
