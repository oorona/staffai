# Token Consumption Tracking Feature

## Overview

The bot now tracks token consumption per user and provides admin commands to view statistics and automated reports showing top consumers.

## Features

### 1. **Automatic Token Tracking**
- Tracks every LLM interaction (conversation responses, activity updates)
- Stores three types of data in Redis:
  - **Total tokens** (all-time consumption per user)
  - **Daily tokens** (resets daily, kept for 7 days)
  - **Usage log** (last 1000 interactions with timestamps)

### 2. **Admin Command: `/tokenstats`**
- **Access**: Super user roles only (configured via `SUPER_ROLE_IDS`)
- **Format**: Ephemeral embed (only visible to command issuer)
- **Shows**:
  - Total tokens consumed
  - Today's token usage
  - Estimated cost (based on gpt-4o-mini pricing: $0.15/1M tokens)
  - Recent activity log (last 10 interactions)
  - User's avatar and username

**Usage**: `/tokenstats @username` or `/tokenstats user:@username`

### 3. **Automated Reports**
- **Frequency**: Configurable via `STATS_REPORT_INTERVAL_SECONDS` (default: 24 hours)
- **Channel**: Sent to channel specified by `STATS_REPORT_CHANNEL_ID`
- **Content**: 
  - Top N users by token consumption (N = `STATS_REPORT_TOP_USERS`)
  - Ranking with medals (🥇🥈🥉)
  - Total tokens, daily tokens, estimated costs
  - Grand totals for all tracked users

## Configuration

### Environment Variables

Add to your `.env` file:

```bash
# Super User Roles (Bypass rate limits, access admin commands like /tokenstats, and use special LLM toolsets)
# Comma-separated list of role IDs
SUPER_ROLE_IDS=1234567890,9876543210

# Token Consumption Stats & Reporting
# Channel where automated token usage reports will be sent
STATS_REPORT_CHANNEL_ID=1234567890
# How often to send automated reports (in seconds). Default: 86400 = 24 hours
STATS_REPORT_INTERVAL_SECONDS=86400
# Number of top users to show in automated reports
STATS_REPORT_TOP_USERS=10
```

### Required Setup

1. **Super User Roles**: Add at least one role ID to `SUPER_ROLE_IDS` to enable `/tokenstats` command access
2. **Stats Channel**: Set `STATS_REPORT_CHANNEL_ID` to enable automated reports
3. **Report Interval**: Adjust `STATS_REPORT_INTERVAL_SECONDS` for report frequency
4. **Top Users Count**: Set `STATS_REPORT_TOP_USERS` for ranking size

## Redis Keys

The feature uses the following Redis key patterns:

```
token_stats:total:{guild_id}:{user_id}        # Cumulative total (permanent)
token_stats:daily:{guild_id}:{date}:{user_id} # Daily usage (7 day TTL)
token_stats:log:{guild_id}:{user_id}          # Sorted set (last 1000 entries)
```

## Implementation Details

### Files Modified/Created

1. **`cogs/stats_cog.py`** (NEW - 489 lines)
   - `StatsCog` class with token tracking and reporting
   - Methods:
     - `record_token_usage()` - Store token consumption
     - `get_user_token_stats()` - Retrieve user statistics
     - `get_top_users_by_tokens()` - Get top consumers
     - `tokenstats_command()` - Slash command handler
     - `send_token_report_loop()` - Scheduled task for reports

2. **`utils/message_handler.py`** (MODIFIED)
   - Added stats recording after LLM response
   - Calls `bot.stats_cog.record_token_usage()` with:
     - `user_id` - Discord user ID
     - `guild_id` - Discord server ID
     - `tokens` - Token count from LLM response
     - `message_type` - Interaction type (mention/reply/random)

3. **`bot.py`** (MODIFIED)
   - Added configuration parameters:
     - `stats_report_channel_id` - Channel for reports
     - `stats_report_interval_seconds` - Report frequency
     - `stats_report_top_users` - Ranking size
   - Added `stats_cog` reference (set after cog loads)
   - Loads `cogs.stats_cog` extension in `setup_hook()`
   - Uses `super_role_ids_set` for admin access control

4. **`main.py`** (MODIFIED)
   - Loads `STATS_REPORT_CHANNEL_ID`, `STATS_REPORT_INTERVAL_SECONDS`, `STATS_REPORT_TOP_USERS`
   - Passes all stats configuration to `AIBot` constructor
   - Super user roles (SUPER_ROLE_IDS) grant admin command access

5. **`.env` and `.env.example`** (MODIFIED)
   - Consolidated `ADMIN_ROLE_IDS` into `SUPER_ROLE_IDS`
   - Added `STATS_REPORT_CHANNEL_ID` configuration
   - Added `STATS_REPORT_INTERVAL_SECONDS` (default: 86400)
   - Added `STATS_REPORT_TOP_USERS` (default: 10)

## Cost Calculation

The bot estimates costs based on gpt-4o-mini pricing:
- **Rate**: $0.15 per 1 million tokens
- **Formula**: `(tokens / 1,000,000) * 0.15`

To adjust for different models, modify the `COST_PER_MILLION_TOKENS` constant in `cogs/stats_cog.py`.

## Usage Examples

### Admin Command

```
/tokenstats @JohnDoe
```

**Response** (ephemeral embed):
```
📊 Token Usage Stats - @JohnDoe

💰 Total Tokens: 125,430
📅 Today's Usage: 3,245 tokens
💵 Estimated Cost: $0.02

Recent Activity:
• 2024-01-15 14:23 - 145 tokens (mention)
• 2024-01-15 13:45 - 230 tokens (reply)
• 2024-01-15 12:10 - 89 tokens (random)
[... 7 more entries ...]

(Last updated: 2024-01-15 14:30:00)
```

### Automated Report

**Channel**: #stats-reports (configured via `STATS_REPORT_CHANNEL_ID`)

```
📊 Token Consumption Report
Period: Last 24 hours

🏆 Top 10 Users by Token Consumption

🥇 1. @Alice
   Total: 45,230 tokens | Today: 12,450 tokens
   Estimated Cost: $0.01

🥈 2. @Bob
   Total: 38,920 tokens | Today: 9,870 tokens
   Estimated Cost: $0.01

🥉 3. @Charlie
   Total: 32,105 tokens | Today: 7,230 tokens
   Estimated Cost: $0.00

[... 7 more users ...]

📈 Grand Totals
Total Tokens: 234,567 | Total Cost: $0.04

(Generated: 2024-01-15 00:00:00)
```

## Troubleshooting

### Command Not Appearing

1. Ensure `SUPER_ROLE_IDS` is set in `.env`
2. Verify the bot has loaded `stats_cog` (check logs for "Loaded extension: cogs.stats_cog")
3. Check that your role ID is in the `SUPER_ROLE_IDS` list
4. Slash commands may take up to 1 hour to sync globally

### Reports Not Sending

1. Verify `STATS_REPORT_CHANNEL_ID` is set and valid
2. Ensure the bot has permission to send messages in the channel
3. Check logs for task start: "Token report task started"
4. Verify Redis connection is working

### Stats Not Recording

1. Check logs for "Recording token usage" messages
2. Verify Redis is accessible (check `REDIS_HOST` and `REDIS_PORT`)
3. Ensure `stats_cog` reference is set in bot (check logs for "StatsCog reference stored in bot")
4. Verify message_handler is calling `bot.stats_cog.record_token_usage()`

## Technical Notes

### Thread Safety

All Redis operations in `stats_cog` are wrapped with `discord.utils.asyncio.to_thread()` to avoid blocking the event loop.

### Data Retention

- **Total tokens**: Stored permanently (no expiry)
- **Daily tokens**: Stored with 7-day TTL (auto-deleted after 7 days)
- **Usage log**: Limited to last 1000 entries per user (FIFO)

### Performance

- Redis sorted sets used for efficient log queries
- Atomic operations (ZINCRBY, ZADD) for concurrent safety
- Minimal overhead on message processing (~1ms per interaction)

### Scalability

- Per-guild isolation (stats are guild-specific)
- Configurable report intervals to reduce channel spam
- Efficient top-N queries using ZREVRANGE

## Future Enhancements

Potential improvements:

1. **Per-Model Cost Tracking**: Track costs separately for different LLM models
2. **Weekly/Monthly Reports**: Add longer-term reporting periods
3. **User Quotas**: Set token limits per user with warnings
4. **Cost Alerts**: Notify when costs exceed thresholds
5. **Export Functionality**: CSV/JSON export of statistics
6. **Web Dashboard**: Real-time stats via web interface
