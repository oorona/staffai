# Discord AI Bot (StaffAI Enhanced)

A highly configurable Discord bot using `discord.py` that interacts with users via an OpenWebUI-compatible LLM API. It features persistent context, advanced rate limiting, role-based restrictions/exemptions, and more.

## Features

* **LLM Integration:** Connects to any OpenWebUI-compatible API (Ollama w/ OpenAI endpoint, LM Studio, vLLM, etc.) to generate conversational responses.
* **Configurable Personality:** Define the bot's personality and response style using a dedicated prompt file (`utils/prompts/personality_prompt.txt`).
* **Context Management:** Maintains conversation history **per user per channel**, providing relevant context for ongoing dialogues. History is persisted in **Redis** to survive bot restarts.
* **Modular Message Processing:** Core message handling logic (engagement decisions, rate limiting, context preparation, LLM calls) is encapsulated in a dedicated `MessageHandler` class (`utils/message_handler.py`) for better organization and maintainability.
* **Context Injection:** When replying to a bot message that was part of another user's conversation, the bot intelligently injects the necessary context for a coherent reply and merges that thread into the replier's history.
* **Interaction Triggers:** Responds when mentioned (`@Bot`), when replied to, or based on a configurable random chance.
* **Advanced Rate Limiting:** Protects against spam and overuse:
    * Limits based on **message count** per user within a configurable time window.
    * Limits based on **total LLM tokens** consumed per user within the time window (uses `tiktoken` and API usage data).
* **Restriction System:**
    * Automatically assigns a configurable "Restricted User" role when rate limits are exceeded.
    * Restricts users with this role to interact with the bot only in a specific, configurable channel.
    * Notifies users publicly via channel replies when they are restricted or try to use the bot outside the designated channel while restricted. Notifications for rate limit restrictions are suppressed if the interaction was initiated by the bot's "Random Chance" feature.
* **Rate Limit Exemptions:** Allows users with specific, configurable roles to bypass all rate limits.
* **Global Ignore List:** Allows configuring specific roles whose members the bot will completely ignore.
* **Automatic Restriction Expiry:** The "Restricted User" role can be automatically removed after a configurable duration, with the bot periodically checking for expired restrictions using Redis.
* **Dynamic Logging:** Set the application's logging level (DEBUG, INFO, WARNING, etc.) via an environment variable. Logs to both console and file (`bot.log`).
* **Dockerized:** Includes `Dockerfile` and `docker-compose.yaml` for easy containerization and deployment, including a Redis service.

## Prerequisites

* **Python:** Version 3.8 or higher recommended.
* **Discord Bot Token:** Obtainable from the [Discord Developer Portal](https://discord.com/developers/applications).
    * **Required Intents:** `Server Members Intent` and `Message Content Intent` (Privileged Intents).
    * **Required Permissions:** `View Channel`, `Send Messages`, `Manage Roles` (Crucial for restriction system).
* **OpenWebUI Compatible API:** A running LLM endpoint (e.g., Ollama, LM Studio) accessible via HTTP. You need its URL and the model name.
* **Redis Server:** A running Redis instance (v5.0+) accessible by the bot for history persistence and rate limiting. Can be run via Docker.
* **tiktoken:** The `tiktoken` library (`pip install tiktoken`) is used for token counting if the LLM API doesn't provide usage stats.
* **Discord IDs:** You'll need IDs for:
    * Restricted User Role (`RESTRICTED_USER_ROLE_ID`) - *You must create this role manually in your Discord server(s).*
    * Restricted Channel (`RESTRICTED_CHANNEL_ID`)
    * Ignored Roles (`IGNORED_ROLE_IDS`) (Optional)
    * Rate Limit Exempt Roles (`RATE_LIMIT_EXEMPT_ROLE_IDS`) (Optional)
    * *(Enable Developer Mode in Discord: User Settings > Advanced > Developer Mode. Right-click channels/roles/users to copy IDs.)*

## Configuration (`.env` file)

Create a `.env` file in the project root. Use the following template, replacing placeholder values:

```dotenv
# Discord Bot Configuration
DISCORD_BOT_TOKEN=YOUR_DISCORD_BOT_TOKEN_HERE

# Logging Configuration
LOG_LEVEL=INFO # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

# OpenWebUI/LLM API Configuration
OPENWEBUI_API_URL=http://localhost:8080 # Or your actual API URL (use service name like http://llm:8080 if using docker-compose)
OPENWEBUI_MODEL=your_model_name         # e.g., llama3:latest
OPENWEBUI_API_KEY=your_optional_api_key # Optional API Key for the endpoint

# Channel IDs
RESTRICTED_CHANNEL_ID=YOUR_RESTRICTED_CHANNEL_ID_HERE # Where restricted users must interact

# Bot Behavior
RESPONSE_CHANCE=0.05      # e.g., 0.05 for 5% random response chance
MAX_HISTORY_PER_USER=20   # Max message pairs (user/assistant) stored per user/channel context in Redis

# Redis Configuration
REDIS_HOST=localhost      # Or 'redis' if using docker-compose service name 'redis'
REDIS_PORT=6379
REDIS_DB=0                # Primary DB for history, rate limits, and restriction expiry
# REDIS_PASSWORD=your_redis_password # Uncomment if needed
REDIS_DB_TEST=9           # Separate DB used if running tests in webui_api.py

# Role-based Ignore List (Bot completely ignores users with these roles)
IGNORED_ROLE_IDS= # Comma-separated Role IDs, e.g., 1111,2222

# Rate Limiting & Restriction System
RATE_LIMIT_COUNT=15         # Max messages per user per window before restriction
RATE_LIMIT_WINDOW_SECONDS=60  # Time window (seconds) for message limit
TOKEN_RATE_LIMIT_COUNT=20000  # Max LLM tokens per user per window before restriction
RESTRICTED_USER_ROLE_ID=YOUR_ACTUAL_RESTRICTED_ROLE_ID_HERE # Role assigned when limits are hit
# Notification message templates (use <#{channel_id}> for placeholder)
RATE_LIMIT_MESSAGE_USER="You've sent messages too frequently. To continue using the bot, please use the <#{channel_id}> channel. This restriction will be reviewed periodically."
RESTRICTED_CHANNEL_MESSAGE_USER="Due to previous high activity, you can currently only interact with me in the <#{channel_id}> channel."

# Rate Limiting Exemptions (Users with these roles bypass limits)
RATE_LIMIT_EXEMPT_ROLE_IDS= # Comma-separated Role IDs, e.g., 3333,4444

# Restriction Duration
RESTRICTION_DURATION_SECONDS=86400 # Duration in seconds for restriction (Default: 24 hours)
RESTRICTION_CHECK_INTERVAL_SECONDS=300 # How often to check for expired restrictions (Default: 5 minutes)

# Knowledge & Tools (Optional - for WebUIAPI if used)
LIST_TOOLS=
KNOWLEDGE_ID=
```
# Installation & Running

## Method 1: Running Directly with Python

1. Clone/Create Files: Ensure you have all project files (`main.py`, `bot.py`, `requirements.txt`, `.env`, `Dockerfile`, `docker-compose.yaml`, `cogs/listener_cog.py`, `utils/webui_api.py`, `utils/message_handler.py`).
2. Create Prompts: Create the `utils/prompts` directory and add `personality_prompt.txt` with your desired content.
3. Install Dependencies:
```Bash
pip install -r requirements.txt
```
4. Configure `.env`: Fill in your details in the `.env` file.

5. Setup Redis: Ensure a Redis server is running and accessible at the host/port specified in .env.

6. Setup Discord Role: Manually create the "Restricted User" role in your Discord server(s) and put its ID in `RESTRICTED_USER_ROLE_ID`. Ensure the bot's role is higher in the hierarchy than this restricted role.

7. Run the bot:
```Bash
    python main.py
```
## Method 2: Running with Docker Compose (Recommended)

1. Install Docker and Docker Compose.

2. Clone/Create Files: As above.

3. Create Prompts: As above.

4. Configure `.env`: Fill in your details. Important: If using the provided `docker-compose.yaml`, set `REDIS_HOST=redis` in your `.env` file, as `redis` is the service name within the Docker network.

5. Setup Discord Role: As above.

6. Build and Run: Open a terminal in the project's root directory:
```Bash
docker-compose up --build -d
```
- `--build`: Rebuilds the bot image if code changed.
- `-d`: Runs containers in detached mode (background).
- The `docker-compose.yaml` included starts both the bot and a Redis service.

7. View Logs: `docker-compose logs -f staffai` (or your bot service name)

8. Stop: `docker-compose down`

# Project Structure
```
staffai/
├── .env                  # Environment variables (sensitive, DO NOT COMMIT)
├── Dockerfile            # Instructions to build the bot Docker image
├── docker-compose.yaml   # Defines bot and Redis services for Docker
├── main.py               # Entry point, loads config, starts bot
├── bot.py                # Defines the AIBot class (instantiates WebUIAPI, Redis client), loads cogs
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── specs.txt             # Detailed project specifications
├── bot.log               # Log file output
│
├── cogs/                 # Discord.py Cogs (extensions)
│   └── listener_cog.py   # Handles Discord events, delegates to MessageHandler, manages restrictions
│
└── utils/                # Utility modules and files
    ├── webui_api.py      # Handles LLM API communication and Redis history (used by AIBot)
    ├── message_handler.py # NEW: Encapsulates message processing logic (engagement, rate limits, LLM calls)
    ├── prompts/          # Directory for prompt template files
    │   ├── personality_prompt.txt
```