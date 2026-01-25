# Installation Guide

This guide covers the prerequisites, dependencies, and setup instructions for StaffAI.

---

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.8+ | Runtime environment |
| Redis | 5.0+ | Conversation persistence, rate limiting |
| LiteLLM Proxy | Latest | Universal LLM gateway |
| Docker (optional) | 20.10+ | Containerized deployment |
| Docker Compose (optional) | 2.0+ | Multi-container orchestration |

### Discord Bot Setup

1. **Create Discord Application**
   - Go to [Discord Developer Portal](https://discord.com/developers/applications)
   - Click "New Application" and name it
   - Navigate to "Bot" section and click "Add Bot"

2. **Enable Privileged Intents**
   - Under "Bot" → "Privileged Gateway Intents":
     - ✅ Server Members Intent
     - ✅ Message Content Intent

3. **Get Bot Token**
   - Under "Bot" → "Token" → "Copy"
   - Keep this secret!

4. **Set Bot Permissions**
   - Under "OAuth2" → "URL Generator":
     - Scopes: `bot`, `applications.commands`
     - Bot Permissions: `View Channels`, `Send Messages`, `Manage Roles`
   - Use generated URL to invite bot to your server

5. **Create Restricted User Role**
   - In your Discord server, create a role named "Restricted User"
   - Ensure the bot's role is **higher** in the role hierarchy
   - Copy the Role ID (Developer Mode → Right-click role → Copy ID)

---

## Installation Methods

### Method 1: Direct Python Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/staffai.git
   cd staffai
   ```

2. **Create Virtual Environment (Recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # or
   .venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration (see Configuration Guide)
   ```

5. **Create Prompt Files**
   ```bash
   mkdir -p utils/prompts
   # Create personality_prompt.txt with bot personality
   # Create base_activity_system_prompt.txt for activity generation
   ```

6. **Ensure External Services**
   - Redis server running and accessible
   - LiteLLM proxy running and accessible

7. **Run the Bot**
   ```bash
   python main.py
   ```

### Method 2: Docker Installation (Recommended for Production)

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/staffai.git
   cd staffai
   ```

2. **Create Secret Files**
   ```bash
   mkdir -p secrets
   
   # Discord bot token
   echo "YOUR_DISCORD_BOT_TOKEN" > secrets/discord_bot_token.txt
   
   # LiteLLM API key
   echo "YOUR_LITELLM_API_KEY" > secrets/litellm_api_key.txt
   
   # Secure the files
   chmod 600 secrets/*.txt
   ```

3. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env - Set REDIS_HOST=redis for Docker networking
   ```

4. **Create External Networks** (if not existing)
   ```bash
   docker network create bot
   docker network create dbnet
   ```

5. **Build and Run**
   ```bash
   docker-compose up --build -d
   ```

6. **View Logs**
   ```bash
   docker-compose logs -f staffai
   ```

7. **Stop the Bot**
   ```bash
   docker-compose down
   ```

---

## Dependencies

### Python Packages

```txt
# Core
discord.py>=2.3.0      # Discord API wrapper
python-dotenv>=1.0.0   # Environment configuration
litellm                # Universal LLM gateway
openai>=1.0.0          # AsyncOpenAI client
redis>=5.0.0           # Redis client

# MCP Support
fastmcp>=2.0.0         # Model Context Protocol client
aiohttp>=3.9.0         # Async HTTP client

# Utilities
psutil                 # System statistics
rich                   # Formatted terminal output
```

### External Services

| Service | Purpose | Default Port |
|---------|---------|--------------|
| Redis | Data persistence | 6379 |
| LiteLLM Proxy | LLM gateway | 4000 |
| MCP Servers | Tool calling | Varies |

---

## Verification

After installation, verify the bot starts correctly:

```bash
python main.py
```

Expected output:
```
INFO:__main__:Logging level set to: INFO (20)
INFO:__main__:Loading environment variables from .env file...
INFO:__main__:Retrieving and validating configuration...
INFO:utils.litellm_client:Initialized LiteLLM client with model: <model_name>
INFO:bot:Attempting to load cogs...
INFO:discord.client:Logged in as <BotName> (ID: <BotID>)
```

---

## Troubleshooting

### Bot doesn't start

1. **Check .env file exists** and has required variables
2. **Verify Redis connectivity**: `redis-cli ping` should return `PONG`
3. **Check LiteLLM proxy**: Ensure it's accessible at configured URL
4. **Review logs**: Check `bot.log` for detailed error messages

### Bot doesn't respond

1. **Check bot is in server** and has correct permissions
2. **Verify privileged intents** are enabled in Discord Developer Portal
3. **Check RESPONSE_CHANCE** isn't set to 0
4. **Review IGNORED_ROLE_IDS** to ensure user isn't being ignored

### Docker networking issues

1. **Create external networks** before running docker-compose
2. **Check network names** match in docker-compose.yaml
3. **Verify REDIS_HOST=redis** in .env for Docker

---

## Next Steps

- [Configuration Reference](CONFIGURATION.md) — Configure all environment variables
- [Architecture Overview](ARCHITECTURE.md) — Understand the codebase structure
