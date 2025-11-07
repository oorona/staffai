# Bot API Test Tool

Interactive CLI tool and automated test suite for testing the Discord LLM bot's API calls, LLM responses, and MCP tool calling functionality.

## Quick Start

```bash
# Interactive mode (menu-based)
./run.sh      # Run test tool
./run.sh -r   # Rebuild and run

# Automated testing (all models, all prompts)
./run.sh -t   # Run automated test suite
./run.sh -rt  # Rebuild and run automated tests
```

## Automated Testing

The automated test suite (`run_automated_test.py`) tests all configured models with all test prompts:

**What it tests:**
- ✅ Tool calling with MCP servers (does LLM select the right tool?)
- ✅ Structured JSON output validation (valid response schema?)
- ✅ Complete flow: tools → execution → structured response

**Usage:**
```bash
# Docker:
./run.sh -t

# Direct Python:
python bot_client/run_automated_test.py
```

**Output:**
- 📊 Comprehensive pass/fail report per model
- 🔧 Tool calling success rate
- 📋 Structured output validation rate
- ⏱️  Average response time per model
- 📝 Detailed error messages for failures

**Configuration:**
- Edit `bot_client/prompts/test_prompts.txt` to add/modify test prompts
- Lines starting with `#` are comments (ignored)
- Each non-comment line is a separate test prompt

The script will:
- Check/create Docker networks (dbnet, bot)
- Build and start the container
- Launch the interactive menu OR run automated tests

## Structure

```
bot_client/
├── bot_api_test.py          ← Interactive test tool
├── run_automated_test.py    ← Automated test suite (NEW)
├── run.sh                   ← Quick start script
├── requirements.txt         ← Python dependencies
├── Dockerfile               ← Docker build config
├── docker-compose.yaml      ← Docker deployment
└── prompts/
    └── test_prompts.txt     ← Test prompts (edit as needed)
```

## Interactive Features

- 🚀 Test single or multiple models
- 📝 Load prompts from file or enter interactively  
- 🔧 Enable/disable MCP tools
- 📊 View Redis data (context, stats, rate limits)
- 💾 Test with real conversation context
- 📈 View test results history

## Main Menu

1. **Run Test** - Test models with configured prompts
2. **Manage Prompts** - Add/edit/remove test prompts
3. **Configure Models** - Select which models to test
4. **Configure MCP** - Enable/disable MCP tools
5. **View Redis Data** - Inspect Redis keys and values
6. **Test with Real Context** - Use actual conversation history
7. **View History** - See past test results
8. **Settings** - Configure test parameters
