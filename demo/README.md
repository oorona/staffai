# Demo - Tool Calling Test# StaffAI Demo - Tool Calling & Structured Output Test



This folder contains the tool calling test that validates MCP function calling and structured output across all configured models.## Overview



## FilesThis demo tests 6 LLM models (via LiteLLM) for **Integrated Tool Calling & Structured Output**:



- `test_tool_calling.py` - Tests each model for tool calling + structured outputThe flow is: **Tool Request → LLM Calls Tool → Tool Executes → Result Fed Back → LLM Returns JSON**

- `run.sh` - Script to run the test (with optional rebuild)

- `docker-compose.yml` - Docker compose configurationSee [INTEGRATED_FLOW.md](INTEGRATED_FLOW.md) for detailed explanation of how this works!

- `Dockerfile` - Docker image definition

- `.env` - Environment configuration## Files

- `response_schema.json` - JSON schema for structured output

- `test_tool_calling.py` - Main test script

## Usage- `docker-compose.yml` - Docker setup with networks and secrets

- `.env` - Configuration (LITELLM_API_URL, LITELLM_MODELS, MCP_SERVERS)

### Run Test- `secrets/litellm_api_key.txt` - API key for LiteLLM proxy

```bash- `personality_prompt.txt` - Bot personality instructions

./run.sh- `response_schema.json` - Expected JSON schema for structured output

```- `run.sh` - Helper script to run tests



### Rebuild and Run## Quick Start

```bash

./run.sh -r### Prerequisites

```- Docker and Docker Compose installed

- LiteLLM proxy running on `http://litellm:4000`

## What It Tests- MCP servers running (tenormcp, cvemcp, pistonmcp, wamcp, usersmcp, ytmcp)

- Networks created: `bot`, `dbnet`, `intranet` (external Docker networks)

For each model in `LITELLM_MODELS`:

1. **Tool Calling**: Verifies LLM calls the correct MCP tool### Run Tests

2. **Tool Execution**: Simulates tool execution with mock results

3. **Structured Output**: Validates final response is valid JSON matching schema```bash

# Run from demo folder

## Expected Outputcd demo



```# Run test (will show detailed LLM outputs)

════════════════════════════════════════════════════════════════════════════════./run.sh test

🤖 MODEL 1/6: gemini/gemini-2.5-flash-lite

════════════════════════════════════════════════════════════════════════════════# Rebuild Docker image and run

./run.sh -r test

  📋 INTEGRATED TEST: TOOL CALLING → TOOL EXECUTION → STRUCTURED OUTPUT

  User Request: 'Find me a funny cat image'# Or directly with docker-compose

  Tools Available: 47docker compose run --rm test python3 -u test_tool_calling.py

```

  Step 1: LLM attempts to call tools

    ✅ Response received from LLM## What the Test Does



  Step 2: Check tool execution### Step 0: Configuration

    ✅ YES - LLM called 1 tool(s):Shows what's loaded:

- ✅ API Key (from Docker secrets)

  Step 3: Execute tools and collect results- ✅ Personality Prompt (from file)

    • Executing: search_tenor_gifs- ✅ Response Schema (from JSON file)

      Args: {"query": "funny cat"}- ✅ Models (6 total)

      ✅ Tool result: https://media.tenor.com/images/funny-cat-gif.gif- ✅ MCP Servers (from environment)



  Step 4: Send tool results to LLM for structured response### Step 1: Initialize Client

    ✅ Structured response receivedCreates OpenAI-compatible client for LiteLLM proxy



  Step 5: Validate structured output### Step 2: Load MCP Tools

    ✅ VALID JSON responseConnects to MCP servers and loads tools:

       Type: gif- Shows DNS resolution status

       Response: Here's a cat that's probably judging your life choices...- Shows tool count per server

       Data (URL): https://media.tenor.com/images/funny-cat-gif.gif- Displays first few tool names

- Falls back to 1 test tool if MCP fails

  🎯 RESULT FOR gemini/gemini-2.5-flash-lite:

     Overall: ✅ PASS### Step 3: Test Each Model

     Tool Calling: ✅

     Structured Output: ✅**TEST A: Function Calling**

``````

Prompt: "Find me a funny cat image"

## Reference ImplementationWith: 8 MCP tools available

Expects: Model calls search_tenor_gifs or similar tool

This test represents the **correct** way to implement MCP tool calling:

- Uses `tool_choice="auto"` Output shows:

- Fresh conversations (no history)- Raw response content (if any)

- Raw message objects- Which tools were called

- No max_tokens limit- Tool arguments used

- Explicit timeout=60.0```



If the main bot has issues but this test passes, compare the bot code line-by-line with `test_tool_calling.py`.**TEST B: Structured Output**

```
Prompt: "Tell me a joke about programming"
With: Personality prompt + response schema enforced
Expects: Valid JSON with {type, response, data} fields

Output shows:
- Full raw JSON response
- Validation result
- Field values and lengths
- Failure reason (if invalid)
```

## Expected Results

When LiteLLM and MCP servers are running:

```
Model                               Function Calling     Structured Output   
---------------------------------------------------------------------------
gemini/gemini-2.5-flash-lite        ✅ PASS               ✅ PASS              
gemini/gemini-2.5-flash             ✅ PASS               ✅ PASS              
xai/grok-3-mini                     ✅ PASS               ✅ PASS              
xai/grok-3-mini-fast                ✅ PASS               ✅ PASS              
openai/gpt-5-mini                   ✅ PASS               ✅ PASS              
openai/gpt-5-nano                   ✅ PASS               ✅ PASS              

Function Calling: 6/6 models
Structured Output: 6/6 models
```

## Troubleshooting

### "API Key: ❌ NOT SET"
- Check `secrets/litellm_api_key.txt` exists
- Verify docker-compose.yml mounts the secret

### "DNS failed: cvemcp"
- MCP server not running or on different network
- Check network is `intranet`
- Verify DNS resolution: `docker exec staffai-test nslookup cvemcp`

### "Connection error" on LLM calls
- LiteLLM proxy not running
- Start: `docker compose -f ../docker-compose.yaml up litellm`
- Verify: `curl http://litellm:4000/health`

### "INVALID response: Invalid JSON"
- LLM not respecting structured output format
- Check `response_schema.json` is valid
- Verify model supports response_format parameter

## Fully Self-Contained

Everything is in the `demo/` folder:
- ✅ `test_tool_calling.py` - Main script
- ✅ `docker-compose.yml` - Docker setup
- ✅ `.env` - Configuration
- ✅ `secrets/` - API keys
- ✅ `personality_prompt.txt` - Bot personality
- ✅ `response_schema.json` - Expected output format
- ✅ `run.sh` - Helper script

No dependencies on parent `staffai` folder except:
- Docker images (python:3.13-slim)
- Docker networks (bot, dbnet, intranet) - must exist

## Detailed Output Example

When working, test shows:

```
📋 TEST A: FUNCTION CALLING WITH TOOLS
Step A1: Prompt & tools
  Prompt: 'Find me a funny cat image'
  Tools: 8 available
Step A2: Call model with tool_choice=auto
  ✅ Response received
  📝 Response content: {"type":"function_calls","calls":[...]}

Step A3: Check if model called tools
  ✅ YES - Called 1 tool(s)
     • search_tenor_gifs
       Args: {"query":"funny cat"}

📋 TEST B: STRUCTURED OUTPUT (NO TOOLS)
Step B1: Prompt
  Prompt: 'Tell me a joke about programming'
Step B2: Call model WITHOUT tools but WITH structured output
  ✅ Response received
  📝 Raw response (178 chars):
     {"type":"text","response":"Why do programmers prefer dark mode? Because light attracts bugs.","data":""}

Step B3: Validate structured output
  ✅ VALID JSON response
     Type: text
     Response: Why do programmers prefer dark mode? Because light attracts bugs.
     Data length: 0 chars
```

## Architecture

```
test_tool_calling.py
├── load_mcp_tools() → MCP Servers (tenormcp, etc.)
├── validate_structured_output() → Response Schema validation
├── llm_client → LiteLLM Proxy → Models
└── PERSONALITY_PROMPT → System message for models

Docker Networks:
- bot (LiteLLM, shared services)
- dbnet (Redis)
- intranet (MCP servers, host utilities)
```
