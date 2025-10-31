# OpenWebUI MCP Test Client

Standalone test tool to validate OpenWebUI API calls with multiple MCP servers.

## Setup

1. Create and activate a virtual environment:
   ```bash
   # Navigate to the test_mcp_client folder
   cd test_mcp_client
   
   # Create virtual environment
   python3 -m venv venv
   
   # Activate virtual environment
   # On Linux/Mac:
   source venv/bin/activate
   
   # On Windows:
   # venv\Scripts\activate
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

4. Edit `.env` with your actual configuration:
   ```bash
   nano .env  # or your preferred editor
   ```

5. Configure the following variables:
   - `OPENWEBUI_API_URL`: Your OpenWebUI instance URL
   - `OPENWEBUI_MODEL`: The model name to use
   - `OPENWEBUI_API_KEY`: Your API key (if required)
   - `LIST_TOOLS`: Comma-separated list of MCP servers (e.g., `server:mcp:1,server:mcp:2`)
   - `PERSONALITY_PROMPT_PATH`: Path to personality prompt file (default: `../utils/prompts/personality_prompt.txt`)

## Usage

```bash
# Test both streaming and non-streaming modes
python test_mcp_client.py "dame un trending gif"

# Test only streaming mode (should work with multiple MCP)
python test_mcp_client.py "hello" --stream

# Test only non-streaming mode (may fail with multiple MCP)
python test_mcp_client.py "hello" --no-stream

# Explicitly test both modes
python test_mcp_client.py "gif de un gato bailando" --both

# Run comprehensive test suite (RECOMMENDED for debugging)
python test_mcp_client.py "hello" --test-all
```

## Comprehensive Test Mode (`--test-all`)

The `--test-all` mode runs 5 sequential tests to isolate MCP server issues:

1. **Test 1: No MCP servers** (stream=True)
   - Baseline test without any MCP servers
   - Should always work

2. **Test 2: First MCP server only** (stream=True)
   - Tests with `server:mcp:1` alone
   - Should work

3. **Test 3: Second MCP server only** (stream=True)
   - Tests with `server:mcp:2` alone
   - Should work

4. **Test 4: Both MCP servers** (stream=False)
   - Tests with both servers using non-streaming
   - **Expected to FAIL** due to OpenWebUI TaskGroup bug

5. **Test 5: Both MCP servers** (stream=True)
   - Tests with both servers using streaming
   - Should work (this is the fix)

This helps identify exactly where the issue occurs.

## What It Tests

### Non-Streaming Mode (`stream=False`)
- Tests the original approach that fails with multiple MCP servers
- Shows the exact error response from OpenWebUI
- Helps identify the TaskGroup cleanup bug

### Streaming Mode (`stream=True`)
- Tests the workaround that should work with multiple MCP servers
- Shows how the response is assembled from chunks
- Validates the streaming approach used by the bot

## Output

The tool shows:
- ✅ Configuration loaded from `.env`
- 📤 Full request payload sent to OpenWebUI
- 📥 Response status and headers
- 📝 Complete response content
- 🔍 JSON parsing validation
- 📊 Usage statistics (tokens)
- ❌ Detailed error messages if failures occur

## Example

```bash
# Navigate to the test folder
cd test_mcp_client

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Setup configuration
cp .env.example .env
nano .env  # Edit with your settings

# Run the test
python test_mcp_client.py "dame un trending gif" --both

# Or run comprehensive test suite
python test_mcp_client.py "hello" --test-all
```

This will run both non-streaming (likely to fail) and streaming (should work) tests, helping you confirm the MCP bug and validate the fix.

The `--test-all` mode is particularly useful as it runs 5 different test scenarios to pinpoint exactly where the MCP issue occurs.

## Deactivating the Virtual Environment

When you're done testing:
```bash
deactivate
```
