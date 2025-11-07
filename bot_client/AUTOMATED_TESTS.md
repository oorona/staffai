Automated Test Suite - Quick Reference
======================================

## Running Tests

### Docker (Recommended):
```bash
cd bot_client
./run.sh -t                  # Run automated tests
./run.sh -rt                 # Rebuild and run tests
```

### Direct Python:
```bash
python bot_client/run_automated_test.py
```

## Test Configuration

### Test Prompts (bot_client/prompts/test_prompts.txt):
```
# Lines starting with # are comments
give me a gif of a cat
show me a funny dog gif
search for celebration gifs
```

### Models Tested:
- Reads from `.env`: `LITELLM_MODELS` or `LITELLM_MODEL`
- Tests ALL configured models automatically

### MCP Servers:
- Reads from `.env`: `MCP_SERVERS`
- Tests tool calling if servers are configured

## Test Flow

For each model + prompt combination:
1. ✅ Load MCP tools (once, cached for all tests)
2. ✅ Call LLM with tools available
3. ✅ Check if tool was called (if applicable)
4. ✅ Validate structured JSON output
5. ✅ Report success/failure

## Output Report

```
┌─────────────────────────────────────────────────┐
│           Overall Results                        │
├─────────────┬────────┬────────┬────────┬────────┤
│ Model       │ Total  │ ✓ OK   │ ✗ Fail │ 🔧 Tool│
├─────────────┼────────┼────────┼────────┼────────┤
│ gpt-4o      │   3    │   3    │   0    │   3    │
│ claude-3    │   3    │   2    │   1    │   2    │
│ gemini-2.5  │   3    │   3    │   0    │   3    │
└─────────────┴────────┴────────┴────────┴────────┘

Final Summary:
  Total Tests: 9
  Successful: 8 (88.9%)
  Failed: 1 (11.1%)
  
  Tool Calling Success: 8/9 (88.9%)
  Structured Output Valid: 8/9 (88.9%)
```

## Success Criteria

✅ PASS if:
- Structured JSON output is valid
- Contains required fields: `type`, `response`, `data`
- No exceptions during execution

❌ FAIL if:
- Invalid JSON
- Missing required fields
- Exception during LLM call
- Timeout

## Exit Codes

- `0` - All tests passed
- `1` - Some tests failed
- `130` - Interrupted by user

## Adding New Tests

Edit `bot_client/prompts/test_prompts.txt`:
```
# Your test category
test prompt 1
test prompt 2

# Another category  
test prompt 3
```

The automated test will pick up changes automatically!
