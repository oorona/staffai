User Memory Prompt Files
========================

Runtime-editable files used to maintain compact per-user memory:
- `memory_update_system_prompt.txt`
- `memory_update_user_prompt.txt`
- `memory_update_response_schema.json`
- `tiny_worthwhile_system_prompt.txt`
- `tiny_worthwhile_user_prompt.txt`
- `tiny_worthwhile_response_schema.json`
- `tiny_extract_system_prompt.txt`
- `tiny_extract_user_prompt.txt`
- `tiny_extract_response_schema.json`
- `tiny_compact_system_prompt.txt`
- `tiny_compact_user_prompt.txt`
- `tiny_compact_response_schema.json`
- `frontier_core_extract_system_prompt.txt`
- `frontier_core_extract_user_prompt.txt`
- `frontier_core_extract_response_schema.json`

Pipeline usage:
- `tiny_extract` mode: extracts short memory pieces and appends them; when threshold is reached it compacts via tiny compact files.
- `frontier_pipeline` mode: uses tiny worthwhile files first, then memory_update files.
- `tiny_gate_frontier_core` mode: uses tiny worthwhile files first, then frontier_core_extract files.
- `disabled` mode: no extraction calls.

Template variables:
- `{{CURRENT_MEMORY}}`
- `{{USER_MESSAGE}}`
- `{{ACCUMULATED_MEMORY}}`

Behavior notes:
- Output must remain short and useful for conversation personalization.
- Files are loaded at runtime; edits apply without image rebuild.
