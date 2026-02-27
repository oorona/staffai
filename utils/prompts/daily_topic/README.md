Daily Topic Prompt Files
========================

These files control daily-topic generation and are loaded at runtime by `DailyTopicCog`.

Files:
- `topic_generation_system_prompt.txt`
- `topic_generation_user_prompt.txt`
- `body_generation_system_prompt.txt`
- `body_generation_user_prompt.txt`
- `topic_generation_response_schema.json`
- `body_generation_response_schema.json`

Template variables:
- In `topic_generation_user_prompt.txt`:
  - `{{PREVIOUS_TOPIC_RULE}}`
  - `{{CATEGORY}}`
- In `body_generation_user_prompt.txt`:
  - `{{TOPIC_CATEGORY}}`
  - `{{TOPIC_TITLE}}`
  - `{{TOPIC_DESCRIPTION}}`
  - `{{IMPORTANCE_REASONING}}`

Notes:
- Keep output format instructions as valid JSON-only responses.
- Response schemas are loaded at runtime and passed as `response_format`.
- Since these files are under `utils/prompts`, Docker bind-mount already makes them editable live.
