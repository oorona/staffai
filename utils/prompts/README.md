Prompt Packs
============

All runtime prompts/schemas are consolidated under `utils/prompts/<purpose>/`.

Chat response pack files:
- `system_prompt.txt`
- `persona_prompt.txt`
- `schema.json`

Dynamic persona placeholder:
- `{DYNAMIC_STYLE_TRAITS}` can be used in `persona_prompt.txt` (default and channel overrides).
- `{DYNAMIC_STYLE_TRAITS|be calm, courteous, uplifting, practical.}` sets an inline fallback in the prompt itself.
- `{DYNAMIC_EXPERTISE_LEVEL|intermediate}` can be used in `persona_prompt.txt` to adapt response depth.
- It is replaced at runtime with a learned style line (for example: `be direct, practical, courteous.`).
- Expertise placeholder is replaced at runtime with `beginner`, `intermediate`, or `advanced`.
- Style resolution order: learned per-user style -> prompt inline fallback (channel override if present) -> global default line.
- Expertise resolution order: learned per-user expertise -> prompt inline fallback (channel override if present) -> `intermediate`.

All other purpose folders contain:
- `system_prompt.txt`
- `user_prompt.txt`
- `schema.json`

Current purpose folders:
- `chat_response` - Main conversation system/persona/schema pack.
- `activity_status` - Presence/status generation pack.
- `daily_topic_topic_generation` - Daily-topic proposal generation pack.
- `daily_topic_body_generation` - Daily-topic post body generation pack.
- `user_memory_frontier_update` - Frontier full-memory update pack.
- `user_memory_tiny_worthwhile` - Tiny worthwhile-message classifier pack.
- `user_memory_tiny_extract` - Tiny direct memory extraction pack.
- `user_memory_tiny_compact` - Tiny accumulated-memory compaction pack.
- `user_memory_frontier_core_extract` - Frontier core-memory extraction pack.
- `user_memory_injection_guard` - Frontier prompt-injection guard for memory/profile pipeline (blocks only high confidence).
- `user_memory_injection_guard` also produces a user-facing notice string consumed directly by runtime when blocked (same user language and matching user tone/style).
- `user_style_extract` - Tiny communication-style trait extraction pack (reassessed every 8 worthwhile messages).
- `user_expertise_extract` - Tiny global expertise-level extraction pack (same reassessment cadence as style).

Channel overrides:
- `utils/prompts/channels/<channel_id>/chat_response/system_prompt.txt`
- `utils/prompts/channels/<channel_id>/chat_response/persona_prompt.txt`
