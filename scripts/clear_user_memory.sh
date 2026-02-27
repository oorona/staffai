#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Clear one user's memory from disk + Redis (inside the bot container).

Usage:
  scripts/clear_user_memory.sh --user-id <discord_user_id> [options]

Options:
  --user-id <id>       Required. Discord user ID to purge.
  --guild-id <id>      Optional. If set, scrub only llm_calls:recent:<guild_id>.
                       If omitted, scrub all llm_calls:recent:* lists.
  --container <name>   Docker container name (default: staffai).
  --no-backup          Delete memory file without creating .bak timestamp copy.
  -h, --help           Show this help.

Examples:
  scripts/clear_user_memory.sh --user-id 811781544784035881
  scripts/clear_user_memory.sh --user-id 811781544784035881 --guild-id 1423815821674676267
EOF
}

CONTAINER_NAME="staffai"
USER_ID=""
GUILD_ID=""
NO_BACKUP="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user-id)
      USER_ID="${2:-}"
      shift 2
      ;;
    --guild-id)
      GUILD_ID="${2:-}"
      shift 2
      ;;
    --container)
      CONTAINER_NAME="${2:-}"
      shift 2
      ;;
    --no-backup)
      NO_BACKUP="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$USER_ID" ]]; then
  echo "Error: --user-id is required." >&2
  usage
  exit 1
fi

if ! [[ "$USER_ID" =~ ^[0-9]+$ ]]; then
  echo "Error: --user-id must be numeric." >&2
  exit 1
fi

if [[ -n "$GUILD_ID" ]] && ! [[ "$GUILD_ID" =~ ^[0-9]+$ ]]; then
  echo "Error: --guild-id must be numeric when provided." >&2
  exit 1
fi

if ! docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  echo "Error: container '$CONTAINER_NAME' is not running." >&2
  exit 1
fi

echo "==> Target container: $CONTAINER_NAME"
echo "==> Target user: $USER_ID"
if [[ -n "$GUILD_ID" ]]; then
  echo "==> LLM audit scope: guild $GUILD_ID"
else
  echo "==> LLM audit scope: all guilds (llm_calls:recent:*)"
fi

echo "==> Removing user memory file from /app/data/user_memory/users ..."
docker exec \
  -e TARGET_USER_ID="$USER_ID" \
  -e NO_BACKUP="$NO_BACKUP" \
  "$CONTAINER_NAME" \
  sh -lc '
set -eu
file_path="/app/data/user_memory/users/${TARGET_USER_ID}.json"
if [ -f "$file_path" ]; then
  if [ "${NO_BACKUP}" = "1" ]; then
    rm -f "$file_path"
    echo "file: deleted ${file_path} (no backup)"
  else
    backup_path="${file_path}.bak.$(date +%Y%m%d_%H%M%S)"
    cp "$file_path" "$backup_path"
    rm -f "$file_path"
    echo "file: backup created ${backup_path}"
    echo "file: deleted ${file_path}"
  fi
else
  echo "file: not found ${file_path} (nothing to delete)"
fi
'

echo "==> Removing Redis memory key and scrubbing audit lists ..."
docker exec \
  -e TARGET_USER_ID="$USER_ID" \
  -e TARGET_GUILD_ID="$GUILD_ID" \
  "$CONTAINER_NAME" \
  sh -lc 'python - <<'"'"'PY'"'"'
import json
import os
import sys

import redis

user_id = os.environ["TARGET_USER_ID"].strip()
guild_id = os.environ.get("TARGET_GUILD_ID", "").strip()

redis_host = os.getenv("REDIS_HOST", "redis")
redis_port = int(os.getenv("REDIS_PORT", "6379"))
redis_db = int(os.getenv("REDIS_DB", "0"))
redis_password = os.getenv("REDIS_PASSWORD") or None

try:
    r = redis.Redis(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        password=redis_password,
        decode_responses=True,
    )
    r.ping()
except Exception as e:
    print(f"redis: connection_failed host={redis_host}:{redis_port}/{redis_db} error={e}", file=sys.stderr)
    sys.exit(2)

deleted = r.delete(f"user_memory:{user_id}")
print(f"redis: DEL user_memory:{user_id} -> {deleted}")

def scrub_list(key: str, target_user_id: str) -> int:
    entries = r.lrange(key, 0, -1)
    if not entries:
        return 0

    kept = []
    removed = 0
    for raw in entries:
        try:
            payload = json.loads(raw)
        except Exception:
            kept.append(raw)
            continue

        if str(payload.get("user_id")) == target_user_id:
            removed += 1
        else:
            kept.append(raw)

    if removed > 0:
        pipe = r.pipeline()
        pipe.delete(key)
        if kept:
            pipe.rpush(key, *kept)
        pipe.execute()
    return removed

removed_pipeline = scrub_list("user_memory_pipeline:recent", user_id)
print(f"redis: scrub user_memory_pipeline:recent removed={removed_pipeline}")

if guild_id:
    keys = [f"llm_calls:recent:{guild_id}"]
else:
    keys = [k for k in r.scan_iter("llm_calls:recent:*")]

total_removed = 0
for key in keys:
    removed = scrub_list(key, user_id)
    total_removed += removed
    print(f"redis: scrub {key} removed={removed}")

print(f"redis: llm_calls_removed_total={total_removed}")
PY'

echo "==> Done. User memory purge completed for user $USER_ID."
