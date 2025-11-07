#!/bin/bash

# Launcher for tool calling test with optional rebuild
# Usage:
#   ./run.sh           # Run tool calling test
#   ./run.sh -r        # Rebuild and run tool calling test

REBUILD=""

# Check if rebuild flag is provided
if [ "$1" == "-r" ] || [ "$1" == "--rebuild" ]; then
    REBUILD="-r"
fi

# Change to project root if running from demo dir
if [ "$(basename "$PWD")" == "demo" ]; then
    cd ..
fi

if [ "$REBUILD" == "-r" ]; then
    echo "🔨 Rebuilding Docker image..."
    if [ -f "demo/docker-compose.yml" ] && (command -v docker-compose &> /dev/null || docker compose version &> /dev/null); then
        cd demo
        if command -v docker-compose &> /dev/null; then
            docker-compose build --no-cache test || exit 1
        else
            docker compose build --no-cache test || exit 1
        fi
        cd ..
    else
        docker build -f demo/Dockerfile -t staffai-validation-test . || exit 1
    fi
fi

echo "🚀 Running tool calling test..."

# Try to use docker compose first (has networks), fall back to docker run
if [ -f "demo/docker-compose.yml" ] && (command -v docker-compose &> /dev/null || docker compose version &> /dev/null); then
    if command -v docker-compose &> /dev/null; then
        cd demo
        docker-compose run --rm test python3 test_tool_calling.py
        cd ..
    else
        cd demo
        docker compose run --rm test python3 test_tool_calling.py
        cd ..
    fi
else
    docker run --rm \
        --network bot \
        --network dbnet \
        --env LITELLM_API_URL \
        --env LITELLM_API_KEY \
        --env LITELLM_MODELS \
        --env MCP_SERVERS \
        staffai-validation-test python3 test_tool_calling.py
fi
