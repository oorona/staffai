#!/bin/bash
# Bot API Test Tool - Quick Start Script

set -e

# Parse arguments
REBUILD=false
AUTO_TEST=false
if [ "$1" == "--rebuild" ] || [ "$1" == "-r" ]; then
    REBUILD=true
elif [ "$1" == "--test" ] || [ "$1" == "-t" ]; then
    AUTO_TEST=true
    echo "💡 Tip: Prompt file changes are live-mounted - no rebuild needed!"
elif [ "$1" == "--rebuild-test" ] || [ "$1" == "-rt" ]; then
    REBUILD=true
    AUTO_TEST=true
fi

echo "🚀 Bot API Test Tool"
echo "===================="
if [ "$AUTO_TEST" = true ]; then
    echo "   Mode: AUTOMATED TEST"
    if [ "$REBUILD" = true ]; then
        echo "        (with rebuild)"
    fi
elif [ "$REBUILD" = true ]; then
    echo "   Mode: REBUILD + RUN"
else
    echo "   Mode: RUN (use -r to rebuild, -t for automated test)"
fi
echo

# Check if we're in the right directory
if [ ! -f "bot_api_test.py" ]; then
    echo "❌ Error: bot_api_test.py not found"
    echo "   Please run from bot_client/ directory"
    exit 1
fi

# Check if parent .env exists
if [ ! -f "../.env" ]; then
    echo "❌ Error: ../.env not found"
    echo "   Please create .env file in parent directory"
    exit 1
fi

# Detect docker compose command
COMPOSE_CMD=""
if docker compose version &> /dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    echo "❌ Error: Docker Compose not found"
    exit 1
fi

# Check Docker networks
echo "🔍 Checking Docker networks..."
if ! docker network inspect dbnet &> /dev/null; then
    echo "   Creating 'dbnet' network..."
    docker network create dbnet
fi
if ! docker network inspect bot &> /dev/null; then
    echo "   Creating 'bot' network..."
    docker network create bot
fi
echo "✅ Networks ready"
echo

# Remove existing container
if docker ps -a --format '{{.Names}}' | grep -q '^bot_api_test$'; then
    echo "🧹 Removing existing container..."
    docker rm -f bot_api_test > /dev/null 2>&1
fi

# Build and start
if [ "$REBUILD" = true ]; then
    echo "🔨 Rebuilding Docker image..."
    $COMPOSE_CMD build --no-cache
fi

# Run automated test if requested
if [ "$AUTO_TEST" = true ]; then
    echo "🧪 Running automated test suite..."
    echo
    $COMPOSE_CMD run --rm bot_api_test python /app/bot_client/run_automated_test.py
    echo
    echo "✅ Tests complete!"
    exit $?
fi

echo "🐳 Starting container..."
$COMPOSE_CMD up -d

echo
echo "✅ Container started!"
echo "🎯 Attaching to interactive session..."
echo "   Press Ctrl+C to detach (container keeps running)"
echo

# Attach to the container
docker attach bot_api_test
