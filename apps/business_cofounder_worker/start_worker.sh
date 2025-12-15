#!/bin/bash
# Start Celery worker for Business Co-Founder

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default pool (solo = single-threaded, avoids thread creation issues)
POOL="${CELERY_POOL:-solo}"
LOGLEVEL="${CELERY_LOGLEVEL:-info}"

echo "Starting Business Co-Founder Celery Worker"
echo "============================================"
echo "Project root: $PROJECT_ROOT"
echo "Pool: $POOL"
echo "Log level: $LOGLEVEL"
echo ""

# Check Redis connection (default to host.docker.internal:6379)
if [ -n "$CELERY_BROKER_URL" ]; then
    echo "Using broker: $CELERY_BROKER_URL"
else
    export CELERY_BROKER_URL="redis://host.docker.internal:6379/0"
    echo "Using default broker: $CELERY_BROKER_URL"
fi

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/libs/deepagents:$PROJECT_ROOT/libs/deepagents-cli:$PYTHONPATH"

# Start worker
cd "$PROJECT_ROOT"
exec celery -A apps.business_cofounder_worker.celery_app worker \
    --loglevel="$LOGLEVEL" \
    --pool="$POOL" \
    "$@"

