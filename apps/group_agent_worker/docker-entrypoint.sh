#!/bin/sh
# Group Agent worker entrypoint (REQ-032-FIX1).
# Image already runs as USER celery.
set -eu
mkdir -p "${HOME:-/home/celery}/.deepagents/group_agent_api" 2>/dev/null || true
exec "$@"
