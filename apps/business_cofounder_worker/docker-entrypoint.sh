#!/bin/bash
# Docker entrypoint script for Celery worker
# Fixes permissions for mounted volumes
# This script runs as root, then switches to celery user before executing the command

set -e

# Only fix permissions if running as root (during container startup)
if [ "$(id -u)" = "0" ]; then
    # Ensure .deepagents directory exists and is writable by celery user
    if [ -d "/home/celery/.deepagents" ]; then
        # Fix ownership if needed (in case volume was mounted as root)
        chown -R celery:celery /home/celery/.deepagents 2>/dev/null || true
        chmod -R u+rwX /home/celery/.deepagents 2>/dev/null || true
    fi

    # Ensure the specific app directory exists and is writable
    APP_DIR="/home/celery/.deepagents/business_cofounder_api"
    if [ ! -d "$APP_DIR" ]; then
        mkdir -p "$APP_DIR" 2>/dev/null || true
    fi
    chown -R celery:celery "$APP_DIR" 2>/dev/null || true
    chmod -R u+rwX "$APP_DIR" 2>/dev/null || true

    # Ensure docs directory exists
    DOCS_DIR="$APP_DIR/docs"
    if [ ! -d "$DOCS_DIR" ]; then
        mkdir -p "$DOCS_DIR" 2>/dev/null || true
    fi
    chown -R celery:celery "$DOCS_DIR" 2>/dev/null || true
    chmod -R u+rwX "$DOCS_DIR" 2>/dev/null || true

    # Switch to celery user and execute the command using su
    # Build the command string with proper quoting
    CMD_STR=""
    for arg in "$@"; do
        CMD_STR="$CMD_STR '${arg//\'/\'\\\'\'}'"
    done
    exec su celery -s /bin/bash -c "exec $CMD_STR"
else
    # Already running as celery user, just execute the command
    exec "$@"
fi

