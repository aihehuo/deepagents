#!/bin/bash
# Docker entrypoint script for Group Agent API
# Container runs as non-root user (UID 1000)

set -e

APPUSER_HOME="/home/appuser"
if [ ! -d "$APPUSER_HOME" ]; then
    mkdir -p "$APPUSER_HOME"
fi

DEEPAGENTS_DIR="${APPUSER_HOME}/.deepagents"
if [ ! -d "$DEEPAGENTS_DIR" ]; then
    mkdir -p "$DEEPAGENTS_DIR"
fi

APP_DIR="${DEEPAGENTS_DIR}/group_agent_api"
if [ ! -d "$APP_DIR" ]; then
    mkdir -p "$APP_DIR"
fi

# Verify the app runtime directory is writable
if [ ! -w "$APP_DIR" ]; then
    echo "Error: Runtime directory '$APP_DIR' is not writable by UID $(id -u)" >&2
    exit 1
fi

# Execute the main command
exec "$@"
