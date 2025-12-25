#!/bin/bash
# Docker entrypoint script for Business Co-Founder API
# Ensures directories exist and are writable
# Container runs as non-root user (UID 1000) via --user flag

set -e

CURRENT_UID=$(id -u)
CURRENT_GID=$(id -g)

# Ensure .deepagents directory exists and is writable
# When running with --user 1000:1000, we're already the correct user
APPUSER_HOME="/home/appuser"
if [ ! -d "$APPUSER_HOME" ]; then
    mkdir -p "$APPUSER_HOME" 2>/dev/null || true
fi

# Use appuser's home directory for .deepagents (matching the volume mount)
# The volume is mounted to /home/appuser/.deepagents/business_cofounder_api
if [ ! -d "/home/appuser/.deepagents" ]; then
    mkdir -p "/home/appuser/.deepagents" 2>/dev/null || true
fi

# Ensure the specific app directory exists and is writable
APP_DIR="/home/appuser/.deepagents/business_cofounder_api"
if [ ! -d "$APP_DIR" ]; then
    mkdir -p "$APP_DIR" 2>/dev/null || true
fi

# Ensure subdirectories exist and are writable
for subdir in docs skills; do
    SUBDIR_PATH="$APP_DIR/$subdir"
    if [ ! -d "$SUBDIR_PATH" ]; then
        mkdir -p "$SUBDIR_PATH" 2>/dev/null || true
    fi
done

# Execute the original command (already running as non-root user)
exec "$@"

