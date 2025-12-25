#!/bin/bash
#
# Deploy a DeepAgents app Docker container to remote prod2 server
#
# Usage:
#   ./apps/deploy_to_prod2.sh <app_name> [port]
#
# Example:
#   ./apps/deploy_to_prod2.sh business_cofounder_api
#   ./apps/deploy_to_prod2.sh business_cofounder_worker
#   ./apps/deploy_to_prod2.sh business_cofounder_api 8003
#

set -e

if [ $# -lt 1 ]; then
  echo "Usage: $0 <app_name> [port]"
  echo ""
  echo "Examples:"
  echo "  $0 business_cofounder_api"
  echo "  $0 business_cofounder_worker"
  echo "  $0 business_cofounder_api 8003"
  exit 1
fi

APP_NAME="$1"
PORT="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${SCRIPT_DIR}/${APP_NAME}"

if [ ! -d "$APP_DIR" ]; then
  echo "Error: App directory not found: $APP_DIR"
  exit 1
fi

DEPLOY_ENV_FILE="${DEPLOY_ENV_FILE:-${SCRIPT_DIR}/.deploy.env}"
if [ -f "$DEPLOY_ENV_FILE" ]; then
  echo "Loading deploy environment from $DEPLOY_ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$DEPLOY_ENV_FILE"
  set +a
fi

# Registry configuration (prefer values from .deploy.env)
REGISTRY="${ALIYUN_DOCKER_REGISTRY:-${REGISTRY:-}}"
USERNAME="${ALIYUN_DOCKER_USERNAME:-${USERNAME:-}}"

# Default image name based on app name (can be overridden in .deploy.env)
DEFAULT_IMAGE_NAME="aihehuo/${APP_NAME//_/-}"
IMAGE_NAME="${DOCKER_IMAGE_NAME:-${IMAGE_NAME:-${DEFAULT_IMAGE_NAME}}}"
TAG="${TAG:-${DOCKER_IMAGE_TAG:-latest}}"

if [ -z "$REGISTRY" ] || [ -z "$USERNAME" ]; then
  echo "Error: registry/username not configured."
  echo "Set ALIYUN_DOCKER_REGISTRY and ALIYUN_DOCKER_USERNAME (recommended via ${DEPLOY_ENV_FILE})."
  exit 1
fi

FULL_IMAGE="$REGISTRY/$IMAGE_NAME:$TAG"

# Remote server configuration
REMOTE_HOST="${REMOTE_HOST:-prod2}"
REMOTE_USER="${REMOTE_USER:-root}"

# Default remote directory based on app name
DEFAULT_REMOTE_DIR="/mnt/${APP_NAME//_/-}"
REMOTE_DIR="${REMOTE_DIR:-${DEFAULT_REMOTE_DIR}}"

# Default container name based on app name
DEFAULT_CONTAINER_NAME="${APP_NAME//_/-}"
CONTAINER_NAME="${CONTAINER_NAME:-${DEFAULT_CONTAINER_NAME}}"

# Determine if we need port mapping
# For workers, skip port mapping unless explicitly provided
IS_WORKER=false
if [[ "$APP_NAME" == *"worker"* ]]; then
  IS_WORKER=true
fi

# Default container port (8001 for API, can be overridden in .deploy.env)
# Only used if PORT is set
CONTAINER_PORT="${CONTAINER_PORT:-8001}"

# If no port provided and not a worker, use default
if [ -z "$PORT" ] && [ "$IS_WORKER" = false ]; then
  PORT="8001"
fi

if [ -z "$REMOTE_HOST" ]; then
  echo "Error: REMOTE_HOST is not set."
  echo "Set REMOTE_HOST in ${DEPLOY_ENV_FILE} or export REMOTE_HOST=..."
  exit 1
fi

# Container internal config
# For workers, use /home/celery (non-root user); for API, use /home/appuser (non-root user)
if [[ "$APP_NAME" == *"worker"* ]]; then
  CONTAINER_DATA_DIR="/home/celery/.deepagents/${APP_NAME}"
else
  CONTAINER_DATA_DIR="/home/appuser/.deepagents/${APP_NAME}"
fi

if [ -z "$ALIYUN_DOCKER_PASSWORD" ]; then
  echo "Error: ALIYUN_DOCKER_PASSWORD is not set"
  echo "Set it in ${DEPLOY_ENV_FILE} or export it in your shell."
  exit 1
fi

# Local env file (same convention as other scripts)
ENV_FILE="${APP_DIR}/.docker.env"
if [ ! -f "$ENV_FILE" ]; then
  echo "Error: $ENV_FILE not found. Create it (e.g. copy .docker.env.example)."
  exit 1
fi

echo "Deploying $APP_NAME to ${REMOTE_USER}@${REMOTE_HOST}"
echo "Image: $FULL_IMAGE"
echo "Remote dir: $REMOTE_DIR"
if [ -n "$PORT" ]; then
  echo "Port: $PORT -> ${CONTAINER_PORT}"
fi
echo ""

# Build env file content to ship to remote as a temporary file
DOCKER_ENV_FILE_CONTENT=""
while IFS='=' read -r key value; do
  [[ -z "$key" || "$key" =~ ^# ]] && continue
  value=$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
  DOCKER_ENV_FILE_CONTENT+="$key=$value"$'\n'
done < "$ENV_FILE"

# Ensure HOME points to correct directory inside container
# For workers, use /home/celery; for API, use /home/appuser
if [[ "$APP_NAME" == *"worker"* ]]; then
  DOCKER_ENV_FILE_CONTENT+="HOME=/home/celery"$'\n'
else
  DOCKER_ENV_FILE_CONTENT+="HOME=/home/appuser"$'\n'
fi

ssh "$REMOTE_USER@$REMOTE_HOST" bash <<REMOTE_SCRIPT_END
set -e

REGISTRY="$REGISTRY"
USERNAME="$USERNAME"
FULL_IMAGE="$FULL_IMAGE"
CONTAINER_NAME="$CONTAINER_NAME"
PORT="$PORT"
REMOTE_DIR="$REMOTE_DIR"
CONTAINER_PORT="$CONTAINER_PORT"
CONTAINER_DATA_DIR="$CONTAINER_DATA_DIR"
ALIYUN_DOCKER_PASSWORD="$ALIYUN_DOCKER_PASSWORD"
NETWORK_NAME="$NETWORK_NAME"
IS_WORKER="$IS_WORKER"

echo "Logging into registry..."
echo "\$ALIYUN_DOCKER_PASSWORD" | docker login --username "\$USERNAME" --password-stdin "\$REGISTRY"

echo "Preparing directories..."
mkdir -p "\$REMOTE_DIR/data"

# Fix permissions for mounted volume
if [[ "$APP_NAME" == *"worker"* ]]; then
  echo "Setting permissions for celery user (UID 1000)..."
  # Ensure the data directory and its parent structure are owned by celery user
  chown -R 1000:1000 "\$REMOTE_DIR/data" || true
  # Also ensure parent directories exist and are accessible
  mkdir -p "\$(dirname "\$REMOTE_DIR/data")" || true
else
  echo "Setting permissions for appuser (UID 1000) for API container..."
  # For API containers running as appuser (non-root), ensure the directory is owned by UID 1000
  # Ensure the data directory and its parent structure are owned by appuser (UID 1000)
  chown -R 1000:1000 "\$REMOTE_DIR/data" || true
  # Also ensure parent directories exist and are accessible
  mkdir -p "\$(dirname "\$REMOTE_DIR/data")" || true
fi

echo "Pulling image: \$FULL_IMAGE"
docker pull "\$FULL_IMAGE"

if [ "\$(docker ps -a -q -f name=\$CONTAINER_NAME)" ]; then
  echo "Stopping existing container: \$CONTAINER_NAME"
  docker stop "\$CONTAINER_NAME" >/dev/null 2>&1 || true
  echo "Removing existing container: \$CONTAINER_NAME"
  docker rm "\$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

ENV_FILE_TMP="\$REMOTE_DIR/.docker.env.tmp"
cat > "\$ENV_FILE_TMP" <<ENVEOF
$(echo -n "$DOCKER_ENV_FILE_CONTENT")
ENVEOF

# Detect host gateway IP for host.docker.internal (Linux hosts)
HOST_GATEWAY_IP="172.17.0.1"
if command -v ip >/dev/null 2>&1; then
  DOCKER_BRIDGE_IP=\$(ip addr show docker0 2>/dev/null | grep 'inet ' | awk '{print \$2}' | cut -d/ -f1)
  if [ -n "\$DOCKER_BRIDGE_IP" ]; then
    HOST_GATEWAY_IP="\$DOCKER_BRIDGE_IP"
  else
    DOCKER_ROUTE=\$(ip route show | grep '172.17.0.1/16.*docker0' | awk '{for(i=1;i<=NF;i++) if(\$i=="src") print \$(i+1)}')
    if [ -n "\$DOCKER_ROUTE" ]; then
      HOST_GATEWAY_IP="\$DOCKER_ROUTE"
    fi
  fi
fi

echo "Starting container: \$CONTAINER_NAME"
echo "Attached to network: \$NETWORK_NAME"
# Determine user for container (non-root)
if [[ "$APP_NAME" == *"worker"* ]]; then
  CONTAINER_USER="1000:1000"  # celery user
else
  CONTAINER_USER="1000:1000"  # appuser (same UID as celery for consistency)
fi
echo "Running as user: \$CONTAINER_USER (non-root)"

if [ -n "\$PORT" ] && [ "\$IS_WORKER" != "true" ]; then
  docker run -d \\
    --name "\$CONTAINER_NAME" \\
    --user "\$CONTAINER_USER" \\
    --network "\$NETWORK_NAME" \\
    -p "\$PORT:\$CONTAINER_PORT" \\
    -v "\$REMOTE_DIR/data:\$CONTAINER_DATA_DIR" \\
    --add-host=host.docker.internal:\$HOST_GATEWAY_IP \\
    --env-file "\$ENV_FILE_TMP" \\
    -e CELERY_BROKER_URL="$CELERY_BROKER_URL_PROD" \\
    --restart unless-stopped \\
    --security-opt seccomp=unconfined \\
    "\$FULL_IMAGE"
else
  docker run -d \\
    --name "\$CONTAINER_NAME" \\
    --user "\$CONTAINER_USER" \\
    --network "\$NETWORK_NAME" \\
    -v "\$REMOTE_DIR/data:\$CONTAINER_DATA_DIR" \\
    --add-host=host.docker.internal:\$HOST_GATEWAY_IP \\
    --env-file "\$ENV_FILE_TMP" \\
    -e CELERY_BROKER_URL="$CELERY_BROKER_URL_PROD" \\
    --restart unless-stopped \\
    --security-opt seccomp=unconfined \\
    "\$FULL_IMAGE"
fi

rm -f "\$ENV_FILE_TMP"

sleep 3
docker ps -f name="\$CONTAINER_NAME"
docker logs --tail=30 "\$CONTAINER_NAME"

REMOTE_SCRIPT_END

echo ""
echo "Deployment completed."

# Only show health endpoint for HTTP services (not workers) and if port is set
if [[ "$APP_NAME" != *"worker"* ]] && [ -n "$PORT" ] && [ "$CONTAINER_PORT" != "" ] && [ "$CONTAINER_PORT" != "0" ]; then
  echo "Health: http://${REMOTE_HOST}:${PORT}/health"
elif [[ "$APP_NAME" == *"worker"* ]]; then
  echo "Worker is running. Check logs with: docker logs -f $CONTAINER_NAME"
fi

