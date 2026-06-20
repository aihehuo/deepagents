#!/bin/bash
#
# Pull and run a DeepAgents app Docker image from remote registry
#
# Usage:
#   ./apps/pull_and_run.sh <app_name> [port]
#
# Example:
#   ./apps/pull_and_run.sh business_cofounder_api
#   ./apps/pull_and_run.sh business_cofounder_worker
#   ./apps/pull_and_run.sh business_cofounder_api 8003
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
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
APP_DIR="${SCRIPT_DIR}/${APP_NAME}"

if [ ! -d "$APP_DIR" ]; then
  echo "Error: App directory not found: $APP_DIR"
  exit 1
fi

DEPLOY_ENV_FILE="${DEPLOY_ENV_FILE:-${SCRIPT_DIR}/.deploy.env}"
if [ -f "$DEPLOY_ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$DEPLOY_ENV_FILE"
  set +a
fi

# Configuration (values can come from .deploy.env or environment)
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

# Container configuration
DEFAULT_CONTAINER_NAME="${APP_NAME//_/-}-remote"
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

echo "App:            $APP_NAME"
echo "Container name: $CONTAINER_NAME"
if [ -n "$PORT" ]; then
  echo "Port:           $PORT -> $CONTAINER_PORT"
fi
echo ""

# Local persistence folder (host) -> container checkpoint folder
DATA_DIR="${DATA_DIR:-${APP_DIR}/.tmp_home/.deepagents/${APP_NAME}}"

<<<<<<< HEAD
# For workers, use /home/celery (non-root user); for API, use /root
if [[ "$APP_NAME" == *"worker"* ]]; then
  CONTAINER_DATA_DIR="/home/celery/.deepagents/${APP_NAME}"
else
  CONTAINER_DATA_DIR="/root/.deepagents/${APP_NAME}"
fi
=======
# Container internal config.
# For workers, use /home/celery; for API, use /home/appuser.
if [[ "$APP_NAME" == *"worker"* ]]; then
  CONTAINER_HOME="/home/celery"
else
  CONTAINER_HOME="/home/appuser"
fi
CONTAINER_DATA_DIR="${CONTAINER_HOME}/.deepagents/${APP_NAME}"
CONTAINER_USER="${CONTAINER_USER:-1000:1000}"
>>>>>>> main

# Check if ALIYUN_DOCKER_PASSWORD is set (can be sourced from .deploy.env)
if [ -z "$ALIYUN_DOCKER_PASSWORD" ]; then
  echo "Error: ALIYUN_DOCKER_PASSWORD is not set"
  echo "Set it in ${DEPLOY_ENV_FILE} or export it in your shell."
  exit 1
fi

# Check if .docker.env exists (in app folder)
ENV_FILE="${APP_DIR}/.docker.env"
if [ ! -f "$ENV_FILE" ]; then
  echo "⚠️  Warning: $ENV_FILE not found"
  echo "Creating from template..."
  if [ -f "${APP_DIR}/.docker.env.example" ]; then
    cp "${APP_DIR}/.docker.env.example" "$ENV_FILE"
    echo "✅ Created $ENV_FILE from template"
    echo "⚠️  Please edit $ENV_FILE and add your API keys before continuing"
    read -p "Press Enter to continue after editing $ENV_FILE, or Ctrl+C to cancel..."
  else
    echo "❌ Error: Neither $ENV_FILE nor .docker.env.example found"
    echo "Please create $ENV_FILE with your API keys"
    exit 1
  fi
fi

echo "Logging into Docker registry..."
echo "$ALIYUN_DOCKER_PASSWORD" | docker login --username "$USERNAME" --password-stdin "$REGISTRY"

# Stop and remove existing container if it exists
if [ "$(docker ps -a -q -f name=$CONTAINER_NAME)" ]; then
  echo "🛑 Stopping existing container: $CONTAINER_NAME"
  docker stop "$CONTAINER_NAME" > /dev/null 2>&1 || true
  echo "🗑️  Removing existing container: $CONTAINER_NAME"
  docker rm "$CONTAINER_NAME" > /dev/null 2>&1 || true
fi

echo "📥 Pulling Docker image: $FULL_IMAGE"
docker pull "$FULL_IMAGE"

# Create persistence directory
mkdir -p "$DATA_DIR"

<<<<<<< HEAD
# Fix permissions for non-root user (if app is a worker)
if [[ "$APP_NAME" == *"worker"* ]]; then
  echo "Setting permissions for celery user (UID 1000)..."
  sudo chown -R 1000:1000 "$DATA_DIR" 2>/dev/null || chown -R 1000:1000 "$DATA_DIR" 2>/dev/null || echo "⚠️  Note: Could not set permissions. You may need to run: chown -R 1000:1000 $DATA_DIR"
fi
=======
# Fix permissions for the non-root container user.
echo "Setting permissions for container user ($CONTAINER_USER)..."
sudo chown -R "$CONTAINER_USER" "$DATA_DIR" 2>/dev/null || chown -R "$CONTAINER_USER" "$DATA_DIR" 2>/dev/null || echo "⚠️  Note: Could not set permissions. You may need to run: chown -R $CONTAINER_USER $DATA_DIR"
>>>>>>> main

# Prepare env vars from .docker.env
ENV_ARGS=()
while IFS='=' read -r key value; do
  [[ -z "$key" || "$key" =~ ^# ]] && continue
  value=$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
  ENV_ARGS+=("-e" "$key=$value")
done < "$ENV_FILE"

<<<<<<< HEAD
# Ensure HOME points to correct directory inside container
# For workers, use /home/celery; for API, use /root
if [[ "$APP_NAME" == *"worker"* ]]; then
  ENV_ARGS+=("-e" "HOME=/home/celery")
else
  ENV_ARGS+=("-e" "HOME=/root")
fi
=======
# Ensure HOME points to the non-root user's home directory inside container.
ENV_ARGS+=("-e" "HOME=$CONTAINER_HOME")
>>>>>>> main

# Host gateway mapping
HOST_GATEWAY_IP="host-gateway"

echo "🚀 Starting container: $CONTAINER_NAME"
echo "📦 Image: $FULL_IMAGE"
if [ -n "$PORT" ]; then
  echo "🌐 Port: $PORT -> $CONTAINER_PORT"
fi
<<<<<<< HEAD
=======
echo "👤 User:  $CONTAINER_USER"
>>>>>>> main
echo "📁 Data (host): $DATA_DIR"
echo "📁 Data (container): $CONTAINER_DATA_DIR"
echo ""

if [ -n "$PORT" ] && [ "$IS_WORKER" = false ]; then
  docker run -d \
    --name "$CONTAINER_NAME" \
<<<<<<< HEAD
=======
    --user "$CONTAINER_USER" \
>>>>>>> main
    -p "$PORT:$CONTAINER_PORT" \
    -v "$DATA_DIR:$CONTAINER_DATA_DIR" \
    --add-host=host.docker.internal:$HOST_GATEWAY_IP \
    "${ENV_ARGS[@]}" \
    --restart unless-stopped \
    "$FULL_IMAGE"
else
  docker run -d \
    --name "$CONTAINER_NAME" \
<<<<<<< HEAD
=======
    --user "$CONTAINER_USER" \
>>>>>>> main
    -v "$DATA_DIR:$CONTAINER_DATA_DIR" \
    --add-host=host.docker.internal:$HOST_GATEWAY_IP \
    "${ENV_ARGS[@]}" \
    --restart unless-stopped \
    "$FULL_IMAGE"
fi

echo ""
echo "⏳ Waiting for container to start..."
sleep 3

if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
  echo "✅ Container is running!"
  echo ""
  docker ps -f name="$CONTAINER_NAME"
  echo ""
  echo "📝 Container logs (last 30 lines):"
  docker logs --tail=30 "$CONTAINER_NAME"
  echo ""
  # Only show health endpoint if port is set and it's likely an HTTP service
  if [ -n "$PORT" ] && [ "$CONTAINER_PORT" != "" ] && [ "$CONTAINER_PORT" != "0" ]; then
    echo "🔗 Access the service at:"
    echo "   - Health Check: http://localhost:$PORT/health"
    echo ""
  fi
  echo "📋 Useful commands:"
  echo "   - View logs: docker logs -f $CONTAINER_NAME"
  echo "   - Stop:      docker stop $CONTAINER_NAME"
  echo "   - Remove:    docker rm $CONTAINER_NAME"
  echo "   - Restart:   docker restart $CONTAINER_NAME"
else
  echo "❌ Container failed to start. Check logs with:"
  echo "   docker logs $CONTAINER_NAME"
  exit 1
fi
<<<<<<< HEAD

=======
>>>>>>> main
