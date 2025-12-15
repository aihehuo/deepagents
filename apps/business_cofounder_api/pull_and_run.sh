#!/bin/bash
#
# Pull and run Business Co-Founder API Docker image from remote registry
# Modeled after BPGenerationAgent/pull_and_run.sh (DO NOT edit the BPGenerationAgent script).
#
# This script:
# - logs into Aliyun Container Registry
# - pulls the image
# - runs the API on port 8001 (configurable)
# - bind-mounts a local folder to /root/.deepagents/business_cofounder_api for checkpoint persistence
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

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
IMAGE_NAME="${DOCKER_IMAGE_NAME:-${IMAGE_NAME:-aihehuo/business-cofounder-api}}"
TAG="${TAG:-${DOCKER_IMAGE_TAG:-latest}}"

if [ -z "$REGISTRY" ] || [ -z "$USERNAME" ]; then
  echo "Error: registry/username not configured."
  echo "Set ALIYUN_DOCKER_REGISTRY and ALIYUN_DOCKER_USERNAME (recommended via ${DEPLOY_ENV_FILE})."
  exit 1
fi

FULL_IMAGE="$REGISTRY/$IMAGE_NAME:$TAG"

# Container configuration
CONTAINER_NAME="${CONTAINER_NAME:-business-cofounder-api-remote}"
PORT="${PORT:-8002}"
echo "Container name: $CONTAINER_NAME"
echo "Port: $PORT"

# Local persistence folder (host) -> container checkpoint folder
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/.tmp_home/.deepagents/business_cofounder_api}"
CONTAINER_DATA_DIR="/root/.deepagents/business_cofounder_api"

# Check if ALIYUN_DOCKER_PASSWORD is set (can be sourced from .deploy.env)
if [ -z "$ALIYUN_DOCKER_PASSWORD" ]; then
  echo "Error: ALIYUN_DOCKER_PASSWORD is not set"
  echo "Set it in ${DEPLOY_ENV_FILE} or export it in your shell."
  exit 1
fi

# Check if .docker.env exists (in this folder)
ENV_FILE="${SCRIPT_DIR}/.docker.env"
if [ ! -f "$ENV_FILE" ]; then
  echo "⚠️  Warning: $ENV_FILE not found"
  echo "Creating from template..."
  if [ -f "${SCRIPT_DIR}/.docker.env.example" ]; then
    cp "${SCRIPT_DIR}/.docker.env.example" "$ENV_FILE"
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

# Prepare env vars from .docker.env
ENV_ARGS=()
while IFS='=' read -r key value; do
  [[ -z "$key" || "$key" =~ ^# ]] && continue
  value=$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
  ENV_ARGS+=("-e" "$key=$value")
done < "$ENV_FILE"

# Ensure HOME points to /root inside container
ENV_ARGS+=("-e" "HOME=/root")

# Host gateway mapping (kept similar to BPGenerationAgent for consistency)
HOST_GATEWAY_IP="host-gateway"

echo "🚀 Starting container: $CONTAINER_NAME"
echo "📦 Image: $FULL_IMAGE"
echo "🌐 Port: $PORT"
echo "📁 Data (host): $DATA_DIR"
echo "📁 Data (container): $CONTAINER_DATA_DIR"
echo ""

docker run -d \
  --name "$CONTAINER_NAME" \
  -p "$PORT:8001" \
  -v "$DATA_DIR:$CONTAINER_DATA_DIR" \
  --add-host=host.docker.internal:$HOST_GATEWAY_IP \
  "${ENV_ARGS[@]}" \
  --restart unless-stopped \
  "$FULL_IMAGE"

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
  echo "🔗 Access the API at:"
  echo "   - Health Check: http://localhost:$PORT/health"
  echo "   - Chat:         http://localhost:$PORT/chat"
  echo ""
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


