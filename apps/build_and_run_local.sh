#!/bin/bash
#
# Build and run a DeepAgents app locally for live tests.
#
# This script intentionally does NOT log in to any registry, push images, or
# pull images. It builds from the current working tree and runs the resulting
# local image.
#
# Usage:
#   ./apps/build_and_run_local.sh <app_name> [port]
#
# Examples:
#   ./apps/build_and_run_local.sh business_cofounder_api
#   ./apps/build_and_run_local.sh business_cofounder_api 8003
#   ./apps/build_and_run_local.sh business_cofounder_worker
#

set -e

if [ $# -lt 1 ]; then
  echo "Usage: $0 <app_name> [port]"
  echo ""
  echo "Examples:"
  echo "  $0 business_cofounder_api"
  echo "  $0 business_cofounder_api 8003"
  echo "  $0 business_cofounder_worker"
  exit 1
fi

APP_NAME="$1"
PORT="${2:-${PORT:-}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
APP_DIR="${SCRIPT_DIR}/${APP_NAME}"

if [ ! -d "$APP_DIR" ]; then
  echo "Error: App directory not found: $APP_DIR"
  exit 1
fi

if [ ! -f "${APP_DIR}/Dockerfile" ]; then
  echo "Error: Dockerfile not found: ${APP_DIR}/Dockerfile"
  exit 1
fi

DEFAULT_IMAGE_NAME="aihehuo/${APP_NAME//_/-}-local"
IMAGE_NAME="${LOCAL_IMAGE_NAME:-${IMAGE_NAME:-${DEFAULT_IMAGE_NAME}}}"
TAG="${TAG:-local}"
FULL_IMAGE="${IMAGE_NAME}:${TAG}"

DEFAULT_CONTAINER_NAME="${APP_NAME//_/-}-local"
CONTAINER_NAME="${CONTAINER_NAME:-${DEFAULT_CONTAINER_NAME}}"

IS_WORKER=false
if [[ "$APP_NAME" == *"worker"* ]]; then
  IS_WORKER=true
fi

CONTAINER_PORT="${CONTAINER_PORT:-8001}"
if [ -z "$PORT" ] && [ "$IS_WORKER" = false ]; then
  PORT="8001"
fi

ENV_FILE="${ENV_FILE:-${APP_DIR}/.docker.env}"
ENV_TEMPLATE="${APP_DIR}/docker.env.example"
DATA_DIR="${DATA_DIR:-${APP_DIR}/.tmp_home/.deepagents/${APP_NAME}}"

if [[ "$APP_NAME" == *"worker"* ]]; then
  CONTAINER_HOME="/home/celery"
else
  CONTAINER_HOME="/home/appuser"
fi
CONTAINER_DATA_DIR="${CONTAINER_HOME}/.deepagents/${APP_NAME}"

if [ ! -f "$ENV_FILE" ]; then
  echo "Warning: $ENV_FILE not found."
  if [ -f "$ENV_TEMPLATE" ]; then
    cp "$ENV_TEMPLATE" "$ENV_FILE"
    echo "Created $ENV_FILE from docker.env.example."
    echo "Edit $ENV_FILE and add required API keys before running live tests."
    exit 1
  fi
  echo "Error: $ENV_TEMPLATE not found. Create $ENV_FILE with app environment settings."
  exit 1
fi

mkdir -p "$DATA_DIR"

echo "Building local Docker image"
echo "App:        $APP_NAME"
echo "Image:      $FULL_IMAGE"
echo "Dockerfile: ${APP_DIR}/Dockerfile"
echo "Context:    $REPO_ROOT"
echo ""

docker build \
  -t "$FULL_IMAGE" \
  -f "${APP_DIR}/Dockerfile" \
  "$REPO_ROOT"

if [ "$(docker ps -a -q -f name="^/${CONTAINER_NAME}$")" ]; then
  echo "Stopping existing container: $CONTAINER_NAME"
  docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
  echo "Removing existing container: $CONTAINER_NAME"
  docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

ENV_ARGS=()
while IFS='=' read -r key value; do
  [[ -z "$key" || "$key" =~ ^# ]] && continue
  value=$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
  ENV_ARGS+=("-e" "$key=$value")
done < "$ENV_FILE"

ENV_ARGS+=("-e" "HOME=$CONTAINER_HOME")

HOST_GATEWAY_IP="host-gateway"

echo "Starting local container"
echo "Container: $CONTAINER_NAME"
if [ -n "$PORT" ]; then
  echo "Port:      $PORT -> $CONTAINER_PORT"
fi
echo "Data dir:  $DATA_DIR"
echo ""

if [ -n "$PORT" ] && [ "$IS_WORKER" = false ]; then
  docker run -d \
    --name "$CONTAINER_NAME" \
    -p "$PORT:$CONTAINER_PORT" \
    -v "$DATA_DIR:$CONTAINER_DATA_DIR" \
    --add-host=host.docker.internal:$HOST_GATEWAY_IP \
    "${ENV_ARGS[@]}" \
    "$FULL_IMAGE"
else
  docker run -d \
    --name "$CONTAINER_NAME" \
    -v "$DATA_DIR:$CONTAINER_DATA_DIR" \
    --add-host=host.docker.internal:$HOST_GATEWAY_IP \
    "${ENV_ARGS[@]}" \
    "$FULL_IMAGE"
fi

echo ""
echo "Waiting for container startup..."
sleep 3

if [ "$(docker ps -q -f name="^/${CONTAINER_NAME}$")" ]; then
  echo "Container is running."
  echo ""
  docker ps -f name="$CONTAINER_NAME"
  echo ""
  echo "Logs:"
  docker logs --tail=40 "$CONTAINER_NAME"
  echo ""
  if [ -n "$PORT" ]; then
    echo "Live test environment:"
    echo "  export BC_API_BASE_URL=http://127.0.0.1:${PORT}"
    echo "  export BC_API_LIVE=1"
    echo "  export BC_API_LIVE_E2E=1"
    echo ""
  fi
  echo "Useful commands:"
  echo "  docker logs -f $CONTAINER_NAME"
  echo "  docker stop $CONTAINER_NAME"
  echo "  docker rm $CONTAINER_NAME"
else
  echo "Container failed to start. Check logs with:"
  echo "  docker logs $CONTAINER_NAME"
  exit 1
fi
