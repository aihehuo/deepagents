#!/bin/bash
#
# Build and push Docker image for a DeepAgents app.
# Registry + username are loaded from app's .deploy.env (recommended) or environment variables.
#
# Usage:
#   ./apps/build_and_push.sh <app_name> [tag]
#
# Example:
#   ./apps/build_and_push.sh business_cofounder_api
#   ./apps/build_and_push.sh business_cofounder_worker 0.0.1
#

set -e

if [ $# -lt 1 ]; then
  echo "Usage: $0 <app_name> [tag]"
  echo ""
  echo "Examples:"
  echo "  $0 business_cofounder_api"
  echo "  $0 business_cofounder_worker 0.0.1"
  exit 1
fi

APP_NAME="$1"
TAG="${2:-latest}"

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

DEPLOY_ENV_FILE="${DEPLOY_ENV_FILE:-${SCRIPT_DIR}/.deploy.env}"
if [ -f "$DEPLOY_ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$DEPLOY_ENV_FILE"
  set +a
fi

REGISTRY="${ALIYUN_DOCKER_REGISTRY:-${REGISTRY:-}}"
USERNAME="${ALIYUN_DOCKER_USERNAME:-${USERNAME:-}}"

# Default image name based on app name (can be overridden in .deploy.env)
DEFAULT_IMAGE_NAME="aihehuo/${APP_NAME//_/-}"
IMAGE_NAME="${DOCKER_IMAGE_NAME:-${IMAGE_NAME:-${DEFAULT_IMAGE_NAME}}}"

if [ -z "$REGISTRY" ] || [ -z "$USERNAME" ]; then
  echo "Error: registry/username not configured."
  echo "Set ALIYUN_DOCKER_REGISTRY and ALIYUN_DOCKER_USERNAME (recommended via ${DEPLOY_ENV_FILE})."
  exit 1
fi

if [ -z "$ALIYUN_DOCKER_PASSWORD" ]; then
  echo "Error: ALIYUN_DOCKER_PASSWORD is not set"
  echo "Set it in ${DEPLOY_ENV_FILE} or export it in your shell."
  exit 1
fi

echo "Building and pushing image"
echo "App:      $APP_NAME"
echo "Registry: $REGISTRY"
echo "Image:    $IMAGE_NAME:$TAG"
echo "Context:  $REPO_ROOT"
echo ""

echo "$ALIYUN_DOCKER_PASSWORD" | docker login --username "$USERNAME" --password-stdin "$REGISTRY"

docker build -t "$IMAGE_NAME:$TAG" -f "${APP_DIR}/Dockerfile" "$REPO_ROOT"
docker build -t "$IMAGE_NAME:latest" -f "${APP_DIR}/Dockerfile" "$REPO_ROOT"

FULL_IMAGE="$REGISTRY/$IMAGE_NAME:$TAG"
FULL_IMAGE_LATEST="$REGISTRY/$IMAGE_NAME:latest"

docker tag "$IMAGE_NAME:$TAG" "$FULL_IMAGE"
docker tag "$IMAGE_NAME:latest" "$FULL_IMAGE_LATEST"

docker push "$FULL_IMAGE"
docker push "$FULL_IMAGE_LATEST"

echo ""
echo "Pushed:"
echo " - $FULL_IMAGE"
echo " - $FULL_IMAGE_LATEST"

