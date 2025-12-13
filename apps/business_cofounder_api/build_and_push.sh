#!/bin/bash
#
# Build and push Docker image for Business Co-Founder API.
# Registry + username are loaded from .deploy.env (recommended) or environment variables.
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

REGISTRY="${ALIYUN_DOCKER_REGISTRY:-${REGISTRY:-}}"
USERNAME="${ALIYUN_DOCKER_USERNAME:-${USERNAME:-}}"
IMAGE_NAME="${DOCKER_IMAGE_NAME:-${IMAGE_NAME:-aihehuo/business-cofounder-api}}"
TAG="${TAG:-${DOCKER_IMAGE_TAG:-0.0.1}}"

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
echo "Registry: $REGISTRY"
echo "Image:    $IMAGE_NAME:$TAG"
echo "Context:  $REPO_ROOT"

echo "$ALIYUN_DOCKER_PASSWORD" | docker login --username "$USERNAME" --password-stdin "$REGISTRY"

docker build -t "$IMAGE_NAME:$TAG" -f "$REPO_ROOT/apps/business_cofounder_api/Dockerfile" "$REPO_ROOT"
docker build -t "$IMAGE_NAME:latest" -f "$REPO_ROOT/apps/business_cofounder_api/Dockerfile" "$REPO_ROOT"

FULL_IMAGE="$REGISTRY/$IMAGE_NAME:$TAG"
FULL_IMAGE_LATEST="$REGISTRY/$IMAGE_NAME:latest"

docker tag "$IMAGE_NAME:$TAG" "$FULL_IMAGE"
docker tag "$IMAGE_NAME:latest" "$FULL_IMAGE_LATEST"

docker push "$FULL_IMAGE"
docker push "$FULL_IMAGE_LATEST"

echo "Pushed:"
echo " - $FULL_IMAGE"
echo " - $FULL_IMAGE_LATEST"


