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

# group_agent_api / group_agent_worker: SHA-only tags, clean tree, no :latest push
_require_group_agent_sha_tag() {
  local app="$1"
  if [ $# -lt 2 ] || [ -z "${2:-}" ] || [ "${2:-}" = "latest" ]; then
    echo "Error: $app requires an explicit full 40-character commit SHA tag. 'latest' or empty tag is rejected."
    exit 1
  fi
  if [[ ! "$TAG" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "Error: $app tag must be a full 40-character commit SHA, got: $TAG"
    exit 1
  fi
  CURRENT_HEAD="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)"
  if [ "$TAG" != "$CURRENT_HEAD" ]; then
    echo "Error: Provided tag ($TAG) does not match current git HEAD ($CURRENT_HEAD)"
    exit 1
  fi
  TRACKED_CHANGES="$(git -C "$REPO_ROOT" status --porcelain -uno 2>/dev/null || true)"
  if [ -n "$TRACKED_CHANGES" ]; then
    echo "Error: Working tree has tracked modifications. Commit all changes before building/deploying $app:"
    echo "$TRACKED_CHANGES"
    exit 1
  fi
}

if [ "$APP_NAME" = "group_agent_api" ]; then
  _require_group_agent_sha_tag "$APP_NAME" "${2:-}"
  COPY_UNTRACKED="$(git -C "$REPO_ROOT" status --porcelain libs/deepagents apps/group_agent_api 2>/dev/null | grep '^\?\?' || true)"
  if [ -n "$COPY_UNTRACKED" ]; then
    echo "Error: Docker COPY source paths (libs/deepagents, apps/group_agent_api) contain untracked files:"
    echo "$COPY_UNTRACKED"
    exit 1
  fi
fi

if [ "$APP_NAME" = "group_agent_worker" ]; then
  _require_group_agent_sha_tag "$APP_NAME" "${2:-}"
  COPY_UNTRACKED="$(git -C "$REPO_ROOT" status --porcelain libs/deepagents apps/group_agent_api apps/group_agent_worker 2>/dev/null | grep '^\?\?' || true)"
  if [ -n "$COPY_UNTRACKED" ]; then
    echo "Error: Docker COPY source paths (libs/deepagents, apps/group_agent_api, apps/group_agent_worker) contain untracked files:"
    echo "$COPY_UNTRACKED"
    exit 1
  fi
fi

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
FULL_IMAGE="$REGISTRY/$IMAGE_NAME:$TAG"
docker tag "$IMAGE_NAME:$TAG" "$FULL_IMAGE"
docker push "$FULL_IMAGE"

if [ "$APP_NAME" != "group_agent_api" ] && [ "$APP_NAME" != "group_agent_worker" ]; then
  docker build -t "$IMAGE_NAME:latest" -f "${APP_DIR}/Dockerfile" "$REPO_ROOT"
  FULL_IMAGE_LATEST="$REGISTRY/$IMAGE_NAME:latest"
  docker tag "$IMAGE_NAME:latest" "$FULL_IMAGE_LATEST"
  docker push "$FULL_IMAGE_LATEST"
fi

echo ""
echo "Pushed:"
echo " - $FULL_IMAGE"
if [ "$APP_NAME" != "group_agent_api" ] && [ "$APP_NAME" != "group_agent_worker" ]; then
  echo " - $FULL_IMAGE_LATEST"
fi

echo ""
echo "Cleaning up local old images for ${REGISTRY}/${IMAGE_NAME} (retaining latest 3)..."
LOCAL_REPO_TO_PRUNE="${REGISTRY}/${IMAGE_NAME}"
LOCAL_IMAGE_IDS="$(docker images "$LOCAL_REPO_TO_PRUNE" --format '{{.CreatedAt}}	{{.ID}}' | sort -r | awk -F'	' '{print $2}' | awk '!seen[$0]++')"
LOCAL_TOTAL_COUNT="$(echo "$LOCAL_IMAGE_IDS" | grep -v '^$' | wc -l | tr -d ' ')"
if [ "$LOCAL_TOTAL_COUNT" -gt 3 ]; then
  LOCAL_OLD_IMAGE_IDS="$(echo "$LOCAL_IMAGE_IDS" | tail -n +4)"
  LOCAL_USED_IMAGE_IDS="$(docker ps -a --format '{{.Image}}' | tr ' ' '\n' | sort -u)"
  for img_id in $LOCAL_OLD_IMAGE_IDS; do
    if [ -n "$img_id" ]; then
      if echo "$LOCAL_USED_IMAGE_IDS" | grep -q "$img_id"; then
        echo "  Skipping $img_id: in use by a local container"
      else
        echo "  Removing old local image: $img_id"
        docker rmi "$img_id" 2>/dev/null || true
      fi
    fi
  done
else
  echo "  Local image count ($LOCAL_TOTAL_COUNT) <= 3. No old local images pruned."
fi
docker image prune -f >/dev/null 2>&1 || true

