#!/bin/bash
#
# Deploy a DeepAgents app to prod3 using docker-compose.prod3.yml
#
# Usage:
#   ./apps/deploy_to_prod3.sh <app_name> [tag]
#
# Example:
#   ./apps/deploy_to_prod3.sh business_cofounder_api
#   ./apps/deploy_to_prod3.sh wu_tanchang_api 0.1.0
#   ./apps/deploy_to_prod3.sh group_agent_api <40-char-sha>
#   ./apps/deploy_to_prod3.sh group_agent_worker <40-char-sha>
#

set -e

if [ $# -lt 1 ]; then
  echo "Usage: $0 <app_name> [tag]"
  echo ""
  echo "Examples:"
  echo "  $0 business_cofounder_api"
  echo "  $0 wu_tanchang_api 0.1.0"
  echo "  $0 group_agent_api <40-char-sha>"
  echo "  $0 group_agent_worker <40-char-sha>"
  exit 1
fi

APP_NAME="$1"
TAG="${2:-latest}"
SERVICE_NAME="${APP_NAME//_/-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

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

COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.prod3.yml"
if [ ! -f "$COMPOSE_FILE" ]; then
  echo "Error: Compose file not found: $COMPOSE_FILE"
  exit 1
fi

if ! grep -q "^  ${SERVICE_NAME}:" "$COMPOSE_FILE"; then
  echo "Error: Service ${SERVICE_NAME} not found in ${COMPOSE_FILE}"
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
DEFAULT_IMAGE_NAME="aihehuo/${APP_NAME//_/-}"
IMAGE_NAME="${DOCKER_IMAGE_NAME:-${IMAGE_NAME:-${DEFAULT_IMAGE_NAME}}}"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"

REMOTE_EXPORT=""
COMPOSE_SERVICES="$SERVICE_NAME"
VERIFY_IMAGE_STRICT=0
if [ "$APP_NAME" = "group_agent_api" ]; then
  REMOTE_EXPORT="export GROUP_AGENT_API_IMAGE=\"${FULL_IMAGE}\""
  VERIFY_IMAGE_STRICT=1
elif [ "$APP_NAME" = "group_agent_worker" ]; then
  # Same image for worker replicas + dedicated beat.
  REMOTE_EXPORT="export GROUP_AGENT_WORKER_IMAGE=\"${FULL_IMAGE}\""
  COMPOSE_SERVICES="group-agent-worker group-agent-beat"
  VERIFY_IMAGE_STRICT=1
elif [ "$APP_NAME" = "wu_tanchang_api" ]; then
  REMOTE_EXPORT="export WU_TANCHANG_API_IMAGE=\"${FULL_IMAGE}\""
  VERIFY_IMAGE_STRICT=1
elif [ "$APP_NAME" = "business_cofounder_api" ]; then
  REMOTE_EXPORT="export BUSINESS_COFOUNDER_API_IMAGE=\"${FULL_IMAGE}\""
  VERIFY_IMAGE_STRICT=1
fi

# 1. Build and push image for linux/amd64
echo "========================================="
echo "1. Building and pushing image for linux/amd64..."
echo "========================================="
DOCKER_DEFAULT_PLATFORM=linux/amd64 "$SCRIPT_DIR/build_and_push.sh" "$APP_NAME" "$TAG"

# 2. Deploy on prod3 via docker-compose.prod3.yml
echo ""
echo "========================================="
echo "2. SSH to prod3 and pulling/restarting service(s): $COMPOSE_SERVICES"
echo "========================================="
ssh root@prod3 bash <<REMOTE_SCRIPT_END
set -e
cd /mnt/deepagents
${REMOTE_EXPORT}

echo "Pulling image for service(s): $COMPOSE_SERVICES ($FULL_IMAGE)"
# shellcheck disable=SC2086
docker compose -f docker-compose.prod3.yml pull $COMPOSE_SERVICES

echo "Recreating container(s) for service(s): $COMPOSE_SERVICES"
# shellcheck disable=SC2086
docker compose -f docker-compose.prod3.yml up -d $COMPOSE_SERVICES

echo "Waiting for container to stabilize..."
sleep 5

echo "Checking container status:"
# shellcheck disable=SC2086
docker compose -f docker-compose.prod3.yml ps $COMPOSE_SERVICES

echo "Container logs:"
# shellcheck disable=SC2086
docker compose -f docker-compose.prod3.yml logs --tail 30 $COMPOSE_SERVICES

echo "Verifying deployed container image reference & ID:"
for svc in $COMPOSE_SERVICES; do
  CONTAINER_ID="\$(docker compose -f docker-compose.prod3.yml ps -q "\$svc")"
  if [ -z "\$CONTAINER_ID" ]; then
    echo "Error: Deployed container ID for \$svc is empty!" >&2
    exit 1
  fi

  ACTUAL_IMAGE="\$(docker inspect --format '{{.Config.Image}}' "\$CONTAINER_ID")"
  if [ "$VERIFY_IMAGE_STRICT" = "1" ] && [ "\$ACTUAL_IMAGE" != "$FULL_IMAGE" ]; then
    echo "Error: Deployed container image for \$svc (\$ACTUAL_IMAGE) does not match expected image ($FULL_IMAGE)!" >&2
    exit 1
  fi

  IMAGE_ID="\$(docker inspect --format '{{.Image}}' "\$CONTAINER_ID")"
  echo "Deployment verified for \$svc:"
  echo "  Container ID:    \$CONTAINER_ID"
  echo "  Image Reference: \$ACTUAL_IMAGE"
  echo "  Image ID:        \$IMAGE_ID"
done

echo ""
echo "Cleaning up old images for ${REGISTRY}/${IMAGE_NAME} on prod3 (retaining latest 3)..."
REPO_TO_PRUNE="${REGISTRY}/${IMAGE_NAME}"
IMAGE_IDS="\$(docker images "\$REPO_TO_PRUNE" --format '{{.CreatedAt}}	{{.ID}}' | sort -r | awk -F'	' '{print \$2}' | awk '!seen[\$0]++')"
TOTAL_COUNT="\$(echo "\$IMAGE_IDS" | grep -v '^$' | wc -l | tr -d ' ')"
if [ "\$TOTAL_COUNT" -gt 3 ]; then
  OLD_IMAGE_IDS="\$(echo "\$IMAGE_IDS" | tail -n +4)"
  USED_IMAGE_IDS="\$(docker ps -a --format '{{.Image}}' | tr ' ' '\n' | sort -u)"
  for img_id in \$OLD_IMAGE_IDS; do
    if [ -n "\$img_id" ]; then
      if echo "\$USED_IMAGE_IDS" | grep -q "\$img_id"; then
        echo "  Skipping \$img_id: in use by a container"
      else
        echo "  Removing old image: \$img_id"
        docker rmi "\$img_id" 2>/dev/null || true
      fi
    fi
  done
else
  echo "  Image count (\$TOTAL_COUNT) <= 3. No old images pruned."
fi
docker image prune -f >/dev/null 2>&1 || true
REMOTE_SCRIPT_END

echo ""
echo "Deployment to prod3 completed successfully!"
