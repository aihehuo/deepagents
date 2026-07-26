#!/bin/bash
#
# Example deployment script for a DeepAgents app.
#
# This is a GENERIC template. It intentionally contains no real hostnames,
# users, directories, registries, or credentials. Copy it to your own private,
# untracked deploy script and fill in the required environment variables for
# your infrastructure.
#
# Required environment variables (no defaults — the script fails closed if any
# is missing, so it can never connect, build, push, or deploy to an
# unconfigured target):
#
#   DEPLOY_HOST   Target host to deploy to (e.g. deploy.example.invalid)
#   DEPLOY_USER   SSH user on the target host
#   DEPLOY_DIR    Absolute path of the compose project on the target host
#   IMAGE_REF     Fully-qualified image reference to pull (registry/name:tag)
#
# Usage:
#   DEPLOY_HOST=... DEPLOY_USER=... DEPLOY_DIR=... IMAGE_REF=... \
#     ./apps/deploy_to_prod.example.sh <app_name> [tag]
#
# Example (non-routable placeholder target):
#   DEPLOY_HOST=deploy.example.invalid \
#   DEPLOY_USER=deployer \
#   DEPLOY_DIR=/srv/deepagents \
#   IMAGE_REF=registry.example.invalid/deepagents/business-cofounder-api:latest \
#     ./apps/deploy_to_prod.example.sh business_cofounder_api

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <app_name> [tag]"
  echo ""
  echo "Example:"
  echo "  DEPLOY_HOST=deploy.example.invalid DEPLOY_USER=deployer \\"
  echo "  DEPLOY_DIR=/srv/deepagents \\"
  echo "  IMAGE_REF=registry.example.invalid/deepagents/business-cofounder-api:latest \\"
  echo "    $0 business_cofounder_api"
  exit 1
fi

APP_NAME="$1"
TAG="${2:-latest}"
SERVICE_NAME="${APP_NAME//_/-}"

# Fail closed: every deploy target must be supplied explicitly. There are no
# baked-in production defaults, so a misconfigured environment aborts here
# instead of touching real infrastructure.
: "${DEPLOY_HOST:?DEPLOY_HOST is required}"
: "${DEPLOY_USER:?DEPLOY_USER is required}"
: "${DEPLOY_DIR:?DEPLOY_DIR is required}"
: "${IMAGE_REF:?IMAGE_REF is required}"

# Compose file also has no default — point it at your own private compose file.
COMPOSE_FILE="${COMPOSE_FILE:?COMPOSE_FILE is required (path to your private docker-compose file)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Build and push image (delegates to your own build_and_push.sh).
echo "========================================="
echo "1. Building and pushing image for ${APP_NAME}:${TAG} ..."
echo "========================================="
"${SCRIPT_DIR}/build_and_push.sh" "$APP_NAME" "$TAG"

# 2. Pull and restart the service on the deploy target over SSH.
# The remote host must have the private compose file present at
# ${DEPLOY_DIR}/${COMPOSE_BASENAME}. IMAGE_REF is exported so the compose file's
# ${..._IMAGE:?...} required variable resolves to exactly the image we built.
COMPOSE_BASENAME="$(basename "${COMPOSE_FILE}")"
IMAGE_ENV_VAR="$(echo "${APP_NAME}" | tr '[:lower:]' '[:upper:]')_IMAGE"

echo ""
echo "========================================="
echo "2. Connecting to ${DEPLOY_USER}@${DEPLOY_HOST} and restarting ${SERVICE_NAME} ..."
echo "========================================="
ssh "${DEPLOY_USER}@${DEPLOY_HOST}" bash <<REMOTE_SCRIPT_END
set -euo pipefail
cd "${DEPLOY_DIR}"

# Bind the required image variable for this service to the exact built image.
export ${IMAGE_ENV_VAR}="${IMAGE_REF}"

echo "Pulling image for service: ${SERVICE_NAME} (${IMAGE_REF})"
docker compose -f "${COMPOSE_BASENAME}" pull "${SERVICE_NAME}"

echo "Recreating container for service: ${SERVICE_NAME}"
docker compose -f "${COMPOSE_BASENAME}" up -d "${SERVICE_NAME}"

echo "Checking container status:"
docker compose -f "${COMPOSE_BASENAME}" ps "${SERVICE_NAME}"

echo "Verifying deployed image matches ${IMAGE_REF} ..."
CONTAINER_ID="\$(docker compose -f "${COMPOSE_BASENAME}" ps -q "${SERVICE_NAME}")"
if [ -z "\$CONTAINER_ID" ]; then
  echo "Error: deployed container for ${SERVICE_NAME} is empty" >&2
  exit 1
fi
ACTUAL_IMAGE="\$(docker inspect --format '{{.Config.Image}}' "\$CONTAINER_ID")"
if [ "\$ACTUAL_IMAGE" != "${IMAGE_REF}" ]; then
  echo "Error: deployed image (\$ACTUAL_IMAGE) does not match expected (${IMAGE_REF})" >&2
  exit 1
fi
REMOTE_SCRIPT_END

echo ""
echo "Deployment of ${APP_NAME} completed."
