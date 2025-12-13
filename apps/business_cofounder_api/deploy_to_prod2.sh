#!/bin/bash
#
# Deploy Business Co-Founder API Docker container to remote prod2 server
# Modeled after BPGenerationAgent/deploy_to_prod2.sh (DO NOT edit BPGenerationAgent scripts).
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEPLOY_ENV_FILE="${DEPLOY_ENV_FILE:-${SCRIPT_DIR}/.deploy.env}"
if [ -f "$DEPLOY_ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$DEPLOY_ENV_FILE"
  set +a
fi

# Registry configuration (prefer values from .deploy.env)
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

# Remote server configuration
REMOTE_HOST="${REMOTE_HOST:-${REMOTE_HOST:-}}"
REMOTE_USER="${REMOTE_USER:-${REMOTE_USER:-root}}"
REMOTE_DIR="${REMOTE_DIR:-${REMOTE_DIR:-/mnt/business-cofounder-api}}"
CONTAINER_NAME="${CONTAINER_NAME:-${CONTAINER_NAME:-business-cofounder-api}}"
PORT="${PORT:-${PORT:-8001}}"

if [ -z "$REMOTE_HOST" ]; then
  echo "Error: REMOTE_HOST is not set."
  echo "Set REMOTE_HOST in ${DEPLOY_ENV_FILE} or export REMOTE_HOST=..."
  exit 1
fi

# Container internal config
CONTAINER_PORT="8001"
CONTAINER_DATA_DIR="/root/.deepagents/business_cofounder_api"

if [ -z "$ALIYUN_DOCKER_PASSWORD" ]; then
  echo "Error: ALIYUN_DOCKER_PASSWORD is not set"
  exit 1
fi

# Local env file (same convention as other scripts)
ENV_FILE="${SCRIPT_DIR}/.docker.env"
if [ ! -f "$ENV_FILE" ]; then
  echo "Error: $ENV_FILE not found. Create it (e.g. copy .docker.env.example)."
  exit 1
fi

echo "Deploying Business Co-Founder API to ${REMOTE_USER}@${REMOTE_HOST}"
echo "Image: $FULL_IMAGE"
echo "Remote dir: $REMOTE_DIR"
echo "Port: $PORT -> ${CONTAINER_PORT}"
echo ""

# Build env file content to ship to remote as a temporary file
DOCKER_ENV_FILE_CONTENT=""
while IFS='=' read -r key value; do
  [[ -z "$key" || "$key" =~ ^# ]] && continue
  value=$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
  DOCKER_ENV_FILE_CONTENT+="$key=$value"$'\n'
done < "$ENV_FILE"

# Ensure HOME is /root for Path.home() usage and checkpoint location
DOCKER_ENV_FILE_CONTENT+="HOME=/root"$'\n'

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

echo "Logging into registry..."
echo "\$ALIYUN_DOCKER_PASSWORD" | docker login --username "\$USERNAME" --password-stdin "\$REGISTRY"

echo "Preparing directories..."
mkdir -p "\$REMOTE_DIR/data"

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
    DOCKER_ROUTE=\$(ip route show | grep '172.17.0.0/16.*docker0' | awk '{for(i=1;i<=NF;i++) if(\$i=="src") print \$(i+1)}')
    if [ -n "\$DOCKER_ROUTE" ]; then
      HOST_GATEWAY_IP="\$DOCKER_ROUTE"
    fi
  fi
fi

echo "Starting container: \$CONTAINER_NAME"
docker run -d \\
  --name "\$CONTAINER_NAME" \\
  -p "\$PORT:\$CONTAINER_PORT" \\
  -v "\$REMOTE_DIR/data:\$CONTAINER_DATA_DIR" \\
  --add-host=host.docker.internal:\$HOST_GATEWAY_IP \\
  --env-file "\$ENV_FILE_TMP" \\
  --restart unless-stopped \\
  "\$FULL_IMAGE"

rm -f "\$ENV_FILE_TMP"

sleep 3
docker ps -f name="\$CONTAINER_NAME"
docker logs --tail=30 "\$CONTAINER_NAME"

REMOTE_SCRIPT_END

echo ""
echo "Deployment completed."
echo "Health: http://${REMOTE_HOST}:${PORT}/health"


