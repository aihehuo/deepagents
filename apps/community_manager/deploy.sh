#!/usr/bin/env bash
set -euo pipefail
: "${COMMUNITY_MANAGER_IMAGE:?Set an immutable manager image}"
: "${COMMUNITY_MANAGER_ENV_FILE:?Set the private manager env file path}"
if [[ ! "$COMMUNITY_MANAGER_IMAGE" =~ @sha256:[0-9a-f]{64}$ && ! "$COMMUNITY_MANAGER_IMAGE" =~ :[0-9a-f]{40}$ ]]; then
  echo 'Manager image must use a digest or full commit tag' >&2; exit 1
fi
if [[ ! -f "$COMMUNITY_MANAGER_ENV_FILE" ]]; then echo 'Private env file is missing' >&2; exit 1; fi
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
compose_args=(-p community-manager-v1 -f "$script_dir/compose.yml")
if [[ -n "${COMMUNITY_MANAGER_NETWORK:-}" ]]; then compose_args+=(-f "$script_dir/compose.network.yml"); fi
# Project and service names are fixed: existing customer deployments are never targeted.
docker compose "${compose_args[@]}" config --quiet
docker compose "${compose_args[@]}" pull
docker compose "${compose_args[@]}" up -d community-manager-api community-manager-worker community-manager-scheduler
