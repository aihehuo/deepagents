#!/usr/bin/env bash
set -euo pipefail
: "${MANAGER_RUNTIME_IMAGE:?Set an immutable runtime image digest or full commit tag}"
: "${COMMUNITY_MANAGER_IMAGE_REPOSITORY:?Set the manager image repository}"
if [[ ! "$MANAGER_RUNTIME_IMAGE" =~ @sha256:[0-9a-f]{64}$ && ! "$MANAGER_RUNTIME_IMAGE" =~ :[0-9a-f]{40}$ ]]; then
  echo 'Runtime image must use a digest or full commit tag' >&2; exit 1
fi
repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"
if [[ -n "$(git status --porcelain -- apps/community_manager)" ]]; then
  echo 'Commit manager source before building a release image' >&2; exit 1
fi
manager_revision="$(git rev-parse HEAD)"
manager_image="${COMMUNITY_MANAGER_IMAGE_REPOSITORY}:${manager_revision}"
docker build --build-arg "MANAGER_RUNTIME_IMAGE=$MANAGER_RUNTIME_IMAGE" -f apps/community_manager/Dockerfile -t "$manager_image" .
printf '%s\n' "$manager_image"
