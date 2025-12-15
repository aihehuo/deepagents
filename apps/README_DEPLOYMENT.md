# Deployment Scripts

Generic deployment scripts for DeepAgents apps. These scripts work with any app in the `apps/` directory.

## Scripts

1. **`build_and_push.sh`** - Build and push Docker image to registry
2. **`pull_and_run.sh`** - Pull and run Docker container locally
3. **`deploy_to_prod2.sh`** - Deploy Docker container to production server

## Usage

All scripts take the app name as the first parameter:

```bash
# Build and push
./apps/build_and_push.sh <app_name> [tag]

# Pull and run locally
./apps/pull_and_run.sh <app_name> [port]

# Deploy to production
./apps/deploy_to_prod2.sh <app_name> [port]
```

## Examples

### Business Co-Founder API

```bash
# Build and push
./apps/build_and_push.sh business_cofounder_api
./apps/build_and_push.sh business_cofounder_api 0.0.3

# Run locally
./apps/pull_and_run.sh business_cofounder_api
./apps/pull_and_run.sh business_cofounder_api 8003

# Deploy to production
./apps/deploy_to_prod2.sh business_cofounder_api
./apps/deploy_to_prod2.sh business_cofounder_api 8001
```

### Business Co-Founder Worker

```bash
# Build and push
./apps/build_and_push.sh business_cofounder_worker
./apps/build_and_push.sh business_cofounder_worker 0.0.1

# Run locally
./apps/pull_and_run.sh business_cofounder_worker

# Deploy to production
./apps/deploy_to_prod2.sh business_cofounder_worker
```

## Configuration

Each app should have its own `.deploy.env` file in its directory (e.g., `apps/business_cofounder_api/.deploy.env`):

```bash
# Registry
ALIYUN_DOCKER_REGISTRY=crpi-lp1jelcmhkef5y0u.cn-qingdao.personal.cr.aliyuncs.com
ALIYUN_DOCKER_USERNAME=your-username
ALIYUN_DOCKER_PASSWORD=your-password

# Image (optional, defaults to aihehuo/<app-name>)
DOCKER_IMAGE_NAME=aihehuo/business-cofounder-api
DOCKER_IMAGE_TAG=latest

# Production deployment (optional, used by deploy_to_prod2.sh)
REMOTE_HOST=prod2
REMOTE_USER=root
REMOTE_DIR=/mnt/business-cofounder-api
CONTAINER_NAME=business-cofounder-api
PORT=8001
```

## Defaults

- **Image name**: `aihehuo/<app-name>` (e.g., `business_cofounder_api` → `aihehuo/business-cofounder-api`)
- **Container name**: `<app-name>-remote` (for local) or `<app-name>` (for production)
- **Port**: `8001` (can be overridden)
- **Data directory**: `apps/<app-name>/.tmp_home/.deepagents/<app-name>` (local) or `/mnt/<app-name>/data` (production)

## Migration from App-Specific Scripts

The old scripts in `apps/business_cofounder_api/` can be removed. The new generic scripts in `apps/` replace them:

- `apps/business_cofounder_api/build_and_push.sh` → `apps/build_and_push.sh business_cofounder_api`
- `apps/business_cofounder_api/pull_and_run.sh` → `apps/pull_and_run.sh business_cofounder_api`
- `apps/business_cofounder_api/deploy_to_prod2.sh` → `apps/deploy_to_prod2.sh business_cofounder_api`

