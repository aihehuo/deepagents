# Supply an immutable runtime image tag/digest that already contains the installed dependencies.
# Building this image never rebuilds or restarts the customer containers.
ARG MANAGER_RUNTIME_IMAGE
FROM ${MANAGER_RUNTIME_IMAGE}
USER root
COPY --chown=1000:1000 apps/community_manager /app/apps/community_manager
# Ship the exact shared pure-data contract; never import customer agent graphs.
COPY --chown=1000:1000 apps/group_agent_api/app/wecom_journey.py /app/apps/group_agent_api/app/wecom_journey.py
USER 1000:1000
WORKDIR /app
ENV PYTHONPATH=/app PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT []
HEALTHCHECK NONE
CMD ["python", "-m", "uvicorn", "apps.community_manager.api:app", "--host", "0.0.0.0", "--port", "8092"]
