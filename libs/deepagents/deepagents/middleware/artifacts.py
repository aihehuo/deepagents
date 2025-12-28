"""Middleware for tracking artifact URLs (e.g., uploaded HTML files).

This middleware tracks artifact URLs that are produced by different skills,
typically HTML files that have been uploaded to cloud storage.
"""

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Annotated, NotRequired, TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import InjectedToolCallId
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.runtime import Runtime
from langgraph.types import Command


class ArtifactMetadata(TypedDict):
    """Metadata for a single artifact."""

    url: str
    """The URL of the uploaded artifact."""

    artifact_type: NotRequired[str]
    """The type of artifact (e.g., 'html', 'pdf', 'document'). Defaults to 'html'."""

    name: NotRequired[str]
    """Optional name or description of the artifact."""

    created_at: NotRequired[str]
    """ISO 8601 timestamp when the artifact was recorded."""


def _artifacts_reducer(left: list[ArtifactMetadata] | None, right: list[ArtifactMetadata] | None) -> list[ArtifactMetadata]:
    """Reducer for artifacts list that appends new artifacts.
    
    Args:
        left: Existing artifacts list (may be None during initialization).
        right: New artifacts to append.
        
    Returns:
        Combined list with new artifacts appended to existing ones.
    """
    if left is None:
        return right if right is not None else []
    if right is None:
        return left
    # Append new artifacts to existing list
    return left + right


class ArtifactsState(AgentState):
    """State for tracking artifact URLs."""

    artifacts: Annotated[NotRequired[list[ArtifactMetadata]], _artifacts_reducer]
    """List of artifact metadata, typically HTML files uploaded to cloud storage."""


class ArtifactsStateUpdate(TypedDict):
    """A state update for the artifacts middleware."""

    artifacts: NotRequired[list[ArtifactMetadata]]
    """List of artifact metadata."""


ADD_ARTIFACT_TOOL_DESCRIPTION = """Record an artifact URL in the agent state.

Use this tool after uploading a deliverable (typically an HTML file) to record its URL.
This allows the agent to track all artifacts that have been created and uploaded during
the conversation.

Args:
    url: The URL of the uploaded artifact (required).
    artifact_type: The type of artifact (e.g., 'html', 'pdf', 'document'). Defaults to 'html'.
    name: Optional name or description of the artifact.

This tool updates the agent state to track the artifact URL for later reference."""


def _get_add_artifact_tool() -> StructuredTool:
    """Create a tool that records an artifact URL."""
    
    def add_artifact(
        url: Annotated[
            str,
            "The URL of the uploaded artifact.",
        ],
        artifact_type: Annotated[
            str,
            "The type of artifact (e.g., 'html', 'pdf', 'document'). Defaults to 'html'.",
        ] = "html",
        name: Annotated[
            str,
            "Optional name or description of the artifact.",
        ] = "",
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command:
        """Record an artifact URL in the agent state.
        
        Args:
            url: The URL of the uploaded artifact.
            artifact_type: The type of artifact. Defaults to 'html'.
            name: Optional name or description of the artifact.
            tool_call_id: Injected tool call ID.
        
        Returns:
            Command with state update adding the artifact to the list.
        """
        artifact: ArtifactMetadata = {
            "url": url,
            "artifact_type": artifact_type,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        if name:
            artifact["name"] = name
        
        # The reducer will append this artifact to the existing list
        return Command(
            update={
                "artifacts": [artifact],  # Reducer appends to existing list
                "messages": [
                    ToolMessage(
                        content=f"Recorded artifact: {url} (type: {artifact_type})",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )
    
    return StructuredTool.from_function(
        func=add_artifact,
        name="add_artifact",
        description=ADD_ARTIFACT_TOOL_DESCRIPTION,
    )




ARTIFACTS_SYSTEM_PROMPT = """## Artifact Tracking

You have access to an artifact tracking tool to record URLs of uploaded deliverables.

- add_artifact: Record an artifact URL (typically HTML files) that have been uploaded

**When to use:**
- After successfully uploading an HTML file or other deliverable using upload_asset
- When you need to track artifacts for later reference
- The artifact URL should be recorded for documentation and tracking purposes

**Important:** Only record artifacts that have been successfully uploaded and have a valid URL."""


class ArtifactsMiddleware(AgentMiddleware):
    """Middleware for tracking artifact URLs (e.g., uploaded HTML files).

    This middleware:
    1. Tracks artifact URLs that are produced by different skills
    2. Typically used for HTML files uploaded to cloud storage
    3. Provides a tool to record artifact URLs in the agent state

    Example:
        ```python
        from deepagents.middleware.artifacts import ArtifactsMiddleware
        from langchain.agents import create_agent

        agent = create_agent(
            model="anthropic:claude-sonnet-4-20250514",
            middleware=[ArtifactsMiddleware()],
        )
        ```
    """

    state_schema = ArtifactsState

    def __init__(
        self,
        *,
        system_prompt_template: str | None = None,
    ) -> None:
        """Initialize the ArtifactsMiddleware.

        Args:
            system_prompt_template: Optional custom system prompt template.
                If None, uses the default template.
        """
        self.system_prompt_template = (
            system_prompt_template or ARTIFACTS_SYSTEM_PROMPT
        )
        self.tools = [
            _get_add_artifact_tool(),
        ]

    def before_agent(
        self,
        state: ArtifactsState,
        runtime: Runtime,
    ) -> ArtifactsStateUpdate | None:
        """Initialize artifacts state if not present.

        Args:
            state: Current agent state.
            runtime: Runtime context.

        Returns:
            Updated state with artifacts initialized to empty list if not present.
        """
        if "artifacts" not in state:
            return ArtifactsStateUpdate(artifacts=[])
        return None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject artifact tracking guidance into the system prompt.

        This runs on every model call to ensure the tracking information is always available.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        if self.system_prompt_template:
            if request.system_prompt:
                new_system_prompt = request.system_prompt + "\n\n" + self.system_prompt_template
            else:
                new_system_prompt = self.system_prompt_template

            return handler(request.override(system_prompt=new_system_prompt))
        
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Inject artifact tracking guidance into the system prompt.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        if self.system_prompt_template:
            if request.system_prompt:
                new_system_prompt = request.system_prompt + "\n\n" + self.system_prompt_template
            else:
                new_system_prompt = self.system_prompt_template

            return await handler(request.override(system_prompt=new_system_prompt))
        
        return await handler(request)

