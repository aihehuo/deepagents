"""Middleware for automatically generating and managing todos for business idea development.

This middleware extends the todo list functionality by automatically generating a todo list
based on the sequential business idea development progression. It monitors the BusinessIdeaState
and creates/updates todos as milestones are completed.

The sequential progression:
1. business-idea-evaluation → mark_business_idea_complete
2. persona-clarification → mark_persona_clarified
3. painpoint-enhancement → mark_painpoint_enhanced
4. 60s-pitch-creation → mark_pitch_created
5. baseline-pricing-optimization → mark_pricing_optimized
6. business-model-pivot-exploration
"""

from collections.abc import Awaitable, Callable
from typing import Annotated, Literal, NotRequired, TypedDict

from langchain.agents.middleware.todo import TodoListMiddleware
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    OmitFromInput,
)
from langgraph.runtime import Runtime

from deepagents.middleware.business_idea_tracker import BusinessIdeaState


class Todo(TypedDict):
    """A single todo item with content and status."""

    content: str
    """The content/description of the todo item."""

    status: Literal["pending", "in_progress", "completed"]
    """Current status of the todo item."""


class BusinessIdeaDevelopmentState(AgentState):
    """State schema that combines business idea tracking and todos."""

    # From BusinessIdeaState
    business_idea_complete: NotRequired[bool]
    materialized_business_idea: NotRequired[str | None]
    persona_clarified: NotRequired[bool]
    painpoint_enhanced: NotRequired[bool]
    pitch_created: NotRequired[bool]
    pricing_optimized: NotRequired[bool]

    # From TodoListMiddleware
    todos: Annotated[NotRequired[list[Todo]], OmitFromInput]


# Define the sequential todo items
BUSINESS_IDEA_DEVELOPMENT_TODOS = [
    {
        "content": "Evaluate the business idea using the business-idea-evaluation skill. Assess across painpoint, technology, and future vision perspectives. If complete, call mark_business_idea_complete tool.",
        "milestone": "business_idea_complete",
    },
    {
        "content": "Clarify the target user persona using the persona-clarification skill. Create a detailed persona with demographics, goals, pain points, and behaviors. Then call mark_persona_clarified tool.",
        "milestone": "persona_clarified",
    },
    {
        "content": "Enhance the pain point using the painpoint-enhancement skill. Evaluate across six emotional-resonance dimensions (urgency, frequency, economic cost, universality, viral spread, regulatory pressure). Then call mark_painpoint_enhanced tool.",
        "milestone": "painpoint_enhanced",
    },
    {
        "content": "Create a 60-second pitch using the 60s-pitch-creation skill. Include pain point resonance, team advantage statement, and call to action. Then call mark_pitch_created tool.",
        "milestone": "pitch_created",
    },
    {
        "content": "Establish baseline pricing and optimization using the baseline-pricing-and-optimization skill. Apply 1/10 value rule, generate pricing tactics, and identify key partners. Then call mark_pricing_optimized tool.",
        "milestone": "pricing_optimized",
    },
    {
        "content": "Explore business model pivots using the business-model-pivot-exploration skill. Test-fit the product/service into seven business model archetypes and identify the most promising alternatives.",
        "milestone": None,  # Final step, no milestone to mark
    },
]


BUSINESS_IDEA_DEVELOPMENT_SYSTEM_PROMPT = """## Business Idea Development Workflow

You are helping develop a business idea through a structured, sequential workflow. A todo list has been automatically generated for you based on the current progress of the business idea development.

**Important:**
- Work through the todos in order - each step unlocks the next
- Complete each todo fully before moving to the next
- Mark todos as completed immediately after finishing each step
- Use the appropriate milestone marking tools after completing each skill:
  - After business-idea-evaluation → call `mark_business_idea_complete`
  - After persona-clarification → call `mark_persona_clarified`
  - After painpoint-enhancement → call `mark_painpoint_enhanced`
  - After 60s-pitch-creation → call `mark_pitch_created`
  - After baseline-pricing-optimization → call `mark_pricing_optimized`

The todo list will automatically update as you complete each milestone. Focus on the current todo and complete it thoroughly before proceeding."""


class BusinessIdeaDevelopmentMiddleware(AgentMiddleware):
    """Middleware that automatically generates and manages todos for business idea development.

    This middleware:
    1. Monitors BusinessIdeaState to track progression
    2. Automatically generates todos based on current state
    3. Updates todos as milestones are completed
    4. Guides the agent through the sequential progression

    It extends TodoListMiddleware functionality by providing business-idea-specific todo generation.

    Example:
        ```python
        from deepagents.middleware.business_idea_development import BusinessIdeaDevelopmentMiddleware
        from langchain.agents import create_agent

        agent = create_agent(
            model="anthropic:claude-sonnet-4-20250514",
            middleware=[
                BusinessIdeaTrackerMiddleware(),  # Required: tracks milestones
                BusinessIdeaDevelopmentMiddleware(),  # Auto-generates todos
            ],
        )
        ```
    """

    state_schema = BusinessIdeaDevelopmentState

    def __init__(
        self,
        *,
        system_prompt_template: str | None = None,
    ) -> None:
        """Initialize the BusinessIdeaDevelopmentMiddleware.

        Args:
            system_prompt_template: Optional custom system prompt template.
                If None, uses the default template.
        """
        self.system_prompt_template = (
            system_prompt_template or BUSINESS_IDEA_DEVELOPMENT_SYSTEM_PROMPT
        )
        # We don't provide tools - we rely on TodoListMiddleware for write_todos
        # and BusinessIdeaTrackerMiddleware for milestone marking tools
        self.tools = []

    def _generate_todos_from_state(self, state: BusinessIdeaDevelopmentState) -> list[Todo]:
        """Generate todos based on current business idea development state.

        Args:
            state: Current agent state with business idea tracking fields.

        Returns:
            List of todos in the correct order, with appropriate statuses.
        """
        business_idea_complete = state.get("business_idea_complete", False)
        persona_clarified = state.get("persona_clarified", False)
        painpoint_enhanced = state.get("painpoint_enhanced", False)
        pitch_created = state.get("pitch_created", False)
        pricing_optimized = state.get("pricing_optimized", False)

        todos = []
        current_step = 0

        # Step 1: Business Idea Evaluation
        if business_idea_complete:
            todos.append(
                Todo(
                    content=BUSINESS_IDEA_DEVELOPMENT_TODOS[0]["content"],
                    status="completed",
                )
            )
            current_step = 1
        else:
            todos.append(
                Todo(
                    content=BUSINESS_IDEA_DEVELOPMENT_TODOS[0]["content"],
                    status="pending",
                )
            )
            return todos  # Can't proceed until idea is complete

        # Step 2: Persona Clarification
        if persona_clarified:
            todos.append(
                Todo(
                    content=BUSINESS_IDEA_DEVELOPMENT_TODOS[1]["content"],
                    status="completed",
                )
            )
            current_step = 2
        else:
            todos.append(
                Todo(
                    content=BUSINESS_IDEA_DEVELOPMENT_TODOS[1]["content"],
                    status="pending",
                )
            )
            return todos  # Can't proceed until persona is clarified

        # Step 3: Painpoint Enhancement
        if painpoint_enhanced:
            todos.append(
                Todo(
                    content=BUSINESS_IDEA_DEVELOPMENT_TODOS[2]["content"],
                    status="completed",
                )
            )
            current_step = 3
        else:
            todos.append(
                Todo(
                    content=BUSINESS_IDEA_DEVELOPMENT_TODOS[2]["content"],
                    status="pending",
                )
            )
            return todos  # Can't proceed until painpoint is enhanced

        # Step 4: 60-Second Pitch Creation
        if pitch_created:
            todos.append(
                Todo(
                    content=BUSINESS_IDEA_DEVELOPMENT_TODOS[3]["content"],
                    status="completed",
                )
            )
            current_step = 4
        else:
            todos.append(
                Todo(
                    content=BUSINESS_IDEA_DEVELOPMENT_TODOS[3]["content"],
                    status="pending",
                )
            )
            return todos  # Can't proceed until pitch is created

        # Step 5: Baseline Pricing and Optimization
        if pricing_optimized:
            todos.append(
                Todo(
                    content=BUSINESS_IDEA_DEVELOPMENT_TODOS[4]["content"],
                    status="completed",
                )
            )
            current_step = 5
        else:
            todos.append(
                Todo(
                    content=BUSINESS_IDEA_DEVELOPMENT_TODOS[4]["content"],
                    status="pending",
                )
            )
            return todos  # Can't proceed until pricing is optimized

        # Step 6: Business Model Pivot Exploration
        todos.append(
            Todo(
                content=BUSINESS_IDEA_DEVELOPMENT_TODOS[5]["content"],
                status="pending",  # This is the final step, no milestone to track
            )
        )

        return todos

    def before_agent(
        self,
        state: BusinessIdeaDevelopmentState,
        runtime: Runtime,
    ) -> dict | None:
        """Initialize or update todos based on current business idea development state.

        This method is called before each agent invocation. It automatically generates
        or updates the todo list based on the current milestone completion status.

        Args:
            state: Current agent state.
            runtime: Runtime context.

        Returns:
            State update with generated todos, or None if todos already exist and are current.
        """
        # Generate todos based on current state
        generated_todos = self._generate_todos_from_state(state)

        # Check if todos already exist in state
        existing_todos = state.get("todos", [])

        # If no todos exist, initialize them
        if not existing_todos:
            return {"todos": generated_todos}

        # Compare existing todos with generated ones
        # If the number of todos changed, update
        if len(existing_todos) != len(generated_todos):
            return {"todos": generated_todos}

        # Check if any todo status should change based on milestone completion
        # We preserve "in_progress" status if the agent is currently working on a todo,
        # but we update "pending" to "completed" if the milestone is now complete
        needs_update = False
        updated_todos = []
        
        for i, (existing, generated) in enumerate(zip(existing_todos, generated_todos)):
            existing_status = existing.get("status", "pending")
            generated_status = generated.get("status", "pending")
            
            # If milestone is complete, the todo should be completed
            if generated_status == "completed" and existing_status != "completed":
                updated_todos.append(generated)
                needs_update = True
            # If milestone is not complete but todo was marked completed, revert to pending
            elif generated_status == "pending" and existing_status == "completed":
                updated_todos.append(generated)
                needs_update = True
            # Preserve in_progress status if agent is working on it
            elif existing_status == "in_progress" and generated_status == "pending":
                updated_todos.append(existing)  # Keep in_progress
            else:
                updated_todos.append(existing)  # Keep existing status
        
        if needs_update:
            return {"todos": updated_todos}

        # Todos are up to date, no update needed
        return None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject business idea development guidance into the system prompt.

        This runs on every model call to ensure the guidance is always available.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        # Inject the system prompt
        full_prompt = self.system_prompt_template

        if request.system_prompt:
            new_system_prompt = request.system_prompt + "\n\n" + full_prompt
        else:
            new_system_prompt = full_prompt

        return handler(request.override(system_prompt=new_system_prompt))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Inject business idea development guidance into the system prompt.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        # Inject the system prompt
        full_prompt = self.system_prompt_template

        if request.system_prompt:
            new_system_prompt = request.system_prompt + "\n\n" + full_prompt
        else:
            new_system_prompt = full_prompt

        return await handler(request.override(system_prompt=new_system_prompt))

