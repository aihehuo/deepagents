"""Optional LangChain adapter for agents without a script execution tool."""

import asyncio
import http.client
import json
import uuid
from pathlib import Path
from typing import Annotated

from langchain.tools import InjectedToolCallId
from langchain_core.tools import BaseTool, StructuredTool

from skillpacks.aihehuo.client import HOST, build_path, fetch
from skillpacks.aihehuo.evidence import record_result
from skillpacks.aihehuo.manage import _catalog


class _CompatibleStructuredTool(StructuredTool):
    """Keep legacy direct dict invocation while preserving real agent call IDs."""

    def _parse_input(
        self, tool_input: str | dict[str, object], tool_call_id: str | None
    ) -> str | dict[str, object]:
        if tool_call_id is None and isinstance(tool_input, dict):
            tool_call_id = f"direct-{uuid.uuid4().hex}"
        return super()._parse_input(tool_input, tool_call_id)


def create_aihehuo_tool(*, skills_dir: Path) -> BaseTool:
    """Create a read-only tool restricted to installed catalog capabilities.

    !!! warning
        This application integration is experimental. Recreate the tool after
        changing the installed skill selection.

    Args:
        skills_dir: Actual filesystem directory holding installed skill packages.

    Returns:
        A LangChain tool usable in `create_deep_agent`'s tools list.

    Raises:
        ValueError: No catalog packages are installed or their manifests differ.
        OSError: Installed package metadata cannot be read.
    """
    resources: set[str] = set()
    for name, metadata in _catalog().items():
        directory = skills_dir / name
        if not (directory / "SKILL.md").is_file():
            continue
        manifest = json.loads((directory / "package.json").read_text(encoding="utf-8"))
        if (
            manifest.get("name") != name
            or manifest.get("resources") != metadata["resources"]
        ):
            msg = f"Invalid installed package manifest: {name}"
            raise ValueError(msg)
        resources.update(metadata["resources"])
    if not resources:
        msg = "Install at least one AIHEHUO catalog skill before creating its tool"
        raise ValueError(msg)

    def query(
        resource: str,
        keyword: str | None = None,
        q: str | None = None,
        page: int | None = None,
        per: int | None = None,
        created_since: str | None = None,
        event_id: str | None = None,
        output_format: str = "json",
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> str:
        """Retrieve public AIHEHUO data for an installed skill resource."""
        if resource not in resources:
            return (
                f"Resource is not installed. Available resources: {sorted(resources)}"
            )
        params: dict[str, str | int] = {
            key: value
            for key, value in {
                "keyword": keyword,
                "q": q,
                "page": page,
                "per": per,
                "created_since": created_since,
            }.items()
            if value is not None
        }
        output = "md" if resource == "weekly" else output_format
        execution = None
        reservation = None
        try:
            from apps.group_agent_api.agent_factory.intent_routing.execution import (
                QueryPlanError,
                current_intent_execution,
            )

            execution = current_intent_execution()
            if execution is not None:
                adopted = execution.adopted_result(resource)
                if adopted is not None:
                    record_result(resource, adopted)
                    return adopted
                reservation = execution.reserve(
                    resource=resource,
                    parameters={
                        "keyword": keyword,
                        "q": q,
                        "page": page,
                        "per": per,
                        "created_since": created_since,
                        "event_id": event_id,
                        "output_format": output,
                    },
                    tool_call_id=tool_call_id,
                )
            path = build_path(
                resource, params=params, output_format=output, event_id=event_id
            )
            if reservation is not None:
                execution.record_search(reservation)
                response = fetch(
                    path,
                    output_format=output,
                    timeout=reservation.hard_timeout_ms / 1000,
                )
            else:
                response = fetch(path, output_format=output)
            source_url = f"https://{HOST}{path}"
            if output == "json":
                document = json.loads(response)
                if isinstance(document, dict):
                    document["_source_url"] = source_url
                result = json.dumps(document, ensure_ascii=False)
            else:
                result = f"Source API: {source_url}\n\n{response}"
            if execution is not None and reservation is not None:
                execution.commit_result(reservation, result)
        except QueryPlanError as error:
            return f"AIHEHUO query rejected: {error.code}"
        except (ValueError, RuntimeError, OSError, http.client.HTTPException) as error:
            if execution is not None and reservation is not None:
                timed_out = (
                    isinstance(error, TimeoutError) or "timed out" in str(error).lower()
                )
                try:
                    execution.commit_failure(reservation, timed_out=timed_out)
                except QueryPlanError as commit_error:
                    return f"AIHEHUO query rejected: {commit_error.code}"
            return f"AIHEHUO query failed: {error}"
        else:
            record_result(resource, result)
            return result

    async def query_async(
        resource: str,
        keyword: str | None = None,
        q: str | None = None,
        page: int | None = None,
        per: int | None = None,
        created_since: str | None = None,
        event_id: str | None = None,
        output_format: str = "json",
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> str:
        """Retrieve public data while awaiting every progress acknowledgement."""
        if resource not in resources:
            return (
                f"Resource is not installed. Available resources: {sorted(resources)}"
            )
        params: dict[str, str | int] = {
            key: value
            for key, value in {
                "keyword": keyword,
                "q": q,
                "page": page,
                "per": per,
                "created_since": created_since,
            }.items()
            if value is not None
        }
        output = "md" if resource == "weekly" else output_format
        execution = None
        reservation = None
        try:
            from apps.group_agent_api.agent_factory.intent_routing.execution import (
                QueryPlanError,
                current_intent_execution,
            )

            execution = current_intent_execution()
            if execution is not None:
                adopted = execution.adopted_result(resource)
                if adopted is not None:
                    record_result(resource, adopted)
                    return adopted
                reservation = await execution.reserve_async(
                    resource=resource,
                    parameters={
                        "keyword": keyword,
                        "q": q,
                        "page": page,
                        "per": per,
                        "created_since": created_since,
                        "event_id": event_id,
                        "output_format": output,
                    },
                    tool_call_id=tool_call_id,
                )
            path = build_path(
                resource, params=params, output_format=output, event_id=event_id
            )
            if reservation is not None:
                await execution.record_search_async(reservation)
                response = await asyncio.to_thread(
                    fetch,
                    path,
                    output_format=output,
                    timeout=reservation.hard_timeout_ms / 1000,
                )
            else:
                response = await asyncio.to_thread(fetch, path, output_format=output)
            source_url = f"https://{HOST}{path}"
            if output == "json":
                document = json.loads(response)
                if isinstance(document, dict):
                    document["_source_url"] = source_url
                result = json.dumps(document, ensure_ascii=False)
            else:
                result = f"Source API: {source_url}\n\n{response}"
            if execution is not None and reservation is not None:
                await execution.commit_result_async(reservation, result)
        except QueryPlanError as error:
            return f"AIHEHUO query rejected: {error.code}"
        except (ValueError, RuntimeError, OSError, http.client.HTTPException) as error:
            if execution is not None and reservation is not None:
                timed_out = (
                    isinstance(error, TimeoutError) or "timed out" in str(error).lower()
                )
                try:
                    execution.commit_failure(reservation, timed_out=timed_out)
                except QueryPlanError as commit_error:
                    return f"AIHEHUO query rejected: {commit_error.code}"
            return f"AIHEHUO query failed: {error}"
        else:
            record_result(resource, result)
            return result

    return _CompatibleStructuredTool.from_function(
        query,
        coroutine=query_async,
        name="query_aihehuo_public_data",
        description=(
            f"Read public AIHEHUO data. Installed resources: {', '.join(sorted(resources))}. "
            "Only users and ideas support q. Use keyword for faqs, groups, projects, users or ideas; "
            "omit keyword and q for blog and group-members. Weekly accepts no pagination parameters. "
            "output_format must be json or md. No contact or write operations."
            " Results include the actual official source API URL (_source_url in JSON). "
            "Use it as the citation if an individual item has no url."
        ),
    )
