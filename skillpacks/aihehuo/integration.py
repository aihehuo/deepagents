"""Optional LangChain adapter for agents without a script execution tool."""

import http.client
import json
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool

from skillpacks.aihehuo.client import HOST, build_path, fetch
from skillpacks.aihehuo.manage import _catalog


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
        if manifest.get("name") != name or manifest.get("resources") != metadata["resources"]:
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
    ) -> str:
        """Retrieve public AIHEHUO data for an installed skill resource."""
        if resource not in resources:
            return f"Resource is not installed. Available resources: {sorted(resources)}"
        params: dict[str, str | int] = {
            key: value
            for key, value in {"keyword": keyword, "q": q, "page": page, "per": per, "created_since": created_since}.items()
            if value is not None
        }
        output = "md" if resource == "weekly" else output_format
        try:
            path = build_path(resource, params=params, output_format=output, event_id=event_id)
            response = fetch(path, output_format=output)
            source_url = f"https://{HOST}{path}"
            if output == "json":
                document = json.loads(response)
                if isinstance(document, dict):
                    document["_source_url"] = source_url
                return json.dumps(document, ensure_ascii=False)
            result = f"Source API: {source_url}\n\n{response}"
        except (ValueError, RuntimeError, OSError, http.client.HTTPException) as error:
            return f"AIHEHUO query failed: {error}"
        else:
            return result

    return StructuredTool.from_function(
        query,
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
