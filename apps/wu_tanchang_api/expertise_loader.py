"""Expertise loader for loading domain-specific expertise templates.

This module loads expertise definitions from markdown files with YAML frontmatter.
Each expertise file defines:
- Metadata (name, description)
- Canvas template structure
- Expert role and methodology

Similar to the skills loading pattern, but for expertise templates.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

# Maximum size for expertise .md files (10MB)
MAX_EXPERTISE_FILE_SIZE = 10 * 1024 * 1024


class ExpertiseMetadata(TypedDict):
    """Metadata for an expertise template."""
    
    name: str
    """Name of the expertise (e.g., 'business_cofounder')."""
    
    description: str
    """Description of what this expertise provides."""
    
    path: str
    """Path to the expertise .md file."""


class ExpertiseDefinition(TypedDict):
    """Complete expertise definition."""
    
    name: str
    """Name of the expertise."""
    
    description: str
    """Description of the expertise."""
    
    system_prompt: str
    """Expert role and methodology content (markdown body)."""
    
    canvas_template: str
    """JSON template/schema for canvas structure."""


def _parse_expertise_frontmatter(expertise_path: Path) -> ExpertiseDefinition | None:
    """Parse YAML frontmatter and content from expertise .md file.
    
    Args:
        expertise_path: Path to the expertise .md file
        
    Returns:
        ExpertiseDefinition with parsed data, or None if parsing fails
    """
    try:
        # Check file size
        if expertise_path.stat().st_size > MAX_EXPERTISE_FILE_SIZE:
            return None
        
        content = expertise_path.read_text(encoding="utf-8")
        
        # Parse YAML frontmatter
        # Format: ---\nkey: value\n---\nmarkdown content
        frontmatter_match = re.match(
            r"^---\s*\n(.*?)\n---\s*\n(.*)$",
            content,
            re.DOTALL,
        )
        
        if not frontmatter_match:
            return None
        
        frontmatter_text = frontmatter_match.group(1)
        markdown_content = frontmatter_match.group(2).strip()
        
        # Parse YAML frontmatter (simple parsing)
        metadata = {}
        canvas_template_lines = []
        in_canvas_template = False
        
        for line in frontmatter_text.split("\n"):
            line = line.rstrip()
            
            if in_canvas_template:
                # Continue collecting canvas template lines
                if line and not line.startswith(" ") and ":" in line:
                    # New key found, stop canvas template
                    in_canvas_template = False
                    # Process this line as a new key
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()
                else:
                    # Part of canvas template (indented or continuation)
                    canvas_template_lines.append(line[2:] if line.startswith("  ") else line)
            elif ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                
                if key == "canvas_template" and value == "|":
                    # Multi-line canvas template starts
                    in_canvas_template = True
                    canvas_template_lines = []
                else:
                    metadata[key] = value
        
        # Store canvas template
        if canvas_template_lines:
            metadata["canvas_template"] = "\n".join(canvas_template_lines).strip()
        
        # Validate required fields
        if "name" not in metadata or "description" not in metadata:
            return None
        
        return ExpertiseDefinition(
            name=metadata["name"],
            description=metadata["description"],
            system_prompt=markdown_content,
            canvas_template=metadata.get("canvas_template", "{}"),
        )
        
    except (OSError, UnicodeDecodeError):
        return None


def list_expertise(expertise_dir: Path) -> list[ExpertiseMetadata]:
    """Discover all expertise templates in the directory.
    
    Args:
        expertise_dir: Directory containing expertise .md files
        
    Returns:
        List of expertise metadata (name, description, path)
    """
    if not expertise_dir.exists() or not expertise_dir.is_dir():
        return []
    
    expertise_list: list[ExpertiseMetadata] = []
    
    for file_path in sorted(expertise_dir.glob("*.md")):
        if not file_path.is_file():
            continue
        
        expertise_def = _parse_expertise_frontmatter(file_path)
        if expertise_def:
            expertise_list.append(
                ExpertiseMetadata(
                    name=expertise_def["name"],
                    description=expertise_def["description"],
                    path=str(file_path),
                )
            )
    
    return expertise_list


def load_expertise(expertise_type: str, expertise_dir: Path) -> ExpertiseDefinition:
    """Load a specific expertise by name.
    
    Args:
        expertise_type: Name of the expertise (e.g., 'business_cofounder')
        expertise_dir: Directory containing expertise files
        
    Returns:
        ExpertiseDefinition with full content
        
    Raises:
        FileNotFoundError: If expertise file doesn't exist
        ValueError: If expertise file is invalid
    """
    # Try exact filename match
    expertise_path = expertise_dir / f"{expertise_type}.md"
    
    if not expertise_path.exists():
        raise FileNotFoundError(
            f"Expertise '{expertise_type}' not found at {expertise_path}. "
            f"Available expertise: {[e['name'] for e in list_expertise(expertise_dir)]}"
        )
    
    expertise_def = _parse_expertise_frontmatter(expertise_path)
    
    if not expertise_def:
        raise ValueError(
            f"Failed to parse expertise file: {expertise_path}. "
            "Check that it has valid YAML frontmatter with 'name' and 'description' fields."
        )
    
    return expertise_def
