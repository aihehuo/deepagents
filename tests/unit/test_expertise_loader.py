"""Unit tests for expertise loader functionality."""

from __future__ import annotations

from pathlib import Path

import pytest

from apps.business_cofounder_api.expertise_loader import (
    ExpertiseDefinition,
    ExpertiseMetadata,
    list_expertise,
    load_expertise,
)


class TestListExpertise:
    """Test list_expertise function for discovering expertise templates."""

    def test_list_expertise_empty_directory(self, tmp_path: Path) -> None:
        """Should return empty list for empty directory."""
        expertise_dir = tmp_path / "expertise"
        expertise_dir.mkdir()

        result = list_expertise(expertise_dir)
        assert result == []

    def test_list_expertise_nonexistent_directory(self, tmp_path: Path) -> None:
        """Should return empty list for non-existent directory."""
        expertise_dir = tmp_path / "nonexistent"

        result = list_expertise(expertise_dir)
        assert result == []

    def test_list_expertise_finds_valid_files(self, tmp_path: Path) -> None:
        """Should discover all .md files with valid frontmatter."""
        expertise_dir = tmp_path / "expertise"
        expertise_dir.mkdir()

        # Create two valid expertise files
        (expertise_dir / "expertise1.md").write_text("""---
name: expertise1
description: First expertise
canvas_template: |
  {"field": "value"}
---

# Expertise 1
Content here.
""")

        (expertise_dir / "expertise2.md").write_text("""---
name: expertise2
description: Second expertise
canvas_template: |
  {"field": "value"}
---

# Expertise 2
More content.
""")

        result = list_expertise(expertise_dir)
        
        assert len(result) == 2
        assert result[0]["name"] == "expertise1"
        assert result[0]["description"] == "First expertise"
        assert result[1]["name"] == "expertise2"
        assert result[1]["description"] == "Second expertise"

    def test_list_expertise_ignores_invalid_files(self, tmp_path: Path) -> None:
        """Should skip files without proper YAML frontmatter."""
        expertise_dir = tmp_path / "expertise"
        expertise_dir.mkdir()

        # Valid file
        (expertise_dir / "valid.md").write_text("""---
name: valid
description: Valid expertise
---

Content
""")

        # Invalid: no frontmatter
        (expertise_dir / "invalid1.md").write_text("""# No Frontmatter
Just content
""")

        # Invalid: missing name
        (expertise_dir / "invalid2.md").write_text("""---
description: Missing name field
---

Content
""")

        # Invalid: not a .md file
        (expertise_dir / "notmd.txt").write_text("""---
name: test
description: Test
---

Content
""")

        result = list_expertise(expertise_dir)
        
        assert len(result) == 1
        assert result[0]["name"] == "valid"

    def test_list_expertise_returns_metadata(self, tmp_path: Path) -> None:
        """Should return complete metadata for each expertise."""
        expertise_dir = tmp_path / "expertise"
        expertise_dir.mkdir()

        expertise_file = expertise_dir / "test_expertise.md"
        expertise_file.write_text("""---
name: test_expertise
description: Test expertise description
---

# Test Expertise
""")

        result = list_expertise(expertise_dir)
        
        assert len(result) == 1
        metadata = result[0]
        assert metadata["name"] == "test_expertise"
        assert metadata["description"] == "Test expertise description"
        assert Path(metadata["path"]) == expertise_file


class TestLoadExpertise:
    """Test load_expertise function for loading specific expertise templates."""

    def test_load_expertise_success(self, tmp_path: Path) -> None:
        """Should load valid expertise with all fields."""
        expertise_dir = tmp_path / "expertise"
        expertise_dir.mkdir()

        expertise_file = expertise_dir / "test_expertise.md"
        expertise_file.write_text("""---
name: test_expertise
description: Test expertise for testing
canvas_template: |
  {
    "field1": "value1",
    "insights": []
  }
---

# Test Expertise

This is test content for the expertise.

## Analysis Tasks

- Task 1
- Task 2
""")

        expertise = load_expertise("test_expertise", expertise_dir)

        assert expertise["name"] == "test_expertise"
        assert expertise["description"] == "Test expertise for testing"
        assert "field1" in expertise["canvas_template"]
        assert "value1" in expertise["canvas_template"]
        assert "This is test content" in expertise["system_prompt"]
        assert "Analysis Tasks" in expertise["system_prompt"]

    def test_load_expertise_file_not_found(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError for non-existent expertise."""
        expertise_dir = tmp_path / "expertise"
        expertise_dir.mkdir()

        with pytest.raises(FileNotFoundError) as exc_info:
            load_expertise("nonexistent", expertise_dir)

        assert "nonexistent" in str(exc_info.value)

    def test_load_expertise_missing_name_field(self, tmp_path: Path) -> None:
        """Should raise ValueError if 'name' field missing."""
        expertise_dir = tmp_path / "expertise"
        expertise_dir.mkdir()

        (expertise_dir / "invalid.md").write_text("""---
description: Missing name field
---

Content
""")

        with pytest.raises(ValueError) as exc_info:
            load_expertise("invalid", expertise_dir)

        assert "invalid" in str(exc_info.value).lower()

    def test_load_expertise_missing_description_field(self, tmp_path: Path) -> None:
        """Should raise ValueError if 'description' field missing."""
        expertise_dir = tmp_path / "expertise"
        expertise_dir.mkdir()

        (expertise_dir / "invalid.md").write_text("""---
name: invalid
---

Content
""")

        with pytest.raises(ValueError) as exc_info:
            load_expertise("invalid", expertise_dir)

        assert "invalid" in str(exc_info.value).lower()

    def test_load_expertise_canvas_template_multiline(self, tmp_path: Path) -> None:
        """Should correctly parse multiline canvas_template YAML."""
        expertise_dir = tmp_path / "expertise"
        expertise_dir.mkdir()

        (expertise_dir / "multiline.md").write_text("""---
name: multiline
description: Test multiline template
canvas_template: |
  {
    "level1": {
      "level2": {
        "level3": "deep value"
      }
    },
    "array": [1, 2, 3],
    "string": "multiline\\nstring"
  }
---

# Content
""")

        expertise = load_expertise("multiline", expertise_dir)

        assert "level1" in expertise["canvas_template"]
        assert "level2" in expertise["canvas_template"]
        assert "level3" in expertise["canvas_template"]
        assert "array" in expertise["canvas_template"]

    def test_load_expertise_empty_canvas_template(self, tmp_path: Path) -> None:
        """Should handle missing canvas_template with default empty object."""
        expertise_dir = tmp_path / "expertise"
        expertise_dir.mkdir()

        (expertise_dir / "no_template.md").write_text("""---
name: no_template
description: No canvas template
---

# Content
""")

        expertise = load_expertise("no_template", expertise_dir)

        assert expertise["canvas_template"] == "{}"

    def test_load_expertise_large_file_rejected(self, tmp_path: Path) -> None:
        """Should reject files exceeding MAX_EXPERTISE_FILE_SIZE."""
        expertise_dir = tmp_path / "expertise"
        expertise_dir.mkdir()

        # Create a file larger than 10MB
        large_content = "x" * (11 * 1024 * 1024)  # 11MB
        (expertise_dir / "large.md").write_text(f"""---
name: large
description: Large file
---

{large_content}
""")

        # Should return None from parser (file too large)
        with pytest.raises(ValueError):
            load_expertise("large", expertise_dir)

    def test_load_expertise_complex_yaml(self, tmp_path: Path) -> None:
        """Should handle complex YAML structures in frontmatter."""
        expertise_dir = tmp_path / "expertise"
        expertise_dir.mkdir()

        (expertise_dir / "complex.md").write_text("""---
name: complex
description: Complex YAML test
canvas_template: |
  {
    "metadata": {
      "version": "1.0",
      "author": "test"
    },
    "fields": ["field1", "field2", "field3"]
  }
additional_field: Some additional value
---

# Complex Expertise

Content here.
""")

        expertise = load_expertise("complex", expertise_dir)

        assert expertise["name"] == "complex"
        assert "metadata" in expertise["canvas_template"]
        assert "version" in expertise["canvas_template"]
        assert "fields" in expertise["canvas_template"]

    def test_load_expertise_no_frontmatter(self, tmp_path: Path) -> None:
        """Should raise ValueError for file without frontmatter."""
        expertise_dir = tmp_path / "expertise"
        expertise_dir.mkdir()

        (expertise_dir / "no_frontmatter.md").write_text("""# Just Markdown

No YAML frontmatter here.
""")

        with pytest.raises(ValueError):
            load_expertise("no_frontmatter", expertise_dir)

    def test_load_expertise_unicode_content(self, tmp_path: Path) -> None:
        """Should handle Unicode content correctly."""
        expertise_dir = tmp_path / "expertise"
        expertise_dir.mkdir()

        (expertise_dir / "unicode.md").write_text("""---
name: unicode
description: Unicode test 中文测试
canvas_template: |
  {
    "field": "值"
  }
---

# Unicode Expertise

Content with unicode: 你好世界 🚀
""", encoding="utf-8")

        expertise = load_expertise("unicode", expertise_dir)

        assert expertise["name"] == "unicode"
        assert "中文测试" in expertise["description"]
        assert "你好世界" in expertise["system_prompt"]
        assert "🚀" in expertise["system_prompt"]
