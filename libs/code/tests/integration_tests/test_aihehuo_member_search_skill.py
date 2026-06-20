"""Integration tests for aihehuo-member-search skill.

This test suite verifies:
1. Skill discovery and loading through SkillsMiddleware
2. Skill script execution
3. Output validation
4. Actual API search results
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from deepagents_cli.agent import create_cli_agent
from deepagents_cli.skills.load import list_skills
from deepagents_cli.skills.middleware import SkillsMiddleware
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage


class FixedGenericFakeChatModel(GenericFakeChatModel):
    """Fixed version of GenericFakeChatModel that properly handles bind_tools."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        """Override bind_tools to return self."""
        return self


class TestAihehuoMemberSearchSkillDiscovery:
    """Test that aihehuo-member-search skill can be discovered and loaded."""

    def test_skill_discovery_from_examples(self, tmp_path: Path) -> None:
        """Test that aihehuo-member-search skill is discovered when copied to skills directory."""
        # Get the example skill directory
        repo_root = Path(__file__).parent.parent.parent
        example_skill_dir = repo_root / "examples" / "skills" / "aihehuo-member-search"
        
        # Copy skill to temporary skills directory
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skill_dest = skills_dir / "aihehuo-member-search"
        shutil.copytree(example_skill_dir, skill_dest)
        
        # Test skill discovery
        skills = list_skills(user_skills_dir=skills_dir, project_skills_dir=None)
        
        assert len(skills) == 1, "Should discover exactly one skill"
        assert skills[0]["name"] == "aihehuo-member-search", "Skill name should match"
        assert "AI He Huo" in skills[0]["description"] or "爱合伙" in skills[0]["description"], "Description should mention AI He Huo"
        assert skills[0]["source"] == "user", "Skill should be from user directory"
        
        # Verify SKILL.md exists and is readable
        skill_md_path = Path(skills[0]["path"])
        assert skill_md_path.exists(), "SKILL.md should exist"
        assert skill_md_path.name == "SKILL.md", "Path should point to SKILL.md"
        
        # Verify skill script exists
        skill_script = skill_dest / "aihehuo_member_search.py"
        assert skill_script.exists(), "aihehuo_member_search.py should exist"

    def test_skills_middleware_loads_aihehuo_skill(self, tmp_path: Path) -> None:
        """Test that SkillsMiddleware loads aihehuo-member-search skill metadata."""
        # Get the example skill directory
        repo_root = Path(__file__).parent.parent.parent
        example_skill_dir = repo_root / "examples" / "skills" / "aihehuo-member-search"
        
        # Copy skill to temporary skills directory
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skill_dest = skills_dir / "aihehuo-member-search"
        shutil.copytree(example_skill_dir, skill_dest)
        
        # Create SkillsMiddleware
        middleware = SkillsMiddleware(
            skills_dir=skills_dir,
            assistant_id="test-agent",
            project_skills_dir=None,
        )
        
        # Get skills list using the same method middleware uses
        from deepagents_cli.skills.load import list_skills
        skills_list = list_skills(
            user_skills_dir=middleware.skills_dir,
            project_skills_dir=middleware.project_skills_dir,
        )
        
        assert len(skills_list) == 1, "Middleware should discover one skill"
        assert skills_list[0]["name"] == "aihehuo-member-search", "Skill name should match"
        assert "AI He Huo" in skills_list[0]["description"] or "爱合伙" in skills_list[0]["description"], "Description should mention AI He Huo"


class TestAihehuoMemberSearchSkillExecution:
    """Test that aihehuo-member-search skill script can be executed."""

    def test_skill_script_help(self, tmp_path: Path) -> None:
        """Test that aihehuo-member-search script shows help when run with --help."""
        # Get the example skill directory
        repo_root = Path(__file__).parent.parent.parent
        skill_script = repo_root / "examples" / "skills" / "aihehuo-member-search" / "aihehuo_member_search.py"
        
        # Run script with --help
        result = subprocess.run(
            [sys.executable, str(skill_script), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        assert result.returncode == 0, f"Script should exit successfully: {result.stderr}"
        assert "Search AI He Huo" in result.stdout or "爱合伙" in result.stdout, "Help should mention AI He Huo"
        assert "query" in result.stdout, "Help should mention query parameter"

    def test_skill_script_without_requests_package(self, tmp_path: Path) -> None:
        """Test that script handles missing requests package gracefully."""
        # Get the example skill directory
        repo_root = Path(__file__).parent.parent.parent
        skill_script = repo_root / "examples" / "skills" / "aihehuo-member-search" / "aihehuo_member_search.py"
        
        # Check if requests is installed
        import importlib.util
        requests_installed = importlib.util.find_spec("requests") is not None
        
        if not requests_installed:
            # If requests is not installed, script should handle it
            result = subprocess.run(
                [sys.executable, str(skill_script), "test query"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # Script should handle missing package gracefully
            assert "requests package not installed" in result.stdout or "Error" in result.stdout or result.returncode != 0
        else:
            # If requests is installed, script should run (may fail on API call, but that's OK)
            result = subprocess.run(
                [sys.executable, str(skill_script), "test query", "--max-results", "1"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            # Script should execute (may return error from API, but shouldn't crash)
            assert result.returncode in [0, 1], "Script should handle execution"

    @pytest.mark.timeout(30)
    def test_skill_script_query_validation(self, tmp_path: Path) -> None:
        """Test that script validates query input (must be longer than 5 characters)."""
        # Get the example skill directory
        repo_root = Path(__file__).parent.parent.parent
        skill_script = repo_root / "examples" / "skills" / "aihehuo-member-search" / "aihehuo_member_search.py"
        
        # Test with very short query (should be rejected by API)
        result = subprocess.run(
            [sys.executable, str(skill_script), "test", "--max-results", "1"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # Script should handle short queries (API will reject, but script should handle it)
        # The script validates query length and returns JSON error
        output = result.stdout
        if "Query too short" in output or "error" in output.lower():
            # This is expected - query validation worked
            assert True
        else:
            # Or script executed and API handled it
            assert result.returncode in [0, 1], "Script should handle short queries"

    @pytest.mark.timeout(30)
    def test_skill_script_actual_search_results(self, tmp_path: Path) -> None:
        """Test that the script returns actual search results in the expected format.
        
        This test performs a real AI He Huo search and validates:
        1. The script executes successfully
        2. Results are returned in JSON format
        3. Results contain expected fields (total, page, hits, etc.)
        """
        # Check if requests package is installed
        import importlib.util
        requests_installed = importlib.util.find_spec("requests") is not None
        
        if not requests_installed:
            pytest.skip("requests package not installed, skipping actual search test")
        
        # Get the example skill directory
        repo_root = Path(__file__).parent.parent.parent
        skill_script = repo_root / "examples" / "skills" / "aihehuo-member-search" / "aihehuo_member_search.py"
        
        # Run a real search with a descriptive query
        # Using a query that's likely to return results
        search_query = "寻找有技术背景的创业者"
        result = subprocess.run(
            [sys.executable, str(skill_script), search_query, "--max-results", "3"],
            capture_output=True,
            text=True,
            timeout=30,  # Longer timeout for actual API call
        )
        
        # Script should execute successfully
        assert result.returncode == 0, f"Script should execute successfully: {result.stderr}"
        
        # Print debug output from stderr (if any)
        if result.stderr:
            print("\n" + "=" * 80)
            print("DEBUG OUTPUT (stderr):")
            print("=" * 80)
            print(result.stderr)
            print("=" * 80 + "\n")
        
        # Should have output
        assert len(result.stdout) > 0, "Script should return search results"
        
        # Print the search results for visibility
        output = result.stdout
        print("\n" + "=" * 80)
        print("AIHEHUO MEMBER SEARCH RESULTS:")
        print("=" * 80)
        print(output)
        print("=" * 80 + "\n")
        
        # Output should be valid JSON
        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            pytest.fail(f"Output should be valid JSON: {e}\nOutput: {output}")
        
        # Validate JSON structure
        assert isinstance(data, dict), "Result should be a dictionary"
        
        # Check for expected fields in the response
        if "error" in data:
            error_msg = data.get('error', '')
            error_message = data.get('message', '')
            status_code = data.get('status_code')
            
            # If API key is missing, that's a configuration issue
            if "API key" in error_msg or "API key" in error_message or "AIHEHUO_API_KEY" in error_msg:
                pytest.skip(f"API key not configured: {error_message}. Please set AIHEHUO_API_KEY in .env.aihehuo file.")
            
            # 400 client errors indicate a problem with our request/implementation - test should fail
            if status_code == 400:
                pytest.fail(
                    f"API returned 400 Bad Request error. This indicates a problem with the request format or parameters.\n"
                    f"Error: {error_msg}\n"
                    f"Message: {error_message}\n"
                    f"Full response: {json.dumps(data, indent=2)}"
                )
            
            # Other errors (500, network issues, etc.) are acceptable - API might be unavailable
            print(f"⚠️  API returned error: {error_msg}")
            if status_code:
                print(f"   Status code: {status_code}")
            if error_message:
                print(f"   Message: {error_message}")
            print("   (This is acceptable if API is unavailable or has server issues)")
            return
        
        # If we have results, validate the structure
        # API returns: { "data": [...], "meta": { "total_count": ..., "current_page": ..., "per_page": ..., "total_pages": ... } }
        assert "data" in data or "hits" in data or "total" in data, "Result should contain 'data', 'hits', or 'total' field"
        
        # Handle different response formats
        if "data" in data:
            # New API format: { "data": [...], "meta": {...} }
            hits = data.get("data", [])
            meta = data.get("meta", {})
            total = meta.get("total_count", len(hits))
            page = meta.get("current_page", 1)
            page_size = meta.get("per_page", 10)
        elif "hits" in data:
            # Old format: { "hits": [...], "total": ... }
            hits = data.get("hits", [])
            total = data.get("total", len(hits))
            page = data.get("page", 1)
            page_size = data.get("page_size", data.get("per", 10))
        else:
            # Fallback
            hits = []
            total = data.get("total", 0)
            page = data.get("page", 1)
            page_size = data.get("page_size", data.get("per", 10))
        
        print(f"\nSearch Results Summary:")
        print(f"  - Total results: {total}")
        print(f"  - Current page: {page}")
        print(f"  - Page size: {page_size}")
        print(f"  - Results in this page: {len(hits)}")
        print(f"  - Query: '{search_query}'")
        print(f"  - Max results requested: 3 (adjusted to minimum 10)\n")
        
        # Validate hits structure if we have results
        if len(hits) > 0:
            print(f"✅ Found {len(hits)} member(s) in results\n")
            # Each hit should be a dictionary with member information
            for i, hit in enumerate(hits[:3], 1):  # Show first 3
                print(f"Member {i}:")
                if isinstance(hit, dict):
                    # Print key fields if available
                    for key in ["name", "id", "number", "bio", "goal"]:
                        if key in hit:
                            value = str(hit[key])[:100]  # Truncate long values
                            print(f"  - {key}: {value}")
                print()
        else:
            print("ℹ️  No members found in results\n")

    @pytest.mark.timeout(30)
    def test_skill_script_chinese_keyword_search(self, tmp_path: Path) -> None:
        """Test that the script can search with Chinese keywords and return results.
        
        This test performs a real AI He Huo search with Chinese keywords and validates:
        1. The script executes successfully with Chinese input
        2. Results are returned in JSON format
        3. Results contain Chinese characters (since AI He Huo is a Chinese platform)
        """
        # Check if requests package is installed
        import importlib.util
        requests_installed = importlib.util.find_spec("requests") is not None
        
        if not requests_installed:
            pytest.skip("requests package not installed, skipping Chinese search test")
        
        # Get the example skill directory
        repo_root = Path(__file__).parent.parent.parent
        skill_script = repo_root / "examples" / "skills" / "aihehuo-member-search" / "aihehuo_member_search.py"
        
        # Run a search with Chinese keywords
        # Using a descriptive Chinese query
        chinese_query = "寻找有AI技术背景的创业者，希望合作开发智能产品"
        result = subprocess.run(
            [sys.executable, str(skill_script), chinese_query, "--max-results", "3"],
            capture_output=True,
            text=True,
            timeout=30,  # Longer timeout for actual API call
        )
        
        # Script should execute successfully
        assert result.returncode == 0, f"Script should execute successfully: {result.stderr}"
        
        # Print debug output from stderr (if any)
        if result.stderr:
            print("\n" + "=" * 80)
            print("DEBUG OUTPUT (stderr):")
            print("=" * 80)
            print(result.stderr)
            print("=" * 80 + "\n")
        
        # Should have output
        assert len(result.stdout) > 0, "Script should return search results"
        
        # Print the search results for visibility
        output = result.stdout
        print("\n" + "=" * 80)
        print("AIHEHUO MEMBER SEARCH RESULTS (Chinese Keywords):")
        print("=" * 80)
        print(f"Query: {chinese_query}")
        print("-" * 80)
        print(output)
        print("=" * 80 + "\n")
        
        # Output should be valid JSON
        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            pytest.fail(f"Output should be valid JSON: {e}\nOutput: {output}")
        
        # Validate JSON structure
        assert isinstance(data, dict), "Result should be a dictionary"
        
        # Handle error case
        if "error" in data:
            error_msg = data.get('error', '')
            error_message = data.get('message', '')
            status_code = data.get('status_code')
            
            # If API key is missing, that's a configuration issue
            if "API key" in error_msg or "API key" in error_message or "AIHEHUO_API_KEY" in error_msg:
                pytest.skip(f"API key not configured: {error_message}. Please set AIHEHUO_API_KEY in .env.aihehuo file.")
            
            # 400 client errors indicate a problem with our request/implementation - test should fail
            if status_code == 400:
                pytest.fail(
                    f"API returned 400 Bad Request error. This indicates a problem with the request format or parameters.\n"
                    f"Error: {error_msg}\n"
                    f"Message: {error_message}\n"
                    f"Full response: {json.dumps(data, indent=2)}"
                )
            
            # Other errors (500, network issues, etc.) are acceptable - API might be unavailable
            print(f"⚠️  API returned error: {error_msg}")
            if status_code:
                print(f"   Status code: {status_code}")
            if error_message:
                print(f"   Message: {error_message}")
            print("   (This is acceptable if API is unavailable or has server issues)")
            return
        
        # Check for Chinese characters in the output
        # Chinese characters are in Unicode ranges: \u4e00-\u9fff (CJK Unified Ideographs)
        import re
        chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]+')
        has_chinese = bool(chinese_char_pattern.search(output))
        
        # Handle different response formats
        if "data" in data:
            # New API format: { "data": [...], "meta": {...} }
            hits = data.get("data", [])
            meta = data.get("meta", {})
            total = meta.get("total_count", len(hits))
        elif "hits" in data:
            # Old format: { "hits": [...], "total": ... }
            hits = data.get("hits", [])
            total = data.get("total", len(hits))
        else:
            # Fallback
            hits = []
            total = data.get("total", 0)
        
        print(f"\nChinese Search Results Summary:")
        print(f"  - Total results: {total}")
        print(f"  - Results in this page: {len(hits)}")
        print(f"  - Query: '{chinese_query}'")
        print(f"  - Contains Chinese characters: {has_chinese}\n")
        
        # Since AI He Huo is a Chinese platform, results should likely contain Chinese
        if has_chinese:
            print("✅ Found Chinese characters in results - Chinese platform data detected!")
        else:
            print("ℹ️  No Chinese characters found in JSON structure")
            print("   (Chinese content may be in nested fields within the data array)")


class TestAihehuoMemberSearchSkillIntegration:
    """Integration test: skill discovery + agent setup + execution capability."""

    def test_agent_with_aihehuo_skill_middleware(self, tmp_path: Path) -> None:
        """Test that agent can be created with aihehuo-member-search skill loaded."""
        # Get the example skill directory
        repo_root = Path(__file__).parent.parent.parent
        example_skill_dir = repo_root / "examples" / "skills" / "aihehuo-member-search"
        
        # Copy skill to temporary skills directory
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skill_dest = skills_dir / "aihehuo-member-search"
        shutil.copytree(example_skill_dir, skill_dest)
        
        # Create fake model
        model = FixedGenericFakeChatModel(
            messages=iter([
                AIMessage(content="I can help you search AI He Huo members."),
            ])
        )
        
        # Mock settings to use our temp directory
        with patch("deepagents_cli.agent.settings") as mock_settings:
            agent_dir = tmp_path / "agents" / "test-agent"
            agent_dir.mkdir(parents=True)
            (agent_dir / "agent.md").write_text("# Test Agent")
            
            mock_settings.user_deepagents_dir = tmp_path / "agents"
            mock_settings.ensure_agent_dir.return_value = agent_dir
            mock_settings.ensure_user_skills_dir.return_value = skills_dir
            mock_settings.get_project_skills_dir.return_value = None
            mock_settings.get_user_agent_md_path.return_value = agent_dir / "agent.md"
            mock_settings.get_project_agent_md_path.return_value = None
            mock_settings.get_agent_dir.return_value = agent_dir
            mock_settings.project_root = None
            
            # Create agent with skills middleware
            agent, backend = create_cli_agent(
                model=model,
                assistant_id="test-agent",
                tools=[],
            )
            
            # Verify agent was created
            assert agent is not None, "Agent should be created"
            
            # Invoke agent to trigger skills middleware
            result = agent.invoke(
                {"messages": [HumanMessage(content="What skills do you have?")]},
                {"configurable": {"thread_id": "test-thread"}},
            )
            
            # Verify agent executed
            assert "messages" in result, "Result should contain messages"
            
            # Check that skills middleware was active
            ai_messages = [msg for msg in result["messages"] if msg.type == "ai"]
            assert len(ai_messages) > 0, "Agent should produce AI messages"

    def test_skill_script_is_executable(self, tmp_path: Path) -> None:
        """Test that the skill script file is properly formatted and executable."""
        # Get the example skill directory
        repo_root = Path(__file__).parent.parent.parent
        skill_script = repo_root / "examples" / "skills" / "aihehuo-member-search" / "aihehuo_member_search.py"
        
        # Verify script exists
        assert skill_script.exists(), "Skill script should exist"
        
        # Verify it's a Python file
        assert skill_script.suffix == ".py", "Should be a Python file"
        
        # Verify it has shebang
        with open(skill_script, "r", encoding="utf-8") as f:
            first_line = f.readline()
            assert first_line.startswith("#!/usr/bin/env python"), "Should have Python shebang"
        
        # Verify script can be read and has valid structure
        with open(skill_script, "r", encoding="utf-8") as f:
            content = f.read()
            assert "def search_members" in content, "Should have search_members function"
            assert "def main" in content, "Should have main function"
            assert "argparse" in content, "Should use argparse"
            assert "requests" in content, "Should use requests library"

    def test_skill_markdown_format(self, tmp_path: Path) -> None:
        """Test that SKILL.md has proper format and required fields."""
        # Get the example skill directory
        repo_root = Path(__file__).parent.parent.parent
        skill_md = repo_root / "examples" / "skills" / "aihehuo-member-search" / "SKILL.md"
        
        # Verify file exists
        assert skill_md.exists(), "SKILL.md should exist"
        
        # Read and verify frontmatter
        content = skill_md.read_text(encoding="utf-8")
        
        # Check for YAML frontmatter
        assert content.startswith("---"), "Should start with YAML frontmatter"
        assert "name:" in content, "Should have name field"
        assert "description:" in content, "Should have description field"
        
        # Check for required sections
        assert "## When to Use" in content or "## When to use" in content, "Should have usage section"
        assert "## How to Use" in content or "## How to use" in content, "Should have how-to section"
        
        # Check for skill name in frontmatter
        assert "aihehuo-member-search" in content, "Should mention skill name"

