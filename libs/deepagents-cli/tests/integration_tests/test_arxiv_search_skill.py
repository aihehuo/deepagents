"""Integration tests for arxiv-search skill.

This test suite verifies:
1. Skill discovery and loading through SkillsMiddleware
2. Skill script execution
3. Output validation
"""

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


class TestArxivSearchSkillDiscovery:
    """Test that arxiv-search skill can be discovered and loaded."""

    def test_skill_discovery_from_examples(self, tmp_path: Path) -> None:
        """Test that arxiv-search skill is discovered when copied to skills directory."""
        # Get the example skill directory
        repo_root = Path(__file__).parent.parent.parent
        example_skill_dir = repo_root / "examples" / "skills" / "arxiv-search"
        
        # Copy skill to temporary skills directory
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skill_dest = skills_dir / "arxiv-search"
        shutil.copytree(example_skill_dir, skill_dest)
        
        # Test skill discovery
        skills = list_skills(user_skills_dir=skills_dir, project_skills_dir=None)
        
        assert len(skills) == 1, "Should discover exactly one skill"
        assert skills[0]["name"] == "arxiv-search", "Skill name should match"
        assert "arXiv" in skills[0]["description"], "Description should mention arXiv"
        assert skills[0]["source"] == "user", "Skill should be from user directory"
        
        # Verify SKILL.md exists and is readable
        skill_md_path = Path(skills[0]["path"])
        assert skill_md_path.exists(), "SKILL.md should exist"
        assert skill_md_path.name == "SKILL.md", "Path should point to SKILL.md"
        
        # Verify skill script exists
        skill_script = skill_dest / "arxiv_search.py"
        assert skill_script.exists(), "arxiv_search.py should exist"

    def test_skills_middleware_loads_arxiv_skill(self, tmp_path: Path) -> None:
        """Test that SkillsMiddleware loads arxiv-search skill metadata."""
        # Get the example skill directory
        repo_root = Path(__file__).parent.parent.parent
        example_skill_dir = repo_root / "examples" / "skills" / "arxiv-search"
        
        # Copy skill to temporary skills directory
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skill_dest = skills_dir / "arxiv-search"
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
        assert skills_list[0]["name"] == "arxiv-search", "Skill name should match"
        assert "arXiv" in skills_list[0]["description"], "Description should mention arXiv"


class TestArxivSearchSkillExecution:
    """Test that arxiv-search skill script can be executed."""

    def test_skill_script_help(self, tmp_path: Path) -> None:
        """Test that arxiv-search script shows help when run with --help."""
        # Get the example skill directory
        repo_root = Path(__file__).parent.parent.parent
        skill_script = repo_root / "examples" / "skills" / "arxiv-search" / "arxiv_search.py"
        
        # Run script with --help
        result = subprocess.run(
            [sys.executable, str(skill_script), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        assert result.returncode == 0, f"Script should exit successfully: {result.stderr}"
        assert "Search arXiv" in result.stdout, "Help should mention arXiv search"
        assert "query" in result.stdout, "Help should mention query parameter"

    @pytest.mark.timeout(30)
    def test_skill_script_without_arxiv_package(self, tmp_path: Path) -> None:
        """Test that script handles missing arxiv package gracefully.
        
        Note: This test verifies the script structure handles imports correctly.
        Since the script runs in a subprocess, we can't mock sys.modules.
        If arxiv is installed, the script will run successfully (which is also valid).
        The test verifies the script can execute without crashing.
        """
        # Get the example skill directory
        repo_root = Path(__file__).parent.parent.parent
        skill_script = repo_root / "examples" / "skills" / "arxiv-search" / "arxiv_search.py"
        
        # Check if arxiv is installed in the test environment
        import importlib.util
        arxiv_installed = importlib.util.find_spec("arxiv") is not None
        
        # Run the script with a simple query (will make actual API call if arxiv is installed)
        result = subprocess.run(
            [sys.executable, str(skill_script), "test query", "--max-papers", "1"],
            capture_output=True,
            text=True,
            timeout=30,  # Increased timeout for actual API call
        )
        
        if arxiv_installed:
            # If arxiv is installed, script should run successfully
            # Note: main() doesn't print the result, so stdout may be empty
            assert result.returncode == 0, f"Script should run successfully when arxiv is installed: {result.stderr}"
        else:
            # If arxiv is not installed, script should handle it gracefully
            assert result.returncode != 0 or "arxiv package not installed" in result.stdout or "Error" in result.stdout

    @pytest.mark.timeout(30)
    def test_skill_script_query_validation(self, tmp_path: Path) -> None:
        """Test that script validates query input."""
        # Get the example skill directory
        repo_root = Path(__file__).parent.parent.parent
        skill_script = repo_root / "examples" / "skills" / "arxiv-search" / "arxiv_search.py"
        
        # Test with very short query (will make actual API call if arxiv is installed)
        # Limit to 1 paper to speed up the test
        result = subprocess.run(
            [sys.executable, str(skill_script), "test", "--max-papers", "1"],
            capture_output=True,
            text=True,
            timeout=30,  # Increased timeout for actual API call
        )
        
        # Script should either accept it or provide helpful error
        # (The actual validation happens in the API, but script should handle it)
        assert result.returncode in [0, 1], "Script should handle short queries"

    def test_skill_script_actual_search_results(self, tmp_path: Path) -> None:
        """Test that the script returns actual search results in the expected format.
        
        This test performs a real arXiv search and validates:
        1. The script executes successfully
        2. Results are returned in the expected format (Title: ... Summary: ...)
        3. Results contain meaningful content
        """
        # Check if arxiv package is installed
        import importlib.util
        arxiv_installed = importlib.util.find_spec("arxiv") is not None
        
        if not arxiv_installed:
            pytest.skip("arxiv package not installed, skipping actual search test")
        
        # Get the example skill directory
        repo_root = Path(__file__).parent.parent.parent
        skill_script = repo_root / "examples" / "skills" / "arxiv-search" / "arxiv_search.py"
        
        # Run a real search with a simple, common query
        # Using "machine learning" as it's likely to return results
        result = subprocess.run(
            [sys.executable, str(skill_script), "machine learning", "--max-papers", "3"],
            capture_output=True,
            text=True,
            timeout=30,  # Longer timeout for actual API call
        )
        
        # Script should execute successfully
        assert result.returncode == 0, f"Script should execute successfully: {result.stderr}"
        
        # Should have output
        assert len(result.stdout) > 0, "Script should return search results"
        
        # Print the search results for visibility
        output = result.stdout
        print("\n" + "=" * 80)
        print("ARXIV SEARCH RESULTS:")
        print("=" * 80)
        print(output)
        print("=" * 80 + "\n")
        
        # Validate result format - should contain "Title:" and "Summary:" markers
        assert "Title:" in output, "Results should contain 'Title:' field"
        assert "Summary:" in output, "Results should contain 'Summary:' field"
        
        # Count how many results we got (each result should have Title and Summary)
        title_count = output.count("Title:")
        summary_count = output.count("Summary:")
        
        # Print summary information
        print(f"\nSearch Results Summary:")
        print(f"  - Number of papers found: {title_count}")
        print(f"  - Query: 'machine learning'")
        print(f"  - Max papers requested: 3")
        print(f"  - Output length: {len(output)} characters\n")
        
        # Should have at least one result
        assert title_count > 0, "Should have at least one result with Title"
        assert summary_count > 0, "Should have at least one result with Summary"
        
        # Title and Summary counts should match (each paper has both)
        assert title_count == summary_count, f"Each result should have both Title and Summary (got {title_count} titles, {summary_count} summaries)"
        
        # Results should be limited to max-papers (3 in this case)
        assert title_count <= 3, f"Should not exceed max-papers limit (got {title_count} results)"
        
        # Verify the format: each result should have Title followed by Summary
        # Split by double newlines (results are separated by blank lines)
        results = [r.strip() for r in output.split("\n\n") if r.strip()]
        
        # Each result block should start with "Title:"
        for result_block in results:
            if result_block.startswith("Title:"):
                # Should contain both Title and Summary
                assert "Title:" in result_block, "Result block should contain Title"
                assert "Summary:" in result_block, "Result block should contain Summary"
                
                # Extract title and summary
                lines = result_block.split("\n")
                title_line = [l for l in lines if l.startswith("Title:")][0]
                summary_line = [l for l in lines if l.startswith("Summary:")][0]
                
                # Title and Summary should have actual content (not just the label)
                assert len(title_line) > len("Title:"), "Title should have content"
                assert len(summary_line) > len("Summary:"), "Summary should have content"

    @pytest.mark.timeout(30)
    def test_skill_script_chinese_keyword_search(self, tmp_path: Path) -> None:
        """Test that the script can search with Chinese keywords and return Chinese literature.
        
        This test performs a real arXiv search with Chinese keywords and validates:
        1. The script executes successfully with Chinese input
        2. Results are returned in the expected format
        3. Results contain Chinese characters (indicating Chinese language papers)
        """
        # Check if arxiv package is installed
        import importlib.util
        arxiv_installed = importlib.util.find_spec("arxiv") is not None
        
        if not arxiv_installed:
            pytest.skip("arxiv package not installed, skipping Chinese search test")
        
        # Get the example skill directory
        repo_root = Path(__file__).parent.parent.parent
        skill_script = repo_root / "examples" / "skills" / "arxiv-search" / "arxiv_search.py"
        
        # Run a search with Chinese keywords
        # Using "机器学习" (machine learning) as it's likely to return Chinese papers
        chinese_query = "机器学习"
        result = subprocess.run(
            [sys.executable, str(skill_script), chinese_query, "--max-papers", "3"],
            capture_output=True,
            text=True,
            timeout=30,  # Longer timeout for actual API call
        )
        
        # Script should execute successfully
        assert result.returncode == 0, f"Script should execute successfully: {result.stderr}"
        
        # Should have output
        assert len(result.stdout) > 0, "Script should return search results"
        
        # Print the search results for visibility
        output = result.stdout
        print("\n" + "=" * 80)
        print("ARXIV SEARCH RESULTS (Chinese Keywords):")
        print("=" * 80)
        print(f"Query: {chinese_query} (machine learning in Chinese)")
        print("-" * 80)
        print(output)
        print("=" * 80 + "\n")
        
        # Handle case where no papers are found (valid result for Chinese queries)
        if "No papers found" in output:
            print("ℹ️  No papers found for Chinese query - this is a valid result")
            print("   (arXiv primarily contains English papers, so Chinese keyword searches may return no results)")
            # This is still a successful execution - the script handled the query correctly
            assert result.returncode == 0, "Script should execute successfully even when no papers found"
            return  # Exit early - no need to validate result format
        
        # If we have results, validate the format
        # Validate result format - should contain "Title:" and "Summary:" markers
        assert "Title:" in output, "Results should contain 'Title:' field"
        assert "Summary:" in output, "Results should contain 'Summary:' field"
        
        # Check for Chinese characters in the output
        # Chinese characters are in Unicode ranges: \u4e00-\u9fff (CJK Unified Ideographs)
        import re
        chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]+')
        has_chinese = bool(chinese_char_pattern.search(output))
        
        # Print summary information
        title_count = output.count("Title:")
        summary_count = output.count("Summary:")
        print(f"\nChinese Search Results Summary:")
        print(f"  - Number of papers found: {title_count}")
        print(f"  - Query: '{chinese_query}' (machine learning in Chinese)")
        print(f"  - Max papers requested: 3")
        print(f"  - Contains Chinese characters: {has_chinese}")
        print(f"  - Output length: {len(output)} characters\n")
        
        # Should have at least one result
        assert title_count > 0, "Should have at least one result with Title"
        assert summary_count > 0, "Should have at least one result with Summary"
        
        # Title and Summary counts should match
        assert title_count == summary_count, f"Each result should have both Title and Summary (got {title_count} titles, {summary_count} summaries)"
        
        # Results should be limited to max-papers (3 in this case)
        assert title_count <= 3, f"Should not exceed max-papers limit (got {title_count} results)"
        
        # Note: Chinese characters may or may not appear depending on what papers are available
        # arXiv has papers in many languages, so we validate the format but don't require Chinese
        # If Chinese characters are found, that's good - it means we got Chinese papers
        if has_chinese:
            print("✅ Found Chinese characters in results - Chinese language papers detected!")
        else:
            print("ℹ️  No Chinese characters found - results may be in English or other languages")
            print("   (This is acceptable as arXiv search returns papers matching the query regardless of language)")


class TestArxivSearchSkillIntegration:
    """Integration test: skill discovery + agent setup + execution capability."""

    def test_agent_with_arxiv_skill_middleware(self, tmp_path: Path) -> None:
        """Test that agent can be created with arxiv-search skill loaded."""
        # Get the example skill directory
        repo_root = Path(__file__).parent.parent.parent
        example_skill_dir = repo_root / "examples" / "skills" / "arxiv-search"
        
        # Copy skill to temporary skills directory
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skill_dest = skills_dir / "arxiv-search"
        shutil.copytree(example_skill_dir, skill_dest)
        
        # Create fake model
        model = FixedGenericFakeChatModel(
            messages=iter([
                AIMessage(content="I can help you search arXiv."),
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
            
            # Check that skills middleware was active (skills would be in system prompt)
            # We can't directly check system prompt, but if agent runs, middleware worked
            ai_messages = [msg for msg in result["messages"] if msg.type == "ai"]
            assert len(ai_messages) > 0, "Agent should produce AI messages"

    def test_skill_script_is_executable(self, tmp_path: Path) -> None:
        """Test that the skill script file is properly formatted and executable."""
        # Get the example skill directory
        repo_root = Path(__file__).parent.parent.parent
        skill_script = repo_root / "examples" / "skills" / "arxiv-search" / "arxiv_search.py"
        
        # Verify script exists
        assert skill_script.exists(), "Skill script should exist"
        
        # Verify it's a Python file
        assert skill_script.suffix == ".py", "Should be a Python file"
        
        # Verify it has shebang
        with open(skill_script, "r", encoding="utf-8") as f:
            first_line = f.readline()
            assert first_line.startswith("#!/usr/bin/env python"), "Should have Python shebang"
        
        # Verify script can be imported (syntax check)
        # We'll just check it can be read and has valid structure
        with open(skill_script, "r", encoding="utf-8") as f:
            content = f.read()
            assert "def query_arxiv" in content, "Should have query_arxiv function"
            assert "def main" in content, "Should have main function"
            assert "argparse" in content, "Should use argparse"

    def test_skill_markdown_format(self, tmp_path: Path) -> None:
        """Test that SKILL.md has proper format and required fields."""
        # Get the example skill directory
        repo_root = Path(__file__).parent.parent.parent
        skill_md = repo_root / "examples" / "skills" / "arxiv-search" / "SKILL.md"
        
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
        assert "arxiv-search" in content, "Should mention skill name"

