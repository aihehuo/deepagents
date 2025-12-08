---
name: aihehuo-member-search
description: Search AI He Huo (爱合伙) platform for entrepreneurs, investors, and members using semantic vector search
---

# AI He Huo Member Search Skill

This skill provides access to the AI He Huo (爱合伙) platform, a Chinese entrepreneurship and networking platform. It allows you to search for entrepreneurs, investors, and members using semantic vector search.

## When to Use This Skill

Use this skill when you need to:
- Find entrepreneurs or co-founders with specific backgrounds or expertise
- Search for investors interested in particular industries or technologies
- Discover members with relevant experience for collaboration
- Find people with specific skills, industries, or project experience
- Locate WeChat-reachable members for direct communication

## How to Use

The skill provides a Python script that searches the AI He Huo platform and returns formatted results.

### Basic Usage

**Note:** Always use the absolute path from your skills directory (shown in the system prompt above).

If running deepagents from a virtual environment:
```bash
.venv/bin/python [YOUR_SKILLS_DIR]/aihehuo-member-search/aihehuo_member_search.py "your search query" [--max-results N] [--page N]
```

Or for system Python:
```bash
python3 [YOUR_SKILLS_DIR]/aihehuo-member-search/aihehuo_member_search.py "your search query" [--max-results N] [--page N]
```

Replace `[YOUR_SKILLS_DIR]` with the absolute skills directory path from your system prompt (e.g., `~/.deepagents/agent/skills` or the full absolute path).

**Arguments:**
- `query` (required): The search query string (must be longer than 5 characters). Use coherent, descriptive sentences rather than simple keywords for best results with semantic search.
- `--max-results` (optional): Maximum number of results per page (default: 10)
- `--page` (optional): Page number for pagination (default: 1)
- `--wechat-reachable-only` (optional): Only return users who are reachable on WeChat
- `--investor` (optional): Only search for investors
- `--excluded-ids` (optional): Comma-separated list of user IDs to exclude from results

### Examples

Search for AI technology entrepreneurs:
```bash
.venv/bin/python ~/.deepagents/agent/skills/aihehuo-member-search/aihehuo_member_search.py "寻找有AI技术背景的创业者，希望合作开发智能产品" --max-results 5
```

Search for mobile app development partners:
```bash
.venv/bin/python ~/.deepagents/agent/skills/aihehuo-member-search/aihehuo_member_search.py "需要寻找有丰富经验的技术合伙人，擅长移动应用开发"
```

Search for investors in education technology:
```bash
.venv/bin/python ~/.deepagents/agent/skills/aihehuo-member-search/aihehuo_member_search.py "寻找对教育科技领域感兴趣的投资人" --investor
```

Search for WeChat-reachable members:
```bash
.venv/bin/python ~/.deepagents/agent/skills/aihehuo-member-search/aihehuo_member_search.py "寻找有技术背景的创业者" --wechat-reachable-only
```

Search with multiple filters:
```bash
.venv/bin/python ~/.deepagents/agent/skills/aihehuo-member-search/aihehuo_member_search.py "寻找早期投资人，关注教育科技领域" --investor --wechat-reachable-only --max-results 15
```

## Output Format

The script returns JSON-formatted results with:
- **total**: Total number of matching results
- **page**: Current page number
- **page_size**: Number of results per page
- **hits**: Array of member objects, each containing:
  - User information (name, ID, number, etc.)
  - Background and experience
  - Project information (if applicable)
  - Contact information (if available)

## Features

- **Semantic vector search**: Uses AI-powered semantic search for better relevance
- **Fast retrieval**: Direct API access to AI He Huo platform
- **Flexible filtering**: Filter by investor status, WeChat reachability, and exclude specific users
- **Pagination support**: Navigate through large result sets
- **Automatic configuration**: Reads API base URL and API key from `.env.aihehuo` file in repository root

## Dependencies

This skill requires the `requests` Python package. The script will detect if it's missing and show an error.

**If you see "Error: requests package not installed":**

If running deepagents from a virtual environment (recommended), use the venv's Python:
```bash
.venv/bin/python -m pip install requests
```

Or for system-wide install:
```bash
python3 -m pip install requests
```

The package is not included in deepagents by default since it's skill-specific. Install it on-demand when first using this skill.

## Configuration

The skill automatically reads the API base URL and API key from the `.env.aihehuo` file in the repository root. The file should contain:

```
AIHEHUO_API_BASE=https://new-api.aihehuo.com
AIHEHUO_API_KEY=your_api_key_here
```

**Important**: The API key is required for authentication. If the API key is not found, the script will return an error.

If the file is not found, the script will try to read from environment variables (`AIHEHUO_API_BASE` and `AIHEHUO_API_KEY`), and defaults to `https://new-api.aihehuo.com` for the base URL if not set.

## Important Notes

### Query Format

Since this skill uses **semantic vector search**, it's important to use coherent, descriptive sentences rather than simple keyword lists:

✅ **Good queries:**
- "寻找有AI技术背景的创业者，希望合作开发智能产品"
- "需要寻找有丰富经验的技术合伙人，擅长移动应用开发"
- "寻找对教育科技领域感兴趣的投资人"

❌ **Poor queries:**
- "AI 技术"
- "创业者 投资人"
- "移动应用"

### Query Length

- **Minimum length**: 6 characters (the API requires queries longer than 5 characters)
- **Recommended**: Use full sentences describing what you're looking for

### Search Tips

1. **Be specific**: Describe the type of person, their background, and what you're looking for
2. **Use natural language**: Write queries as you would describe the person to a colleague
3. **Combine filters**: Use `--investor` and `--wechat-reachable-only` together for targeted searches
4. **Pagination**: Use `--page` to navigate through results if you need more than the first page

## Notes

- AI He Huo is a Chinese entrepreneurship and networking platform
- Results include entrepreneurs, investors, and members with various backgrounds
- The platform supports semantic search for better matching
- WeChat reachability filter helps find members you can contact directly
- Results are returned in JSON format for easy parsing and integration

