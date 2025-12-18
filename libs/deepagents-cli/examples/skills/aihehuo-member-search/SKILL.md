---
name: aihehuo-member-search
description: Match entrepreneurs with potential co-founders, investors, and partners on the AI He Huo (爱合伙) platform
---

# AI He Huo Member Search Skill

This skill helps you connect entrepreneurs and business idea creators with the right people on the AI He Huo (爱合伙) platform - a Chinese entrepreneurship and networking platform. Use this skill to find co-founders, investors, domain experts, and partners who can help bring business ideas to life.

## When to Use This Skill

Use this skill when you need to:
- **Match entrepreneurs with co-founders** - Find people with complementary skills and backgrounds
- **Connect with investors** - Identify investors interested in specific industries, technologies, or business models
- **Find domain experts** - Locate professionals with relevant experience in specific industries or technologies
- **Build founding teams** - Discover potential team members for different roles (technical, business, marketing, etc.)
- **Explore similar ideas** - Find related business ideas and projects to learn from or collaborate with

## Core Capabilities

This skill empowers you to:

1. **Extract and formulate search queries** from business ideas, business plans, or user requirements
2. **Break down complex needs** into multiple targeted searches for different roles and requirements
3. **Match people with opportunities** using semantic search to find the best fits
4. **Discover complementary partners** who can fill gaps in skills, experience, or resources

## How to Use

This skill provides access to two powerful tools:

- **`aihehuo_search_members`**: Search for members, entrepreneurs, and investors
- **`aihehuo_search_ideas`**: Search for business ideas and projects

### Workflow: From Business Idea to Matched Partners

#### Step 1: Analyze and Extract Key Requirements

When given a business idea or business plan, start by:
- **Summarizing the core concept** - What problem does it solve? What market does it target?
- **Identifying required roles** - What skills and expertise are needed? (e.g., technical co-founder, business development, marketing, domain expertise)
- **Defining investor profile** - What type of investors would be interested? (e.g., early-stage, EdTech-focused, AI/ML investors)
- **Extracting key phrases** - What are the important keywords, technologies, industries, or domains?

#### Step 2: Create Targeted Search Queries

For each role or need, create a specific, descriptive search query. **Use natural language sentences** rather than simple keywords for best results with semantic search.

**Example: Breaking down an AI-powered educational platform idea**

Instead of one generic search, create multiple targeted searches:

1. **Technical Co-Founder Search:**
   - Query: "寻找有AI技术背景的创业者，擅长教育科技领域，希望合作开发智能教育产品"
   - Focus: Technical expertise in AI and EdTech

2. **Business Development Co-Founder Search:**
   - Query: "需要寻找有教育行业背景的创业者，熟悉在线教育市场，擅长业务拓展"
   - Focus: Industry experience and business development

3. **Investor Search:**
   - Query: "寻找对教育科技领域感兴趣的投资人，关注AI驱动的教育创新项目"
   - Focus: EdTech and AI-focused investors
   - Use `investor=True` parameter

4. **Domain Expert Search:**
   - Query: "寻找有在线教育平台运营经验的专业人士，了解学生学习行为分析"
   - Focus: Domain expertise in online learning

5. **Related Ideas Search:**
   - Use `aihehuo_search_ideas` with query: "AI驱动的教育平台项目，帮助学生通过互动编程挑战学习编程"
   - Focus: Find similar projects to learn from or collaborate with

#### Step 3: Execute Multiple Searches

Execute each search query separately to get targeted results. This approach:
- **Improves relevance** - Each search is focused on a specific need
- **Enables comparison** - You can evaluate candidates for each role independently
- **Provides comprehensive coverage** - Ensures you find people for all necessary roles

### Search Query Best Practices

#### ✅ Good Query Examples

**For Co-Founders:**
- "寻找有AI技术背景的创业者，希望合作开发智能产品"
- "需要寻找有丰富经验的技术合伙人，擅长移动应用开发和产品设计"
- "寻找有教育行业背景的创业者，熟悉在线教育市场，擅长业务拓展和用户增长"

**For Investors:**
- "寻找对教育科技领域感兴趣的投资人，关注早期阶段的AI教育创新项目"
- "寻找关注SaaS和B2B教育产品的投资人，有EdTech行业投资经验"

**For Domain Experts:**
- "寻找有在线教育平台运营经验的专业人士，了解学生学习行为分析和个性化推荐"
- "寻找有编程教育背景的专家，熟悉青少年编程教学和课程设计"

**For Ideas:**
- "AI驱动的教育平台项目，帮助学生通过互动编程挑战学习编程"
- "移动应用开发项目，专注于在线教育领域"

#### ❌ Poor Query Examples (Avoid These)

- "AI 技术" - Too generic, lacks context
- "创业者 投资人" - Just keywords, not descriptive
- "移动应用" - No specific need or context
- "教育" - Too broad, doesn't describe what you're looking for

### Query Guidelines

1. **Use full sentences** - Describe what you're looking for as you would to a colleague
2. **Be specific** - Include industry, technology, role, or domain context
3. **Minimum length** - Member searches require queries longer than 5 characters (6+ recommended)
4. **Natural language** - Write queries in natural, conversational language
5. **One need per query** - Create separate queries for different roles or requirements

### Tool Parameters

**For `aihehuo_search_members`:**
- `query` (required): Natural language search query (6+ characters)
- `max_results` (optional): Number of results per page (default: 10, minimum: 10)
- `page` (optional): Page number for pagination (default: 1)
- `wechat_reachable_only` (optional): Filter for WeChat-reachable members (default: False)
- `investor` (optional): Filter for investors only (default: None, searches all)
- `excluded_ids` (optional): List of user IDs to exclude from results

**For `aihehuo_search_ideas`:**
- `query` (required): Natural language search query
- `max_results` (optional): Number of results per page (default: 10)
- `page` (optional): Page number for pagination (default: 1)

## Strategic Approach: Multi-Role Matching

When helping an entrepreneur find partners, follow this strategic approach:

### 1. Understand the Business Idea
- Read and summarize the business idea or plan
- Identify the core value proposition and target market
- Understand the business model and go-to-market strategy

### 2. Identify Gaps and Needs
- What skills are missing from the current team?
- What expertise is needed to execute the idea?
- What resources or connections are required?

### 3. Create Role-Specific Searches
- **Technical roles**: Focus on specific technologies, frameworks, or technical domains
- **Business roles**: Emphasize industry experience, business development, or market knowledge
- **Investor searches**: Highlight industry focus, stage preference, and investment thesis alignment
- **Domain experts**: Target specific industry knowledge or specialized expertise

### 4. Execute and Synthesize
- Run multiple searches in parallel for different roles
- Compare and evaluate candidates for each role
- Identify the best matches based on background, experience, and project fit
- Consider complementary skills and team dynamics

## Example: Complete Matching Workflow

**Scenario:** An entrepreneur has an idea for an AI-powered language learning app for children.

**Step 1 - Analysis:**
- Core concept: AI-powered language learning for children (ages 5-12)
- Target market: Parents seeking educational apps for children
- Key technologies: AI/ML, mobile app development, educational content

**Step 2 - Identify Needs:**
- Technical co-founder with AI/ML and mobile app experience
- Educational content expert with children's education background
- Business co-founder with EdTech market knowledge
- Investors interested in EdTech and AI

**Step 3 - Create Searches:**

```python
# Search 1: Technical Co-Founder
aihehuo_search_members(
    query="寻找有AI和机器学习技术背景的创业者，擅长移动应用开发，希望合作开发儿童教育产品",
    max_results=10
)

# Search 2: Educational Content Expert
aihehuo_search_members(
    query="寻找有儿童教育背景的专业人士，熟悉语言学习教学法，有教育内容开发经验",
    max_results=10
)

# Search 3: Business Co-Founder
aihehuo_search_members(
    query="寻找有教育科技行业背景的创业者，熟悉儿童教育产品市场，擅长用户增长和业务拓展",
    max_results=10
)

# Search 4: Investors
aihehuo_search_members(
    query="寻找对教育科技和AI驱动的儿童教育产品感兴趣的投资人",
    investor=True,
    max_results=10
)

# Search 5: Related Ideas
aihehuo_search_ideas(
    query="AI驱动的儿童语言学习应用项目，专注于移动端教育产品"
)
```

**Step 4 - Synthesize Results:**
- Review candidates for each role
- Identify best matches based on experience and fit
- Compile a comprehensive report with recommendations

## Output Format

Search results are returned in JSON format with:
- **total**: Total number of matching results
- **page**: Current page number
- **page_size**: Number of results per page
- **hits**: Array of member/idea objects containing:
  - User/project information (name, ID, number, etc.)
  - Background and experience
  - Project information (if applicable)
  - Contact information (if available)

## Key Features

- **Semantic vector search**: AI-powered semantic search for better relevance and matching
- **Multi-role matching**: Create targeted searches for different roles and needs
- **Flexible filtering**: Filter by investor status, WeChat reachability, and exclude specific users
- **Pagination support**: Navigate through large result sets
- **Idea discovery**: Find related business ideas to learn from or collaborate with

## Important Notes

- **Use natural language**: Write queries as descriptive sentences, not keyword lists
- **One role per search**: Create separate searches for different roles or needs
- **Be specific**: Include industry, technology, and context in your queries
- **Multiple searches**: Don't try to find everything in one search - break it down by role
- **Synthesize results**: Compare candidates across different searches to find the best matches

## Platform Information

- AI He Huo (爱合伙) is a Chinese entrepreneurship and networking platform
- The platform supports semantic search for better matching
- Results include entrepreneurs, investors, and members with various backgrounds
- WeChat reachability filter helps find members you can contact directly
- Results are returned in JSON format for easy parsing and integration
