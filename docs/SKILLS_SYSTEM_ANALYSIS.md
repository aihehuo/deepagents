# Skills System Analysis: Prompt-Based vs Script-Based Skills

## Executive Summary

**Current State**: The skills system **already supports prompt-based skills** (skills without scripts), though the documentation and system prompt may not make this clear.

**Finding**: Skills are **not bound to scripts** - they are documentation files that guide the agent's behavior. Scripts are optional supporting files.

---

## 1. How Skills Currently Work

### 1.1 Skill Discovery Process

**Location**: `libs/deepagents-cli/deepagents_cli/skills/load.py`

**Process**:
1. **Scan directories**: `list_skills()` scans `~/.deepagents/{agent}/skills/` and `{project}/.deepagents/skills/`
2. **Find SKILL.md files**: Looks for `SKILL.md` in subdirectories
3. **Parse YAML frontmatter**: Extracts `name` and `description` from frontmatter
4. **Return metadata**: Returns `SkillMetadata` with name, description, path, source

**Key Point**: Only `SKILL.md` is required. Scripts are never discovered or validated.

### 1.2 Skills Metadata Injection

**Location**: `libs/deepagents-cli/deepagents_cli/skills/middleware.py`

**Process**:
1. **Before agent execution**: `before_agent()` loads skills metadata
2. **On every model call**: `wrap_model_call()` injects skills list into system prompt
3. **Progressive disclosure**: Only name + description are shown initially

**System Prompt Injection**:
```python
SKILLS_SYSTEM_PROMPT = """
## Skills System

**Available Skills:**

{skills_list}

**How to Use Skills (Progressive Disclosure):**
1. Recognize when a skill applies
2. Read the skill's full instructions (using read_file on SKILL.md path)
3. Follow the skill's instructions
4. Access supporting files (scripts, configs, etc.) - OPTIONAL
"""
```

**Key Point**: Skills metadata is injected, but scripts are never mentioned in metadata.

### 1.3 Skill Application

**How the agent uses skills**:

1. **Discovery**: Agent sees skills list in system prompt (name + description)
2. **Recognition**: Agent matches user request to skill description
3. **Reading**: Agent uses `read_file` to read full `SKILL.md` content
4. **Following instructions**: Agent follows the instructions in `SKILL.md`
5. **Script execution (optional)**: If `SKILL.md` mentions scripts, agent executes them using `execute` tool

**Key Point**: Skill application is entirely agent-driven. There's no automatic script execution.

---

## 2. Current Skill Examples Analysis

### 2.1 Pure Prompt-Based Skills (No Scripts)

#### Example 1: `langgraph-docs`

**Structure**:
```
langgraph-docs/
└── SKILL.md    # Only file, no scripts
```

**Content**: Instructions on how to fetch and use LangGraph documentation using `fetch_url` tool.

**Key Characteristics**:
- ✅ No scripts
- ✅ Pure prompt/instructions
- ✅ Guides agent behavior
- ✅ Uses existing tools (`fetch_url`)

#### Example 2: `web-research`

**Structure**:
```
web-research/
└── SKILL.md    # Only file, no scripts
```

**Content**: Structured workflow for research using subagents, file operations, and `task` tool.

**Key Characteristics**:
- ✅ No scripts
- ✅ Methodology/workflow definition
- ✅ Guides agent behavior
- ✅ Uses existing tools (`task`, `write_file`, `read_file`)

### 2.2 Script-Based Skills

#### Example 3: `arxiv-search`

**Structure**:
```
arxiv-search/
├── SKILL.md
└── arxiv_search.py    # Python script
```

**Content**: Instructions + script execution commands.

**Key Characteristics**:
- ✅ Has script
- ✅ Instructions tell agent to execute script
- ✅ Script provides specialized functionality (arXiv API)

#### Example 4: `aihehuo-member-search`

**Structure**:
```
aihehuo-member-search/
├── SKILL.md
└── aihehuo_member_search.py    # Python script
```

**Content**: Instructions + script execution commands.

**Key Characteristics**:
- ✅ Has script
- ✅ Instructions tell agent to execute script
- ✅ Script provides specialized functionality (API access)

---

## 3. Key Insight: Skills Are Documentation, Not Executables

### 3.1 Skills System Does NOT Execute Scripts

**Important**: The skills system (`SkillsMiddleware`, `load.py`) **never executes scripts**.

**What it does**:
- ✅ Discovers `SKILL.md` files
- ✅ Parses YAML frontmatter
- ✅ Injects metadata into system prompt
- ❌ Does NOT discover scripts
- ❌ Does NOT execute scripts
- ❌ Does NOT validate script existence

**Script execution happens**:
- When the **agent reads `SKILL.md`** and sees instructions to run a script
- The agent uses the **`execute` tool** (from FilesystemMiddleware)
- This is **agent-driven**, not skill-system-driven

### 3.2 Skills Are Guidance, Not Automation

**Skills provide**:
- **Instructions**: Step-by-step workflows
- **Methodologies**: Design thinking, research patterns, etc.
- **Best practices**: How to approach specific tasks
- **Tool usage guidance**: Which tools to use and when

**Skills do NOT provide**:
- Automatic execution
- Tool creation
- Built-in functionality

**Conclusion**: Skills are **documentation that guides the agent**, not executable code.

---

## 4. Support for Prompt-Based Skills

### 4.1 Current Support Status

**✅ Fully Supported**: Prompt-based skills work today!

**Evidence**:
- `langgraph-docs` skill has no scripts
- `web-research` skill has no scripts
- Skills system only requires `SKILL.md`
- No validation or requirement for scripts

### 4.2 Why It Might Seem Script-Required

**Potential Confusion Sources**:

1. **System Prompt Wording**:
   ```markdown
   **Executing Skill Scripts:**
   Skills may contain Python scripts or other executable files.
   ```
   - Says "may contain" (optional), but emphasizes scripts

2. **Example Skills**:
   - Many example skills happen to include scripts
   - Could give impression scripts are required

3. **Documentation**:
   - May not explicitly state scripts are optional

---

## 5. Design Implications for Prompt-Based Skills

### 5.1 What Prompt-Based Skills Can Provide

**Methodologies**:
- Design thinking frameworks
- Lean startup methodologies
- Structured thinking patterns
- Decision-making frameworks

**Specialized Knowledge**:
- Domain-specific guidance
- Industry best practices
- Problem-solving approaches
- Interview/question frameworks

**Workflow Definitions**:
- Multi-step processes
- Quality checklists
- Review procedures
- Planning templates

**Policy/Behavior Guidelines**:
- Communication styles
- Response patterns
- Interaction protocols
- Ethical guidelines

### 5.2 Example: Business Co-Founder Skill (Prompt-Based)

**Structure**:
```
business-cofounder/
└── SKILL.md
```

**Content** (hypothetical):
```markdown
---
name: business-cofounder
description: Act as a business co-founder applying design thinking and lean startup principles
---

# Business Co-Founder Skill

You are acting as a business co-founder and co-CEO, helping entrepreneurs develop startup ideas.

## Core Principles

### Design Thinking Application

1. **Empathize Phase**:
   - Help user understand their target customers
   - Guide user persona creation
   - Identify pain points through structured questioning

2. **Define Phase**:
   - Help clearly define problems
   - Prioritize pain points
   - Create problem statements

3. **Ideate Phase**:
   - Brainstorm solutions
   - Evaluate ideas
   - Guide MVP scope definition

[... more methodologies ...]

## Information Gathering Framework

When the user's idea is incomplete, ask structured questions:

1. **Market Questions**:
   - Who is your target customer?
   - What problem are you solving?
   - Why is this problem urgent?

2. **Solution Questions**:
   - How does your solution work?
   - What makes it unique?
   - Why will customers choose you?

[... more frameworks ...]

## Artifact Generation Guidelines

### User Persona Artifact
- Include: Demographics, psychographics, pain points, goals
- Format: Structured HTML with clear sections
- Template: [detailed template]

### Pain Point Artifact
- Include: Problem statement, affected users, severity, frequency
- Format: Prioritized list in HTML
- Template: [detailed template]

[... more artifact guidelines ...]
```

**No scripts needed!** This skill provides:
- ✅ Methodologies (design thinking, lean startup)
- ✅ Frameworks (question structures)
- ✅ Guidelines (artifact templates)
- ✅ Behavioral instructions (how to act as co-founder)

---

## 6. Recommendations

### 6.1 System Prompt Clarification

**Current** (may be confusing):
```markdown
**Executing Skill Scripts:**
Skills may contain Python scripts or other executable files. Always use absolute paths from the skill list.
```

**Suggested Update**:
```markdown
**Skill Types:**
Skills can be:
- **Prompt-based**: Pure instructions, methodologies, frameworks (just SKILL.md)
- **Script-based**: Includes Python scripts or executables (SKILL.md + script files)

Both types work the same way - read SKILL.md and follow instructions. Scripts are optional supporting files.
```

### 6.2 Documentation Updates

**Add explicit section**:
```markdown
## Skill Types

### Prompt-Based Skills (No Scripts)
Skills can be pure documentation providing:
- Methodologies and frameworks
- Specialized knowledge and guidance
- Workflow definitions
- Behavioral instructions

Example: `web-research` skill provides a research workflow without any scripts.

### Script-Based Skills (With Scripts)
Skills can include Python scripts or executables for specialized functionality.

Example: `arxiv-search` skill includes a Python script to query arXiv API.

**Both types are equally valid!** Scripts are optional.
```

### 6.3 Enhanced Skill Metadata (Optional Enhancement)

**Current Metadata**:
```python
SkillMetadata = {
    "name": str,
    "description": str,
    "path": str,  # Path to SKILL.md
    "source": str  # "user" or "project"
}
```

**Potential Enhancement** (for better discovery):
```python
SkillMetadata = {
    "name": str,
    "description": str,
    "path": str,
    "source": str,
    "type": str,  # "prompt-only" | "script-based" | "mixed"
    "has_scripts": bool,  # Auto-detected
}
```

**Implementation**: Scan skill directory for executable files after parsing `SKILL.md`.

**Benefit**: Agent could prioritize prompt-only skills for methodology tasks, script-based for execution tasks.

---

## 7. Use Cases for Prompt-Based Skills

### 7.1 Business Co-Founder Skill

**Perfect for prompt-based**:
- Design thinking methodology
- Lean startup frameworks
- Question generation frameworks
- Artifact generation templates
- Conversation patterns

**No scripts needed** - all guidance and methodology.

### 7.2 Other Prompt-Based Skill Examples

**Design Thinking Skill**:
- Empathize → Define → Ideate → Prototype → Test framework
- Question templates for each phase
- Evaluation criteria

**Product Management Skill**:
- PRD templates
- Feature prioritization frameworks
- User story formats
- Roadmap planning

**Code Review Skill**:
- Review checklist
- Security considerations
- Performance evaluation
- Best practices

**Interview Skill**:
- Question frameworks
- Evaluation rubrics
- Follow-up patterns

---

## 8. Implementation Recommendations

### 8.1 No Code Changes Needed (Current State)

**For pure prompt-based skills**:
- ✅ Create `SKILL.md` with YAML frontmatter
- ✅ Write instructions, methodologies, frameworks
- ✅ No scripts required
- ✅ Works immediately

### 8.2 Optional Enhancements

#### Enhancement 1: Skill Type Detection

**Purpose**: Help agent understand skill capabilities

**Implementation**:
```python
def _detect_skill_type(skill_dir: Path) -> str:
    """Detect if skill has scripts."""
    has_scripts = any(
        f.suffix in ['.py', '.sh', '.js'] and f.name != '__init__.py'
        for f in skill_dir.iterdir()
        if f.is_file()
    )
    return "script-based" if has_scripts else "prompt-only"
```

#### Enhancement 2: Enhanced System Prompt

**Update system prompt to explicitly mention both types**:
```markdown
**Skill Types:**
- **Prompt-based skills**: Provide methodologies, frameworks, and guidance (no scripts needed)
- **Script-based skills**: Include executable files for specialized functionality

Read SKILL.md to understand how to use each skill.
```

#### Enhancement 3: Skill Examples

**Add more prompt-only skill examples**:
- `design-thinking` - Design thinking methodology
- `product-management` - Product management frameworks
- `code-review` - Review checklists and best practices

---

## 9. Current Limitations (Minor)

### 9.1 Potential Confusion

**Issue**: System prompt may make scripts seem required

**Impact**: Low - skills work regardless, but agent might be confused

**Solution**: Update system prompt wording

### 9.2 No Type Differentiation

**Issue**: Can't distinguish prompt-only from script-based in metadata

**Impact**: Low - agent reads SKILL.md to understand anyway

**Solution**: Optional enhancement for better discovery

### 9.3 Documentation Gap

**Issue**: Documentation may not explicitly state scripts are optional

**Impact**: Medium - users might think scripts are required

**Solution**: Update documentation with clear examples

---

## 10. Conclusion

### Summary

1. **✅ Prompt-based skills are fully supported** - no code changes needed
2. **Skills are documentation, not executables** - scripts are optional
3. **Current system is flexible** - supports both types equally
4. **Minor improvements possible** - clearer documentation and system prompt

### Answer to Your Question

**Q: Can we have skills not bound to a script?**

**A: Yes! Skills are already not bound to scripts.** 

- Skills are just `SKILL.md` files with instructions
- Scripts are optional supporting files
- Examples like `langgraph-docs` and `web-research` are already prompt-only
- The system never validates or requires scripts

**For your business co-founder use case**, a pure prompt-based skill would be perfect:
- Design thinking methodology → Instructions in SKILL.md
- Lean startup principles → Instructions in SKILL.md
- Question frameworks → Instructions in SKILL.md
- Artifact templates → Instructions in SKILL.md

No scripts needed!

### Next Steps

1. **Create prompt-based skill**: Just create `SKILL.md` with your methodologies
2. **Optional**: Update system prompt for clarity
3. **Optional**: Add skill type detection for better metadata
4. **Optional**: Update documentation with prompt-only examples

---

## 11. Appendix: Current Skill Discovery Code

**Key Function**: `list_skills()` in `libs/deepagents-cli/deepagents_cli/skills/load.py`

```python
def _list_skills(skills_dir: Path, source: str) -> list[SkillMetadata]:
    """List all skills from a single skills directory."""
    skills: list[SkillMetadata] = []
    
    for skill_dir in skills_dir.iterdir():
        if not skill_dir.is_dir():
            continue
        
        # Look for SKILL.md file
        skill_md_path = skill_dir / "SKILL.md"
        if not skill_md_path.exists():
            continue  # Skip directories without SKILL.md
        
        # Parse metadata from SKILL.md
        metadata = _parse_skill_metadata(skill_md_path, source=source)
        if metadata:
            skills.append(metadata)
    
    return skills
```

**Key Points**:
- ✅ Only checks for `SKILL.md`
- ✅ Never checks for scripts
- ✅ Scripts are completely ignored in discovery
- ✅ Only parses YAML frontmatter (name, description)

**Conclusion**: Scripts are never discovered or validated. Skills work perfectly fine without them!

