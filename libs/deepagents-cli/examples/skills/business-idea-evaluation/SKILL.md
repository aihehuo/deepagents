---
name: business-idea-evaluation
description: Evaluates whether the conversation currently contains a clear, materialized business idea from at least one of three valid perspectives. This skill is used only until a complete business idea has been identified; once an idea is recognized, the skill becomes irrelevant and should not be invoked again.
---

# Skill: Business Idea Evaluation (Pre-Idea Phase Only)

## Purpose
This skill is used during the **early phase of the conversation**, before a complete business idea has been clearly formulated. It evaluates whether the user’s current input contains enough substance to qualify as a business idea under one or more of the following perspectives:
1. Painpoint / target-user need  
2. Technology-driven opportunity  
3. Future vision or scenario  

Once a clear business idea has been recognized in the conversation, this skill must not be used again.

## When This Skill Should Be Used
- Use this skill **whenever the conversation has not yet produced a clear, materialized business idea**.  
- The skill may be invoked on the first message, or on early follow-up messages, as long as the agent has not yet detected a valid idea.
- As soon as the user provides a sufficiently clear business idea (according to one of the valid perspectives), the conversation transitions into the **post-idea phase**, and this skill becomes irrelevant.
- Do NOT invoke this skill again after a valid idea has been identified.

## Valid Perspectives for a Business Idea

### 1. Painpoint / Target-User Perspective
A description of a real problem or need experienced by a specific user segment.  
Minimum requirement:  
- Mentions a user group, OR  
- Describes a painpoint clearly enough to infer a user segment.

### 2. Technology-Based Opportunity Perspective
A new technology, capability, or invention that unlocks new possibilities.  
Minimum requirement:  
- Mentions the technology AND  
- Describes at least one application, implication, or outcome.

### 3. Future Vision / Scenario Perspective
A forward-looking or imagined scenario that implies a new need or product opportunity.  
Minimum requirement:  
- Describes a scenario or vision AND  
- Implies a problem, need, or opportunity.

## What This Skill Produces
The output includes (all formatted in strict Markdown for readability):
- Whether the current message qualifies as a complete business idea.
- Which of the three perspectives it satisfies.
- A brief explanation with clear reasoning.
- If complete: a concise, well-formatted summary of the idea.
- If incomplete: constructive feedback + 3-5 clarifying questions to guide the user (in a numbered list).

**All output MUST use proper Markdown formatting** (headers, bold text, lists, proper spacing) as specified in the Output Format section below.

## Completeness Gate (Mandatory)
**Important:** A message can “touch” one of the three perspectives but still be **incomplete**.

Before you declare **Is business idea: Yes** (and before you call `mark_business_idea_complete`), you MUST be able to answer **all three** of the following with concrete, non-generic specifics:

1. **WHO** is the target user/customer segment?  
   - A specific segment (e.g., “mid-career product managers in tech companies”) is OK.  
   - Generic segments like “people”, “everyone”, “students” without context are **NOT** enough.
2. **WHAT** is the specific pain point / need?  
   - Must be a real, concrete problem (frequency/urgency or clear consequence helps).  
   - Generic “productivity is hard” is **NOT** enough.
3. **HOW** will the solution work at a high level?  
   - Must include a concrete mechanism, workflow, or differentiating capability.  
   - “An AI app for productivity” without explaining what it does is **NOT** enough.

If any of the above is missing or vague, the idea is **incomplete** even if the user mentions “AI”, “tech”, or a future vision.

## Incomplete Idea Criteria
The idea is incomplete if the message:
- Does not describe a user need or painpoint, AND  
- Does not describe a technological opportunity, AND  
- Does not describe a future scenario with implied opportunities.

In this case, the skill:
- Provides constructive feedback  
- Identifies what is missing  
- Asks for clarification  
- Continues to be used until a valid idea emerges

### Common “False Positive” Cases (Treat as Incomplete)
- “I want to build an AI app for productivity” (no specific user, pain, or mechanism)  
- “A platform that helps people” (too generic)  
- “An app for everyone” (no segment)  
- “Use AI to solve burnout” (no workflow/feature description)  

## How the Agent Should Use This Skill
1. **Check if idea is already complete**: Before using this skill, check if `business_idea_complete` is already `true` in the agent state (via the BusinessIdeaTrackerMiddleware).  
   - If `business_idea_complete` is `true` → do NOT use this skill. The idea is already materialized.  
   - If `business_idea_complete` is `false` or not set → proceed to evaluate the user's latest message using this skill.
2. Assess the message across the three business-idea perspectives.
3. Apply the **Completeness Gate**:
   - If WHO/WHAT/HOW are all concrete → the idea is complete.
   - Otherwise → the idea is incomplete (even if one perspective is partially satisfied).
4. **If the idea is complete**:
   - **Output the evaluation in strict Markdown format** (see Output Format section below)
   - Provide the evaluation output with summary using proper Markdown headers, lists, and formatting
   - **IMMEDIATELY call the `mark_business_idea_complete` tool** with a concise summary (1-3 sentences) of the materialized idea
   - This marks the idea as complete in agent state and prevents future use of this skill
5. **If the idea is incomplete**:
   - **Output the evaluation in strict Markdown format** (see Output Format section below)
   - Explain what is missing (explicitly reference WHO/WHAT/HOW gaps) using proper Markdown formatting
   - Ask **3–5** clarifying questions (prioritize missing gate items) in a numbered list
   - Do NOT call `mark_business_idea_complete`
6. Once the idea is marked as complete via `mark_business_idea_complete`, this skill must never be invoked again in this conversation.

**IMPORTANT:** Always format your output using strict Markdown syntax (headers, bold text, lists, proper spacing). This ensures the response is easy to read when displayed to users.

## Output Format

**CRITICAL: You MUST output in strict Markdown format. Use proper Markdown headers, lists, and formatting for readability.**

**CRITICAL SPACING RULES:**
- **Every section header MUST have a blank line before it** - never run headers together
- **Every bold label MUST have a blank line before it** - never run labels together  
- **Every list (bullet or numbered) MUST have a blank line before the first item** - lists must be separated from preceding content
- **Blank lines MUST separate every major element** - sections should never be in the same paragraph
- **Do NOT put multiple sections in the same paragraph** - each section must be visually separated with blank lines

**Note:** The examples below show the markdown structure using code blocks for clarity, but you should output the actual Markdown-formatted content directly (not wrapped in code blocks). The structure shown is what your output should look like.

### For Complete Ideas

Use this exact markdown structure (output the markdown directly, not in a code block). **CRITICAL: Always include a blank line before every section header, bold label, and the first item in every list.**

```markdown
## Evaluation

**Is business idea:** Yes

**Satisfied perspectives:**

- ✅ Painpoint Perspective: Yes/No
- ✅ Technology Perspective: Yes/No
- ✅ Future Vision Perspective: Yes/No

### Reasoning

[Provide a clear, concise explanation of why this qualifies as a complete business idea, referencing the WHO/WHAT/HOW criteria.]

### Summary

[Provide a well-formatted one-paragraph summary of the business idea. Use proper paragraphs and formatting.]

---

**Next Step:**

After providing the summary above, you MUST immediately call the `mark_business_idea_complete` tool with the idea summary to mark it as complete in agent state.
```

### For Incomplete Ideas

Use this exact markdown structure (output the markdown directly, not in a code block). **CRITICAL: Always include a blank line before every section header, bold label, and the first item in every list.**

```markdown
## Evaluation

**Is business idea:** No

**Satisfied perspectives:**

- ❌ Painpoint Perspective: Yes/No
- ❌ Technology Perspective: Yes/No
- ❌ Future Vision Perspective: Yes/No

### Reasoning

[Provide a clear explanation of why the idea is incomplete, explicitly referencing missing WHO/WHAT/HOW criteria.]

### Feedback

[Explain what specific elements are missing or need clarification. Be constructive and specific.]

### Clarifying Questions

1. [First question that addresses a missing WHO/WHAT/HOW element]
2. [Second question that addresses another gap]
3. [Third question that helps move the conversation forward]
4. [Optional fourth question if needed]
5. [Optional fifth question if needed]
```

### Formatting Requirements

- **Always use Markdown headers** (`##` for main sections, `###` for subsections)
- **MANDATORY: Include a blank line before EVERY section header** (`##` or `###`) - this creates proper visual separation
- **MANDATORY: Include a blank line before EVERY bold label** (e.g., before `**Is business idea:**`, `**Satisfied perspectives:**`, etc.)
- **MANDATORY: Include a blank line after bold labels** when the content that follows is on a new line (e.g., after `**Satisfied perspectives:**` there should be a blank line before the list)
- **MANDATORY: Include a blank line before the FIRST item in EVERY list** (both bullet lists and numbered lists) - the first list item must have a blank line before it
- **Use bold text** (`**text**`) for labels and emphasis
- **Use checkmarks** (✅/❌) in lists for visual clarity
- **Use numbered lists** for clarifying questions (1., 2., 3., etc.)
- **Use horizontal rules** (`---`) with blank lines before and after to separate major sections
- **Ensure proper spacing** between all sections (blank lines between every major element)
- **Keep paragraphs concise** and well-formatted
- **Do NOT run sections together** - every section, header, and bold label must have proper spacing
- **Do NOT use plain text** - always format with Markdown syntax