---
name: persona-clarification
description: Helps transform a vague or rough business idea into a clear, detailed, and actionable target user persona by inferring missing attributes and asking clarifying questions when needed.
---

# Skill: Persona Clarification

## Purpose
This skill refines the user's rough or vague business idea into a well-defined target user persona.  
It guides the agent to infer demographic and psychographic characteristics, identify realistic user contexts, and ask targeted clarifying questions to fill information gaps.

## When to Use
Invoke this skill when:
- The user provides a business idea without specifying target users.
- The description of the target users is too broad (e.g., "everyone", "young people", "entrepreneurs").
- The business plan requires a clearer understanding of who the product is for.
- The agent needs persona detail for downstream tasks such as business planning, experiment design, or co-founder matching.

## What This Skill Produces
A refined primary persona including:
- Name (fictional, optional)
- Age range
- Background
- Occupation / role
- Income level (optional if irrelevant)
- Location or environment
- Goals and motivations
- Specific pain points relevant to the idea
- Behaviors and habits
- Context of product usage (where / when / how)

Additionally:
- Up to 3–5 clarifying questions if required information is missing.

## How the Agent Should Use This Skill
1. Parse the user's business idea and extract any implied user segments.
2. Infer likely demographic and psychographic details, remaining realistic and grounded.
3. When essential details are missing, generate concise clarifying questions.
4. Output a structured persona that other agent components can use (business plan, task engine, market research, etc.).
5. Avoid generating multiple personas; choose the most practical primary persona unless explicitly requested.

## Output Format
The output should follow this exact structure:

Persona:
- Name:
- Age:
- Background:
- Occupation:
- Income range:
- Location:
- Goals:
- Core pain points:
- Behaviors:
- Environment of product use:

Clarifying Questions (if needed):
1.
2.
3.