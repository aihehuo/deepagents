# Test Skill Unlock Logic Issues

This document identifies issues with the skill unlock and dependency logic in the test files based on the new sequential skill progression.

## Sequential Skill Unlock Progression

1. **business-idea-evaluation** → UNLOCKED when: `business_idea_complete == False`
   - When complete → call `mark_business_idea_complete` → unlocks persona-clarification

2. **persona-clarification** → UNLOCKED when: `business_idea_complete == True`
   - When complete → call `mark_persona_clarified` → unlocks painpoint-enhancement

3. **painpoint-enhancement** → UNLOCKED when: `persona_clarified == True`
   - When complete → call `mark_painpoint_enhanced` → unlocks 60s-pitch-creation

4. **60s-pitch-creation** → UNLOCKED when: `persona_clarified == True AND painpoint_enhanced == True`
   - When complete → call `mark_pitch_created` → unlocks baseline-pricing-and-optimization

5. **baseline-pricing-and-optimization** → UNLOCKED when: `pitch_created == True`
   - When complete → call `mark_pricing_optimized` → unlocks business-model-pivot-exploration

6. **business-model-pivot-exploration** → UNLOCKED when: `pricing_optimized == True`

---

## Issues Found

### 1. `test_persona_clarification_skill.py`

**Issue**: Does not use `BusinessIdeaTrackerMiddleware` at all.

**Problems**:
- Cannot test unlock logic
- Cannot verify that persona-clarification is locked when `business_idea_complete == False`
- Cannot verify that persona-clarification unlocks after business idea is complete

**Required Fix**:
- Add `BusinessIdeaTrackerMiddleware()` to middleware list
- Add `checkpointer=MemorySaver()` for state persistence
- Update test to:
  1. First mark business idea as complete using `mark_business_idea_complete`
  2. Then test persona-clarification skill (should be unlocked)
  3. Verify `persona_clarified` is set after skill completes
  4. Call `mark_persona_clarified` after persona clarification

---

### 2. `test_painpoint_enhancement_skill.py`

**Issue**: Only marks business idea as complete, but painpoint-enhancement requires `persona_clarified == True`.

**Problems**:
- Test jumps directly from business idea complete to painpoint-enhancement
- Missing prerequisite: persona should be clarified first
- Test doesn't call `mark_persona_clarified` before using painpoint-enhancement

**Required Fix**:
- Update test flow to:
  1. Mark business idea as complete (`mark_business_idea_complete`)
  2. Use persona-clarification skill
  3. Call `mark_persona_clarified` after persona clarification
  4. Then use painpoint-enhancement skill (should now be unlocked)
  5. Call `mark_painpoint_enhanced` after pain point enhancement

**Current Test Flow (WRONG)**:
```
Step 1: Identify business idea → mark_business_idea_complete
Step 2: Use painpoint-enhancement (should be LOCKED - persona not clarified!)
```

**Correct Test Flow**:
```
Step 1: Identify business idea → mark_business_idea_complete
Step 2: Clarify persona → mark_persona_clarified
Step 3: Enhance pain point → mark_painpoint_enhanced
```

---

### 3. `test_60s_pitch_creation_skill.py`

**Issue**: Only marks business idea as complete, but 60s-pitch-creation requires `persona_clarified == True AND painpoint_enhanced == True`.

**Problems**:
- Test jumps directly from business idea complete to pitch creation
- Missing prerequisites: persona should be clarified AND painpoint should be enhanced
- Test doesn't call `mark_persona_clarified` or `mark_painpoint_enhanced` before using pitch creation

**Required Fix**:
- Update test flow to:
  1. Mark business idea as complete (`mark_business_idea_complete`)
  2. Use persona-clarification skill → `mark_persona_clarified`
  3. Use painpoint-enhancement skill → `mark_painpoint_enhanced`
  4. Then use 60s-pitch-creation skill (should now be unlocked)
  5. Call `mark_pitch_created` after pitch creation

**Current Test Flow (WRONG)**:
```
Step 1: Identify business idea → mark_business_idea_complete
Step 2: Use 60s-pitch-creation (should be LOCKED - persona/painpoint not done!)
```

**Correct Test Flow**:
```
Step 1: Identify business idea → mark_business_idea_complete
Step 2: Clarify persona → mark_persona_clarified
Step 3: Enhance pain point → mark_painpoint_enhanced
Step 4: Create pitch → mark_pitch_created
```

---

### 4. `test_baseline_pricing_optimization_skill.py`

**Issue**: Only marks business idea as complete, but baseline-pricing-optimization requires `pitch_created == True`.

**Problems**:
- Test jumps directly from business idea complete to pricing optimization
- Missing prerequisites: pitch must be created first
- Test doesn't follow the full sequence

**Required Fix**:
- Update test flow to:
  1. Mark business idea as complete (`mark_business_idea_complete`)
  2. Use persona-clarification skill → `mark_persona_clarified`
  3. Use painpoint-enhancement skill → `mark_painpoint_enhanced`
  4. Use 60s-pitch-creation skill → `mark_pitch_created`
  5. Then use baseline-pricing-optimization skill (should now be unlocked)
  6. Call `mark_pricing_optimized` after pricing optimization

**Current Test Flow (WRONG)**:
```
Step 1: Identify business idea → mark_business_idea_complete
Step 2: Use baseline-pricing-optimization (should be LOCKED - pitch not created!)
```

**Correct Test Flow**:
```
Step 1: Identify business idea → mark_business_idea_complete
Step 2: Clarify persona → mark_persona_clarified
Step 3: Enhance pain point → mark_painpoint_enhanced
Step 4: Create pitch → mark_pitch_created
Step 5: Optimize pricing → mark_pricing_optimized
```

---

### 5. `test_business_model_pivot_exploration_skill.py`

**Issue**: Only marks business idea as complete, but business-model-pivot-exploration requires `pricing_optimized == True`.

**Problems**:
- Test jumps directly from business idea complete to pivot exploration
- Missing prerequisites: all previous steps must be completed
- Test doesn't follow the full sequence

**Required Fix**:
- Update test flow to:
  1. Mark business idea as complete (`mark_business_idea_complete`)
  2. Use persona-clarification skill → `mark_persona_clarified`
  3. Use painpoint-enhancement skill → `mark_painpoint_enhanced`
  4. Use 60s-pitch-creation skill → `mark_pitch_created`
  5. Use baseline-pricing-optimization skill → `mark_pricing_optimized`
  6. Then use business-model-pivot-exploration skill (should now be unlocked)

**Current Test Flow (WRONG)**:
```
Step 1: Identify business idea → mark_business_idea_complete
Step 2: Use business-model-pivot-exploration (should be LOCKED - pricing not optimized!)
```

**Correct Test Flow**:
```
Step 1: Identify business idea → mark_business_idea_complete
Step 2: Clarify persona → mark_persona_clarified
Step 3: Enhance pain point → mark_painpoint_enhanced
Step 4: Create pitch → mark_pitch_created
Step 5: Optimize pricing → mark_pricing_optimized
Step 6: Explore business model pivots
```

---

### 6. `test_business_idea_evaluation_skill.py`

**Status**: ✅ Mostly correct

**Current Behavior**:
- Correctly marks business idea as complete
- Correctly checks that skill is locked after completion

**Possible Enhancement**:
- Could add a test to verify persona-clarification unlocks after business idea is marked complete

---

## Summary of Required Changes

### For Each Test File:

1. **Add missing milestone marking tools** to system prompts:
   - `mark_persona_clarified`
   - `mark_painpoint_enhanced`
   - `mark_pitch_created`
   - `mark_pricing_optimized`

2. **Follow the complete sequence**:
   - Don't skip prerequisite steps
   - Call milestone marking tools after each skill completes
   - Verify state flags are set correctly

3. **Add validation checks**:
   - Verify prerequisite states are `True` before using a skill
   - Verify skill is locked when prerequisites aren't met
   - Verify milestone flags are set after skill completion

4. **Update system prompts** to instruct the agent to:
   - Follow the sequential progression
   - Call milestone marking tools after completing each skill
   - Only use skills when their prerequisites are met

---

## Example: Correct Test Pattern

```python
# Step 1: Identify business idea
result_1 = agent.invoke({"messages": [HumanMessage("I have a business idea...")]}, config)
assert result_1.get("business_idea_complete") == True
# Verify mark_business_idea_complete was called

# Step 2: Clarify persona (now unlocked)
result_2 = agent.invoke({
    "messages": result_1["messages"] + [HumanMessage("Help clarify my target persona")]
}, config)
assert result_2.get("persona_clarified") == True
# Verify mark_persona_clarified was called

# Step 3: Enhance pain point (now unlocked)
result_3 = agent.invoke({
    "messages": result_2["messages"] + [HumanMessage("Enhance my pain point")]
}, config)
assert result_3.get("painpoint_enhanced") == True
# Verify mark_painpoint_enhanced was called

# Step 4: Create pitch (now unlocked)
result_4 = agent.invoke({
    "messages": result_3["messages"] + [HumanMessage("Create a 60-second pitch")]
}, config)
assert result_4.get("pitch_created") == True
# Verify mark_pitch_created was called

# Continue for remaining steps...
```

