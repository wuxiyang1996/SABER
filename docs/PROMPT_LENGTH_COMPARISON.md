# Prompt length: current vs GitHub (origin/main)

## Summary

**We did add a non-trivial amount of text.** The extra content is **unlikely to exceed 8192 tokens by itself**, but it does reduce headroom for tool schemas and multi-turn responses.

---

## 1. Changes in `agent/vla_rollout.py`

| Change | Approx. size |
|--------|---------------|
| **Character-Edit Budget section** (new) | **~733 chars (~183 tokens)** |
| ACTION_INFLATION-specific stealth_rule | ~220 chars (~55 tokens) |
| PROMPT tool description: `max_added_chars` sentence | ~130 chars (~33 tokens) |
| **Total in system prompt (vs GitHub)** | **~1,080 chars (~270 tokens)** |

GitHub’s `_build_vla_system_prompt` had:
- No `max_edit_chars` parameter
- No "## Character-Edit Budget" block
- Only two stealth_rule branches (task_failure vs else), no ACTION_INFLATION

So for **task_failure** we add the budget block + the one PROMPT line (~200 tokens). For **action_inflation** we also add the inflation stealth_rule.

---

## 2. Changes in `rwd_func/rwd.py` (`get_objective_system_prompt`)

- **CONSTRAINT_VIOLATION**: Short description replaced with a long one (four violation channels, ~600 chars).
- **ACTION_INFLATION**: Full branch added with long strategy (prompt + token inflation tactics) — **~1,400 chars (~350 tokens)** when `prompt` in tool sets.
- **CONSTRAINT_VIOLATION**: Long focus_note with prompt/token strategy — **~1,800 chars (~450 tokens)** when prompt in tool sets.

So for **task_failure** we did **not** add much in `rwd.py` (token/prompt strategy was already there). The big additions are for **action_inflation** and **constraint_violation**.

---

## 3. Rough token budget (8192 max)

- **System prompt (current, task_failure, char+token+prompt)**: ~1,300–1,500 tokens  
- **User message**: ~80–120 tokens  
- **Tool schemas** (LangChain/LangGraph): ~500–1,500 tokens (many tools)  
- **Total input before any turn**: ~2,000–3,200 tokens  
- **Remaining for 8 turns** (assistant + tool results): ~5,000–6,000 tokens  

So we are **not** over 8192 from the prompt alone. The risk is:

1. **Tool results** (FIND/APPLY) can be large; `_truncate_result(2000)` limits each result but multiple turns add up.
2. If **training** used a shorter prompt and we later added the budget + longer objective text, the **same 8192** now fits fewer turns of response.

---

## 4. Recommendations

1. **Keep max_seq_length=8192** and **max_model_len=8192** in both train and eval (already done).
2. **Optional trim**: Shorten the "Character-Edit Budget" block to 2–3 bullets (e.g. “Total budget: 200 chars. Per-call max_added_chars: 200. Read edited_chars/budget_remaining after each APPLY.”) to save ~400 chars (~100 tokens).
3. **Monitor**: If you see truncation or “no-attack” on long-horizon episodes, reduce `max_turns` (e.g. 5 instead of 8) or shorten the objective strategy in `get_objective_system_prompt` for the objective you use most.
