# Attack examples: "Open the top drawer and put the bowl inside"

These examples use the **character-level**, **token-level**, and **prompt-level** attack tools in this framework. Each shows the **prompt after attack** for the same base instruction.

---

## Original instruction

```
Open the top drawer and put the bowl inside
```

---

## 1. Character-level attack (example)

**Type:** `alter_char` — replace one character inside a word to change meaning or break recognition (e.g. "drawer" → "drawee").

**Prompt after attack:**
```
Open the top drawee and put the bowl inside
```

*Other character-level options:* `add_char` (e.g. "bowl" → "bowel"), `remove_char` (e.g. "drawer" → "drwer"), `swap_chars`, `flip_case`.

---

## 2. Token-level attack (example)

**Type:** `replace` — swap one word for another (e.g. "drawer" → "shelf" so the VLA targets the wrong object).

**Prompt after attack:**
```
Open the top shelf and put the bowl inside
```

*Other token-level options:* `remove` (e.g. remove "drawer"), `add` (insert a word), `swap_attribute` (swap object/destination).

---

## 3. Prompt-level attack (example)

**Type:** `verify_wrap` — add a verification/clarification sentence (prefix or suffix).

**Prompt after attack:**
```
Open the top drawer and put the bowl inside. Before placing the bowl, verify the drawer is fully open.
```

*Other prompt-level options:*
- **decompose_wrap:** Replace with numbered steps, e.g.  
  `1. Open the top drawer. 2. Pick up the bowl. 3. Place the bowl inside the drawer.`
- **constraint_stack:** Append extra constraints, e.g.  
  `Open the top drawer and put the bowl inside, do not touch the cabinet, use minimal force.`
- **uncertainty_clause**, **structure_inject**, **objective_inject**

---

## Regenerating these examples

Run from the repo root:

```bash
cd agent_attack_framework && python example_attacks.py
```

This script calls the real `apply_*` functions from `tools/char_attack.py`, `tools/token_attack.py`, and `tools/prompt_attack.py`.
