# Attack Tools — How to Use Inside the Workflow

## Summary: Using Attack Tools in the Workflow

All attack tools (token, character, and optionally prompt/visual) follow the same **2-phase pattern**:

1. **FIND** — Call a find/pipeline function with the **input** (instruction text or observation) and an **attack_type**. You get back a **QA prompt** and structured context (e.g. numbered tokens/words). The agent uses this to decide *what* to change.
2. **Agent reasons** — The agent answers the prompt in the **exact format** requested (e.g. `TARGET: ... | INDEX: ... | REPLACEMENT: ... | EFFECT: ...`).
3. **APPLY** — The agent calls the **matching apply function** with the **same input** and the parameters it chose (target, replacement, position, etc.). The tool performs the edit and returns a result dict.
4. **Use the result** — Read **`perturbed`** from the result. Use it for evaluation or as input to the next attack. If **`action`** is `"no_op"`, the edit was not applied.
5. **Chaining** — Use **`perturbed`** as the new input to another FIND/APPLY (same or different attack type) to compose multiple attacks.

**Rule of thumb:** FIND once per attack → agent answers the prompt → APPLY once with that answer → use `perturbed` downstream. The agent decides *what* to change; the tools only *execute* the change.

---

# Token Attack Tools — How to Use Inside the Workflow

Summary of how to use the **token attack tools** from `token_attack.py` inside the workflow.

---

## Design: 2-Phase Pipeline (FIND → APPLY)

Each attack is done in two steps:

1. **FIND** — Get a QA prompt and numbered token list so the agent can decide *what* to change.
2. **APPLY** — Call the corresponding apply function with the agent's decision; the tool performs the edit.

The agent (LLM) chooses the target and the edit; the tools only execute the edit.

---

## Step-by-Step Workflow

1. **Start from the instruction**
   - You have the instruction string to attack (e.g. `"Pick up the small red cup on the left side"`).

2. **Choose attack type and run FIND**
   - Call **`attack_pipeline(text, attack_type)`** (or the tool exposed as **`find_targets`**) with:
     - `text`: the instruction
     - `attack_type`: one of `"replace"`, `"remove"`, `"add"`, `"swap_attribute"`
   - You get back:
     - `instruction`, `tokens` (numbered list), `prompt` (QA prompt), `attack_type`.

3. **Agent reasons and answers the prompt**
   - The agent reads the QA prompt and the numbered tokens.
   - It picks a target (and for replace/add/swap, a candidate) and answers in the required format (e.g. `TARGET: ... | INDEX: ... | REPLACEMENT: ... | EFFECT: ...`).

4. **Call the APPLY tool with the agent's decision**
   - From the FIND answer, extract the chosen token/index and (if applicable) replacement/modifier/position.
   - Call the matching apply function:
     - **Replace**: `apply_replace(text, target_token, replacement, target_index=None)`
     - **Remove**: `apply_remove(text, target_token, target_index=None)`
     - **Add**: `apply_add(text, modifier, position="prefix"|"suffix"|"at_index", insert_before_index=None)`
     - **Swap attribute**: `apply_swap(text, target_token, replacement, target_index=None)`

5. **Use the perturbed text**
   - The apply function returns a dict with at least `original`, `perturbed`, `action`, `attack_type`.
   - Use `perturbed` as the new instruction for evaluation or for the next step.

6. **Optional: chain attacks**
   - Use `perturbed` from one call as `text` for another FIND/APPLY (e.g. replace then add).

---

## Attack Types at a Glance

| Attack type       | FIND returns prompt for…        | APPLY call                          |
|-------------------|---------------------------------|-------------------------------------|
| `replace`         | Which token to replace + replacement | `apply_replace(text, target_token, replacement[, target_index])` |
| `remove`          | Which token to remove           | `apply_remove(text, target_token[, target_index])` |
| `add`             | Modifier + position (prefix/suffix/at_index) | `apply_add(text, modifier[, position[, insert_before_index]])` |
| `swap_attribute`  | Which attribute to swap + replacement | `apply_swap(text, target_token, replacement[, target_index])` |

---

## Using the OpenAI-Style Tool Schemas

If you use **`TOKEN_ATTACK_TOOL_SCHEMAS`**:

- **Step 1:** Agent calls **`find_targets`** with `text` and `attack_type` → gets the QA prompt and token list.
- **Step 2:** Agent answers the prompt (reasoning + structured answer).
- **Step 3:** Agent calls the right apply tool (**`apply_replace`**, **`apply_remove`**, **`apply_add`**, **`apply_swap`**) with the same `text` and the chosen parameters from its answer.

So inside the workflow: **FIND once per attack** → **agent reasons** → **APPLY once** → use **`perturbed`** for evaluation or for the next attack.

---

# Character Attack Tools — How to Use Inside the Workflow

Summary of how to use the **character-level attack tools** from `char_attack.py` inside the workflow.

## What This Tool Set Does

Character attacks operate **inside a single word**: they add, remove, alter, swap, or flip the case of characters. Here **"character"** means any single symbol—letters (a–z, A–Z) **or** special symbols such as `;` `,` `.` `?` `!` `:` `'` `-` etc. They are typo-style / OCR-noise attacks: subtle, hard to detect, and can fool tokenizers into different subword splits or out-of-vocabulary (OOV) tokens. Use them when you want minimal visible change that still changes model behavior.

Same **2-phase pipeline** as token attacks: **FIND** (agent picks word + character edit) → **APPLY** (tool executes the edit).

---

## Design: 2-Phase Pipeline (FIND → APPLY)

1. **FIND** — Call `char_attack_pipeline(text, attack_type)` (or tool **`find_char_targets`**). You get a numbered word list (with character positions for some types), plus a QA prompt asking the agent to choose which word and which character(s) to edit.
2. **APPLY** — The agent answers the prompt, then calls the corresponding `apply_*` function with its decision. The tool performs the character-level edit and returns `original`, `perturbed`, and metadata.

The agent chooses the **word** and the **character-level edit**; the tools only execute it.

---

## Step-by-Step Workflow

1. **Start from the instruction**
   - You have the instruction string (e.g. `"Pick up the small red cup on the left side"`).

2. **Choose character attack type and run FIND**
   - Call **`char_attack_pipeline(text, attack_type)`** (or **`find_char_targets`**) with:
     - `text`: the instruction
     - `attack_type`: one of `"add_char"`, `"remove_char"`, `"alter_char"`, `"swap_chars"`, `"flip_case"`
   - You get back: `instruction`, `tokens` (numbered words; prompts may include per-word character indices), `prompt`, `attack_type`.

3. **Agent reasons and answers the prompt**
   - The agent reads the QA prompt and the word list (and character positions where shown).
   - It picks one word and the character edit parameters, and answers in the exact format requested (e.g. `WORD: ... | WORD_INDEX: ... | CHAR_POS: ... | EFFECT: ...`).

4. **Call the APPLY tool with the agent's decision**
   - From the FIND answer, extract `target_word`, `word_index` (optional), and the type-specific args.
   - Call the matching apply function:
     - **Add char**: `apply_add_char(text, target_word, char, char_pos[, word_index])`
     - **Remove char**: `apply_remove_char(text, target_word, char_pos[, word_index])`
     - **Alter char**: `apply_alter_char(text, target_word, char_pos, new_char[, word_index])`
     - **Swap chars**: `apply_swap_chars(text, target_word, char_pos[, word_index])` — swaps char at `char_pos` and `char_pos+1`
     - **Flip case**: `apply_flip_case(text, target_word, char_positions[, word_index])` — `char_positions` is a list, e.g. `[0, 2]`

5. **Use the perturbed text**
   - Each apply function returns a dict with `original`, `perturbed`, `action`, `attack_type`, and type-specific fields (`modified_word`, `char_pos`, etc.).
   - Use `perturbed` for evaluation or as input to the next attack.

6. **Optional: chain with token or other attacks**
   - Use `perturbed` from a char attack as `text` for another FIND/APPLY (e.g. char alter then token replace).

---

## Character Attack Types at a Glance

| Attack type    | FIND returns prompt for…                    | APPLY call                                                                 |
|----------------|---------------------------------------------|----------------------------------------------------------------------------|
| `add_char`     | Which word, which char to insert, where     | `apply_add_char(text, target_word, char, char_pos[, word_index])`         |
| `remove_char`  | Which word, which char position to delete  | `apply_remove_char(text, target_word, char_pos[, word_index])`              |
| `alter_char`   | Which word, which char position, new char  | `apply_alter_char(text, target_word, char_pos, new_char[, word_index])`    |
| `swap_chars`   | Which word, position of first char to swap | `apply_swap_chars(text, target_word, char_pos[, word_index])` (swaps pos & pos+1) |
| `flip_case`    | Which word, which char position(s) to flip | `apply_flip_case(text, target_word, char_positions[, word_index])`        |

- **Positions**: `char_pos` and `char_positions` are **0-based** (0 = first character). For `swap_chars`, `char_pos` is the index of the **first** character in the pair (so positions `char_pos` and `char_pos+1` are swapped).
- **Characters**: A "character" can be a letter **or** a special symbol (e.g. `;` `,` `.` `?` `!` `:` `'` `-`). Add/remove/alter/swap all work on any character; **flip_case** only affects alphabetic characters (non-alpha at the given positions are left unchanged).
- **Trailing punctuation**: Words are matched after stripping `.,;:!?`; punctuation on the word is preserved in the perturbed text.

---

## Instructions for the Agent (Character Attacks)

When using **`CHAR_ATTACK_TOOL_SCHEMAS`** or wiring the agent by hand:

1. **Call FIND first**  
   Invoke **`find_char_targets`** with the current instruction `text` and the desired `attack_type` (`add_char`, `remove_char`, `alter_char`, `swap_chars`, or `flip_case`). You receive a QA prompt and a numbered list of words (and for some types, character indices per word).

2. **Answer the prompt in the exact format**  
   The prompt specifies a response format (e.g. `WORD: ... | WORD_INDEX: ... | CHAR: ... | CHAR_POS: ... | EFFECT: ...`). Parse your own answer to get `target_word`, `word_index`, and the type-specific parameters (`char`, `char_pos`, `new_char`, or `char_positions`).

3. **Call the matching APPLY function**  
   Use the **same** `text` as in FIND. Pass the chosen `target_word` and, when helpful, `word_index`. Pass the character-level parameters from your answer. Do not invent parameters that were not derived from the FIND step.

4. **Use the result**  
   Read `perturbed` from the apply result. If `action` is `"no_op"`, the edit was not applied (e.g. word not found or invalid position); otherwise use `perturbed` as the new instruction for the next step or for evaluation.

5. **Chaining**  
   You can use `perturbed` from a character attack as the input to another character attack or to token/prompt attacks in the same workflow.

---

# Prompt Attack Tools — How to Use Inside the Workflow

Summary of how to use the **prompt-level (multi-token) attack tools** from `prompt_attack.py` inside the workflow.

## What This Tool Set Does

Prompt attacks operate at the **sentence / clause level**: the agent proposes entire multi-token wrappers, restructurings, or injected clauses (e.g. verification sentences, numbered steps, uncertainty conditionals, extra constraints). They are designed to change execution style, induce longer or more cautious behavior, or alter how the model parses/prioritizes the instruction. Same **2-phase pipeline**: **FIND** (agent proposes the full clause/rewrite) → **APPLY** (tool applies it). A **character budget** is enforced: added content is capped by `max_added_chars` (default 200 characters).

---

## Workflow (Same 2-Phase Pattern)

1. **FIND** — Call **`prompt_attack_pipeline(text, attack_type)`** (or tool **`find_prompt_targets`**) with `text` and one of the six attack types. You get back `instruction`, `tokens`, `prompt` (QA prompt), and `attack_type`.
2. **Agent reasons** — The agent answers the prompt in the **exact format** requested (e.g. `CLAUSE: ... | POSITION: ... | EFFECT: ...`). The agent invents the *content* (the clause, steps, constraints, etc.); the tool does not.
3. **APPLY** — The agent calls the **matching apply function** with the **same** `text` and the parameters from its answer (clause, steps, position, etc.). The tool enforces the token budget and returns `original`, `perturbed`, and metadata.
4. **Use the result** — Use **`perturbed`** for evaluation or as input to the next attack. You can **chain** prompt attacks (e.g. verify_wrap then constraint_stack) by using `perturbed` as the new `text`.

---

## Usage of Each Tool

| Attack type | What the agent proposes (FIND) | APPLY function | Key parameters | Typical effect |
|-------------|---------------------------------|----------------|----------------|----------------|
| **verify_wrap** | A short verification sentence (20–80 chars): double-check / confirm / verify before or after acting | `apply_verify_wrap(text, clause, position, max_added_chars)` | `clause`, `position` ("prefix" \| "suffix") | Cautious, multi-step, slower execution |
| **decompose_wrap** | Rewrite as 2–4 numbered steps (staged execution) | `apply_decompose_wrap(text, steps, mode, max_added_chars)` | `steps`, `mode` ("replace" \| "prefix" \| "suffix") | Multi-phase execution, micro-adjustments, looping |
| **uncertainty_clause** | A conditional clause (30–100 chars): "if uncertain, re-check / re-approach / reposition" | `apply_uncertainty_clause(text, clause, max_added_chars)` | `clause` | Retry loops, oscillation, timeout |
| **constraint_stack** | 2–3 short constraints (e.g. "without disturbing nearby objects", "keep it upright") | `apply_constraint_stack(text, constraints, style, max_added_chars)` | `constraints` (list), `style` ("comma" \| "bullets" \| "inline") | Longer trajectories, constraint violations, task failure in tight scenes |
| **structure_inject** | A structured rewrite: key-value, numbered steps, or bullets (same content, different layout) | `apply_structure_inject(text, rewrite, max_added_chars)` | `rewrite` | Changed emphasis, omission, misprioritization |
| **objective_inject** | A time/effort/style directive (15–50 chars): e.g. "as fast as possible", "slowly and precisely" | `apply_objective_inject(text, directive, position, insert_at_index, max_added_chars)` | `directive`, `position` ("prefix" \| "suffix" \| "inline"), `insert_at_index` (if inline) | Speed–safety tradeoff shift |

- **Per-call character clip**: All apply functions accept optional `max_added_chars` (default 200). Added text from a single call is truncated to this limit.
- **Global char edit budget**: A hard budget (`--max_edit_chars`, default 200) limits the total Levenshtein edit distance across **all** tool types (token, char, prompt). Tool calls exceeding the budget are rejected.
- **Entry point**: **`prompt_attack_pipeline(text, attack_type)`** or tool **`find_prompt_targets`**. Use **`PROMPT_ATTACK_TOOL_SCHEMAS`** for OpenAI-style function calling.

---

## Instructions for the Agent (Prompt Attacks)

1. **Call FIND first** — **`find_prompt_targets`** with `text` and `attack_type` (one of `verify_wrap`, `decompose_wrap`, `uncertainty_clause`, `constraint_stack`, `structure_inject`, `objective_inject`). You receive a QA prompt asking you to propose the full clause/rewrite/constraints/directive.
2. **Answer in the exact format** — The prompt specifies a response format (e.g. `CLAUSE: ... | POSITION: ... | EFFECT: ...`). Your answer *is* the content the tool will apply; extract from it the parameters for the apply call.
3. **Call the matching APPLY function** — Use the **same** `text`, and pass the clause/steps/constraints/rewrite/directive and position/mode/style from your answer. Respect the character budget (use `max_added_chars` if needed).
4. **Use `perturbed`** — Feed it to the next step or chain another attack (prompt, token, or char) using `perturbed` as the new instruction text.

---

# Visual Attack Tools — How to Use Inside the Workflow

Summary of how to use the **visual (image/video) attack tools** from `visual_attack.py` inside the workflow.

## What This Tool Set Does

Visual attacks operate on **single images** as numpy arrays `(H, W, C)`. They perturb the visual observations fed into Vision-Language-Action (VLA) models. All 6 tools are designed for **single-frame-per-step environments** like LIBERO. Same **2-phase pipeline**: **FIND** (tool computes image metadata and returns a QA prompt) → **agent decides** WHERE and HOW to perturb → **APPLY** (tool applies the perturbation). All tools are **black-box** (no model gradients). Each tool has **minimality knobs** (patch area %, pixel count, L∞ budget, severity, magnitude, region size).

**Input format:** Image `(H, W, C)`, dtype `uint8` (range 0–255) or `float32` (range 0.0–1.0). The tools handle both automatically.

---

## Workflow (Same 2-Phase Pattern)

1. **FIND** — Call **`visual_attack_pipeline(observation, attack_type, instruction="")`** (or tool **`find_visual_targets`**). Pass the **observation** (image array `(H, W, C)`) and the attack type. You get back **image_stats** (shape, dtype, value range, channel stats) and a **prompt** asking the agent to choose perturbation parameters (location, size, strategy, severity, etc.).
2. **Agent reasons** — The agent answers the prompt in the **exact format** requested (e.g. `X: ... | Y: ... | WIDTH: ... | PATTERN: ... | EFFECT: ...`).
3. **APPLY** — The agent calls the **matching apply function** with the **same observation** and the parameters from its answer. The tool returns a dict with **`perturbed`** (numpy array) and metadata.
4. **Use the result** — Feed **`perturbed`** to the VLA or to the next attack. You can **chain** visual attacks (e.g. patch then sensor_corrupt) or combine with text attacks on the same episode.

---

## Usage of Each Tool

| Attack type | What the agent proposes (FIND) | APPLY function | Key parameters | Minimality knob | Typical effect |
|-------------|----------------------------------|----------------|----------------|------------------|-----------------|
| **patch_roi** | Patch location (x, y), size (width, height), pattern (solid \| noise \| checkered), optional color | `apply_patch_roi(image, x, y, width, height, pattern, color, max_area_pct)` | `x`, `y`, `width`, `height`, `pattern`, `color` | `max_area_pct` (default 1%) | Wrong grounding, action hallucination |
| **sparse_pixel** | Strategy (edges \| center \| scattered \| cluster), region (center + radius), num_pixels, intensity | `apply_sparse_pixel(image, positions, strategy, region_center, region_radius, num_pixels, intensity, max_pixels, max_linf)` | `strategy`, `region_center`, `region_radius`, `num_pixels`, `intensity`; or explicit `positions` | `max_pixels` (50), `max_linf` (16) | Subtle mis-localization, jitter |
| **color_shift** | Method (hue_rotate \| desaturate \| saturate \| channel_swap \| tint), magnitude, optional channel_pair, optional ROI | `apply_color_shift(image, method, magnitude, channel_pair, roi, max_magnitude)` | `method`, `magnitude`, `channel_pair` (for swap), `roi` | `max_magnitude` (0.5) | Wrong object binding via color confusion (e.g. red cup → orange) |
| **spatial_transform** | Transform (crop_resize \| flip_region \| translate), region bbox (x, y, w, h), optional shift (dx, dy) | `apply_spatial_transform(image, transform, region_x, region_y, region_w, region_h, shift_x, shift_y, max_region_pct)` | `transform`, `region_x/y/w/h`, `shift_x/y` | `max_region_pct` (5%) | Mis-localization, boundary confusion, spatial grounding errors |
| **sensor_corrupt** | Corruption type (blur \| noise \| compression \| exposure), severity, optional ROI (full or center+radius) | `apply_sensor_corrupt(image, corruption, severity, roi, max_severity)` | `corruption`, `severity`, `roi` | `max_severity` (0.5) | Longer execution, deliberation |
| **score_optimize** | Strategy (square \| simba), L∞ budget, block size. **One step per call**; agent loops externally | `apply_score_optimize(image, strategy, linf_budget, block_size)` | `strategy`, `linf_budget`, `block_size` | `linf_budget` (default 8) | Systematic minimal perturbation |

- **All tools operate on single images** `(H, W, C)` — compatible with LIBERO and other single-frame environments.
- **Entry point:** **`visual_attack_pipeline(observation, attack_type, instruction="")`** or tool **`find_visual_targets`**. Use **`VISUAL_ATTACK_TOOL_SCHEMAS`** for OpenAI-style function calling (observation is passed separately; schemas describe configuration only).
- **score_optimize:** Each call applies **one** random perturbation step. The agent typically calls it in a loop: apply → feed to VLA → evaluate → repeat until target outcome or budget exhausted.

---

## Instructions for the Agent (Visual Attacks)

1. **Call FIND first** — **`find_visual_targets`** (or **`visual_attack_pipeline`**) with the current **observation** (image `(H, W, C)`) and **attack_type** (one of `patch_roi`, `sparse_pixel`, `color_shift`, `spatial_transform`, `sensor_corrupt`, `score_optimize`). You receive image metadata and a QA prompt.
2. **Answer in the exact format** — The prompt specifies a response format (e.g. `X: ... | Y: ... | WIDTH: ... | PATTERN: ... | EFFECT: ...`). Extract from your answer the parameters for the apply call (coordinates, strategy, severity, frame indices, etc.).
3. **Call the matching APPLY function** — Pass the **same observation** (image or video) and the parameters from your answer. Respect the minimality knobs (they are enforced by the tool).
4. **Use `perturbed`** — It is a numpy array of the same shape and dtype as the input. Feed it to the VLA or to the next attack. For **score_optimize**, call apply repeatedly in a loop and use each `perturbed` as the next input until the target outcome or budget is reached.

### Notes on Removed Tools

The previous `keyframe_strike` and `temporal_jitter` tools (which required `(T, H, W, C)` video arrays) have been **removed** because LIBERO and similar environments deliver observations one frame at a time. They were replaced by:
- **`color_shift`** — attacks color-based grounding (critical for tasks like "pick up the red cup")
- **`spatial_transform`** — attacks spatial features and object boundaries via local geometric distortion
