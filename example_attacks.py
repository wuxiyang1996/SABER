#!/usr/bin/env python3
"""Generate example attacked prompts for: Open the top drawer and put the bowl inside.
Uses character-level, token-level, and prompt-level attacks from the framework.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.char_attack import apply_alter_char, apply_add_char, apply_remove_char
from tools.token_attack import apply_replace, apply_remove
from tools.prompt_attack import apply_verify_wrap, apply_decompose_wrap, apply_constraint_stack

INSTRUCTION = "Open the top drawer and put the bowl inside"

def main():
    print("=" * 70)
    print("Original instruction:")
    print(f"  \"{INSTRUCTION}\"")
    print("=" * 70)

    # --- Character-level example: alter "drawer" -> "drawee" (1-char change, different word)
    r = apply_alter_char(INSTRUCTION, target_word="drawer", char_pos=5, new_char="e")
    print("\n### 1. CHARACTER-LEVEL ATTACK (alter_char)")
    print("  Attack: replace char at position 5 in 'drawer' with 'e' → 'drawee'")
    print(f"  Prompt after attack: \"{r['perturbed']}\"")

    # --- Token-level example: replace "drawer" with "shelf"
    r = apply_replace(INSTRUCTION, target_token="drawer", replacement="shelf")
    print("\n### 2. TOKEN-LEVEL ATTACK (replace)")
    print("  Attack: replace token 'drawer' with 'shelf'")
    print(f"  Prompt after attack: \"{r['perturbed']}\"")

    # --- Prompt-level example: verify_wrap (add verification clause)
    r = apply_verify_wrap(
        INSTRUCTION,
        clause="Before placing the bowl, verify the drawer is fully open.",
        position="suffix",
    )
    print("\n### 3. PROMPT-LEVEL ATTACK (verify_wrap)")
    print("  Attack: append verification clause (suffix)")
    print(f"  Prompt after attack: \"{r['perturbed']}\"")

    # --- Optional: one more prompt-level (decompose_wrap)
    steps = "1. Open the top drawer. 2. Pick up the bowl. 3. Place the bowl inside the drawer."
    r = apply_decompose_wrap(INSTRUCTION, steps=steps, mode="replace")
    print("\n### 4. PROMPT-LEVEL ATTACK (decompose_wrap)")
    print("  Attack: replace instruction with numbered steps")
    print(f"  Prompt after attack: \"{r['perturbed']}\"")

    # --- Optional: constraint_stack
    r = apply_constraint_stack(
        INSTRUCTION,
        constraints=["do not touch the cabinet", "use minimal force"],
        style="comma",
    )
    print("\n### 5. PROMPT-LEVEL ATTACK (constraint_stack)")
    print("  Attack: append extra constraints (comma-separated)")
    print(f"  Prompt after attack: \"{r['perturbed']}\"")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
