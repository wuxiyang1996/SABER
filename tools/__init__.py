# --- Token-level tools (word-level perturbations) ---
from .token_attack import (
    # Phase 1: FIND — prompt agent to identify targets
    find_replace_targets,
    find_remove_targets,
    find_add_targets,
    find_swap_targets,
    attack_pipeline,
    FIND_REGISTRY,
    # Phase 2: APPLY — execute the manipulation
    apply_replace,
    apply_remove,
    apply_add,
    apply_swap,
    apply_attack,
    ATTACK_REGISTRY,
    # Backward-compatible aliases
    replace_token,
    remove_token,
    add_token,
    swap_attribute,
    # Tool schemas for agentic use
    TOKEN_ATTACK_TOOL_SCHEMAS,
)

# --- Character-level tools (within-word perturbations) ---
from .char_attack import (
    # Phase 1: FIND
    find_add_char_targets,
    find_remove_char_targets,
    find_alter_char_targets,
    find_swap_chars_targets,
    find_flip_case_targets,
    find_multi_char_targets,
    char_attack_pipeline,
    CHAR_FIND_REGISTRY,
    # Phase 2: APPLY
    apply_add_char,
    apply_remove_char,
    apply_alter_char,
    apply_swap_chars,
    apply_flip_case,
    apply_multi_char_edit,
    apply_char_attack,
    CHAR_ATTACK_REGISTRY,
    # Tool schemas
    CHAR_ATTACK_TOOL_SCHEMAS,
)

# --- Prompt-level tools (multi-token sentence/clause perturbations) ---
from .prompt_attack import (
    # Phase 1: FIND
    find_verify_wrap_targets,
    find_decompose_wrap_targets,
    find_uncertainty_clause_targets,
    find_constraint_stack_targets,
    find_structure_inject_targets,
    find_objective_inject_targets,
    prompt_attack_pipeline,
    PROMPT_FIND_REGISTRY,
    # Phase 2: APPLY
    apply_verify_wrap,
    apply_decompose_wrap,
    apply_uncertainty_clause,
    apply_constraint_stack,
    apply_structure_inject,
    apply_objective_inject,
    apply_prompt_attack,
    PROMPT_ATTACK_REGISTRY,
    # Tool schemas
    PROMPT_ATTACK_TOOL_SCHEMAS,
)

# --- Visual-observation tools (single-image perturbations) ---
from .visual_attack import (
    # Phase 1: FIND
    find_patch_roi_targets,
    find_sparse_pixel_targets,
    find_color_shift_targets,
    find_spatial_transform_targets,
    find_sensor_corrupt_targets,
    find_score_optimize_targets,
    visual_attack_pipeline,
    VISUAL_FIND_REGISTRY,
    # Phase 2: APPLY
    apply_patch_roi,
    apply_sparse_pixel,
    apply_color_shift,
    apply_spatial_transform,
    apply_sensor_corrupt,
    apply_score_optimize,
    apply_visual_attack,
    VISUAL_ATTACK_REGISTRY,
    # Tool schemas
    VISUAL_ATTACK_TOOL_SCHEMAS,
)
