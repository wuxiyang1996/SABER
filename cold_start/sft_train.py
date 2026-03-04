"""Cold-start SFT: fine-tune the attack agent on GPT-5 Mini trajectories.

Converts successful attack trajectories (collected by collect.py) into
multi-turn tool-calling conversations and runs supervised fine-tuning
on Qwen2.5-3B-Instruct with LoRA.  The resulting adapter can be loaded
by the GRPO trainer (train_vla.py) as a warm start.

The pipeline:
  1. Load success_trajectories.jsonl from a cold-start collection run.
  2. Convert each trajectory's message_log from LangGraph format to the
     Qwen2.5 chat template with tool-call metadata.
  3. Fine-tune with TRL SFTTrainer + LoRA (matching GRPO config).
  4. Save the LoRA adapter so train_vla.py --resume can pick it up.

Usage
-----
    # From agent_attack_framework/
    python -m cold_start.sft_train \\
        --data_dir cold_start/outputs/cold_start__gpt-5-mini__20260301_083338__task_failure__char-prompt-token \\
        --output_dir outputs/sft_cold_start

    # Then run GRPO training with warm start:
    python train_vla.py --resume --base_model outputs/sft_cold_start
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("cold_start.sft")


# ============================================================================
# Tool schema definitions (extracted from the attack tools)
# ============================================================================

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "find_targets",
            "description": "Analyse the instruction and find token-level targets for replacement, removal, addition, or swapping.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Current instruction text."},
                    "attack_type": {"type": "string", "enum": ["replace", "remove", "add", "swap"],
                                    "description": "Type of token attack to find targets for."},
                },
                "required": ["text", "attack_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_prompt_targets",
            "description": "Analyse the instruction and find prompt-level injection points.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Current instruction text."},
                    "attack_type": {"type": "string",
                                    "enum": ["objective_inject", "constraint_stack", "decompose_wrap",
                                             "verify_wrap", "uncertainty_clause", "structure_inject"],
                                    "description": "Type of prompt attack."},
                },
                "required": ["text", "attack_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_char_targets",
            "description": "Analyse the instruction and find character-level targets for mutation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Current instruction text."},
                    "attack_type": {"type": "string",
                                    "enum": ["alter_char", "add_char", "remove_char", "flip_case"],
                                    "description": "Type of character attack."},
                },
                "required": ["text", "attack_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_replace",
            "description": "Replace a token in the instruction with a different word.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "target_token": {"type": "string"},
                    "replacement": {"type": "string"},
                    "target_index": {"type": "integer"},
                },
                "required": ["text", "target_token", "replacement", "target_index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_remove",
            "description": "Remove a token from the instruction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "target_token": {"type": "string"},
                    "target_index": {"type": "integer"},
                },
                "required": ["text", "target_token", "target_index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_add",
            "description": "Add a modifier token before a target position.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "modifier": {"type": "string"},
                    "position": {"type": "string"},
                    "insert_before_index": {"type": "integer"},
                },
                "required": ["text", "modifier", "position", "insert_before_index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_swap",
            "description": "Swap a token with a nearby synonym/antonym.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "target_token": {"type": "string"},
                    "replacement": {"type": "string"},
                    "target_index": {"type": "integer"},
                },
                "required": ["text", "target_token", "replacement", "target_index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_objective_inject",
            "description": "Inject a conflicting directive into the instruction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "directive": {"type": "string"},
                    "position": {"type": "string", "enum": ["prefix", "suffix", "inline"]},
                    "insert_at_index": {"type": "integer"},
                    "max_added_chars": {"type": "integer"},
                },
                "required": ["text", "directive", "position"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_constraint_stack",
            "description": "Append contradictory constraints to the instruction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "constraints": {"type": "array", "items": {"type": "string"}},
                    "style": {"type": "string"},
                    "max_added_chars": {"type": "integer"},
                },
                "required": ["text", "constraints"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_decompose_wrap",
            "description": "Wrap the instruction in a decomposed multi-step format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "steps": {"type": "array", "items": {"type": "string"}},
                    "mode": {"type": "string"},
                    "max_added_chars": {"type": "integer"},
                },
                "required": ["text", "steps"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_verify_wrap",
            "description": "Add a verification clause to the instruction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "clause": {"type": "string"},
                    "position": {"type": "string"},
                    "max_added_chars": {"type": "integer"},
                },
                "required": ["text", "clause"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_uncertainty_clause",
            "description": "Add hedging / uncertainty language to the instruction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "clause": {"type": "string"},
                    "max_added_chars": {"type": "integer"},
                },
                "required": ["text", "clause"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_structure_inject",
            "description": "Rewrite the instruction with a structurally different phrasing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "rewrite": {"type": "string"},
                    "max_added_chars": {"type": "integer"},
                },
                "required": ["text", "rewrite"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_alter_char",
            "description": "Alter a character in a target word.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "target_word": {"type": "string"},
                    "word_index": {"type": "integer"},
                    "char_pos": {"type": "integer"},
                    "new_char": {"type": "string"},
                },
                "required": ["text", "target_word", "word_index", "char_pos", "new_char"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_add_char",
            "description": "Insert a character into a target word.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "target_word": {"type": "string"},
                    "word_index": {"type": "integer"},
                    "char_pos": {"type": "integer"},
                    "char": {"type": "string"},
                },
                "required": ["text", "target_word", "word_index", "char_pos", "char"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_remove_char",
            "description": "Remove a character from a target word.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "target_word": {"type": "string"},
                    "word_index": {"type": "integer"},
                    "char_pos": {"type": "integer"},
                },
                "required": ["text", "target_word", "word_index", "char_pos"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_flip_case",
            "description": "Flip the case of characters in a target word.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "target_word": {"type": "string"},
                    "word_index": {"type": "integer"},
                    "char_positions": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["text", "target_word", "word_index", "char_positions"],
            },
        },
    },
]


# ============================================================================
# Data conversion: LangGraph message_log → OpenAI chat format
# ============================================================================

def _make_tool_call_id(traj_idx: int, call_idx: int) -> str:
    return f"call_{traj_idx}_{call_idx}"


def convert_trajectory_to_chat(
    trajectory: Dict[str, Any],
    traj_idx: int = 0,
    min_assistant_turns: int = 2,
) -> Optional[List[Dict[str, Any]]]:
    """Convert a cold-start trajectory into OpenAI chat-completion format.

    Handles the LangGraph message_log with roles {system, human, ai, tool}
    and converts to {system, user, assistant, tool} with proper tool_call
    IDs and function arguments.

    Returns None if the trajectory doesn't meet quality criteria.
    """
    msg_log = trajectory.get("message_log", [])
    if not msg_log:
        return None

    messages: List[Dict[str, Any]] = []
    call_counter = 0
    pending_tool_call_ids: List[str] = []
    assistant_turns = 0

    for msg in msg_log:
        role = msg.get("role", "")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        if role == "system":
            messages.append({"role": "system", "content": content})

        elif role == "human":
            messages.append({"role": "user", "content": content})

        elif role == "ai":
            assistant_msg: Dict[str, Any] = {"role": "assistant"}

            if tool_calls:
                assistant_msg["content"] = content or None
                formatted_calls = []
                pending_tool_call_ids = []
                for tc in tool_calls:
                    tc_id = _make_tool_call_id(traj_idx, call_counter)
                    call_counter += 1
                    pending_tool_call_ids.append(tc_id)
                    formatted_calls.append({
                        "id": tc_id,
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"], ensure_ascii=False),
                        },
                    })
                assistant_msg["tool_calls"] = formatted_calls
                assistant_turns += 1
            else:
                assistant_msg["content"] = content
                if content:
                    assistant_turns += 1

            messages.append(assistant_msg)

        elif role == "tool":
            tc_id = pending_tool_call_ids.pop(0) if pending_tool_call_ids else _make_tool_call_id(traj_idx, call_counter - 1)
            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": content,
            })

    if assistant_turns < min_assistant_turns:
        return None

    return messages


def load_sft_dataset(
    data_path: str,
    min_reward: float = 0.0,
    min_assistant_turns: int = 2,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load and convert cold-start trajectories into SFT chat examples.

    Returns a list of dicts, each with:
      - "messages": list of chat messages in OpenAI format
      - "tools": list of tool schemas
      - "metadata": trajectory metadata (reward, task, etc.)
    """
    examples = []
    path = Path(data_path)

    if path.is_dir():
        jsonl_path = path / "success_trajectories.jsonl"
        if not jsonl_path.exists():
            jsonl_path = path / "trajectories.jsonl"
    else:
        jsonl_path = path

    if not jsonl_path.exists():
        raise FileNotFoundError(f"No trajectory data found at {jsonl_path}")

    logger.info("Loading trajectories from %s ...", jsonl_path)

    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if max_samples and len(examples) >= max_samples:
                break

            traj = json.loads(line)
            if traj.get("reward", 0) < min_reward:
                continue

            messages = convert_trajectory_to_chat(
                traj, traj_idx=i, min_assistant_turns=min_assistant_turns,
            )
            if messages is None:
                continue

            examples.append({
                "messages": messages,
                "tools": TOOL_SCHEMAS,
                "metadata": {
                    "reward": traj.get("reward", 0),
                    "task_suite": traj.get("task_suite", ""),
                    "task_id": traj.get("task_id", -1),
                    "objective": traj.get("objective", ""),
                    "tools_used": traj.get("tools_used", []),
                    "original_instruction": traj.get("original_instruction", ""),
                    "perturbed_instruction": traj.get("perturbed_instruction", ""),
                },
            })

    logger.info("Loaded %d SFT examples (from %d lines).", len(examples), i + 1)
    return examples


# ============================================================================
# Tokenisation: apply chat template with tool-call formatting
# ============================================================================

def _wrap_tools_for_grpo(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Double-wrap tool schemas to match ART's GRPO tokenisation path.

    ART's ``tokenize_trajectory`` (``art/preprocessing/tokenize.py``)
    applies ``[{"type": "function", "function": t} for t in history.tools]``
    where ``history.tools`` is already in OpenAI format
    (``[{"type": "function", "function": {...}}]``).  This produces a
    double-wrapped schema that the tokenizer's chat template renders into
    the ``<tools>`` block.

    We replicate this here so the SFT-trained model sees the *identical*
    tool-schema rendering it will encounter during GRPO rollouts.
    """
    return [{"type": "function", "function": t} for t in tools]


def tokenize_examples(
    examples: List[Dict[str, Any]],
    tokenizer,
    max_seq_length: int = 8192,
) -> dict:
    """Apply the tokenizer's chat template to produce input_ids + labels.

    Only assistant tokens are trained on (tool results and user/system
    messages are masked with -100 in labels).
    """
    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    skipped = 0

    for ex in examples:
        messages = ex["messages"]
        tools = _wrap_tools_for_grpo(ex.get("tools", TOOL_SCHEMAS))

        try:
            full_text = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as e:
            logger.debug("Skipping example (template error): %s", e)
            skipped += 1
            continue

        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        if len(full_ids) > max_seq_length:
            full_ids = full_ids[:max_seq_length]

        # Build labels: mask everything except assistant turns.
        # Strategy: tokenize non-assistant prefix, mark those positions as -100.
        # For efficiency, we use a simpler approach: build the conversation
        # incrementally and find assistant token spans.

        labels = [-100] * len(full_ids)

        # Tokenize up to each assistant boundary to find assistant spans
        prefix_messages = []
        for msg in messages:
            if msg["role"] == "assistant":
                # Tokenize prefix (everything before this assistant turn)
                prefix_text = tokenizer.apply_chat_template(
                    prefix_messages,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prefix_len = len(tokenizer.encode(prefix_text, add_special_tokens=False))

                # Tokenize prefix + this assistant turn
                prefix_messages.append(msg)
                with_assistant_text = tokenizer.apply_chat_template(
                    prefix_messages,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                with_assistant_len = len(tokenizer.encode(with_assistant_text, add_special_tokens=False))

                # Unmask assistant tokens
                for j in range(prefix_len, min(with_assistant_len, len(labels))):
                    labels[j] = full_ids[j]
            else:
                prefix_messages.append(msg)

        # Verify we have some trainable tokens
        trainable = sum(1 for l in labels if l != -100)
        if trainable == 0:
            skipped += 1
            continue

        all_input_ids.append(full_ids)
        all_attention_mask.append([1] * len(full_ids))
        all_labels.append(labels)

    if skipped:
        logger.info("Skipped %d examples during tokenisation.", skipped)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }


# ============================================================================
# Data preview: show what the SFT data looks like
# ============================================================================

def preview_sft_data(
    data_path: str,
    n_samples: int = 2,
    max_msg_len: int = 300,
):
    """Print a human-readable preview of the converted SFT data."""
    examples = load_sft_dataset(data_path, max_samples=n_samples)

    for i, ex in enumerate(examples[:n_samples]):
        meta = ex["metadata"]
        print(f"\n{'='*70}")
        print(f"SFT Example {i+1}")
        print(f"{'='*70}")
        print(f"  Reward:      {meta['reward']:.3f}")
        print(f"  Task:        {meta['task_suite']} task {meta['task_id']}")
        print(f"  Objective:   {meta['objective']}")
        print(f"  Tools used:  {meta['tools_used']}")
        print(f"  Original:    {meta['original_instruction']}")
        print(f"  Perturbed:   {meta['perturbed_instruction']}")
        print(f"  Messages:    {len(ex['messages'])} turns")
        print()

        for j, msg in enumerate(ex["messages"]):
            role = msg["role"].upper()
            if msg.get("tool_calls"):
                calls_str = ", ".join(
                    f"{tc['function']['name']}(...)"
                    for tc in msg["tool_calls"]
                )
                content_preview = msg.get("content", "") or ""
                print(f"  [{j:2d}] {role:10s} → tool_calls: [{calls_str}]")
                if content_preview:
                    print(f"       {'':10s}   content: {content_preview[:max_msg_len]}")
            elif msg.get("tool_call_id"):
                content = msg.get("content", "")
                preview = content[:max_msg_len] + ("..." if len(content) > max_msg_len else "")
                print(f"  [{j:2d}] TOOL       ← {msg['tool_call_id']}: {preview}")
            else:
                content = msg.get("content", "")
                preview = content[:max_msg_len] + ("..." if len(content) > max_msg_len else "")
                print(f"  [{j:2d}] {role:10s} {preview}")

    print(f"\n{'='*70}")
    print(f"Total available: {len(examples)} examples")
    print(f"{'='*70}")


# ============================================================================
# SFT Training
# ============================================================================

def train_sft(args: argparse.Namespace) -> None:
    """Run supervised fine-tuning on cold-start trajectories."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset

    # ---- Load data ----
    examples = load_sft_dataset(
        args.data_dir,
        min_reward=args.min_reward,
        min_assistant_turns=args.min_assistant_turns,
        max_samples=args.max_samples,
    )
    if not examples:
        logger.error("No valid examples found. Check data_dir and filters.")
        return

    # ---- Train/val split ----
    import random
    random.seed(42)
    random.shuffle(examples)
    val_size = max(1, int(len(examples) * args.val_fraction))
    train_examples = examples[val_size:]
    val_examples = examples[:val_size]
    logger.info("Split: %d train, %d val", len(train_examples), len(val_examples))

    # ---- Load tokenizer ----
    logger.info("Loading tokenizer: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Tokenize ----
    logger.info("Tokenizing %d train + %d val examples ...", len(train_examples), len(val_examples))
    train_tokenized = tokenize_examples(train_examples, tokenizer, args.max_seq_length)
    val_tokenized = tokenize_examples(val_examples, tokenizer, args.max_seq_length)

    trainable_per_example = [
        sum(1 for l in labels if l != -100)
        for labels in train_tokenized["labels"]
    ]
    total_trainable = sum(trainable_per_example)
    avg_trainable = total_trainable / len(trainable_per_example) if trainable_per_example else 0
    avg_seq_len = sum(len(ids) for ids in train_tokenized["input_ids"]) / len(train_tokenized["input_ids"]) if train_tokenized["input_ids"] else 0
    logger.info(
        "Tokenized: %d train, %d val  |  avg seq_len=%.0f, avg trainable tokens=%.0f (%.1f%%)",
        len(train_tokenized["input_ids"]), len(val_tokenized["input_ids"]),
        avg_seq_len, avg_trainable,
        100 * avg_trainable / avg_seq_len if avg_seq_len else 0,
    )

    train_dataset = Dataset.from_dict(train_tokenized)
    val_dataset = Dataset.from_dict(val_tokenized)

    # ---- Load model ----
    use_lora = not args.full_finetune
    logger.info("Loading model: %s  (mode: %s)", args.base_model,
                "LoRA" if use_lora else "full-parameter")
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }
    if args.load_in_4bit:
        if not use_lora:
            logger.warning("--load_in_4bit ignored for full-parameter SFT (requires bf16).")
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)

    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        n_params = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Full-parameter SFT: %d params (%.1fM), all trainable.",
                     n_train, n_train / 1e6)

    # ---- Data collator: pad to batch max length ----
    from dataclasses import dataclass
    @dataclass
    class SFTCollator:
        tokenizer: Any
        max_seq_length: int

        def __call__(self, features):
            max_len = min(
                max(len(f["input_ids"]) for f in features),
                self.max_seq_length,
            )
            batch = {"input_ids": [], "attention_mask": [], "labels": []}
            for f in features:
                pad_len = max_len - len(f["input_ids"])
                batch["input_ids"].append(
                    f["input_ids"][:max_len] + [self.tokenizer.pad_token_id] * pad_len
                )
                batch["attention_mask"].append(
                    f["attention_mask"][:max_len] + [0] * pad_len
                )
                batch["labels"].append(
                    f["labels"][:max_len] + [-100] * pad_len
                )

            return {
                k: torch.tensor(v, dtype=torch.long)
                for k, v in batch.items()
            }

    # ---- Build timestamped output directory ----
    from transformers import TrainingArguments, Trainer
    from datetime import datetime

    _timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _model_tag = args.base_model.replace("/", "_")
    # Infer objective from data
    _objectives = set(ex["metadata"].get("objective", "unknown") for ex in examples)
    _obj_tag = "+".join(sorted(_objectives)) if _objectives else "unknown"
    _run_name = f"sft_cold_start__{_model_tag}__{_timestamp}__{_obj_tag}"
    output_dir = os.path.join(args.output_dir, _run_name)
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Run directory: %s", output_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=SFTCollator(tokenizer, args.max_seq_length),
    )

    # ---- Train ----
    logger.info("Starting SFT training ...")
    logger.info("  Base model:      %s", args.base_model)
    logger.info("  Mode:            %s", "full-parameter" if args.full_finetune else f"LoRA r={args.lora_r}, alpha={args.lora_r * 2}")
    logger.info("  Train examples:  %d", len(train_dataset))
    logger.info("  Val examples:    %d", len(val_dataset))
    logger.info("  Epochs:          %d", args.num_epochs)
    logger.info("  Batch size:      %d x %d grad_accum = %d effective",
                args.batch_size, args.gradient_accumulation_steps,
                args.batch_size * args.gradient_accumulation_steps)
    logger.info("  LR:              %s", args.learning_rate)
    logger.info("  Output:          %s", output_dir)

    train_result = trainer.train()
    logger.info("Training complete: %s", train_result.metrics)

    # ---- Save model ----
    model_dir = os.path.join(output_dir, "model")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    if use_lora:
        logger.info("LoRA adapter saved to %s", model_dir)
    else:
        logger.info("Full model saved to %s", model_dir)

    # ---- Save training metadata ----
    meta = {
        "base_model": args.base_model,
        "data_dir": str(args.data_dir),
        "full_finetune": args.full_finetune,
        "lora_r": args.lora_r if use_lora else None,
        "lora_alpha": args.lora_r * 2 if use_lora else None,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_seq_length": args.max_seq_length,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "total_trainable_tokens": total_trainable,
        "avg_trainable_tokens_per_example": avg_trainable,
        "train_metrics": train_result.metrics,
        "min_reward": args.min_reward,
    }
    with open(os.path.join(output_dir, "sft_config.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info("Done! Use this model for warm-start GRPO:")
    logger.info("  python train_vla.py --resume --base_model %s", model_dir)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cold-start SFT: fine-tune attack agent on GPT-5 Mini trajectories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to cold-start output directory (containing success_trajectories.jsonl).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/sft_runs",
        help="Parent directory for SFT runs (a timestamped subdirectory is created).",
    )
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--full_finetune", action="store_true",
                        help="Full-parameter SFT (no LoRA). Qwen2.5-3B is ~6GB bf16, fits on one GPU.")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank (ignored with --full_finetune).")
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load base model in 4-bit (saves memory, slower).")
    parser.add_argument("--min_reward", type=float, default=0.0,
                        help="Minimum reward to include a trajectory.")
    parser.add_argument("--min_assistant_turns", type=int, default=2,
                        help="Minimum assistant turns to include a trajectory.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max trajectories to use (None = all).")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Fraction of data for validation.")
    parser.add_argument(
        "--preview", action="store_true",
        help="Only preview the converted data, don't train.",
    )

    args = parser.parse_args()

    if args.preview:
        preview_sft_data(args.data_dir)
    else:
        train_sft(args)


if __name__ == "__main__":
    main()
