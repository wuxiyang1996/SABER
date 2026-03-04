# Reasoning VLAs: Alternatives and Faster Options vs DeepThinkVLA

## Current setup: DeepThinkVLA

- **Model**: `yinchenghust/deepthinkvla_libero_cot_rl` (2.9B, CoT + RL)
- **Speed**: Slower due to **autoregressive CoT** token generation before each action.
- **LIBERO**: 97% average success; hybrid decoder (reasoning → parallel actions).

## Faster options (in this repo or research)

### 1. **ECoT** (already supported) — use as `--vla_model ecot`

- **Checkpoint**: `Embodied-CoT/ecot-openvla-7b-bridge` (OpenVLA-based, 7B).
- **Reasoning**: Embodied chain-of-thought before action prediction.
- **Speed**: Same class as OpenVLA (single `predict_action` call); typically **faster per step** than DeepThinkVLA if the HF API doesn’t emit long CoT. No separate CoT decode in our wrapper.
- **Use**: Source `rollout_ecot.sh` and pass `--vla_model ecot` in cold-start scripts.

### 2. **DeepThinkVLA with Masked CoT** (same checkpoint, faster path)

- **Idea**: DeepThinkVLA repo supports **masked CoT**: skip generating reasoning tokens and go straight to action decoding via `prompt_cot_predict_action()`.
- **Result**: ~**0.175× latency** vs full autoregressive CoT (paper: 96.5% LIBERO with mask CoT).
- **Where**: `repos/deepthinkvla/src/experiments/deepthinkvla_utils.py` — `get_vla_action_mask_cot()`.
- **To use here**: Add a “mask_cot” mode to `deepthinkvla_rollout_wrapper.py` that calls the model’s `prompt_cot_predict_action()` (and the repo’s token/action handling) instead of `generate()`. Same checkpoint, no new download.

### 3. **LightVLA** (no reasoning, but faster/lighter)

- **Checkpoints**: `TTJiang/LightVLA-libero-*` (per-suite).
- **Reasoning**: None; direct action prediction.
- **Speed**: Lighter architecture; good if you care more about throughput than CoT.
- **Use**: Add `lightvla` to `--vla_model` choices and load via `model_factory` (wrapper already exists).

### 4. **External / future options** (no LIBERO integration yet)

- **Fast-ThinkAct** (arXiv:2601.09708): Verbalizable latent planning, **~9.3× faster** than ThinkAct-style CoT; evaluated on LIBERO. Checkpoints may appear on HuggingFace later.
- **LaRA-VLA**: Latent reasoning; **~90% latency reduction** vs explicit CoT. No public LIBERO checkpoint found.
- **Fast ECoT** (arXiv:2506.07639): Inference-time acceleration for ECoT—**still produces full textual reasoning** (plans, subgoals, visual inferences). Speeds up by thought reuse, parallel reasoning-step generation, and async action decoding. No new checkpoint; applies to existing ECoT. Would need integration (e.g. vLLM + their scheduling) in this codebase.
- **ECoT-Lite**: 3× speedup; learns from reasoning at train time but **no textual reasoning at test time** (latent only). Would need new checkpoints/wrappers.

## Summary

| Option              | Reasoning (text?) | Relative speed vs DeepThinkVLA | In this repo      |
|---------------------|-------------------|---------------------------------|-------------------|
| **ECoT**            | Yes (CoT text)    | Likely faster                   | Yes (`ecot`)      |
| **Fast ECoT**       | Yes (CoT text)    | ~2–7× faster (same ECoT model) | Integration TODO  |
| **DeepThinkVLA mask CoT** | Yes (no gen) | ~5–6× faster (0.175× latency) | Repo has it; wrapper TODO |
| **LightVLA**        | No                | Lighter/faster                  | Yes (factory only) |
| **Fast-ThinkAct**   | Latent only       | ~9× faster                      | No checkpoint yet  |

**Practical next steps**: Use **ECoT** for a faster reasoning VLA today (`rollout_ecot.sh` + `--vla_model ecot`). For maximum speed with DeepThinkVLA, add a mask-CoT path in `deepthinkvla_rollout_wrapper.py` using `prompt_cot_predict_action()`.
