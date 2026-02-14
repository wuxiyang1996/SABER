# HotpotQA Adversarial Attack Framework

An adversarial attack framework built on **ART** (Agent Reinforcement Trainer). A GRPO-trained attack agent (Qwen3-1.7B) learns to craft adversarial suffixes that fool a frozen Qwen2.5-3B-Instruct evaluation model on multi-hop HotpotQA questions.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                 Attack Agent (Qwen3-1.7B, LoRA-tuned)        │
│  Sees: question, context, gold answer                        │
│  Calls: add_suffix(suffix="...")                             │
│  Trained via: ART / GRPO                                     │
└──────────────┬───────────────────────────────────────────────┘
               │  adversarial suffix
               ▼
┌──────────────────────────────────────────────────────────────┐
│   Modified Question = original_question + suffix             │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│            Frozen Eval Model (Qwen2.5-3B-Instruct)           │
│  Input: modified question + reference context                │
│  Output: answer                                              │
│  NOT trained — base weights only                             │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│                     Reward Computation                        │
│  Compare eval model answer vs gold answer (F1/EM)            │
│  reward = 1 - F1  (higher when model is MORE wrong)          │
│  +0.5 bonus if F1 = 0 (complete fool)                        │
│  -0.5 penalty if no suffix provided                          │
└──────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
agent_attack_framework/
├── agent/
│   ├── __init__.py
│   └── rollout.py           # Attack agent rollout (ART-based)
├── dataset/
│   ├── __init__.py
│   └── hotpotqa.py           # HotpotQA loader + F1/EM metrics
├── eval_model/
│   ├── __init__.py
│   └── qa_model.py           # Frozen Qwen2.5-3B QA wrapper
├── tools/
│   ├── __init__.py
│   └── attack.py             # add_suffix tool
├── trainer/
│   ├── __init__.py
│   └── train.py              # ART GRPO training loop
├── eval.py                   # Evaluation: baseline vs attack
├── run.py                    # Convenience entry-point
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Install ART from local checkout
pip install -e /path/to/ART

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) W&B for logging
export WANDB_API_KEY=your_key
```

## Quick Evaluation

Compare baseline (clean questions) vs attack (with suffix):

```bash
cd agent_attack_framework
python eval.py --n 10
```

## Training

```bash
cd agent_attack_framework
python -m trainer.train \
    --steps 50 \
    --lr 5e-6 \
    --traj-per-group 8 \
    --groups-per-step 4 \
    --train-samples 200
```

| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | 50 | Max training steps |
| `--lr` | 5e-6 | Learning rate |
| `--traj-per-group` | 8 | Rollouts per scenario for reward variance |
| `--groups-per-step` | 4 | Scenarios per training step |
| `--train-samples` | 200 | Training scenarios from HotpotQA |
| `--eval-every` | 5 | Validation frequency |

## How It Works

1. **Attack agent** (LoRA on Qwen3-1.7B) sees the question, context, and gold answer.
2. It calls `add_suffix(suffix=...)` to append adversarial text to the question.
3. The **frozen eval model** (Qwen2.5-3B-Instruct, no LoRA) answers the modified question.
4. **Reward** = `1 - F1(eval_answer, gold)` — high when the attack succeeds.
5. **GRPO** uses reward variance across rollouts to update the attack agent's policy.
6. Over training, the agent learns what kinds of suffixes effectively fool the QA model.
