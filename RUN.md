# How to Run the VLA Attack Experiment (pi0.5 in LIBERO)

## Quick run

From `agent_attack_framework/` with the right environment and dependencies:

```bash
python run.py vla --objective task_failure --task_suite libero_spatial --task_ids 0,1,2
```

Or call the script directly (same options):

```bash
python train_vla.py --objective task_failure --task_suite libero_spatial --task_ids 0,1,2
```

### Ensuring the agent can deploy multi-tool attacks

To improve attack success, allow the agent to use **multiple tools** (and more rounds):

1. **Enable several tool families** so the agent can choose token, char, prompt, and/or visual attacks:
   ```bash
   python run.py vla --tool_sets token,char,prompt,visual ...
   ```
   Default is `token,char,prompt`; add `visual` for image perturbations.

2. **Increase tool-call rounds** so the agent can chain tools (e.g. apply one attack, then FIND again and apply another):
   ```bash
   python run.py vla --max_turns 10 ...
   ```
   Default is 8. Use 10–12 if single-tool attacks often fail and you want the agent to try multiple strategies or chain token + prompt, etc.

3. The system prompt already instructs the agent to chain attacks and use multiple tools when a single attack is insufficient; with more tool sets and more turns, the agent has room to succeed.

### Attack budget vs max_turns

| Concept | Meaning | Where it’s set |
|--------|---------|----------------|
| **max_turns** | **How many steps** the agent gets per episode. One “turn” = one ReAct cycle (agent → tool call → result). More turns allow multiple FIND/APPLY pairs or chaining several tools. | `--max_turns` (default 8). |
| **Attack budget(s)** | **How large** each perturbation is allowed to be. These are per-tool or per-call limits on the *size* of an edit, not the number of steps. | Inside each tool family; not a single global flag. |

Examples of **attack budgets** (perturbation size limits):

- **Prompt tools**: `max_added_tokens` (default 40) — caps how many tokens can be added in one prompt-level edit. Enforced inside the prompt_attack tools.
- **Visual tools**: `linf_budget` (e.g. 8) — max per-pixel change (L∞) for image perturbations. Passed when the agent calls the apply function.
- **Reward**: The objective reward includes a **stealth penalty** (`--stealth_weight`); large edits reduce reward, so the agent is encouraged to stay within small effective “budgets” even when tools allow more.

So: **max_turns** = “how many tool-call rounds”; **attack budget** = “how big each edit can be” (per tool / per call).

---

## Prerequisites

1. **Environment**: Use the env from `requirements.txt` (e.g. conda `libero`, Python 3.11, PyTorch + JAX, LIBERO, openpi, openpipe-art). To verify:
   ```bash
   conda activate libero
   python scripts/check_libero_env.py
   ```
   This checks: Python 3.11, RoboTwin/policy/pi05, jax, flax, openpi, libero, openpipe-art, vllm, langgraph, and Pi05LiberoModel. If `vllm` is not 0.13.0, you may see a warning; upgrade with `pip install vllm==0.13.0` if needed.
2. **LIBERO**: Installed and on `PYTHONPATH` (e.g. `pip install -e /path/to/LIBERO --no-deps`).
3. **RoboTwin + pi0.5**: The pi0.5 wrapper expects RoboTwin at `vlm-robot/RoboTwin` (sibling of `agent_attack_framework/`), with `policy/pi05` and the openpi code it ships. If your layout is different, adjust `libero_rollouts/pi05_libero_model.py` (`_ROBOTWIN_ROOT`).
4. **Headless rendering**: On a machine without a display, set before any MuJoCo/PyOpenGL import (the script sets these by default):
   - `MUJOCO_GL=egl`
   - `PYOPENGL_PLATFORM=egl`
5. **GPUs**: GPU 0 is dedicated to all VLA rollouts (Pi0.5 via JAX); all remaining visible GPUs are used for the attack agent (vLLM/ART training). The script auto-detects visible GPUs and excludes `--vla_gpu` from the attack set. For single-GPU runs use `--vla_gpu 0 --attack_gpus 0` (may OOM).

**Version note**: `requirements.txt` pins `vllm==0.13.0`. If your env has a different vllm (e.g. 0.11.2), the check script will warn; upgrade with `pip install vllm==0.13.0` if you need exact compatibility.

**Outputs**: LoRA checkpoints, vLLM logs, and ART trajectories are written under **`agent_attack_framework/outputs/`** (e.g. `outputs/vla-attack-agent/models/<model_name>/checkpoints/`, `.../logs/`). This folder is in `.gitignore`.

---

## Why it might get stuck

The run can appear to hang at a few points. Here’s what usually happens and what to try.

### 1. Right after starting (imports)

- **What you see**: No log for a while after `python run.py vla ...`.
- **Cause**: Heavy imports (PyTorch, JAX, ART, LIBERO, etc.).
- **What to do**: Wait 30–60 s. If it still does nothing, run with `python -u run.py vla ...` so stdout is unbuffered and check for import errors (e.g. missing LIBERO, RoboTwin, or openpi).

### 2. “Loading Pi0.5 VLA model on GPU 0 …”

- **What you see**: Stuck after this log line.
- **Cause**: First JAX run and/or downloading the pi0.5 checkpoint from `gs://openpi-assets/checkpoints/pi05_base` (and possibly normalization assets).
- **What to do**: Ensure network access to the bucket (or have the checkpoint cached under `OPENPI_DATA_HOME`). First load can take several minutes.

### 3. After “Set CUDA_VISIBLE_DEVICES=… for ART subprocess” / “Configuring attack model …”

- **What you see**: No progress after these lines.
- **Cause**: `await attack_model.register(backend)` starts the ART backend (vLLM) in a subprocess: loading Qwen, compiling CUDA graphs, etc. This often takes **2–5+ minutes** and uses a lot of GPU memory.
- **What to do**: Wait at least 5–10 minutes. If you have limited GPU memory, use `--gpu_memory_utilization 0.6` and ensure no other big process is on the attack GPU. Check GPU usage (e.g. `nvidia-smi`) to confirm the process is active.

### 4. “Step 0 — gathering N trajectory groups …”

- **What you see**: Stuck here for a long time.
- **Cause**: Each “group” runs full LIBERO episodes: create env, run baseline pi0.5 rollout, then attack rollout (LLM tool calls + another pi0.5 rollout). With default `--groups_per_step 4` and `--trajectories_per_group 4`, that’s 16 rollouts per step; each rollout can be hundreds of steps.
- **What to do**: For a **quick sanity check**, reduce load (see “Lightweight run” below). Then increase again once it runs.

### 5. “[Warning]: datasets path … does not exist!”

- **What you see**: `[Warning]: datasets path /path/to/libero_pkg/libero/datasets does not exist!`
- **Cause**: The LIBERO package (e.g. `libero_pkg`) expects a `datasets` directory for some benchmarks; the path comes from `~/.libero/config.yaml` or the package default. For VLA attack rollouts you don’t need real dataset files.
- **What to do**: Create the directory so the warning goes away: `mkdir -p /path/to/libero_pkg/libero/datasets` (use your actual `libero_pkg` path). The run will work even if the folder is empty.

### 6. LIBERO / EGL / rendering

- **What you see**: Crash or hang when creating the LIBERO env or on first `env.reset()` / `env.step()`.
- **Cause**: Missing or broken EGL (headless OpenGL), or LIBERO/MuJoCo data path not set.
- **What to do**: Set `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl`; install libEGL/Mesa. If EGL is not available, try `MUJOCO_GL=osmesa` (slower). Ensure LIBERO’s `get_libero_path()` and benchmark data are valid.

### 7. “Engine core initialization failed. See root cause above.”

- **What you see**: `RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}` when the vLLM attack server starts.
- **Cause**: The vLLM worker process exited during startup. The **real error** (often CUDA OOM or driver error) is usually printed **earlier** in the same log.
- **What to do**:
  1. Scroll up in the log and look for the first traceback or `OutOfMemoryError` / `CUDA out of memory`.
  2. Lower vLLM’s memory use: `--gpu_memory_utilization 0.5` (or 0.45 for single-GPU runs).
  3. Ensure the attack GPUs don't overlap with the VLA GPU (default: GPU 0 = VLA rollouts, remaining = agent training).
  4. Try a smaller `--base_model` (e.g. Qwen3-1.7B) if the GPU is small.

### 8. “add_lora” failed: tensor size mismatch

- **What you see**: `ValueError: Call to add_lora method failed: The size of tensor a (…) must match the size of tensor b (…) at non-singleton dimension …`.
- **Cause**: You passed **`--resume`** and an existing LoRA checkpoint was trained for a different `--base_model` or model size; dimensions don’t match.
- **What to do**: **By default the script trains from scratch** (it clears previous checkpoints for this model). If you see this error, you are likely using `--resume`. Run **without** `--resume` so the run starts from the current `--base_model`. Use `--resume` only when you intend to continue a previous run with the same base model.

### 9. Error 500 (HTTP 500 Internal Server Error)

- **What you see**: `Error 500`, `500 Internal Server Error`, or an exception mentioning HTTP 500 when the run is making inference requests (e.g. during “Step 0 — gathering …” or when the ReAct agent calls the model).
- **Cause**: The **vLLM inference server** (started by ART for the attack model) is returning 500. Common reasons:
  1. **OOM on the attack GPU** — vLLM ran out of memory during a forward pass (long sequence, large batch, or LoRA load).
  2. **vLLM worker crash** — the worker process died; the real error is often in the vLLM log (see below).
  3. **ART/vLLM version mismatch** — e.g. ART 0.5.x with vLLM 0.11.x can hit missing APIs; see README “Required patches” for `pause_generation`/`resume_generation` and `sleep`/`wake_up`.
- **What to do**:
  1. **Check vLLM logs**: ART writes logs under `agent_attack_framework/outputs/vla-attack-agent/models/<model_name>/logs/`. Open the latest log and search for the first `Traceback`, `Error`, or `OutOfMemoryError` **before** the 500.
  2. **Lower GPU memory use**: `--gpu_memory_utilization 0.5` (or 0.45 if using a single GPU for both VLA and attack).
  3. **Use separate GPUs**: Ensure `--vla_gpu` (default 0, all rollouts) does not overlap with `--attack_gpus` (defaults to all other visible GPUs).
  4. **Apply ART/vLLM patches** (README): If you are on vLLM < 0.16, add the `pause_generation`/`resume_generation` stubs in `vllm/v1/engine/async_llm.py` and use `llm.sleep()`/`llm.wake_up()` in `art/unsloth/service.py` instead of `run_on_workers(do_sleep/do_wake_up)`.
  5. **Smaller model**: Try `--base_model Qwen/Qwen2.5-1.5B-Instruct` (or Qwen3-1.7B) to reduce vLLM memory.
  6. **Retry**: Sometimes the first request after wake-up fails; the next step may succeed. If 500s are intermittent, increase timeouts or reduce concurrency (e.g. fewer `trajectories_per_group`).

---

## Lightweight run (debug / quick test)

Use one task, one episode, one group, and one trajectory per group so the first step finishes quickly and you can see that nothing is stuck:

```bash
python run.py vla \
  --objective task_failure \
  --task_suite libero_spatial \
  --task_ids 0 \
  --episodes_per_task 1 \
  --groups_per_step 1 \
  --trajectories_per_group 1 \
  --num_epochs 1
```

Optional: shorten the episode so each rollout is faster (e.g. 50 steps):

```bash
python run.py vla \
  --objective task_failure \
  --task_suite libero_spatial \
  --task_ids 0 \
  --episodes_per_task 1 \
  --groups_per_step 1 \
  --trajectories_per_group 1 \
  --num_epochs 1 \
  --max_steps 50
```

If this runs to “Step 0 — gathered … training …” and then “Training complete.” (or a later step), the pipeline is working. Then remove the overrides or increase `task_ids`, `episodes_per_task`, `groups_per_step`, and `trajectories_per_group` for real training.

---

## Single-GPU run (may OOM)

If you only have one GPU:

```bash
python run.py vla \
  --task_suite libero_spatial \
  --task_ids 0 \
  --vla_gpu 0 \
  --attack_gpus 0 \
  --gpu_memory_utilization 0.45 \
  --episodes_per_task 1 \
  --groups_per_step 1 \
  --trajectories_per_group 1
```

Both Pi0.5 (JAX) and the attack model (vLLM) will use GPU 0; 0.45 leaves room for both. If you see OOM, lower `gpu_memory_utilization` or use a smaller base model.

---

## Useful options summary

| Option | Default | Description |
|--------|---------|-------------|
| `--task_suite` | libero_spatial | libero_spatial, libero_object, libero_goal, libero_10, libero_90 |
| `--task_ids` | 0 | Comma-separated task indices |
| `--episodes_per_task` | 3 | Initial states per task |
| `--groups_per_step` | 4 | Scenario groups per training step |
| `--trajectories_per_group` | 4 | Rollouts per group |
| `--vla_gpu` | 0 | GPU for Pi0.5 (JAX) |
| `--attack_gpus` | 1 | GPU(s) for attack model (vLLM) |
| `--max_steps` | (suite default) | Max env steps per episode (use lower for quick tests) |
