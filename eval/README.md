# LIBERO Evaluation (4 Suites)

Unified evaluation on the **four LIBERO evaluation suites**: **spatial**, **object**, **goal**, and **long** (libero_10). Same episode counts, seeds, and logging for comparable results across models.

All models below are available on **GitHub** and/or **Hugging Face**; links are in the tables.

---

## Model index (GitHub & Hugging Face)

| Model | GitHub | Hugging Face |
|-------|--------|--------------|
| **OpenPI π0 / π0.5** | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | [lerobot/pi05_libero_finetuned](https://huggingface.co/lerobot/pi05_libero_finetuned), [lerobot/pi05_libero_base](https://huggingface.co/lerobot/pi05_libero_base) |
| **DeepThinkVLA** | [OpenBMB/DeepThinkVLA](https://github.com/OpenBMB/DeepThinkVLA) | [yinchenghust/deepthinkvla_libero_cot_rl](https://huggingface.co/yinchenghust/deepthinkvla_libero_cot_rl) |
| **MolmoAct** | [allenai/molmoact](https://github.com/allenai/molmoact) | [allenai/MolmoAct-7B-D-LIBERO-Spatial-0812](https://huggingface.co/allenai/MolmoAct-7B-D-LIBERO-Spatial-0812), [allenai/MolmoAct-7B-D-LIBERO-Long-0812](https://huggingface.co/allenai/MolmoAct-7B-D-LIBERO-Long-0812), (+ Object/Goal variants) |
| **ECoT (Embodied CoT)** | [MichalZawalski/embodied-CoT](https://github.com/MichalZawalski/embodied-CoT) | [Embodied-CoT/ecot-openvla-7b-bridge](https://huggingface.co/Embodied-CoT/ecot-openvla-7b-bridge) |
| **InternVLA-M1** | [InternRobotics/InternVLA-M1](https://github.com/InternRobotics/InternVLA-M1) | [InternRobotics/InternVLA-M1-LIBERO-Spatial](https://huggingface.co/InternRobotics/InternVLA-M1-LIBERO-Spatial), [InternRobotics/InternVLA-M1](https://huggingface.co/InternRobotics/InternVLA-M1) |
| **OpenVLA** | [openvla/openvla](https://github.com/openvla/openvla) | [openvla/openvla-7b-finetuned-libero-spatial](https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial), [openvla/openvla-7b-finetuned-libero-10](https://huggingface.co/openvla/openvla-7b-finetuned-libero-10), (+ object/goal) |
| **StarVLA** | (see HF) | [StarVLA/Qwen3-VL-PI-LIBERO-4in1](https://huggingface.co/StarVLA/Qwen3-VL-PI-LIBERO-4in1) |
| **X-VLA** | [2toinf/X-VLA](https://github.com/2toinf/X-VLA), [huggingface/lerobot](https://github.com/huggingface/lerobot) | [lerobot/xvla-libero](https://huggingface.co/lerobot/xvla-libero), [lerobot/xvla-base](https://huggingface.co/lerobot/xvla-base) |
| **LightVLA** | [LiAutoAD/LightVLA](https://github.com/LiAutoAD/LightVLA) | [TTJiang/models](https://huggingface.co/TTJiang/models) (search `lightvla`) |

---

## Feasible model weights (checkpoints for eval)

Concrete checkpoint IDs and URLs for loading each model. Use these with the model’s repo or `from_pretrained`-style APIs.

| Model | Weights / checkpoint source | Notes |
|-------|----------------------------|--------|
| **OpenPI π0** | `gs://openpi-assets/checkpoints/pi0_base` (GCS; repo script), or [lerobot/pi05_libero_base](https://huggingface.co/lerobot/pi05_libero_base) (LeRobot port) | Official base is GCS; [HF issue #260](https://github.com/Physical-Intelligence/openpi/issues/260). Zero-shot LIBERO uses pi0_base. |
| **OpenPI π0.5** | `gs://openpi-assets/checkpoints/pi05_libero` (GCS), [lerobot/pi05_libero_finetuned](https://huggingface.co/lerobot/pi05_libero_finetuned), [lerobot/pi05_libero_base](https://huggingface.co/lerobot/pi05_libero_base) | LIBERO-finetuned or base via LeRobot. |
| **OpenVLA** | [openvla/openvla-7b](https://huggingface.co/openvla/openvla-7b) (base). LIBERO per-suite: [openvla/openvla-7b-finetuned-libero-spatial](https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial), [openvla/openvla-7b-finetuned-libero-object](https://huggingface.co/openvla/openvla-7b-finetuned-libero-object), [openvla/openvla-7b-finetuned-libero-goal](https://huggingface.co/openvla/openvla-7b-finetuned-libero-goal), [openvla/openvla-7b-finetuned-libero-10](https://huggingface.co/openvla/openvla-7b-finetuned-libero-10) | One checkpoint per suite; eval script loads by suite. |
| **X-VLA** | [lerobot/xvla-libero](https://huggingface.co/lerobot/xvla-libero) (LIBERO), [lerobot/xvla-base](https://huggingface.co/lerobot/xvla-base) | Single LIBERO policy; `lerobot-eval --policy.path=lerobot/xvla-libero` auto-downloads. |
| **MolmoAct** | [allenai/MolmoAct-7B-D-LIBERO-Spatial-0812](https://huggingface.co/allenai/MolmoAct-7B-D-LIBERO-Spatial-0812), [allenai/MolmoAct-7B-D-LIBERO-Object-0812](https://huggingface.co/allenai/MolmoAct-7B-D-LIBERO-Object-0812), [allenai/MolmoAct-7B-D-LIBERO-Goal-0812](https://huggingface.co/allenai/MolmoAct-7B-D-LIBERO-Goal-0812), [allenai/MolmoAct-7B-D-LIBERO-Long-0812](https://huggingface.co/allenai/MolmoAct-7B-D-LIBERO-Long-0812). Base: [allenai/MolmoAct-7B-D-0812](https://huggingface.co/allenai/MolmoAct-7B-D-0812) | One HF model per LIBERO suite; [MolmoAct collection](https://huggingface.co/collections/allenai/molmoact). |
| **DeepThinkVLA** | [yinchenghust/deepthinkvla_libero_cot_rl](https://huggingface.co/yinchenghust/deepthinkvla_libero_cot_rl) (3B, LIBERO CoT+RL). Repo: [OpenBMB/DeepThinkVLA](https://github.com/OpenBMB/DeepThinkVLA); LIBERO eval: [wadeKeith/DeepThinkVLA_libero_plus](https://github.com/wadeKeith/DeepThinkVLA_libero_plus) | Hybrid decoder; HF checkpoint for LIBERO. |
| **ECoT** | [Embodied-CoT/ecot-openvla-7b-bridge](https://huggingface.co/Embodied-CoT/ecot-openvla-7b-bridge). Code: [MichalZawalski/embodied-CoT](https://github.com/MichalZawalski/embodied-CoT) | Reasoning-before-acting VLA built on OpenVLA; uses same LIBERO eval stack as OpenVLA with this checkpoint. |
| **InternVLA-M1** | [InternRobotics/InternVLA-M1-LIBERO-Spatial](https://huggingface.co/InternRobotics/InternVLA-M1-LIBERO-Spatial), [InternRobotics/InternVLA-M1-LIBERO-Long](https://huggingface.co/InternRobotics/InternVLA-M1-LIBERO-Long), [InternRobotics/InternVLA-M1](https://huggingface.co/InternRobotics/InternVLA-M1) | Per-suite or base; see repo for Object/Goal if added. |
| **StarVLA** | [StarVLA/Qwen3-VL-PI-LIBERO-4in1](https://huggingface.co/StarVLA/Qwen3-VL-PI-LIBERO-4in1) (LIBERO 4-in-1; ~97.5% at 100k steps) | Single checkpoint for all 4 LIBERO suites. [StarVLA LIBERO collection](https://huggingface.co/collections/StarVLA/libero). |
| **LightVLA** | [TTJiang/LightVLA-libero-spatial](https://huggingface.co/TTJiang/LightVLA-libero-spatial), [TTJiang/LightVLA-libero-object](https://huggingface.co/TTJiang/LightVLA-libero-object), [TTJiang/LightVLA-libero-goal](https://huggingface.co/TTJiang/LightVLA-libero-goal), [TTJiang/LightVLA-libero-10](https://huggingface.co/TTJiang/LightVLA-libero-10). [LiAutoAD/LightVLA](https://github.com/LiAutoAD/LightVLA) LIBERO.md for eval | One HF model per LIBERO suite (8B). |

---

## Download Hugging Face checkpoints (for evaluation)

All evaluation checkpoints listed above are on Hugging Face. To pre-download them so evals can run (or run faster / offline), use:

```bash
pip install huggingface_hub
# Download all checkpoints for all models
python -m eval.download_checkpoints

# Download for specific model(s)
python -m eval.download_checkpoints --model xvla
python -m eval.download_checkpoints --models openvla,starvla,molmoact

# List repo_ids that would be downloaded (no download)
python -m eval.download_checkpoints --list
```

Downloads go to the Hugging Face cache (`$AGENT_ATTACK_CACHE/huggingface` or `./.cache/huggingface`). **All eval entry points** (`run_libero_eval`, `run_one_model`, `run_all_libero_evals_parallel`, `eval.external.run`) set `HF_HOME` / `HF_HUB_CACHE` / `TRANSFORMERS_CACHE` to this same path, so after you run `download_checkpoints`, model checkpoints are **loaded automatically from cache** when you run evaluation (no extra config). The list of repo_ids per model is in `eval/external/configs.py` (`HF_CHECKPOINTS_FOR_EVAL`). OpenPI π0/π0.5 native runs use GCS by default; the script also downloads the LeRobot HF mirrors for reference or LeRobot-based workflows.

---

## Suites and defaults

| Suite        | LIBERO name   | Max steps | Tasks |
|-------------|---------------|-----------|-------|
| spatial     | libero_spatial | 220     | 10    |
| object      | libero_object  | 280     | 10    |
| goal        | libero_goal    | 300     | 10    |
| long        | libero_10      | 520     | 10    |

**Default eval:** all 4 suites, task IDs 0–9, 5 episodes per task, seed 42 → 4×10×5 = 200 episodes total.

**If you see low baseline success when running agent training** (`train_vla.py`): the eval script uses **seed 42** and **replan_steps=5**. Training now defaults to **seed 42** so baseline rollouts match eval conditions; if you previously used the old default (seed 7), different env stochasticity can lower baseline success. Also ensure you are not using `libero_90` for training (Pi0.5 has ~20–30% there); use the four eval suites (spatial, object, goal, long) and tasks 0–6 for high baseline.

**Training vs eval settings (OpenPI π0.5)** — Training is aligned with eval where it matters:

| Setting            | Eval (`run_libero_eval`)     | Training (`train_vla.py`)        |
|--------------------|-----------------------------|----------------------------------|
| Suites             | spatial, object, goal, long | Same (default)                   |
| Seed               | 42                          | 42 (default)                    |
| Replan steps       | 5                           | 5 (default)                     |
| Resolution         | 256                         | 256 (vla_rollout default)       |
| Checkpoint         | pi05_libero (default)       | Same                            |
| Max steps          | Per-suite (220/280/300/520) | 800 (all suites) for action_inflation headroom |
| Train tasks        | —                           | 0–6                             |
| Eval / test tasks  | 0–9 or 7–9                  | 7–9 (held-out)                  |

The only intentional difference is **max_steps**: eval uses per-suite limits for comparable benchmarks; training uses 800 so the attack rollout has room to succeed with more steps (action_inflation). Baseline success in training should match eval on the same tasks when seed/replan/resolution match.

**Per-model default hyperparameters** are set in `eval/model_registry.py` (`MODEL_DEFAULT_HYPERPARAMS` / `get_model_defaults()`). Source-backed where available (e.g. OpenPI: `action_horizon=10`, `replan_steps=5` from [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) config `pi0_libero` / `pi05_libero`). If you omit `--episodes_per_task`, `--seed`, or `--replan_steps`, the script uses the default for the chosen model.

---

## Eval code for each model (4 LIBERO suites)

From `agent_attack_framework` you can run or print eval commands for every model.

| Model | Eval code / command |
|-------|---------------------|
| **OpenPI π0** | `python -m eval.run_libero_eval --model openpi_pi0` (native) |
| **OpenPI π0.5** | `python -m eval.run_libero_eval --model openpi_pi05` (native) |
| **OpenVLA** | `python -m eval.run_libero_eval --model openvla --repo_path /path/to/openvla` or `--print_cmd` to print per-suite commands |
| **X-VLA** | `python -m eval.run_libero_eval --model xvla` (no repo needed; uses `lerobot-eval` if installed) or `--print_cmd` |
| **MolmoAct** | `python -m eval.run_libero_eval --model molmoact --repo_path /path/to/molmoact [--print_cmd]` |
| **DeepThinkVLA** | `python -m eval.run_libero_eval --model deepthinkvla --repo_path /path/to/DeepThinkVLA [--print_cmd]` |
| **ECoT** | `python -m eval.run_libero_eval --model ecot --repo_path /path/to/openvla` (uses OpenVLA repo + checkpoint Embodied-CoT/ecot-openvla-7b-bridge) or `--print_cmd` |
| **InternVLA-M1** | `python -m eval.run_libero_eval --model internvla_m1 --repo_path /path/to/InternVLA-M1 [--print_cmd]` |
| **StarVLA** | `python -m eval.run_libero_eval --model starvla` (no repo; uses `lerobot-eval` + StarVLA/Qwen3-VL-PI-LIBERO-4in1) or `--print_cmd` |
| **LightVLA** | `python -m eval.run_libero_eval --model lightvla --repo_path /path/to/LightVLA [--print_cmd]` |

Native runs (openpi_pi0, openpi_pi05) execute full eval in-process and write `eval/results/<model>_<timestamp>.json`. External models run their repo’s eval script(s); output is logged under `eval/results/<model>_external_<timestamp>.log` and a placeholder JSON is written. Use `--episodes_per_task` and `--seed` for comparable settings (defaults: 5, 42).

### Run each model separately (copy-paste commands)

From `agent_attack_framework` (same defaults: 5 episodes per task, seed 42, all 4 suites). Set `CUDA_VISIBLE_DEVICES` to pick a GPU; add `--output_dir eval/results` if you want a specific dir.

| Model | Command |
|-------|--------|
| **openpi_pi0** | `python -m eval.run_libero_eval --model openpi_pi0` |
| **openpi_pi05** | `python -m eval.run_libero_eval --model openpi_pi05` |
| **openvla** | `python -m eval.run_libero_eval --model openvla --repo_path /path/to/openvla` |
| **xvla** | `python -m eval.run_libero_eval --model xvla` |
| **molmoact** | `python -m eval.run_libero_eval --model molmoact --repo_path /path/to/molmoact` |
| **deepthinkvla** | `python -m eval.run_libero_eval --model deepthinkvla --repo_path /path/to/DeepThinkVLA` |
| **ecot** | `python -m eval.run_libero_eval --model ecot --repo_path /path/to/openvla` |
| **internvla_m1** | `python -m eval.run_libero_eval --model internvla_m1 --repo_path /path/to/InternVLA-M1` |
| **starvla** | `python -m eval.run_libero_eval --model starvla` |
| **lightvla** | `python -m eval.run_libero_eval --model lightvla --repo_path /path/to/LightVLA` |

Or use the single entry-point script (sets GPU and forwards args):

```bash
python -m eval.run_one_model --model openpi_pi05
python -m eval.run_one_model --model openvla --repo_path /path/to/openvla --gpu 0
```

---

## Native evaluation (this repo)

**OpenPI π0 and π0.5** are supported directly. Run from `agent_attack_framework`:

```bash
# Pi0.5 (LIBERO-finetuned) on all 4 suites, 5 episodes per task
python -m eval.run_libero_eval --model openpi_pi05 --suites spatial,object,goal,long --episodes_per_task 5 --seed 42

# Pi0 (zero-shot from pi0_base) on same setup
python -m eval.run_libero_eval --model openpi_pi0 --suites spatial,object,goal,long --episodes_per_task 5 --seed 42

# Test split only (tasks 7–9 per suite), 10 episodes per task
python -m eval.run_libero_eval --model openpi_pi05 --task_ids 7-9 --episodes_per_task 10 --seed 42

# Custom checkpoint and output dir
python -m eval.run_libero_eval --model openpi_pi05 --checkpoint /path/to/ckpt --output_dir eval/results
```

Results are written to `eval/results/<model>_<timestamp>.json` (or `--output_dir`). Each run includes per-suite and per-task success counts and an overall success rate. The `summary.per_task` field gives a breakdown by task category (suite and task_id) for reference.

---

## External models (run in their repos)

The following models are **not** run by `eval.run_libero_eval`; use their official repos and the commands below for the **same 4 suites** (spatial / object / goal / long) so results are comparable. Prefer **same episode counts and seed** (e.g. 5 episodes per task, seed 42) where supported.

### Reasoning-capable (4)

| Model | GitHub | Hugging Face | LIBERO eval |
|-------|--------|--------------|-------------|
| **DeepThinkVLA** | [OpenBMB/DeepThinkVLA](https://github.com/OpenBMB/DeepThinkVLA) | (checkpoints in repo) | Use repo’s LIBERO eval script and released checkpoints |
| **MolmoAct** | [allenai/molmoact](https://github.com/allenai/molmoact) | [allenai/MolmoAct-7B-D-LIBERO-*](https://huggingface.co/allenai?search=MolmoAct+LIBERO) | Run their `run_libero_eval.py` (or equivalent) for the 4 suites; same episodes/seed if configurable |
| **ECoT** | [MichalZawalski/embodied-CoT](https://github.com/MichalZawalski/embodied-CoT) | [Embodied-CoT/ecot-openvla-7b-bridge](https://huggingface.co/Embodied-CoT/ecot-openvla-7b-bridge) | Use OpenVLA repo LIBERO eval with checkpoint above; same episodes/seed |
| **InternVLA-M1** | [InternRobotics/InternVLA-M1](https://github.com/InternRobotics/InternVLA-M1) | [InternRobotics/InternVLA-M1-LIBERO-Spatial](https://huggingface.co/InternRobotics/InternVLA-M1-LIBERO-Spatial) | Use repo's LIBERO reproduction instructions |

### Strong baselines (4)

| Model | GitHub | Hugging Face | LIBERO eval |
|-------|--------|--------------|-------------|
| **OpenVLA** | [openvla/openvla](https://github.com/openvla/openvla) | [openvla/openvla-7b-finetuned-libero-*](https://huggingface.co/openvla?search=libero) | Use repo's LIBERO suite checkpoints and documented eval commands; match episodes/seed |
| **StarVLA** (LIBERO-4in1) | (see HF org) | [StarVLA](https://huggingface.co/StarVLA) | Use LIBERO eval flow and released LIBERO checkpoint; same episodes/seed if supported |
| **X-VLA** | [2toinf/X-VLA](https://github.com/2toinf/X-VLA), [huggingface/lerobot](https://github.com/huggingface/lerobot) | [lerobot/xvla-libero](https://huggingface.co/lerobot/xvla-libero) | Use LeRobot/xvla-libero eval; align with LeRobot-style or OpenVLA-style eval as documented |
| **LightVLA** | [LiAutoAD/LightVLA](https://github.com/LiAutoAD/LightVLA) | [TTJiang/models](https://huggingface.co/TTJiang/models) (search `lightvla`) | Follow LIBERO.md in repo; same episodes/seed where configurable |

---

## Eval style (LeRobot vs OpenVLA)

- **LeRobot-style:** Often single env API, episode loop in Python, success computed from predicates or return.
- **OpenVLA-style:** May use separate scripts or configs per suite, same core metric (task success).

For a **single unified report**, run each model with:

- **Suites:** spatial, object, goal, long (libero_10).
- **Episodes per task:** e.g. 5 (or 10 for test-only).
- **Seed:** e.g. 42.
- **Logging:** success per (suite, task, episode) and success rate per suite + overall.

Then you can aggregate the generated JSONs (or their equivalents) into one table.

---

## Output format (native runs)

`eval/results/<model>_<timestamp>.json`:

```json
{
  "model": "openpi_pi05",
  "suites": ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
  "task_ids": [0, 1, ..., 9],
  "episodes_per_task": 5,
  "seed": 42,
  "per_suite": {
    "libero_spatial": {
      "0": [{"episode_idx": 0, "success": true, "num_steps": 150}, ...],
      ...
    },
    ...
  },
  "summary": {
    "libero_spatial": {"success": 45, "episodes": 50, "success_rate": 0.9},
    ...
    "per_task": {
      "libero_spatial": {"0": {"success": 4, "episodes": 5, "success_rate": 0.8}, "1": {...}, ...},
      ...
    },
    "overall": {"success": 180, "episodes": 200, "success_rate": 0.9}
  }
}
```

Use this format as a template when mapping external model outputs into the same structure for comparison.

---

## Eval all 10 models in parallel (4 GPUs, CPU workers)

The **10 models** are: **openpi_pi0**, **openpi_pi05**, **openvla**, **xvla**, **molmoact**, **deepthinkvla**, **ecot**, **internvla_m1**, **starvla**, **lightvla**. StarVLA runs via `lerobot-eval` with checkpoint [StarVLA/Qwen3-VL-PI-LIBERO-4in1](https://huggingface.co/StarVLA/Qwen3-VL-PI-LIBERO-4in1) (no repo needed).

From `agent_attack_framework`, run all VLAs on LIBERO 4 suites with work distributed across 4 GPUs and parallel MuJoCo/LIBERO envs on CPU:

```bash
# With conda env "vast" (recommended): activate then run
conda activate vast
python -m eval.run_all_libero_evals_parallel --gpus 0,1,2,3 --cpu_workers 4
# Or use the wrapper script (activates vast if conda is available):
./eval/run_all_libero_evals_vast.sh

# All 10 models, 4 GPUs, 4 parallel env threads per native model (without conda)
python -m eval.run_all_libero_evals_parallel --gpus 0,1,2,3 --cpu_workers 4

# Subset of models
python -m eval.run_all_libero_evals_parallel --gpus 0,1,2,3 --models openpi_pi05,openpi_pi0,openvla,xvla

# External models: optional --repo_paths; if omitted, repos are auto-cloned to .cache/repos/<model> (requires git). X-VLA needs no repo (policy from HF).
python -m eval.run_all_libero_evals_parallel --gpus 0,1,2,3 --repo_paths openvla=/path/to/openvla,molmoact=/path/to/molmoact
```

- **Repos directory:** Put cloned repos in **`agent_attack_framework/repos/<model_id>`** (e.g. `repos/openvla`, `repos/molmoact`) and the eval will use them with no `--repo_path` or `--repo_paths`. For ECoT you can use `repos/ecot` or `repos/openvla` (same OpenVLA repo).
- **Auto-clone:** If `--repo_paths` is not set and a repo is not in `repos/<model_id>`, OpenVLA, MolmoAct, DeepThinkVLA, ECoT, InternVLA-M1, and LightVLA are cloned from GitHub into `.cache/repos/<model_id>`. **X-VLA** and **StarVLA** use `lerobot-eval` and pull the policy from HuggingFace (no repo needed).
- **Progress:** Each model logs "Running &lt;model&gt; on GPU &lt;id&gt;..." so long-running native evals don't appear stuck.
- **GPU distribution:** Models are assigned 2–3 per GPU (configurable with `--models_per_gpu`); each GPU runs its models sequentially.
- **CPU parallelism:** For native models (openpi_pi0, openpi_pi05), `--cpu_workers N` runs N LIBERO/MuJoCo envs in parallel (thread pool); model inference is serialized so CPU cores are used for stepping.
- **Output:** Per-model result JSONs in `eval/results/`, plus an aggregated `all_models_metrics_<timestamp>.json` with per-suite and per-task breakdown. The console prints an overall + per-suite table and a per-task breakdown by category (each suite with success rate per task ID).

### Why you might see errors

| Message | Cause | Fix |
|--------|--------|-----|
| **Skipped: no repo** (other external) | An external model has no auto-clone URL and no `--repo_paths` was given. | Provide `--repo_paths <model>=/path` for that model, or add a cloneable repo in code. |
| **No result JSON found** (or **done -> None** for an external model) | The external eval subprocess ran but did not write the expected `&lt;model&gt;_external_&lt;ts&gt;.json` under `eval/results/`. Often the repo’s eval command failed (missing deps, wrong script path, or timeout). | Check `eval/results/&lt;model&gt;_*.log` for the subprocess stdout/stderr. Install the model’s dependencies and ensure its eval script runs when you `cd` into the cloned repo. |
| **Traceback … No module named 'jax'** (openpi_pi0 / openpi_pi05) | Native OpenPI models need JAX and the OpenPI package. | Install JAX and OpenPI (see INSTALL.md), or run only external models with `--models openvla,xvla,molmoact,...`. |

---

## Testing all VLAs

From `agent_attack_framework`:

```bash
python -m eval.test_all_vlas
# or
pytest eval/test_all_vlas.py -v
```

This checks that every model (native + external) has a config, that `get_commands()` returns valid commands for external models, and that `run_libero_eval --model <id> --print_cmd` exits 0 for all 10. Native model load is skipped if OpenPI/JAX is not installed.
