"""
Eval config for each external model on LIBERO 4 suites (spatial, object, goal, long).

Each entry defines how to run that model's official eval so results are comparable
(same suites, episodes_per_task, seed). REPO_PATH is filled by --repo_path or env.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Defaults for comparable evals (match run_libero_eval.py)
DEFAULT_EPISODES_PER_TASK = 5
DEFAULT_SEED = 42
LIBERO_4_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


@dataclass
class ExternalEvalConfig:
    """Config to run one external model's LIBERO eval."""
    id: str
    name: str
    github: str
    hf_org_or_model: str
    suites: List[str] = field(default_factory=lambda: list(LIBERO_4_SUITES))
    episodes_per_task: int = DEFAULT_EPISODES_PER_TASK
    seed: int = DEFAULT_SEED
    # Command template: {repo_path}, {suite}, {episodes_per_task}, {seed}, {checkpoint}
    # One command per suite, or a single command that runs all suites (model-dependent).
    command_per_suite: Optional[str] = None  # e.g. "python run_libero_eval.py --task_suite_name {suite} ..."
    command_all_suites: Optional[str] = None  # e.g. "python eval_libero.py --suites spatial,object,goal,long ..."
    env: Dict[str, str] = field(default_factory=dict)
    notes: str = ""


# Registry of external models: id -> ExternalEvalConfig
EXTERNAL_MODELS: Dict[str, ExternalEvalConfig] = {
    "openvla": ExternalEvalConfig(
        id="openvla",
        name="OpenVLA",
        github="https://github.com/openvla/openvla",
        hf_org_or_model="openvla",
        command_per_suite="{repo_path}/run_libero_eval_openvla.sh {suite} {episodes_per_task} {seed} {checkpoint}",
        notes="Run from openvla repo: experiments/robot/libero/run_libero_eval.py per suite.",
    ),
    "xvla": ExternalEvalConfig(
        id="xvla",
        name="X-VLA (lerobot/xvla-libero)",
        github="https://github.com/huggingface/lerobot",
        hf_org_or_model="lerobot/xvla-libero",
        command_per_suite="lerobot-eval --policy.path=lerobot/xvla-libero --env.type=libero --env.task={suite} --eval.n_episodes={episodes_per_task} --eval.seed={seed}",
        notes="Requires: pip install 'lerobot[xvla]'. One command per suite.",
    ),
    "molmoact": ExternalEvalConfig(
        id="molmoact",
        name="MolmoAct",
        github="https://github.com/allenai/molmoact",
        hf_org_or_model="allenai/MolmoAct-7B-D-LIBERO-*",
        command_per_suite="cd {repo_path} && python run_libero_eval.py --suite {suite} --n_episodes {episodes_per_task} --seed {seed}",
        notes="Repo has per-suite checkpoints (Spatial, Object, Goal, Long). See README 5.2.",
    ),
    "deepthinkvla": ExternalEvalConfig(
        id="deepthinkvla",
        name="DeepThinkVLA",
        github="https://github.com/OpenBMB/DeepThinkVLA",
        hf_org_or_model="(checkpoints in repo)",
        command_all_suites="cd {repo_path} && python eval_libero.py --suites spatial,object,goal,long --n_episodes {episodes_per_task} --seed {seed}",
        notes="LIBERO eval in main repo or wadeKeith/DeepThinkVLA_libero_plus.",
    ),
    "ecot": ExternalEvalConfig(
        id="ecot",
        name="Embodied Chain-of-Thought (ECoT) for OpenVLA",
        github="https://github.com/MichalZawalski/embodied-CoT",
        hf_org_or_model="Embodied-CoT/ecot-openvla-7b-bridge",
        command_per_suite="",  # Not used; run.py get_commands uses _openvla_commands with ECoT checkpoint
        notes="Uses OpenVLA LIBERO eval stack with checkpoint Embodied-CoT/ecot-openvla-7b-bridge. Provide --repo_paths ecot=/path/to/openvla (OpenVLA repo).",
    ),
    "internvla_m1": ExternalEvalConfig(
        id="internvla_m1",
        name="InternVLA-M1",
        github="https://github.com/InternRobotics/InternVLA-M1",
        hf_org_or_model="InternRobotics/InternVLA-M1-LIBERO-Spatial",
        command_per_suite="cd {repo_path} && python eval_libero.py --suite {suite} --n_episodes {episodes_per_task} --seed {seed}",
        notes="Reproduce LIBERO instructions in repo; HF checkpoint per suite.",
    ),
    "starvla": ExternalEvalConfig(
        id="starvla",
        name="StarVLA (LIBERO-4in1)",
        github="(see HF)",
        hf_org_or_model="StarVLA/Qwen3-VL-PI-LIBERO-4in1",
        command_all_suites="cd {repo_path} && python run_libero_eval.py --suites spatial,object,goal,long",
        notes="HF checkpoint: StarVLA/Qwen3-VL-PI-LIBERO-4in1. Provide --repo_paths starvla=/path to StarVLA eval repo.",
    ),
    "lightvla": ExternalEvalConfig(
        id="lightvla",
        name="LightVLA",
        github="https://github.com/LiAutoAD/LightVLA",
        hf_org_or_model="TTJiang/models (search lightvla)",
        command_all_suites="cd {repo_path} && python eval_libero.py --suites spatial,object,goal,long",
        notes="See LIBERO.md in repo for eval instructions and HF caching.",
    ),
}


def get_external_config(model_id: str) -> Optional[ExternalEvalConfig]:
    return EXTERNAL_MODELS.get(model_id.lower().replace("-", "_"))


# Feasible model weights (Hugging Face IDs or official URLs) for reference / automation.
# See eval/README.md "Feasible model weights" for full table and notes.
MODEL_WEIGHTS: Dict[str, List[str]] = {
    "openpi_pi0": [
        "gs://openpi-assets/checkpoints/pi0_base",
        "lerobot/pi05_libero_base",
    ],
    "openpi_pi05": [
        "gs://openpi-assets/checkpoints/pi05_libero",
        "lerobot/pi05_libero_finetuned",
        "lerobot/pi05_libero_base",
    ],
    "openvla": [
        "openvla/openvla-7b",
        "openvla/openvla-7b-finetuned-libero-spatial",
        "openvla/openvla-7b-finetuned-libero-object",
        "openvla/openvla-7b-finetuned-libero-goal",
        "openvla/openvla-7b-finetuned-libero-10",
    ],
    "xvla": ["lerobot/xvla-libero", "lerobot/xvla-base"],
    "molmoact": [
        "allenai/MolmoAct-7B-D-0812",
        "allenai/MolmoAct-7B-D-LIBERO-Spatial-0812",
        "allenai/MolmoAct-7B-D-LIBERO-Object-0812",
        "allenai/MolmoAct-7B-D-LIBERO-Goal-0812",
        "allenai/MolmoAct-7B-D-LIBERO-Long-0812",
    ],
    "deepthinkvla": [
        "yinchenghust/deepthinkvla_libero_cot_rl",
    ],
    "ecot": [
        "Embodied-CoT/ecot-openvla-7b-bridge",
    ],
    "internvla_m1": [
        "InternRobotics/InternVLA-M1",
        "InternRobotics/InternVLA-M1-LIBERO-Spatial",
        "InternRobotics/InternVLA-M1-LIBERO-Long",
    ],
    "starvla": [
        "StarVLA/Qwen3-VL-PI-LIBERO-4in1",
    ],
    "lightvla": [
        "TTJiang/LightVLA-libero-spatial",
        "TTJiang/LightVLA-libero-object",
        "TTJiang/LightVLA-libero-goal",
        "TTJiang/LightVLA-libero-10",
    ],
}

# Hugging Face repo_ids to download for evaluation (used by eval/download_checkpoints.py).
# Only HF-hosted checkpoints; OpenPI GCS checkpoints are fetched by the OpenPI loader at runtime.
HF_CHECKPOINTS_FOR_EVAL: Dict[str, List[str]] = {
    "openpi_pi0": ["lerobot/pi05_libero_base"],
    "openpi_pi05": ["lerobot/pi05_libero_finetuned", "lerobot/pi05_libero_base"],
    "openvla": [
        "openvla/openvla-7b",
        "openvla/openvla-7b-finetuned-libero-spatial",
        "openvla/openvla-7b-finetuned-libero-object",
        "openvla/openvla-7b-finetuned-libero-goal",
        "openvla/openvla-7b-finetuned-libero-10",
    ],
    "xvla": ["lerobot/xvla-libero", "lerobot/xvla-base"],
    "molmoact": [
        "allenai/MolmoAct-7B-D-0812",
        "allenai/MolmoAct-7B-D-LIBERO-Spatial-0812",
        "allenai/MolmoAct-7B-D-LIBERO-Object-0812",
        "allenai/MolmoAct-7B-D-LIBERO-Goal-0812",
        "allenai/MolmoAct-7B-D-LIBERO-Long-0812",
    ],
    "deepthinkvla": ["yinchenghust/deepthinkvla_libero_cot_rl"],
    "ecot": ["Embodied-CoT/ecot-openvla-7b-bridge"],
    "internvla_m1": [
        "InternRobotics/InternVLA-M1",
        "InternRobotics/InternVLA-M1-LIBERO-Spatial",
        "InternRobotics/InternVLA-M1-LIBERO-Long",
    ],
    "starvla": ["StarVLA/Qwen3-VL-PI-LIBERO-4in1"],
    "lightvla": [
        "TTJiang/LightVLA-libero-spatial",
        "TTJiang/LightVLA-libero-object",
        "TTJiang/LightVLA-libero-goal",
        "TTJiang/LightVLA-libero-10",
    ],
}
