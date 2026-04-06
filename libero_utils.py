"""Shared LIBERO environment and VLA rollout utilities.

Extracted from eval_attack_vla.py, eval_replay_attack.py, and baseline eval
scripts to eliminate duplication across the evaluation codebase.
"""

from __future__ import annotations

import threading

import numpy as np

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def parse_task_ids(s):
    """Parse a comma-separated list of task IDs, supporting ranges (e.g. '7-9')."""
    ids = []
    for part in s.replace(" ", "").split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    return sorted(set(ids))


def create_libero_env(task_suite_name, task_id, seed, resolution=256):
    """Create a LIBERO environment for the given task suite and task ID."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import pathlib

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    task_bddl_file = str(
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, initial_states, task.language


def reset_env(env, initial_states, episode_idx=0):
    """Reset a LIBERO environment to a deterministic initial state."""
    env.reset()
    init_state = initial_states[episode_idx % len(initial_states)]
    obs = env.set_init_state(init_state)
    for _ in range(10):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    return obs


def make_generic_policy_fn(vla_model, instruction, replan_steps=5, is_jax=False,
                           vla_lock=None, base_env=None):
    """Create policy_fn(obs, instr) -> (action, reasoning) for any VLA wrapper.

    For JAX models (Pi0/Pi0.5): uses preprocess_image/build_libero_state.
    For PyTorch models: uses the wrapper's predict method directly.
    For obs-predict models (X-VLA): uses predict_from_obs with full obs dict.
    """
    action_buffer = []
    current_instruction = instruction
    lock = vla_lock or threading.Lock()
    _use_obs = getattr(vla_model, 'use_obs_predict', False)

    from libero_rollouts.pi05_libero_model import preprocess_image, build_libero_state

    def policy_fn(obs, instr):
        nonlocal action_buffer, current_instruction
        if instr != current_instruction:
            current_instruction = instr
        if not action_buffer:
            with lock:
                vla_model.set_language(current_instruction)
                if _use_obs:
                    enriched = dict(obs)
                    if base_env is not None and hasattr(base_env, 'robots'):
                        ctrl = base_env.robots[0].controller
                        enriched['_ctrl_ee_pos'] = ctrl.ee_pos.copy()
                        enriched['_ctrl_ee_ori_mat'] = ctrl.ee_ori_mat.copy()
                    actions = vla_model.predict_from_obs(enriched)
                else:
                    agentview = preprocess_image(obs["agentview_image"])
                    wrist = preprocess_image(obs["robot0_eye_in_hand_image"])
                    state = build_libero_state(obs)
                    actions = vla_model.predict(agentview, wrist, state)
            action_buffer.extend(actions[:replan_steps].tolist())
        action = np.array(action_buffer.pop(0), dtype=np.float64)
        return action, ""

    return policy_fn


def run_vla_episode(env, initial_states, vla_model, instruction, episode_idx,
                    max_steps, replan_steps, is_jax=False, vla_lock=None,
                    observation_override=None, fast=False):
    """Run one VLA episode and return a VLARolloutInfo.

    Supports both observation overrides (for attack evaluation) and fast mode
    (skipping predicate collection / scene snapshots for replay evaluation).
    """
    from rwd_func.rwd import collect_libero_rollout_info

    obs = reset_env(env, initial_states, episode_idx)
    if observation_override is not None:
        obs = dict(obs)
        obs["agentview_image"] = observation_override

    base_env = env.env if hasattr(env, "env") else env
    if getattr(vla_model, 'uses_absolute_actions', False):
        for robot in base_env.robots:
            robot.controller.use_delta = False

    policy_fn = make_generic_policy_fn(
        vla_model, instruction, replan_steps=replan_steps,
        is_jax=is_jax, vla_lock=vla_lock, base_env=base_env,
    )
    return collect_libero_rollout_info(
        env=base_env,
        policy_fn=policy_fn,
        instruction=instruction,
        observation=obs,
        max_steps=max_steps,
        collect_predicates=not fast,
        scene_snapshot_interval=0 if fast else 25,
        store_observations=not fast,
    )
