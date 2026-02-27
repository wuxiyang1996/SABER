"""Standalone Pi0.5 baseline test on libero_90 — diagnose low success rate."""
import sys, os, collections, time, pathlib
import numpy as np

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.30"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "openpi/src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "openpi/packages/openpi-client/src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "libero_rollouts"))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from pi05_libero_model import Pi05LiberoModel, preprocess_image, build_libero_state

SUITE = "libero_90"
MAX_STEPS = 400
SEED = 7
TASK_IDS = list(range(90))  # all 90 tasks


def official_preprocess(img, size=224):
    """Matches the official openpi LIBERO evaluation exactly."""
    img = np.ascontiguousarray(img[::-1, ::-1])
    img = image_tools.resize_with_pad(img, size, size)
    img = image_tools.convert_to_uint8(img)
    return img


def run_episode(model, env, initial_states, ep_idx, instruction, max_steps, replan):
    env.reset()
    obs = env.set_init_state(initial_states[ep_idx])
    DUMMY = [0.0] * 6 + [-1.0]
    for _ in range(10):
        obs, _, _, _ = env.step(DUMMY)
    model.set_language(instruction)
    action_plan = collections.deque()
    for step in range(max_steps):
        if not action_plan:
            agentview = official_preprocess(obs["agentview_image"])
            wrist = official_preprocess(obs["robot0_eye_in_hand_image"])
            state = build_libero_state(obs)
            actions = model.predict(agentview, wrist, state)
            action_plan.extend(actions[:replan])
        action = action_plan.popleft()
        obs, reward, done, info = env.step(action.tolist())
        if done:
            return True, step
    return False, max_steps


def main():
    model = Pi05LiberoModel(train_config_name="pi05_libero")
    bd = benchmark.get_benchmark_dict()
    task_suite = bd[SUITE]()

    results = []
    t0 = time.time()
    for task_id in TASK_IDS:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        desc = task.language
        bddl = str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        env.seed(SEED)

        ok, steps = run_episode(model, env, initial_states, 0, desc, MAX_STEPS, replan=5)
        tag = f"PASS@{steps}" if ok else "FAIL"
        results.append((task_id, ok, steps, desc))
        elapsed = time.time() - t0
        print(f"[{task_id:2d}] {tag:12s} ({elapsed:.0f}s)  {desc}", flush=True)
        env.close()

    n = len(results)
    n_pass = sum(1 for _, ok, _, _ in results if ok)
    print(f"\n===== Phase 1: Single-trial scan ({SUITE}, {n} tasks) =====")
    print(f"  Success: {n_pass}/{n} = {n_pass/n*100:.1f}%")

    passed_ids = [tid for tid, ok, _, _ in results if ok]
    failed_ids = [tid for tid, ok, _, _ in results if not ok]
    print(f"  Passed task IDs: {passed_ids}")
    print(f"  Failed task IDs: {failed_ids}")

    # Phase 2: 3-trial reliability check on passed tasks
    print(f"\n===== Phase 2: Reliability check (3 trials on {len(passed_ids)} passed tasks) =====")
    reliable_ids = []
    for task_id in passed_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        desc = task.language
        bddl = str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
        wins = 0
        n_trials = min(3, len(initial_states))
        for trial in range(n_trials):
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
            env.seed(SEED)
            ok2, _ = run_episode(model, env, initial_states, trial, desc, MAX_STEPS, replan=5)
            if ok2:
                wins += 1
            env.close()
        rate = wins / n_trials
        tag = "RELIABLE" if rate >= 0.67 else "FRAGILE"
        if rate >= 0.67:
            reliable_ids.append(task_id)
        print(f"  task {task_id:2d}: {wins}/{n_trials} ({tag})  {desc}", flush=True)

    print(f"\n===== Final: reliable tasks (≥2/3 success) =====")
    print(f"  Count: {len(reliable_ids)}/{n}")
    print(f"  Task IDs: {reliable_ids}")
    # Save as JSON for easy loading
    import json
    outpath = os.path.join(os.path.dirname(__file__), "baseline_reliable_tasks.json")
    with open(outpath, "w") as f:
        json.dump({"suite": SUITE, "reliable_task_ids": reliable_ids,
                   "all_passed_ids": passed_ids, "total_tasks": n}, f, indent=2)
    print(f"  Saved to {outpath}")


if __name__ == "__main__":
    main()
