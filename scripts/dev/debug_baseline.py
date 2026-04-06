"""Debug: run one baseline episode using BOTH the official and our preprocessing,
with both seed=7 (known-good) and seed=42 (used in training), to isolate why
baselines fail during training.
"""
import sys, os, collections
import numpy as np

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.45"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "openpi/src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "openpi/packages/openpi-client/src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "libero_rollouts"))

from openpi_client import image_tools
from pi05_libero_model import Pi05LiberoModel, preprocess_image, build_libero_state
from libero_utils import create_libero_env, reset_env

TASKS = [
    ("libero_goal", 0),
    ("libero_goal", 3),
    ("libero_spatial", 3),
    ("libero_object", 3),
]


def official_preprocess(img, size=224):
    img = np.ascontiguousarray(img[::-1, ::-1])
    img = image_tools.resize_with_pad(img, size, size)
    img = image_tools.convert_to_uint8(img)
    return img


def run_episode(model, env, init_states, ep_idx, instruction, max_steps,
                replan, use_official_preprocess=False, verbose=False):
    obs = reset_env(env, init_states, ep_idx)
    raw_env = env.env if hasattr(env, "env") else env

    model.set_language(instruction)
    action_plan = collections.deque()
    check_history = []

    for step in range(max_steps):
        if getattr(raw_env, "done", False):
            if verbose:
                ts = getattr(raw_env, "timestep", "?")
                print(f"  Step {step}: env.done (timestep={ts})")
            break

        if not action_plan:
            if use_official_preprocess:
                av = official_preprocess(obs["agentview_image"])
                wr = official_preprocess(obs["robot0_eye_in_hand_image"])
            else:
                av = preprocess_image(obs["agentview_image"])
                wr = preprocess_image(obs["robot0_eye_in_hand_image"])
            state = build_libero_state(obs)
            actions = model.predict(av, wr, state)
            action_plan.extend(actions[:replan])

        action = action_plan.popleft()
        obs, reward, done, info = env.step(action.tolist())

        if hasattr(raw_env, "_check_success"):
            cs = raw_env._check_success()
            check_history.append(cs)
            if verbose and cs:
                print(f"  Step {step}: _check_success=True, done={done}")

        if done:
            return True, step, check_history

    final_cs = (hasattr(raw_env, "_check_success")
                and raw_env._check_success())
    return final_cs, max_steps, check_history


def main():
    model = Pi05LiberoModel(train_config_name="pi05_libero")
    model.set_language("warmup")
    _ = model.predict(
        np.zeros((224, 224, 3), dtype=np.uint8),
        np.zeros((224, 224, 3), dtype=np.uint8),
        np.zeros(8, dtype=np.float32),
    )
    print("Model loaded and warmed up.\n")

    for suite, task_id in TASKS:
        env0, _, desc = create_libero_env(suite, task_id, 7)
        env0.close()
        print("=" * 70)
        print(f"TASK: {suite} / task {task_id}: {desc}")
        print("=" * 70)

        for seed in [7, 42]:
            for pname, use_off in [("official_PIL", True),
                                   ("cv2_resize", False)]:
                env, inits, d = create_libero_env(suite, task_id, seed)
                ok, steps, cs = run_episode(
                    model, env, inits, 0, d,
                    max_steps=400, replan=5,
                    use_official_preprocess=use_off)
                n_true = sum(1 for x in cs if x)
                tag = "PASS@%d" % steps if ok else "FAIL"
                print(f"  seed={seed}  {pname:14s}  {tag:12s}  "
                      f"cs_true={n_true}/{len(cs)}")
                env.close()

        env, inits, d = create_libero_env(suite, task_id, 42)
        raw = env.env if hasattr(env, "env") else env
        ok, steps, cs = run_episode(
            model, env, inits, 0, d,
            max_steps=1000, replan=5,
            use_official_preprocess=True, verbose=True)
        n_true = sum(1 for x in cs if x)
        tag = "PASS@%d" % steps if ok else "FAIL"
        hz = getattr(raw, "horizon", "?")
        print(f"  seed=42  max1000  hz={hz}  {tag}  "
              f"cs_true={n_true}/{len(cs)}")
        env.close()
        print()


if __name__ == "__main__":
    main()
