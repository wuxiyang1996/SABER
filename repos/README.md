# Model repos for LIBERO evaluation

Put cloned model repositories here so the eval uses them automatically (no `--repo_path` needed).

| Directory   | Repo to clone |
|------------|----------------|
| `openvla/` | https://github.com/openvla/openvla |
| `molmoact/` | https://github.com/allenai/molmoact |
| `deepthinkvla/` | https://github.com/OpenBMB/DeepThinkVLA |
| `ecot/` or `openvla/` | ECoT uses the OpenVLA repo (clone openvla once; use as `openvla` or `ecot`) |
| `internvla_m1/` | https://github.com/InternRobotics/InternVLA-M1 |
| `lightvla/` | https://github.com/LiAutoAD/LightVLA |

**X-VLA** and **StarVLA** do not need a repo (they use `lerobot-eval` with Hugging Face checkpoints).

Example after cloning:

```
repos/
  openvla/          # git clone https://github.com/openvla/openvla openvla
  molmoact/         # git clone https://github.com/allenai/molmoact molmoact
  deepthinkvla/     # git clone https://github.com/OpenBMB/DeepThinkVLA deepthinkvla
  internvla_m1/     # git clone https://github.com/InternRobotics/InternVLA-M1 internvla_m1
  lightvla/         # git clone https://github.com/LiAutoAD/LightVLA lightvla
```

For ECoT, use the same `openvla/` clone (eval looks for `repos/ecot` or `repos/openvla` when model is ecot).
