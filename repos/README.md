# External model repos

Some VLA models require cloning their source repo for custom model classes
that are not available through HuggingFace `transformers` alone.

Clone the repos below into this directory. The eval wrappers and
`setup_vla_envs.sh` detect them automatically.

| Directory | Repo | Used by |
|---|---|---|
| `deepthinkvla/` | https://github.com/OpenBMB/DeepThinkVLA | DeepThinkVLA (custom PaliGemma model class) |
| `internvla_m1/` | https://github.com/InternRobotics/InternVLA-M1 | InternVLA-M1 (custom architecture) |

```bash
cd repos/
git clone https://github.com/OpenBMB/DeepThinkVLA deepthinkvla
git clone https://github.com/InternRobotics/InternVLA-M1 internvla_m1
```

Other paper models (Pi0.5, OpenVLA, ECoT, MolmoAct) load directly from
HuggingFace and do not need a local repo clone.
