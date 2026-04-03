# FPO++ Code Release

[[project page](https://hongsukchoi.github.io/fpo-control/)] [[arxiv](https://arxiv.org/abs/2602.02481)]

This repository contains experiments for FPO++ (Flow Policy Optimization), which have been cleaned up and refactored for release.

<table><tr><td>
  <strong>Flow Policy Gradients for Robot Control.</strong><br />
  <small></small>Brent&nbsp;Yi<sup>12*</sup>, Hongsuk&nbsp;Choi<sup>12*</sup>, Himanshu&nbsp;Gaurav&nbsp;Singh<sup>12</sup>, Xiaoyu&nbsp;Huang<sup>12</sup>, Takara&nbsp;E.&nbsp;Truong<sup>13</sup>, Carmelo&nbsp;Sferrazza<sup>1</sup>, Yi&nbsp;Ma<sup>24</sup>, Rocky&nbsp;Duan<sup>1†</sup>, Pieter&nbsp;Abbeel<sup>12†</sup>, Guanya&nbsp;Shi<sup>15†</sup>, Karen&nbsp;Liu<sup>13†</sup>, and&nbsp;Angjoo&nbsp;Kanazawa<sup>12&dagger;</sup><small>

</td></tr>
</table>
<sup>1</sup><em>Amazon FAR</em>, <sup>2</sup><em>UC Berkeley</em>, <sup>3</sup><em>Stanford</em>, <sup>4</sup><em>HKU</em>, <sup>5</sup><em>CMU</em>, <sup>*</sup><em>Equal Contribution</em>, <sup>†</sup><em>Amazon FAR Team Co-lead</em>

## Directory Structure

- **[`isaaclab_experiments/`](isaaclab_experiments/)** — Isaac Lab experiments: velocity-conditioned locomotion (6+ robots). Includes training commands and expected training curves.
- **[`manipulation_experiments/`](manipulation_experiments/)** — Manipulation experiments: pretraining and fine-tuning across five manipulation tasks using FPO++, Vanilla FPO, and DPPO variants. Includes training commands and expected training curves.

Each directory is self-contained with its own `setup_env.sh`, `source_env.sh`, and README with setup/usage instructions.

## Setup

Each experiment directory has independent dependencies and conda environments. See the README in each directory for detailed setup instructions. In brief:

```bash
# Isaac Lab experiments
cd isaaclab_experiments
bash setup_env.sh        # one-time setup
source source_env.sh     # activate env (each session)

# Manipulation experiments
cd manipulation_experiments
bash setup_env.sh        # one-time setup
source source_env.sh     # activate env (each session)
```

## Git Submodules

This repository uses git submodules for Isaac Lab extension (under `isaaclab_experiments/`). After cloning, initialize them with:

```bash
git submodule update --init --recursive
```

This is also done automatically by `isaaclab_experiments/setup_env.sh`.

## Licenses

This repository includes and adapts code from the following third-party projects. Original license files and copyright headers are retained in all cases.

### Isaac Lab Experiments (`isaaclab_experiments/`)

| Project | License | Inclusion | What we use/adapt |
|---------|---------|-----------|-------------------|
| [IsaacLab](https://github.com/isaac-sim/IsaacLab) | BSD-3-Clause | Git submodule | Simulation framework, task definitions, robot assets |
| [rsl_rl](https://github.com/leggedrobotics/rsl_rl) | BSD-3-Clause | Adapted (not vendored) | Actor-critic modules, on-policy runner, rollout storage, normalizer |

The `isaaclab_fpo` package adapts code from rsl_rl and IsaacLab (VecEnv wrapper, config dataclasses, training/evaluation scripts). Original copyright headers are retained in all adapted files.

### Manipulation Experiments (`manipulation_experiments/`)

| Project | License | Inclusion | What we use/adapt |
|---------|---------|-----------|-------------------|
| [LeRobot](https://github.com/huggingface/lerobot) | Apache-2.0 | Vendored under `thirdparty/lerobot/` (minor modifications, see `CHANGES.txt`) | Policy architectures and training utilities |
| [robosuite](https://github.com/ARISE-Initiative/robosuite) | MIT | Vendored under `thirdparty/robosuite/` (minor modifications, see `CHANGES.txt`) | Simulation environments for manipulation tasks |
| [DPPO](https://github.com/irom-lab/dppo) | MIT | Adapted file (`src/vit.py`) | Vision Transformer encoder |
| [DexMimicGen](https://github.com/NVlabs/dexmimicgen) | NVIDIA Source Code License (non-commercial) | Cloned at setup time | Dexterous manipulation data generation |
