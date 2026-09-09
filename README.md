# HALight

PyTorch implementation of HALight for multi-agent traffic signal control in CityFlow, with recurrent policies, attention-based neighbor communication, and lane-level missing observations.

## Overview

Each signalized intersection has its own actor, allowing different observation and action dimensions across intersections. Actors exchange learned information with neighboring intersections and use GRU-based policies. Training uses a centralized recurrent critic and sequential actor updates with a PPO-style clipped objective.

The current implementation:

- Samples one traffic flow JSON file per training episode from `-flow_dir`.
- Evaluates the final in-memory policy on every JSON flow file in that same directory after training.
- Supports lane-level observation dropout, controlled by `-obs_drop_prob`.
- Automatically derives the number of agents, observation dimensions, and action dimensions from the road network.

**Evaluation uses the training flow directory; it is not a held-out test split.** There is no separate test-only entry point or checkpoint save/load workflow in this release.

## Repository Structure

```text
.
├── README.md
├── train.sh                     # Batch training script; run from an activated environment
├── cityflow_env_wrapper.py      # CityFlow environment, observations, rewards, and metrics
├── HALight/
│   ├── main.py                  # CLI arguments and training entry point
│   ├── runner.py                # Rollouts, sequential updates, and final validation
│   ├── HALight_Agent.py         # Actor optimization
│   ├── policy_network.py        # Neighbor communication, attention, and recurrent actor
│   ├── critic_network.py        # Centralized recurrent critic
│   ├── actor_buffer.py
│   └── critic_buffer.py
├── models/
│   ├── rnn_layer.py
│   └── Actlayer.py
├── utils/
│   ├── get_neighbour.py         # Build neighbor lists from road-network connectivity
│   ├── comm_buffers.py          # Store communication features across a rollout
│   └── set_hete_graph.py        # Optional heterogeneous-graph utility
├── Nets/
│   ├── Gudang/Gudang.json
│   └── Xiasha/
│       ├── Xiasha.json
│       └── Xiasha.net.xml
└── flows/
    ├── Gudang_high/             # 100 JSON flow files
    ├── Gudang_moderate/         # 100 JSON flow files
    ├── Xiasha_high/             # 100 JSON flow files
    └── Xiasha_moderate/         # 100 JSON flow files
```

## Environment Setup

Linux is recommended. The previous experimental setup used **Python 3.8.20**. Exact dependency versions and external repository revisions are not included in this release, so the instructions below are a setup outline rather than a fully pinned environment.

The training code directly depends on **PyTorch**, **NumPy**, **tqdm**, **CityFlow**, and **HARL**. HARL also installs its own dependencies. `torch_geometric` is needed only when using `utils/set_hete_graph.py`; the main training path does not import that utility.

1. Create and activate an environment:

   ```bash
   conda create -n Cityflow python=3.8.20
   conda activate Cityflow
   ```

2. Install a PyTorch version compatible with your Python version and CUDA/CPU environment. Install NumPy and tqdm in the same environment:

   ```bash
   python -m pip install numpy tqdm
   ```

3. Install [CityFlow](https://cityflow.readthedocs.io/en/latest/install.html), including its native build dependencies. Install [HARL](https://github.com/PKU-MARL/HARL#installation) according to its official instructions. Keep external source checkouts outside this repository.

4. Verify the imports:

   ```bash
   python -c "import torch, numpy, tqdm, cityflow; from harl.common.valuenorm import ValueNorm; print('Imports OK'); print('CUDA:', torch.cuda.is_available())"
   ```

The code automatically selects CUDA when available, otherwise CPU. A `requirements.txt` was not supplied with the current code; export the actual working environment if you need exact reproducibility.

## Datasets

| Network | Signal-controlled intersections | Traffic scenarios | Flow files per scenario |
| --- | ---: | --- | ---: |
| Gudang | 16 | `Gudang_high`, `Gudang_moderate` | 100 |
| Xiasha | 9 | `Xiasha_high`, `Xiasha_moderate` | 100 |

Training uses CityFlow `.json` road networks and the matching scenario directory under `flows/`. The Xiasha SUMO `.net.xml` file is an additional asset, not an input to the current training entry point.

For custom networks, controlled intersection IDs must be numeric strings forming a contiguous range from `0` to `N-1`, because the runner uses these IDs to index tensors. Match the flow routes to the selected network.

## Run a Single Experiment

Run the following from the repository root:

```bash
conda activate Cityflow
python HALight/main.py \
  -net Nets/Gudang/Gudang.json \
  -flow_dir flows/Gudang_high \
  -obs_drop_prob 0.2 \
  -episode 10000 \
  -seed 42
```

Always specify `-net` and `-flow_dir`: the defaults in `main.py` reference Binjiang data, which is not included. Paths are resolved relative to the working directory. Do not use the old script's `../../Nets` and `../../flows` paths with the layout above.

A short execution check can use `-episode 1 -episode_length 10`. It will still run final validation over all 100 files in the selected directory, with the shortened episode length. This is only a smoke check, not an experimental result.

### Main Arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `-net` | Binjiang path; override required | CityFlow road-network JSON |
| `-flow_dir` | Binjiang path; override required | Directory of flow JSON files; not a single file |
| `-obs_drop_prob` | `0.0` | Per-lane missing-observation probability; use values in `[0, 1]` |
| `-episode` | `10000` | Number of training episodes |
| `-episode_length` | `360` | Control steps per episode |
| `-delta_time` | `10` | Simulation seconds per control step |
| `-seed` | `42` | Experiment seed |
| `-lr` | `0.00005` | Actor learning rate |
| `-critic_lr` | `0.0005` | Critic learning rate |
| `-gamma` | `0.95` | Discount factor |
| `-gae_lambda` | `0.9` | GAE parameter |
| `-ppo_epoch` / `-critic_epoch` | `10` / `10` | Actor/critic optimization epochs |
| `-clip_param` | `0.2` | PPO clipping parameter |
| `-entropy_coef` | `0.01` | Entropy coefficient |
| `-infor_dim` | `8` | Communicated information dimension |
| `-ht_dim` / `-attn_dim` | `16` / `16` | Encoded feature and attention dimensions |
| `-hidden_sizes` / `-critic_hidden_sizes` | `64` / `128` | Actor and critic hidden dimensions |

The default rollout length is `360 × 10 = 3,600` simulated seconds. See `python HALight/main.py -h` for the full argument list after installing dependencies. Several boolean arguments use `type=bool`, so passing the text `False` does not reliably disable them; change the defaults or fix their argument parsing when needed. The environment currently hardcodes replay saving to `False` regardless of `-save_replay`.

## Run the Full Experiment Grid

The root-level `train.sh` runs the same 20 configurations as the original experiment script:

- Networks: Gudang and Xiasha.
- Traffic demand: high and moderate.
- Observation dropout: `0.0`, `0.1`, `0.2`, `0.3`, and `0.4`.

```bash
conda activate Cityflow
bash train.sh
```

Experiments run sequentially. Each configuration trains for 10,000 episodes by default, then validates on its 100 flow files. The supplied script resolves paths from its own location and stops if an experiment fails.

## Observation Dropout and Reward

An intersection's observation contains incoming-lane queue counts, incoming-lane vehicle counts, and its current signal phase. For each episode, a lane mask is sampled independently for each intersection and remains fixed throughout that episode. A missing lane has both its queue count and vehicle count replaced by zero; the phase remains visible. New episodes, including validation episodes, receive newly sampled masks.

The reward is the negative sum of incoming-lane waiting vehicle counts, computed from simulator data without applying the observation mask.

## Training, Validation, and Outputs

Training samples flows with replacement using a local random generator seeded by `-seed`. After training, validation visits all flow JSON files in sorted order, uses deterministic actions, and does not update the model. Training environment seeds are `seed + episode`; validation environment seeds are `seed + 999999 + flow_index`, with indices starting at 1.

For example, Gudang/high with dropout `0.2` produces these files in the current working directory:

```text
HALight_RNN_Gudang_0.2_high_training_log.txt
HALight_RNN_Gudang_0.2_high_metric_step.txt
HALight_RNN_Gudang_0.2_high_validate_metric.txt
cityflow.config
```

- `training_log.txt`: total reward per training episode; appended across runs.
- `metric_step.txt`: validation-only step metrics; cleared when validation starts.
- `validate_metric.txt`: one row per validation flow plus mean and population standard deviation across flows; overwritten when validation starts.
- `cityflow.config`: generated simulator configuration; overwritten when the environment is recreated.

Reported metrics include throughput, average queue length, average travel time, average waiting time, and total reward. Average queue length is normalized per intersection and control step, not per lane. Travel time comes from the CityFlow API. The current waiting-time calculation averages only over finished vehicles that have a recorded waiting event; finished vehicles with no such event are excluded.

Output names do not include the seed or a timestamp. Preserve previous results before rerunning the same configuration, and use separate working directories for concurrent runs to avoid configuration and output collisions.

## Reproducibility Notes

The release does not include pretrained weights, automatic checkpoint saving, resume support, or a separate held-out evaluation directory. Validation uses the final in-memory model.

The runner seeds PyTorch in `run()`, after networks have already been constructed. Therefore, the current `-seed` setting alone does not guarantee identical initial weights across independent runs. For strict reproducibility, seed before constructing the runner and record Python/package versions, external dependency commits, hardware, and all experiment arguments.

## Acknowledgements

This implementation draws on the following projects:

- [HARL](https://github.com/PKU-MARL/HARL)
- [RegionLight](https://github.com/HankangGu/RegionLight)
- [CityFlow](https://github.com/cityflow-project/CityFlow)
