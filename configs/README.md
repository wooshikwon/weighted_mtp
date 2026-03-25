# NTP Config Guide

## Config Structure

```
configs/
├── production/
│   ├── baseline.yaml         # Baseline NTP (uniform weights)
│   ├── taw.yaml              # TAW (Token Advantage Weighting, critic-based)
│   ├── random_matched.yaml   # Random-Matched (LogNormal, ablation control)
│   ├── shuffled.yaml         # Shuffled (critic weights permuted, ablation control)
│   └── critic_mlp.yaml       # Critic training (pairwise value learning)
└── local/
    ├── baseline_local.yaml   # Local testing (small model, MPS/CPU)
    └── critic_local.yaml     # Local critic testing
```

**Each config file is self-contained** (no defaults.yaml dependency).

## Execution Order

```
1. baseline.yaml      → Baseline SFT (uniform weights)
2. merge_sft_for_value.py → LoRA merge (creates critic base model)
3. critic_mlp.yaml    → Critic training (pairwise ranking)
4. taw.yaml           → TAW (uses critic checkpoint)
   random_matched.yaml → Random control (parallel with TAW)
   shuffled.yaml      → Shuffled control (parallel with TAW)
5. Evaluation         → python -m weighted_mtp evaluate ...
```

## Pipeline Stages

| Stage | Config | Pipeline | weight_mode |
|-------|--------|----------|-------------|
| Baseline SFT | `baseline.yaml` | `run_baseline.py` | `uniform` |
| TAW | `taw.yaml` | `run_baseline.py` | `critic` |
| Random-Matched | `random_matched.yaml` | `run_baseline.py` | `random` |
| Shuffled | `shuffled.yaml` | `run_baseline.py` | `shuffled` |
| Critic | `critic_mlp.yaml` | `run_critic.py` | N/A |

## CLI Usage

```bash
# Single GPU
python -m weighted_mtp train --config configs/production/baseline.yaml

# 4-GPU FSDP (production)
torchrun --nproc_per_node=4 -m weighted_mtp.pipelines.run_baseline \
  --config configs/production/baseline.yaml

# With overrides
torchrun --nproc_per_node=4 -m weighted_mtp.pipelines.run_baseline \
  --config configs/production/baseline.yaml \
  --override training.batch_size=8 \
  --override data_sampling.n_samples=1000

# Evaluation
python -m weighted_mtp evaluate \
  --checkpoint storage/checkpoints/ntp_baseline/checkpoint_final.pt \
  --dataset humaneval --eval-mode evalplus --num-samples 1
```

## Config Override

All pipelines support `--override` for hierarchical field overrides:

```bash
--override key.subkey=value
```

Format: `key=value` (nested: `key1.key2.key3=value`). Uses OmegaConf `from_dotlist()`.

## Key Fields

### training.weight_mode

Controls token weighting strategy in `run_baseline.py`:
- `uniform`: Standard NTP (all weights = 1.0)
- `critic`: TAW — requires `models.value_model.checkpoint_path`
- `random`: LogNormal random weights (distribution-matched control)
- `shuffled`: Critic weights with positions shuffled (signal control)

### models.value_model (TAW/Shuffled only)

```yaml
models:
  value_model:
    checkpoint_path: storage/checkpoints/critic/.../checkpoint_final.pt
    base_model_path: storage/models/llama3-8b-code-merged
```

### data_sampling

```yaml
data_sampling:
  seed: 42
  n_samples: 200000
  use_pairwise: false    # true for critic only
  difficulty_bins:
    all: [0, 25]
  difficulty_weights:
    all: 1.0
```
