# Configs
All parameters live in YAML under `configs/`. CLI flags and sbatch env vars only override config values.

Many configs include absolute paths from the author's machines (for example `/Users/cfuste/...` or `/home/cfuste/...`). Keep them if you use the same layout, otherwise update them for your environment.

Directory map:
- `configs/backbones/`: backbone-only examples (DINOv2/DINOv3/OpenCLIP)
- `configs/cluster/`: cluster runs and sbatch defaults
- `configs/mac/`: local Mac runs
- `configs/environments/`: minimal conda environment YAMLs

Config notes:
- Set `use_lora: false` + `full_finetune: true` for full fine-tuning (no LoRA).
- Set `data_augmentation: true` to enable train-only online EM augmentation (`flip -> random shift`).
- Use `data_augmentation_prob` (default `0.5`) to control per-sample online augmentation probability.
- Keep `data_augmentation: false` for no train-time augmentation.
- Set `clahe_norm: true` to enable subtle CLAHE preprocessing on all splits before ImageNet normalization.
