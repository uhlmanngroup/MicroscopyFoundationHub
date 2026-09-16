# Configs

Every experiment parameter lives in YAML here. CLI flags and sbatch environment
variables only override config values, never replace the file.

All configs are loaded through `dino_peft.config.load_config()`, which does three
things before the config reaches any script.

## 1. Machine paths live in one file

`paths.yaml` holds the handful of absolute paths that change per machine. Configs
refer to them with `${...}` rather than hardcoding:

```yaml
train_img_dir: "${data_root}/EM/Lucchi++/Train_In"
results_root: "${scratch_root}/dinov3/seg"
```

| key | what it points at |
|---|---|
| `data_root` | datasets (`EM/`, `deepbacs/`, `monusac_448/`, `domain_shift/`) |
| `results_root` | run outputs on persistent storage |
| `scratch_root` | run outputs on scratch, where recent runs write |
| `models_root` | downloaded backbone checkpoints |
| `dinov3_repo` | local clone of facebookresearch/dinov3 |

To run elsewhere, edit `paths.yaml` — or override any entry from the environment
without touching the file, which is how the same config runs on the cluster and
on a laptop:

```bash
export DINO_PEFT_DATA_ROOT=/Users/me/Documents/Data
export DINO_PEFT_SCRATCH_ROOT=/Users/me/Documents/Results
```

An unknown `${placeholder}` is an error, so a typo fails at load rather than
surfacing later as a missing directory.

## 2. Shared values live in `defaults.yaml`

A training config opts in with `extends: defaults` and then only states what makes
it different. Anything the config sets always wins.

```yaml
extends: defaults

experiment_id: "2026-09-08_deepbacs_subtilis_dinov3_lora_seg"
train_img_dir: "${data_root}/deepbacs/single/subtilis/train/images"
use_lora: true
```

Configs that do not train — feature extraction, PCA, OOD, domain analysis — omit
`extends` and receive no defaults.

## 3. Seeds are never implicit

`defaults.yaml` deliberately sets no `seed`. A training config without one still
falls back to seed 0, but now prints a loud warning, because silently seeding
every repetition identically once produced a set of "repeats" with zero variance.
Sweep scripts pass `--seed` per repetition.

## Directory map

- `defaults.yaml` — shared training values
- `paths.yaml` — machine-specific paths
- `backbones/` — one example backbone block per family (DINOv2 / DINOv3 / OpenCLIP)
- `cluster/EM/`, `cluster/deepbacs/`, `cluster/monusac/`, `cluster/domain_shift/`
- `mac/` — local runs

## Key reference

- `modality`: `em` (default), `deepbacs`, or `monusac`. Also selects the results
  subdirectory.
- `use_lora` / `full_finetune`: the adaptation regime. Frozen head-only is both
  `false`; LoRA is `use_lora: true`; end-to-end is `full_finetune: true`.
- `center_crop_size`: paired image/mask crop for `deepbacs` and `monusac`
  (default 448). The old name `deepbacs_center_crop_size` is still accepted.
- `data_augmentation` / `data_augmentation_prob`: train-only online
  `flip -> random shift`.
- `clahe_norm`: subtle CLAHE on all splits before ImageNet normalisation.
- For `deepbacs` and `monusac`, the pipeline forces `img_size.mode: native`.

Run outputs land in `<results_root>/<modality>/<task_type>/<experiment_id>/`.
