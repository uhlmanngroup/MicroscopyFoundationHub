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
results_root: "${scratch_root}/dinov3"
```

| key | what it points at |
|---|---|
| `data_root` | datasets (`EM/`, `deepbacs/`, `monusac_448/`, `domain_shift/`) |
| `results_root` | run outputs on persistent storage |
| `scratch_root` | run outputs on scratch, where recent runs write |
| `models_root` | downloaded backbone checkpoints |
| `ckpt_root` | trained run checkpoints (`best_model.pt`) |
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

`defaults.yaml` deliberately sets no `seed`. A training config without one is a
hard error from `SegTrainer`, because silently seeding every repetition at 0 once
produced a set of "repeats" with zero variance. Sweep scripts pass `--seed` and
`--split-seed` per repetition; the current sweeps use seeds 1-5 with split_seeds
101-105, and every backbone reuses those so the comparison is paired.

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

## 4. Where a run lands

Run outputs land in `<results_root>/<modality>/<task_type>/<experiment_id>/`,
which the configs use to spell one layout:

```
<backbone>/<modality>/<family>/<group>/<experiment_id>/
dinov2/em/seg/lucchi-lora/2025-01-01_A1_lucchi-only_dinov2-base_lora-r16_seg/
```

- **backbone** closes `results_root`: `"${results_root}/dinov2"`.
- **modality** is the imaging domain — `em`, `deepbacs`, `monusac`,
  `openimages`, or `multi` when a run merges several.
- **family** is what the run produces, and opens `task_type`: `seg` for
  segmentation, `feat-analysis` and `domain-analysis` for unsupervised feature
  work, `ood-detection`. Unsupervised results are not segmentation results, so
  they never share a directory.
- **group** closes `task_type`: `task_type: "seg/lucchi-lora"`. It names the
  dataset and tuning mode the repetitions of one experiment share.

A run that needs no group stops at the family (`task_type: "seg"`). DeepBacs
adds the combination between the two, mirroring the dataset tree:
`task_type: "seg/paired/coli-aureus-lora"`.
