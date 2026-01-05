# shift-peft
Parameter-efficient fine-tuning (PEFT) of DINOv2/DINOv3 ViTs for electron microscopy (EM) **foreground segmentation** using **LoRA** adapters.

This is a cleaned-up fork of the original DINOv2 EM segmentation experiments. The DINOv2 pipeline remains intact, and a backbone adapter layer now enables DINOv3.

---

## TL;DR

```bash
# 0) create env & install package (local example)
python -m venv .venv
source .venv/bin/activate
pip install -e .

# 1) train (local)
python scripts/train_em_seg.py --cfg configs/mac/lucchi_lora_mac.yaml

# 2) eval (optional)
python scripts/eval_em_seg.py --cfg configs/mac/lucchi_lora_mac.yaml

# 3) summarize + plots
python scripts/analysis/summarize_seg_results.py --root /path/to/results/seg
python scripts/analysis/plot_seg_summary.py --summary-dir /path/to/results/seg/summary

# 4) features + PCA/UMAP
python scripts/extract_features.py --cfg configs/mac/em_unsupervised_features_mac.yaml
python scripts/run_pca.py --cfg configs/mac/em_pca_mac.yaml

# 5) OOD detection
python scripts/ood_detection.py --cfg configs/mac/ood_detection_mac.yaml
```

Cluster usage stays the same via SLURM sbatch:

```bash
sbatch slurm/single_lucchi.sbatch configs/cluster/lucchi_cluster.yaml
```

---

## Install (minimal)
- Base install: `pip install -e .`
- Optional plotting deps (used by analysis scripts): `pandas`, `matplotlib`, `seaborn`, `plotly`
- Optional dim-reduction deps: `umap-learn`
- Optional OOD extras: `scikit-learn`

If you prefer conda, see `configs/environments/` for minimal environment YAMLs.

---

## Backbone Selection (DINOv2 vs DINOv3)
Backbone choice is config-driven via a `backbone` block. See examples under `configs/backbones/`.

Example (DINOv2):
```yaml
backbone:
  name: dinov2
  variant: small
  load_backend: torchhub
  weights: null
  repo_dir: null
  preprocess:
    preset: em
```

Example (DINOv3):
```yaml
backbone:
  name: dinov3
  variant: vits16
  load_backend: torchhub
  repo_dir: /path/to/facebookresearch/dinov3
  weights: /path/to/dinov3_vits16_pretrain.pth
  preprocess:
    preset: em
```

Legacy configs that only specify `dino_size` still work and default to DINOv2.

## How to Run

### Training / Evaluation (DINOv2 or DINOv3 segmentation)
- Train:
  ```bash
  python scripts/train_em_seg.py --cfg configs/mac/lucchi_lora_mac.yaml
  ```
- Eval (writes metrics + preview triptychs):
  ```bash
  python scripts/eval_em_seg.py --cfg configs/mac/lucchi_lora_mac.yaml
  ```

### Summary plots
- Aggregate runs (writes `summary.csv`, `run_metrics.csv`, `summary.json`):
  ```bash
  python scripts/analysis/summarize_seg_results.py --root /path/to/results/seg
  ```
- Plot summaries:
  ```bash
  python scripts/analysis/plot_seg_summary.py --summary-dir /path/to/results/seg/summary
  ```
- Fancy report plot (customizable, uses hardcoded paths):
  ```bash
  python scripts/analysis/fancy_plot.py
  ```

### PCA / UMAP
1) Extract features:
   ```bash
   python scripts/extract_features.py --cfg configs/mac/em_unsupervised_features_mac.yaml
   ```
2) Run PCA/UMAP:
   ```bash
   python scripts/run_pca.py --cfg configs/mac/em_pca_mac.yaml
   ```

### OOD detection
```bash
python scripts/ood_detection.py --cfg configs/mac/ood_detection_mac.yaml
```

### Domain gap analysis (FDD + linear separability)
```bash
python scripts/run_domain_analysis.py --cfg configs/mac/domain_analysis.yaml
```

---

## Smoke test (lightweight)
This checks that the backbone loads and runs a tiny forward pass:

```bash
python scripts/smoke_test.py --dino-size small --device cpu
```

For DINOv3 (weights required), provide a local path:
```bash
python scripts/smoke_test.py --backbone-name dinov3 --variant vits16 --weights /path/to/dinov3_vits16_pretrain.pth
```

Note: the first DINOv2 run may download weights via `torch.hub` if they are not cached.

## DINOv3 Weights
Access to DINOv3 weights requires an approved request. Once you receive the link, download the checkpoint and set `backbone.weights` in your config:

```bash
# Example (replace with your actual URL + filename)
wget -O /path/to/dinov3_vits16_pretrain.pth "<DINOv3_CHECKPOINT_URL>"
```

Then set:
```yaml
backbone:
  name: dinov3
  variant: vits16
  repo_dir: /path/to/facebookresearch/dinov3
  weights: /path/to/dinov3_vits16_pretrain.pth
```

Also update `img_size.patch_multiple` to 16 when using DINOv3 so resizing snaps to the correct stride.

---

## Results Layout
All runs share a common results layout driven by the YAML config. Each config must define:

- `experiment_id`: unique run name (e.g. `2025-11-20_A1_lucchi+droso_dinov2-base_lora-none_seg`)
- `results_root`: root directory for all outputs
- `task_type`: `seg`, `feats`, `ood-detection`, etc.

Outputs land at:

```
<results_root>/<task_type>/<experiment_id>/
  config_used.yaml
  run_info.txt
  metrics.json
  ckpts/
  figs/
```

`metrics.json` is updated by training/eval/analysis scripts via `update_metrics()`.

---

## Repository Structure
```
shift-peft/
  configs/
    mac/                 # Local configs
    cluster/             # Cluster configs
    backbones/           # Backbone config examples (DINOv2/DINOv3)
    environments/        # Minimal conda env YAMLs
  docs/
    media/               # README assets
  scripts/
    analysis/            # summarize/plot/fancy/OOD comparison
    data/                # dataset composition utilities
    utils/               # log helpers
    train_em_seg.py
    eval_em_seg.py
    extract_features.py
    run_pca.py
    run_domain_analysis.py
    ood_detection.py
    smoke_test.py
  slurm/                 # sbatch entrypoints
  src/                   # core library code
  README.md
  pyproject.toml
```

---

## Datasets
Lucchi, A., Smith, K., Achanta, R., Knott, G., & Fua, P. (2011). Supervoxel-based segmentation of mitochondria in em image stacks with learned shape features. IEEE transactions on medical imaging, 31(2), 474-486. Download [here](https://casser.io/connectomics).

Casser, V., Kang, K., Pfister, H., & Haehn, D. (2020, September). Fast mitochondria detection for connectomics. In Medical Imaging with Deep Learning (pp. 111-120). PMLR. Download [here](https://github.com/unidesigner/groundtruth-drosophila-vnc/tree/master).

For usability purposes, the two datasets are composed into:
```bash
<BASE>/composed-dinopeft/
  train/images, train/masks
  test/images,  test/masks
  mapping.csv
```

To build that layout, edit paths in `scripts/data/compose_em_datasets.py` and run:
```bash
python scripts/data/compose_em_datasets.py
```

---

## Future work
- Add OpenCLIP backbone support.
- Expand backbone coverage (additional checkpoints/variants, richer register-token analysis).
- Add new datasets/modalities (other EM, histology, etc.).

---

## Acknowledgements
This project stands on the shoulders of excellent open-source work and research. We’re grateful to the authors and maintainers of the following projects and papers:

- **DINOv2 (Meta AI / Facebook Research)**  
  We use DINOv2 Vision Transformers and public pretraining weights (loaded via `torch.hub`) as our frozen backbone. DINOv2 provides strong, general-purpose visual representations that we adapt to electron microscopy via parameter-efficient fine-tuning (PEFT).  
  Repo: <https://github.com/facebookresearch/dinov2>

- **FD-DINOv2 (Fréchet Distance with DINOv2 backbone)**  
  We adapt the idea of replacing Inception features in FID with DINOv2 features, as implemented in the FD-DINOv2 repository. In our case, we compute a FID-style Fréchet distance between Gaussian approximations of the DINOv2 feature distributions of two EM datasets (with and without LoRA) to quantify domain shift.  
  Repo: <https://github.com/justin4ai/FD-DINOv2>

- **Exposing Flaws of Generative Model Evaluation Metrics**  
  Stein et al. systematically study generative model evaluation metrics and show that relying on Inception-V3 can unfairly penalize diffusion models. They propose using self-supervised encoders such as DINOv2-ViT-L/14 as richer feature extractors for Fréchet-based scores, which motivates our choice to replace Inception with DINOv2 embeddings when measuring domain distances.  
  Paper: *Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models*, NeurIPS 2023.

- **DINOSim (Electron Microscopy zero-shot evaluation)**  
  DINOSim motivates our evaluation focus: it investigates zero-shot detection/segmentation on EM imagery using DINO features and highlights the domain gap for microscopy. We build on that insight by demonstrating how PEFT (LoRA) improves downstream EM segmentation compared to zero-shot.  
  Project/Paper: *DINOSim: Zero-Shot Object Detection and Semantic Segmentation on Electron Microscopy Images.*

**Licensing note:**  
Please review and respect the licenses of upstream repositories (e.g. DINOv2) and any datasets you use. Their terms apply to model weights, code, and data redistributed or fine-tuned within this project.
