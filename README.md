# DINO-EM-PEFT
Parameter-efficient fine-tuning (PEFT) of DINOv2 ViT for electron microscopy (EM) segmentation using LoRA.

## TL;DR

```bash
# 0) create env & install
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 1) compose datasets into one train/test (runs with no args; edit paths in the script header if needed)
python scripts/compose_em_datasets.py

# 2) train (Mac / local)
python scripts/train_em_seg.py --cfg config/lucchi_lora_mac.yaml

# 3) train (CUDA / cluster, Lucchi++)
python scripts/train_em_seg.py --cfg config/lucchi_lora_cluster.yaml
```

## What's here
* **Backbone**: DINOv2 ViT (`vits14`/`vitb14` etc.) via `torch.hub`.
* **PEFT**: LoRA injected into attention `qkv` and `proj` linear layers; backbone params frozen; only LoRA + segmentation head train.
* **Head**: simple 1×1 conv projection → upsample to image size.
* **Datasets**: utility to compose Drosophila + Lucchi++ into a unified layout.
* **Devices**: `device: "auto"` picks cuda → mps → cpu.

## Datasets
Lucchi, A., Smith, K., Achanta, R., Knott, G., & Fua, P. (2011). Supervoxel-based segmentation of mitochondria in em image stacks with learned shape features. IEEE transactions on medical imaging, 31(2), 474-486. Download [here](https://casser.io/connectomics).

Casser, V., Kang, K., Pfister, H., & Haehn, D. (2020, September). Fast mitochondria detection for connectomics. In Medical Imaging with Deep Learning (pp. 111-120). PMLR. Download [here](https://github.com/unidesigner/groundtruth-drosophila-vnc/tree/master).

For usability purposes, the two dataset are composed into:
````
<BASE>/composed-dinopeft/
  train/images, train/masks
  test/images,  test/masks
  mapping.csv
```
With an 85% split for the Casser et al. dataset. 

To do so, download the original datasets and run:
```bash
python scripts/compose_em_datasets.py
```

## Acknowledgements
This project stands on the shoulders of excellent open-source work and research. We’re grateful to the authors and maintainers of the following projects and papers:

- **DINOv2 (Meta AI / Facebook Research)**  
  We use DINOv2 Vision Transformers and public pretraining weights (loaded via `torch.hub`) as our frozen backbone. DINOv2 provides strong, general-purpose visual representations that we adapt to electron microscopy via parameter-efficient fine-tuning (PEFT).  
  Repo: <https://github.com/facebookresearch/dinov2>

- **RobvanGastel/dinov3-finetune**  
  This repository informed practical design choices for LoRA-based finetuning of DINOv2/DINOv3 encoders: which linear layers to target (e.g., attention `qkv` and `proj`), how to organize training/evaluation code, and how to integrate PEFT cleanly around a frozen backbone.  
  Repo: <https://github.com/RobvanGastel/dinov3-finetune>

- **samar-khanna/ExPLoRA**  
  ExPLoRA provides a strong reference for adapting Vision Transformers under domain shift via extended pre-training. While our current baseline is supervised LoRA on DINOv2, ExPLoRA guides our roadmap toward semi/self-supervised adaptation on unlabeled EM volumes and domain-shifted pretraining strategies.  
  Repo: <https://github.com/samar-khanna/ExPLoRA>

- **DINOSim (Electron Microscopy zero-shot evaluation)**  
  DINOSim motivates our evaluation focus: it investigates zero-shot detection/segmentation on EM imagery using DINO features and highlights the domain gap for microscopy. We build on that insight by demonstrating how PEFT (LoRA) improves downstream EM segmentation compared to zero-shot.  
  Project/Paper: *DINOSim: Zero-Shot Object Detection and Semantic Segmentation on Electron Microscopy Images.*

**Licensing note:**  
Please review and respect the licenses of upstream repositories (DINOv2, `dinov3-finetune`, ExPLoRA) and any datasets you use. Their terms apply to model weights, code, and data redistributed or fine-tuned within this project.
