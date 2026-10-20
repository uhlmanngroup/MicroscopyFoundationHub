# Missing paper cells: 20 October 2026

Eight cells, five explicit seeds each (training seeds 1–5; split seeds 101–105):

- DINOv2-L DeepBacs triplet: frozen, LoRA, full fine-tuning
- OpenCLIP ViT-L-14 DeepBacs: LoRA on S. aureus, E. coli, B. subtilis, and their triplet
- DINOv2-L VNC/Drosophila: LoRA, replacing the earlier three-run estimate

The training outputs and summary are written under `/home/cfuste/scratch/DINO-LoRA`.
The summary checks the seeds, tuning flags, backbone variants, and five evaluated
metrics per cell. It reports pooled and per-species IoU for the DeepBacs triplets.

From the cluster checkout `/home/cfuste/GitHub/DINO-EM-PEFT`, run:

```bash
bash slurm/missing_20261020/submit_missing.sh
```

The command prints eight array job IDs and one dependent summary job ID. After
the summary job finishes, use:

```bash
cat /home/cfuste/scratch/DINO-LoRA/summaries/2026-10-20_missing_paper/status.json
cat /home/cfuste/scratch/DINO-LoRA/summaries/2026-10-20_missing_paper/foreground_iou_mean_std.csv
```

`status.json` is complete only when all 40 runs have valid metrics. The summary
job runs even when an array task fails and records the missing runs in `problems`.

To use a new noncolliding run tag, set `RUN_TAG=YYYY-MM-DD_suffix` when calling
`submit_missing.sh`; use the same tag when looking under `summaries/`.
