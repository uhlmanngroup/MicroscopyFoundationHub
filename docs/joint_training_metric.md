# Joint-training direction alignment

This analysis asks whether datasets require similar local foreground-versus-background directions in frozen pretrained feature space. It reuses the existing spatial-embedding cache and mask occupancies. It does not train or reload a backbone during analysis. The old barycenter, optimal-transport distances and derived risk scores have been retired from new reports.

## Cluster commands

Run from the repository checkout on the cluster using Bash:

```bash
# Check the jobs before submitting.
bash slurm/submit_joint_metric.sh --collection em --backbone dinov3 --metrics-only --dry-run

# Reuse the DINOv3 EM cache already extracted on the cluster.
bash slurm/submit_joint_metric.sh --collection em --backbone dinov3 --metrics-only

# Once other backbone caches exist, analyze all configured combinations.
bash slurm/submit_joint_metric.sh --collection all --backbone all --metrics-only

# If a cache is not present, omit --metrics-only to queue GPU extraction first.
bash slurm/submit_joint_metric.sh --collection em --backbone resnet50
```

The submitter queues CPU analysis and then a summary/index job; GPU extraction is queued only when `--metrics-only` is omitted. Existing cache directories remain under `/home/cfuste/scratch/DINO-LoRA/joint_metric/cache/`. Reports and logs go under `/home/cfuste/data/DINO-LoRA/joint_metric/`. EM source images are configured under `/home/cfuste/data/DINO-LoRA/datasets/EM/` using `joint_em_data_root` in `configs/paths.yaml`. Source images are used only for optional visual alignment checks; the numeric analysis uses cached embeddings and occupancies.

The same entry point accepts `--backbone dinov3`, `resnet50`, or `openclip`, and `--collection em` or `deepbacs`. The current config defaults to EM + DINOv3. `--sensitivity` repeats the analysis for seeds listed in the config sweep. Because directions use every usable local foreground patch, their alignment is seed-independent; seeds resample probe and display patches and change image splits. No backbone inference is repeated.

The existing `dino-peft` environment needs the project and `matplotlib` installed. The optional extra is `pip install -e '.[joint-metric]'`. `torch` and `numpy` are project dependencies. SLURM uses the interpreter from `slurm/lib/common.sh` (overridable with `PY`).

## Method

For each image separately, a patch is clear foreground when its mask occupancy is at least `tau_fg=0.8`, and clear background when occupancy is at most `tau_bg=0.2`. For every clear foreground patch, the analysis searches a radius-2 neighborhood on the feature grid (a 5×5 window, including diagonals) for clear background patches. If none exists, that foreground patch is skipped and counted. Otherwise it computes:

```text
d(f) = foreground embedding − mean(nearby clear-background embeddings)
```

Embeddings are L2-normalized first when `l2_normalize: true`. No intensity threshold or black-border mask is used. Local backgrounds never come from another image. This preserves the full-image preprocessing used during cache extraction while reducing the influence of Kasthuri's large exterior black areas.

Every usable foreground patch contributes to its image's mean direction. Images with local pairs receive equal weight in the dataset mean, so an image with many labeled objects cannot dominate. The mean dataset vector is normalized, and each pair of datasets is compared with the cosine of its normalized vectors. The report displays the symmetric pairwise matrix, each dataset's mean alignment `A_i`, and the mean of all unique pairs `A_collection`. Cosines range from −1 to +1; no arbitrary risk transformation or safe/unsafe threshold is applied.

`M_i` is the image-balanced mean length of local `d(f)` vectors. It describes local feature contrast and is **not** the compatibility score. New runs save all `d(f)` vectors, their dataset/image/token identities and local-background counts in `local_vectors.npz`, allowing aggregation choices to be revisited without feature extraction. The original cache remains the source for different neighborhood or occupancy settings.

## Separate linear-probe diagnostic

The probe is deliberately separate from the zero-training alignment measure. Each usable image supplies up to 32 clear foreground and 32 eligible local-background patch embeddings, sampled reproducibly. Dataset images are split 60% training, 20% validation and the remainder test, with at least one image in each part; images with no valid local pairs are excluded and counted. No patch from an image appears in more than one split.

One L2-regularized binary logistic probe is fit per dataset; another is fit jointly on all datasets. The backbone stays frozen. Training weights give equal total contribution to every dataset, foreground/background class and image. The regularization and decision threshold are fixed in the config; validation scores are reported for inspection and do not tune the model using the test set. The primary held-out result is per-image averaged balanced accuracy on the test images, with `joint − individual` reported per dataset. Probe weights and their pairwise cosines are saved as an additional diagnostic. This patch-level experiment does not reproduce full segmentation training or pixel-level IoU.

## Reading reports

Open `/home/cfuste/data/DINO-LoRA/joint_metric/index.html` to navigate current-format runs. Each run contains:

- `index.html`: pairwise alignment heatmap, `A_i`, secondary `M_i`, individual/joint probe comparison, 2D visualization and image checks.
- `components.json`, `collection.csv`, `per_dataset.csv`, `pairwise_alignment.csv`, `probe_results.csv`: numeric results and provenance.
- `local_vectors.npz`: every usable local direction and its cached patch identity.
- `probe_split.json`, `samples.json`, `probe_weights.npz`: image split, sampled patch identities and fitted linear weights.
- `cloud_projection.npz` and `figs/`: display-only PCA coordinates, PNG/SVG figures and alignment triptychs.

The PCA arrow is the dataset's full-space direction *projected* into two dimensions. All reported cosines use the original feature dimensions. The report embeds PNGs so the HTML alone remains viewable, but download the entire run directory for CSV/JSON/NPZ data.

Existing reports from the retired formulation remain on disk for provenance. The root index and combined CSVs exclude them rather than mixing incompatible scores. New runs have a new schema and run ID, so the cached feature extraction is reused while old outputs are preserved.

Optional observed-performance CSV columns remain `backbone,collection_id,dataset,iou_individual,iou_joint`; matched rows add the observed IoU difference to `per_dataset.csv`. The design hypothesis should be judged against those measured joint-training effects and across multiple backbones/collections, not by tuning cosine thresholds to one known example.

## Local checks

```bash
pip install -e '.[joint-metric,test]'
python -m pytest tests/test_joint_metric.py
```

The small tests check local image boundaries, direction symmetry and image weighting, image-disjoint probe splits, a known opposing-direction probe case, and a cache-to-report run without repeated backbone inference.
