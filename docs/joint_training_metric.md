# Joint-training compatibility

The pipeline has one configuration (`configs/cluster/joint_metric.yaml`), one
analysis command (`scripts/joint_metric.py`), a small bash submitter, and one SLURM
job template. The configuration
references existing training YAMLs for dataset pairing, image geometry and
pretrained backbone weights. It does not run their training or LoRA settings.

## Run on the cluster

From the repository on the cluster, install the extra dependency once in the
existing `dino-peft` environment, then submit through bash:

```bash
pip install -e '.[joint-metric]'
bash slurm/submit_joint_metric.sh --dry-run
bash slurm/submit_joint_metric.sh                 # DINOv3 + EM first
bash slurm/submit_joint_metric.sh --collection all --backbone all
```

The selectors are `em` / `deepbacs` and `dinov3` / `resnet50` / `openclip`.
Add new entries to the same YAML to extend either list. `all` selects every entry.
The submitter queues extraction on one H100, metric computation on CPU after
successful extraction, then an index job after all metric jobs finish. Each job
has 32 GiB RAM and 8 CPUs; extraction gets 12 hours, metrics 8 hours. These limits
have not been benchmarked on the cluster. Submission records and logs share a
timestamped directory under the persistent report root. No email is sent.

The bash entry point uses the existing `slurm/lib/common.sh` interpreter setting:
`$HOME/data/conda/envs/dino-peft/bin/python`, overridable with `PY`. Jobs use that
same interpreter. The submitter only calls `sbatch` on the login node; extraction
and metric computation run in scheduled jobs. Optional scheduler overrides are
`JOINT_METRIC_GPU`, `JOINT_METRIC_CONSTRAINT`, and `JOINT_METRIC_PARTITION`.

Paths continue to come from `configs/paths.yaml`, honoring `DINO_PEFT_*` overrides:

| Purpose | Default |
|---|---|
| Datasets | `/home/cfuste/data/datasets` |
| Raw spatial caches | `/home/cfuste/scratch/DINO-LoRA/joint_metric/cache/` |
| Reports and logs | `/home/cfuste/data/DINO-LoRA/joint_metric/` |
| Pretrained weights | `/home/cfuste/scratch/data/models/` |

DINOv3 requires the existing local checkout and ViT-L/16 pretrained checkpoint.
OpenCLIP and ResNet use the existing pretrained tags; pre-download their weights
or supply local `backbone.weights` in the referenced backbone config for offline
nodes. Jobs put Hugging Face/Torch caches under `models_root`. Scratch retention
applies to the large spatial caches; keep a separate long-term copy if needed.

## The three stages

```bash
# A: frozen raw spatial features + fractional foreground occupancy, once.
python scripts/joint_metric.py cache --collection em --backbone dinov3

# B: distances and risk, entirely from cache.
python scripts/joint_metric.py compute --collection em --backbone dinov3
python scripts/joint_metric.py compute --datasets lucchi vnc   # a pair

# Sampling seeds or other Cartesian-product axes from the config's sweep block.
bash slurm/submit_joint_metric.sh --metrics-only --sensitivity
python scripts/joint_metric.py compute --sensitivity

# C: change only the risk equation; no backbone inference or OT computation.
python scripts/joint_metric.py risk --components /path/to/run/components.json \
  --out-dir /path/to/empty/alternative --formula consensus
# Also supported: --formula ratio --consensus pairwise --epsilon 1e-8
```

Use `--cfg /path/to/config.yaml` for a different configuration, or `--sweep file.yaml`
for an external sweep. Sweep keys are `metric.<field>` with lists of values.
Thresholds, sample cap, seed, L2 normalization, shared PCA, OT epsilon and barycenter
settings are metric-stage choices. Extraction processes one image at a time and
saves all float32 spatial vectors in per-image NPZ shards. It preserves the current
training preprocessing: EM longest-edge resizing; DeepBacs native 448×448 crops;
ImageNet normalization, including OpenCLIP. Input directories include the training
partition's potential validation subset, but never the test partition.

Masks are cropped with images and area-resized to retain fractional occupancy.
ViT cells use exact patch averages; CNN cells use adaptive spatial bins, an
alignment approximation rather than integration over overlapping receptive fields.
Inspect the cache's `README.md` alignment previews before interpreting results.
Completed caches are immutable, checksum-validated, and reused without loading a
backbone. Changed inputs/configs require a new cache location. Input identity uses
paths, sizes and modification times; preserve those only when content is unchanged.

## Read the results

Open `index.html` under the report root. It links to each run's sortable tables,
component figures, sampling counts and convergence diagnostics. Reports work
offline when the whole tree is downloaded. CSVs are available both per run and
combined across runs. Each run saves `components.json`, `config_used.yaml`, exact
sample/image/token identities, fitted `barycenter.npz`, optional `pca.npz`, and
PNG/SVG figures. Rebuild navigation with:

```bash
python scripts/joint_metric.py summary --results-root /path/to/joint_metric
```

Inspect **C** (foreground disagreement), **S** (foreground/background separation),
and **r = C/(S+epsilon)** separately. **G** is mean risk and **L** its population
standard deviation. They are continuous experimental diagnostics, with no fitted
safe/unsafe thresholds. Compare matching settings and inspect near-zero saliency
or convergence warnings. The separately saved pairwise baseline is the mean
foreground distance to all other datasets.

Each new metric run also shows foreground clouds and the fitted barycenter in
one shared 2D PCA, plus foreground/background panels on identical axes. PCA is
fitted on balanced metric samples after their metric preprocessing; the fitted
barycenter is projected with the same transform. These plots never change the
metric or fit a new barycenter in 2D. Axis labels report explained variance;
2D overlap does not prove similarity in the original feature space. Coordinates
and display axes are saved in `cloud_projection.npz`, in the sample order recorded
in `components.json` and `samples.json`.

Three reproducible random training-image triptychs per dataset are saved under
`figs/alignment/` and included in the report when source files are accessible:
processed image, foreground-mask overlay, and patch classes with sampled
locations. They use the cache's saved preprocessing, check size/mtime and
geometry, and compare reconstructed occupancies to the cached values. If source
files have moved, current config paths (including `DINO_PEFT_DATA_ROOT`) are
tried with the saved preprocessing and original file identity. If those fail,
the report shows the first triptychs saved during cache extraction, explicitly
labeled as such; they cannot show sampled markers or be independently rechecked.
No backbone runs for report generation. The near-black fraction is only an
example-image diagnostic (all processed RGB channels <= 2/255), not a tissue
mask, dataset-wide estimate, or automatic exclusion rule.

The cluster's EM images currently live under
`/home/cfuste/data/DINO-LoRA/datasets/EM/`; the joint-metric EM entries use
`joint_em_data_root` from `configs/paths.yaml`. Other training collections keep
their existing `data_root`. The October 13 all-background report can be repaired without rerunning
the metric by submitting a CPU-only preview refresh (replace the run ID when
refreshing a different report):

```bash
bash slurm/submit_joint_metric.sh --previews-only --components \
  /home/cfuste/data/DINO-LoRA/joint_metric/em/dinov3_vitl16/5e69dd369db7697e/components.json
```

The refresh uses the original cache recipe, verifies relocated source file
size/mtime, image/mask geometry and per-patch occupancy, then updates the
report and navigator. If relocation cannot be verified, it labels and shows
the extraction-time cache triptychs instead of claiming random alignment.

PNG figures are embedded in the per-run HTML, so downloading that HTML alone
preserves the visual report. Download the whole result folder for CSV/JSON/NPZ,
vector figures, individual triptychs and navigation between runs.

The shared config uses `sinkhorn_scaling: 0.99`. The previous all-background
run converged at this setting. The current config changes only background
*eligibility* to `near_foreground` with radius 2 patch-grid steps, while keeping
full-image embeddings, foreground eligibility, seed 42, OT epsilon, and the risk
equation unchanged. Reuse the existing cluster cache with:

```bash
bash slurm/submit_joint_metric.sh --collection em --backbone dinov3 --metrics-only
```

The index includes background sampling mode, scaling, convergence and final
gradient RMS to compare runs. At each image's patch grid, the new candidate set
is `q <= 0.2` and within two horizontal/vertical/diagonal grid steps of at
least one `q >= 0.8` patch. Mixed patches remain excluded, and proximity never
crosses images. The same rule applies to every dataset, including DeepBacs;
no pixel-intensity filter is used. A common cap balances foreground and
eligible-background samples. Check that the new run still samples 512 of each;
otherwise compare sampling counts before interpreting score changes.

### Kasthuri black surroundings

The standard EM segmentation configs use full-image longest-edge resizing
(1022 rounded down to a multiple of 16 gives a 1008-pixel long edge for DINOv3).
Kasthuri has no center crop or border-exclusion mask in those configs. Normal
training/evaluation uses bicubic image resize, nearest-neighbor mask resize and
ImageNet normalization. The joint metric uses the same image geometry, with
area-based masks to retain fractional patch occupancy. Separate TRIPLET feature
analysis configs/utilities support center crops; they are not used by standard
Kasthuri segmentation or this metric.

In the original all-background run, black regions labeled background were
eligible background samples. They could increase measured FG/BG separation S
and consequently lower C/S; their effect on the foreground representation and
C must be tested. The near-foreground run keeps the original full-image
embeddings but asks how separable the foreground is from nearby background.
It does not remove every black patch or modify training/evaluation. Foreground Dice/IoU does
not reward true-negative black pixels directly, but predictions of foreground
there count as false positives. Background/mean-class scores can benefit from
easy background, and training loss also sees those pixels. Do not interpret
large S as proof of target/tissue separability until border effects are checked.

In a local replay of the original seed-42 Kasthuri preprocessing and background
sampling, 216 of 512 sampled background patches (42.2%) had at least 95%
near-black pixels (all RGB channels <= 2/255). The replay reproduced the
original report's exact Kasthuri patch counts: 305,235 total, 4,647 clear
foreground, 290,190 clear background, 10,398 mixed. This establishes substantial
border exposure in that sample, but only an exclusion ablation can quantify its
effect on S and the risk ratio.
At radius 2, 40,365 Kasthuri patches are eligible and about 3% of that local
candidate pool are at least 95% near-black. Lucchi and VNC have 80,032 and
8,745 eligible patches respectively, all above the 512-sample cap.

The distance is GeomLoss Sinkhorn divergence with squared Euclidean cost:
`2 * SamplesLoss(p=2, blur=sqrt(ot_epsilon/2))`. All empirical measures have unit
mass. Sampling uses a common cap reduced to the smallest available class; every
dataset contributes exactly 1/N. The barycenter optimizes the mean of this same
divergence over uniform-mass support atoms using LBFGS. Its support approximation,
seed, gradient tolerance and convergence history are saved; neither a global
minimum nor an inner Sinkhorn marginal-residual certificate is claimed. Full
measure duplication preserves mass; capped resampling can still vary between seeds.

Optional observed-performance CSV columns are
`backbone,collection_id,dataset,iou_individual,iou_joint`, with unique keys and
IoUs in [0,1]. Set `performance_csv` in the config or pass `--performance-csv`
to `risk`. Reports add ΔIoU = joint − individual and flag unmatched datasets.

## Tests

```bash
pip install -e '.[joint-metric,test]'
python -m pytest tests/test_joint_metric.py
```

Nine small CPU tests protect the scientific invariants, fractional mask alignment,
shared config resolution, and cache-to-report/risk-only operation. They use tiny
synthetic arrays and a dummy backbone, download no weights, and submit no jobs.
They validate code correctness, not the biological prediction hypothesis.
