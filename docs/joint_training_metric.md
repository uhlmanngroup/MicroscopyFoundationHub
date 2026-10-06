# Joint-training compatibility

The pipeline has one configuration (`configs/cluster/joint_metric.yaml`), one
command (`scripts/joint_metric.py`), and one SLURM job template. The configuration
references existing training YAMLs for dataset pairing, image geometry and
pretrained backbone weights. It does not run their training or LoRA settings.

## Run on the cluster

From the repository, in the existing `dino-peft` environment:

```bash
pip install -e '.[joint-metric]'
python scripts/joint_metric.py submit --dry-run
python scripts/joint_metric.py submit                 # DINOv3 + EM first
python scripts/joint_metric.py submit --collection all --backbone all
```

The selectors are `em` / `deepbacs` and `dinov3` / `resnet50` / `openclip`.
Add new entries to the same YAML to extend either list. `all` selects every entry.
The submitter queues extraction on one H100, metric computation on CPU after
successful extraction, then an index job after all metric jobs finish. Each job
has 32 GiB RAM and 8 CPUs; extraction gets 12 hours, metrics 8 hours. These limits
have not been benchmarked on the cluster. Submission records and logs share a
timestamped directory under the persistent report root. No email is sent.

The existing `slurm/lib/common.sh` provides cluster/environment setup. Jobs use
the Python interpreter that submitted them. Optional scheduler overrides are
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
python scripts/joint_metric.py submit --metrics-only --sensitivity
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
