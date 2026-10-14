"""Portable, offline reports for joint-training compatibility experiments."""
from __future__ import annotations

import csv
import base64
import html
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

STYLE = """
:root{color-scheme:light;--ink:#172f43;--muted:#536777;--accent:#146b82}
*{box-sizing:border-box}body{font:16px/1.55 system-ui,sans-serif;color:var(--ink);background:#f3f6f8;margin:0}
main{max-width:1440px;margin:auto;padding:36px}h1{font-size:30px;line-height:1.2;margin-bottom:8px}
h2{font-size:21px;margin-top:32px}a{color:var(--accent)}.muted,small{color:var(--muted)}
.cards{display:flex;gap:16px;flex-wrap:wrap;margin:24px 0}.card{flex:1;min-width:180px;background:white;border:1px solid #dce4e9;border-radius:10px;padding:20px}.card b{display:block;font-size:30px}
.panel{background:white;border:1px solid #dce4e9;border-radius:10px;padding:20px;margin:20px 0;overflow:auto}
.warn{background:#fff4d6;border-left:4px solid #b87800;padding:14px 20px}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:right;padding:10px 12px;border-bottom:1px solid #e1e8ed;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{background:#eaf1f5;cursor:pointer}tr:hover{background:#f0f8fa}input{font:inherit;padding:8px 12px;border:1px solid #adbdc7;border-radius:6px;width:min(100%,440px);margin:12px 0}
figure{margin:0}img{max-width:100%;height:auto}code{background:#eaf1f5;padding:2px 5px;border-radius:3px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}
nav{display:flex;gap:18px;flex-wrap:wrap}.tag{color:#146b82;font-size:12px;letter-spacing:.1em;text-transform:uppercase}
"""
SCRIPT = """
document.querySelectorAll('input[data-filter]').forEach(x=>x.addEventListener('input',()=>{
let t=document.getElementById(x.dataset.filter);for(let r of t.tBodies[0].rows)r.hidden=!r.textContent.toLowerCase().includes(x.value.toLowerCase());}));
document.querySelectorAll('th').forEach(h=>h.addEventListener('click',()=>{let table=h.closest('table'),i=h.cellIndex,rows=[...table.tBodies[0].rows],asc=h.dataset.asc!=='true';h.dataset.asc=asc;rows.sort((a,b)=>{let x=a.cells[i].textContent,y=b.cells[i].textContent;return (asc?1:-1)*(x.trim()!==''&&y.trim()!==''&&Number.isFinite(+x)&&Number.isFinite(+y)?+x-+y:x.localeCompare(y));});rows.forEach(r=>table.tBodies[0].appendChild(r));}));
"""


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def number(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    return str(value)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def table(rows: list[dict], fields: list[tuple[str, str]], table_id: str, links: dict | None = None) -> str:
    head = "".join(f"<th>{esc(label)}</th>" for _, label in fields)
    body = []
    for row in rows:
        cells = []
        for key, _ in fields:
            value = esc(number(row.get(key)))
            if links and key in links:
                value = f'<a href="{esc(links[key](row))}">{value}</a>'
            cells.append(f"<td>{value}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<input aria-label="Filter table" placeholder="Filter rows…" data-filter="{table_id}"><table id="{table_id}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def document(title: str, body: str) -> str:
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>{STYLE}</style></head><body><main>{body}</main><script>{SCRIPT}</script></body></html>'


def join_performance(result: dict, path: str | Path) -> None:
    """Join pre-aggregated IoUs; ambiguity is an error rather than a guessed match."""
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"backbone", "collection_id", "dataset", "iou_individual", "iou_joint"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Performance CSV requires columns {sorted(required)}")
        lookup = {}
        for line, row in enumerate(reader, 2):
            key = (row["backbone"], row["collection_id"], row["dataset"])
            if key in lookup:
                raise ValueError(f"Duplicate performance key {key} on CSV line {line}; aggregate repeats first")
            a, b = float(row["iou_individual"]), float(row["iou_joint"])
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in (a, b)):
                raise ValueError(f"IoU must be a finite fraction in [0,1], CSV line {line}")
            lookup[key] = {"iou_individual": a, "iou_joint": b, "delta_i": b - a}
    missing = []
    for row in result["per_dataset"]:
        key = (result["backbone"], result["collection_id"], row["dataset"])
        if key in lookup:
            row.update(lookup[key])
        else:
            missing.append(row["dataset"])
    if missing:
        result.setdefault("warnings", []).append("No observed IoU match for: " + ", ".join(missing))
    result["performance_csv"] = str(Path(path).resolve())


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_cloud_projection(result: dict, distributions: dict, output: Path) -> None:
    """One shared display-only PCA on balanced local FG/BG probe samples."""
    groups = [(name, kind, cloud) for name in sorted(distributions)
              for kind, cloud in distributions[name].items()]
    cap = min(len(cloud) for _, _, cloud in groups)
    pooled = np.concatenate([cloud[:cap] for _, _, cloud in groups])
    mean = pooled.mean(axis=0)
    _, singular, vectors = np.linalg.svd(pooled - mean, full_matrices=False)
    axes = np.zeros((2, pooled.shape[1]))
    axes[:min(2, len(vectors))] = vectors[:2]
    pivots = np.argmax(np.abs(axes), axis=1)
    axes *= np.where(axes[np.arange(2), pivots] < 0, -1., 1.)[:, None]
    variance = np.zeros(2)
    if np.square(singular).sum() > 0:
        variance[:min(2, len(singular))] = np.square(singular[:2]) / np.square(singular).sum()
    names = result["dataset_order"]
    starts = np.stack([result["per_dataset"][names.index(name)]["mean_local_background"] for name in names])
    ends = np.stack([result["per_dataset"][names.index(name)]["mean_local_foreground"] for name in names])
    np.savez_compressed(output / "cloud_projection.npz",
                        coordinates=np.concatenate([(cloud - mean) @ axes.T for _, _, cloud in groups]),
                        dataset=np.concatenate([np.repeat(name, len(cloud)) for name, _, cloud in groups]),
                        kind=np.concatenate([np.repeat(kind, len(cloud)) for _, kind, cloud in groups]),
                        arrow_start=(starts - mean) @ axes.T, arrow_end=(ends - mean) @ axes.T,
                        dataset_order=np.asarray(names), mean=mean, axes=axes,
                        explained_variance_ratio=variance)
    result["projection"] = {"method": "shared_balanced_PCA_2D", "display_only": True,
                            "fit": "equal samples per dataset and foreground/local-background class",
                            "input_dimension": int(pooled.shape[1]), "explained_variance_ratio": variance.tolist(),
                            "coordinate_order": "sorted datasets, foreground then background; display subset of probe samples",
                            "artifact": "cloud_projection.npz"}


def alignment_triptych(path: Path, image, mask, occupancy, grid_shape, *, normalization="training",
                      tau_fg=.8, tau_bg=.2, radius=2, title="", selected=None) -> dict:
    """Display the actual processed input, mask overlay, and patch assignments."""
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from .joint_metric import near_foreground_background
    from dino_peft.utils.transforms import IMAGENET_MEAN, IMAGENET_STD, OPENCLIP_MEAN, OPENCLIP_STD
    mean, std = (OPENCLIP_MEAN, OPENCLIP_STD) if normalization == "openclip_native" else (IMAGENET_MEAN, IMAGENET_STD)
    rgb = np.clip(image.numpy().transpose(1, 2, 0) * np.asarray(std) + np.asarray(mean), 0, 1)
    foreground = np.asarray(mask)
    q = np.asarray(occupancy).reshape(grid_shape)
    near = near_foreground_background(np.asarray(occupancy), grid_shape,
                                      tau_fg=tau_fg, tau_bg=tau_bg, radius=radius).reshape(grid_shape)
    classes = np.where(q >= tau_fg, 3, np.where(near, 1, np.where(q <= tau_bg, 0, 2)))
    colors = ["#274e72", "#31a6b8", "#e6ad39", "#31a354"]
    height, width = foreground.shape
    extent = (0, width, height, 0)
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
    for ax in axes:
        ax.imshow(rgb, extent=extent)
        ax.set_axis_off()
    overlay = np.zeros((*foreground.shape, 4))
    overlay[..., 1] = 1
    overlay[..., 3] = foreground * .55
    axes[1].imshow(overlay, extent=extent)
    axes[2].imshow(classes, cmap=ListedColormap(colors), vmin=0, vmax=3, interpolation="nearest", alpha=.5, extent=extent)
    handles = [Patch(color=color, label=label) for color, label in zip(colors, (
        "Other clear BG", f"Local BG: q ≤ {tau_bg:g}", "Mixed (excluded)", f"FG: q ≥ {tau_fg:g}"))]
    for kind, marker in (("foreground", "o"), ("background", "x")):
        tokens = (selected or {}).get(kind, [])
        if tokens:
            row, col = np.divmod(tokens, grid_shape[1])
            axes[2].scatter((col + .5) * width / grid_shape[1], (row + .5) * height / grid_shape[0],
                            s=22, marker=marker, linewidths=.8, **({"facecolors": "none", "edgecolors": "black"} if marker == "o" else {"c": "black"}))
            handles.append(Line2D([], [], marker=marker, color="black", markerfacecolor="none", linestyle="none", label=f"Sampled {kind}"))
    for ax, label in zip(axes, ("1. Processed image", "2. Foreground mask overlay", "3. Patch labels" + (" + sampled locations" if selected is not None else ""))):
        ax.set_title(label, fontsize=11)
    dark = float(np.all(rgb <= 2 / 255, axis=2).mean())
    fig.suptitle(f"{title}\nNear-black processed pixels: {dark:.1%} (diagnostic only; no pixels excluded)", fontsize=11)
    fig.tight_layout(rect=(0, .07, 1, .88))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9, frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {"near_black_pixel_fraction": dark, "near_black_definition": "all processed RGB channels <= 2/255"}


def _image(output: Path, relative: str, alt: str) -> str:
    # A downloaded per-run HTML keeps its figures even without sibling files.
    data = base64.b64encode((output / relative).read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="{esc(alt)}">'


def make_figures(result: dict, output: Path) -> None:
    plt = _pyplot()
    output.mkdir(parents=True, exist_ok=True)
    rows, names = result["per_dataset"], result["dataset_order"]
    matrix = np.asarray(result["pairwise_alignment"])
    fig, ax = plt.subplots(figsize=(max(7.5, len(names) * 1.5), max(4.5, len(names) * 1.1)), layout="constrained")
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    color="white" if abs(matrix[i, j]) > .55 else "#172f43")
    ax.set_title("Local foreground/background direction alignment", fontsize=11)
    fig.colorbar(image, ax=ax, label="Cosine alignment (−1 to +1)")
    for suffix in ("png", "svg"):
        fig.savefig(output / f"alignment_matrix.{suffix}", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(max(9, len(names) * 2.7), 4), layout="constrained")
    for ax, key, title, color in ((axes[0], "A_i", "Mean alignment with other datasets", "#317a9b"),
                                  (axes[1], "M_i", "Local foreground/background magnitude", "#4e8a67")):
        values = [row[key] for row in rows]
        bars = ax.bar(names, values, color=color)
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.spines[["top", "right"]].set_visible(False)
        if key == "A_i":
            ax.set_ylim(-1.05, 1.1)
            ax.axhline(0, color="#82909a", linewidth=1)
        else:
            ax.margins(y=.2)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"dataset_directions.{suffix}", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(7, len(names) * 2.2), 4), layout="constrained")
    positions = np.arange(len(names))
    individual = [row["test_individual_balanced_accuracy"] for row in rows]
    joint = [row["test_joint_balanced_accuracy"] for row in rows]
    ax.bar(positions - .2, individual, width=.4, label="Individual probe", color="#317a9b")
    ax.bar(positions + .2, joint, width=.4, label="Shared probe", color="#d78644")
    for index, row in enumerate(rows):
        ax.text(index, max(individual[index], joint[index]) + .015,
                f"Δ {row['test_joint_minus_individual']:+.3f}", ha="center", fontsize=9)
    ax.set_xticks(positions, names)
    ax.set(ylim=(0, 1.08), ylabel="Test balanced accuracy", title="Frozen-feature linear probe")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"linear_probe.{suffix}", dpi=170)
    plt.close(fig)

    projection = output.parent / "cloud_projection.npz"
    if projection.is_file():
        with np.load(projection, allow_pickle=False) as data:
            xy, dataset, kind = (data[key] for key in ("coordinates", "dataset", "kind"))
            starts, ends, variance = (data[key] for key in ("arrow_start", "arrow_end", "explained_variance_ratio"))
        points = np.concatenate((xy, starts, ends))
        span = np.ptp(points, axis=0)
        margin = np.maximum(span, 1e-3) * .08
        limits = np.array([points.min(axis=0) - margin, points.max(axis=0) + margin]).T
        if span[1] < .1 * span[0]:
            midpoint = np.mean(limits[1])
            limits[1] = midpoint + np.array([-.1, .1]) * span[0]
        fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.7), sharex=True, sharey=True,
                                 squeeze=False, layout="constrained")
        for index, (name, ax) in enumerate(zip(names, axes[0])):
            for label, marker, color in (("background", "x", "#7f8c9a"),
                                         ("foreground", "o", "#247ab0")):
                cloud = xy[(dataset == name) & (kind == label)]
                ax.scatter(*cloud.T, s=12, alpha=.35, marker=marker, color=color, label=label, rasterized=True)
            ax.annotate("", xy=ends[index], xytext=starts[index],
                        arrowprops={"arrowstyle": "-|>", "lw": 3, "color": "#d35400", "mutation_scale": 18})
            ax.scatter(*starts[index], s=45, color="#d35400", zorder=4)
            ax.set(title=name, xlabel=f"PC1 ({variance[0]:.1%})", xlim=limits[0], ylim=limits[1],
                   aspect="auto" if span[1] < .1 * span[0] else "equal")
            ax.legend(fontsize=8)
        axes[0, 0].set_ylabel(f"PC2 ({variance[1]:.1%})")
        fig.suptitle("Foreground and local background · arrows are projected full-space dataset directions")
        for suffix in ("png", "svg"):
            fig.savefig(output / f"local_clouds.{suffix}", dpi=160, bbox_inches="tight")
        plt.close(fig)


def collection_row(result: dict) -> dict:
    row = {"backbone": result["backbone"], "collection_id": result["collection_id"],
           "run_id": result["run_id"], "A_collection": result["collection"]["A_collection"]}
    names = result["dataset_order"]
    for dataset in result["per_dataset"]:
        name = dataset["dataset"]
        for key in ("A_i", "M_i", "test_individual_balanced_accuracy", "test_joint_balanced_accuracy",
                    "test_joint_minus_individual"):
            row[f"{key}__{name}"] = dataset[key]
    matrix = result["pairwise_alignment"]
    for i, left in enumerate(names):
        for j in range(i + 1, len(names)):
            row[f"A__{left}__{names[j]}"] = matrix[i][j]
    row.update(result.get("config", {}))
    row["cache_hash"] = result.get("cache_hash", "")
    row["warning_count"] = len(result.get("warnings", []))
    return row


def write_report(result: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    make_figures(result, output / "figs")
    rows, names = result["per_dataset"], result["dataset_order"]
    for row in rows:
        row.update({"backbone": result["backbone"], "collection_id": result["collection_id"],
                    "run_id": result["run_id"]})
    write_json(output / "components.json", result)
    write_csv(output / "per_dataset.csv", rows,
              [key for key in rows[0] if key not in ("mean_local_foreground", "mean_local_background")])
    write_csv(output / "collection.csv", [collection_row(result)])
    write_csv(output / "pairwise_alignment.csv",
              [{"dataset": name, **dict(zip(names, result["pairwise_alignment"][i]))} for i, name in enumerate(names)])
    write_csv(output / "probe_results.csv", result["linear_probe"]["per_dataset"])
    cfg, warnings = result["config"], result.get("warnings", [])
    title = f'{result["collection_id"]} · {result["backbone"]}'
    fields = [("dataset", "Dataset"), ("A_i", "Mean alignment Aᵢ"),
              ("M_i", "Local magnitude Mᵢ"),
              ("test_individual_balanced_accuracy", "Individual probe"),
              ("test_joint_balanced_accuracy", "Joint probe"),
              ("test_joint_minus_individual", "Joint − individual")]
    counts = [("dataset", "Dataset"), ("n_images_total", "Images"),
              ("n_images_used", "Images with local pairs"), ("n_images_skipped", "Images skipped"),
              ("n_local_vectors", "Local FG pairs"),
              ("foreground_coverage", "Usable FG fraction"),
              ("n_skipped_no_local_background", "FG skipped: no local BG"),
              ("n_background_all", "All clear BG"), ("n_background_eligible", "Local BG"),
              ("n_probe_foreground", "Probe FG"), ("n_probe_background", "Probe BG")]
    body = '<div class="tag">Joint-training direction alignment · experimental diagnostic</div>'
    body += f'<h1>{esc(title)}</h1><p class="muted">Run {esc(result["run_id"])} · {len(rows)} datasets · seed {esc(cfg["seed"])}</p>'
    body += '<nav>'
    if result.get("index_relative"):
        body += f'<a href="{esc(result["index_relative"])}">All experiments</a>'
    body += '<a href="#directions">Direction alignment</a><a href="#probe">Linear probe</a><a href="#clouds">2D view</a><a href="#alignment">Image checks</a>'
    body += '<a href="per_dataset.csv">Dataset CSV</a><a href="pairwise_alignment.csv">Alignment CSV</a><a href="collection.csv">Collection CSV</a><a href="components.json">Raw data</a></nav>'
    body += f'<div class="cards"><div class="card">Collection alignment<b>{number(result["collection"]["A_collection"])}</b><small>Mean cosine across all dataset pairs; −1 to +1</small></div></div>'
    body += '<p>Each foreground patch points from the mean of its nearby clear-background patches toward its foreground embedding. Image directions are averaged equally within each dataset. Dataset directions are compared by cosine in the original feature space. Higher alignment means more similar directions; no risk thresholds are assumed.</p>'
    body += f'<p>Local background uses a {2*cfg["near_foreground_radius_patches"]+1}×{2*cfg["near_foreground_radius_patches"]+1} patch neighborhood within the same image. Clear foreground occupancy ≥ {cfg["tau_fg"]}; clear background occupancy ≤ {cfg["tau_bg"]}. Foreground patches without local clear background are skipped and counted. No intensity or black-pixel filter is applied.</p>'
    if warnings:
        body += '<div class="warn"><b>Checks requiring attention</b><ul>' + ''.join(f'<li>{esc(w)}</li>' for w in warnings) + '</ul></div>'
    body += '<h2 id="directions">Direction alignment</h2><p>The diagonal is 1 by definition. Off-diagonal values compare peer datasets symmetrically. Aᵢ is each dataset’s mean alignment to the others; Mᵢ is a separate magnitude diagnostic.</p>'
    body += '<figure class="panel">' + _image(output, "figs/alignment_matrix.png", "Pairwise direction cosine heatmap") + '<figcaption><a href="figs/alignment_matrix.svg">Download vector figure</a></figcaption></figure>'
    body += '<div class="panel">' + table(rows, fields, "datasets") + '</div>'
    body += '<figure class="panel">' + _image(output, "figs/dataset_directions.png", "Per-dataset direction alignment and local magnitude") + '</figure>'
    body += '<h2 id="probe">Frozen-feature linear probe</h2><p>Individual and shared binary logistic probes use only cached embeddings. The pretrained backbone is frozen. Images are split within each dataset into disjoint training, validation and test sets; the saved split lists their IDs. Training gives equal total weight to every dataset, class and image. The table above and plot below show held-out test balanced accuracy; Δ is joint minus individual. This is a diagnostic, not part of the zero-training alignment metric.</p>'
    body += '<figure class="panel">' + _image(output, "figs/linear_probe.png", "Individual versus joint linear-probe test balanced accuracy") + '<figcaption><a href="figs/linear_probe.svg">Download vector figure</a> · <a href="probe_results.csv">Validation and test CSV</a> · <a href="probe_split.json">Image splits</a> · <a href="probe_weights.npz">Probe weights</a></figcaption></figure>'
    if result.get("projection"):
        retained = sum(result["projection"]["explained_variance_ratio"])
        body += f'<h2 id="clouds">Two-dimensional feature view</h2><p>Dots are sampled foreground and local-background patch embeddings. Orange arrows project each full-space dataset direction from its mean local-background centroid toward foreground. Shared PCA retains {retained:.1%} of displayed feature variance; alignment is computed in {result["projection"]["input_dimension"]} dimensions, not from this picture.</p>'
        body += '<figure class="panel">' + _image(output, "figs/local_clouds.png", "Local foreground and background clouds with direction arrows") + '<figcaption><a href="figs/local_clouds.svg">Download vector figure</a> · <a href="cloud_projection.npz">Projection coordinates</a></figcaption></figure>'
    body += '<h2>Sampling and coverage</h2><div class="panel">' + table(rows, counts, "counts") + '</div>'
    body += '<p><a href="local_vectors.npz">All local d(f) vectors and patch identities</a> · <a href="samples.json">Probe patch identities</a> · <a href="config_used.yaml">Resolved configuration</a></p>'
    alignment = result.get("alignment", {})
    if alignment:
        body += '<h2 id="alignment">Image / mask / patch alignment</h2><p>Source images are reconstructed with the extraction-time preprocessing and checked against cached occupancies when available. Circles and crosses mark probe-sampled foreground and local-background patches. Dark blue marks other clear background; teal marks eligible local background. Near-black percentages are descriptive, not an exclusion rule.</p>'
        for preview in alignment.get("previews", []):
            if preview.get("source") == "cache_preview":
                caption = f'{preview["dataset"]} · {preview["sample_id"]} · extraction-time preview; source not reverified'
            else:
                caption = f'{preview["dataset"]} · {preview["sample_id"]} · occupancy verified · near-black pixels {preview["near_black_pixel_fraction"]:.1%}'
            body += '<figure class="panel">' + _image(output, preview["figure"], caption) + f'<figcaption>{esc(caption)}</figcaption></figure>'
    body += '<details class="panel"><summary>Full method settings and probe-weight comparison</summary><pre>' + esc(json.dumps({"config": cfg, "individual_probe_weight_cosine": result["linear_probe"]["individual_weight_cosine"]}, indent=2)) + '</pre></details>'
    (output / "index.html").write_text(document(title, body))
    md = [f"# {title}", "", f"Run `{result['run_id']}` · {len(rows)} datasets", "",
          f"Collection alignment: {number(result['collection']['A_collection'])}", "",
          "| Dataset | Mean alignment | Local magnitude | Individual probe | Joint probe | Difference |",
          "|---|---:|---:|---:|---:|---:|"]
    md += [f"| {row['dataset']} | {number(row['A_i'])} | {number(row['M_i'])} | {number(row['test_individual_balanced_accuracy'])} | {number(row['test_joint_balanced_accuracy'])} | {number(row['test_joint_minus_individual'])} |" for row in rows]
    md += ["", "Open `index.html` for the matrix, plots, image checks and provenance.", ""]
    (output / "README.md").write_text("\n".join(md))


def summarize_results(root: Path) -> list[dict]:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows, datasets, legacy, invalid = [], [], [], []
    for path in sorted(root.rglob("components.json")):
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        try:
            result = json.loads(path.read_text())
            if result.get("schema_version") != 2:
                legacy.append(path)
                continue
            row = collection_row(result)
            row["report"] = quote(str(path.parent.relative_to(root) / "index.html"), safe="/")
            rows.append(row)
            datasets.extend({key: value for key, value in dataset.items()
                             if key not in ("mean_local_foreground", "mean_local_background")}
                            for dataset in result["per_dataset"])
        except (KeyError, ValueError, TypeError) as exc:
            invalid.append(f"{path.relative_to(root)}: {exc}")
    write_csv(root / "all_collections.csv", rows)
    write_csv(root / "all_datasets.csv", datasets)
    fields = [("collection_id", "Collection"), ("backbone", "Backbone"), ("run_id", "Run"),
              ("A_collection", "Mean pair alignment"), ("seed", "Seed"),
              ("near_foreground_radius_patches", "Local radius"), ("warning_count", "Warnings")]
    body = '<div class="tag">Experiment navigator</div><h1>Local direction alignment</h1>'
    body += f'<p>{len(rows)} current-format runs. Open a run for its pairwise direction matrix, per-dataset alignment, local magnitude and held-out probe comparison.</p>'
    body += '<nav><a href="all_collections.csv">Collection results CSV</a><a href="all_datasets.csv">Dataset results CSV</a></nav>'
    body += '<div class="panel">' + table(rows, fields, "runs", {"run_id": lambda row: row["report"]}) + '</div>' if rows else '<p>No current-format runs found yet.</p>'
    if legacy:
        body += f'<p class="muted">{len(legacy)} historical reports use the retired formulation and are excluded from these comparisons.</p>'
    if invalid:
        body += '<div class="warn">Unreadable result files:<ul>' + ''.join(f'<li>{esc(v)}</li>' for v in invalid) + '</ul></div>'
    (root / "index.html").write_text(document("Local direction alignment experiments", body))
    (root / "README.md").write_text(f"# Local direction alignment\n\n{len(rows)} current-format runs. Open `index.html` to navigate. Historical reports are excluded.\n")
    return rows
