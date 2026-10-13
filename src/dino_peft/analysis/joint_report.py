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
    """One display-only PCA for actual metric samples and the fitted support."""
    from .joint_metric import MetricConfig, _transform_distributions
    clouds, _ = _transform_distributions(distributions, MetricConfig(**result["config"]))
    groups = [(name, kind, cloud) for name in sorted(clouds) for kind, cloud in clouds[name].items()]
    # Balanced datasets/classes determine the axes; the barycenter does not.
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
    np.savez_compressed(output / "cloud_projection.npz",
                        coordinates=np.concatenate([(cloud - mean) @ axes.T for _, _, cloud in groups]),
                        dataset=np.concatenate([np.repeat(name, len(cloud)) for name, _, cloud in groups]),
                        kind=np.concatenate([np.repeat(kind, len(cloud)) for _, kind, cloud in groups]),
                        barycenter=(np.asarray(result["barycenter"]["support"]) - mean) @ axes.T,
                        mean=mean, axes=axes, explained_variance_ratio=variance)
    result["projection"] = {"method": "shared_balanced_PCA_2D", "display_only": True,
                            "fit": "equal samples per dataset and foreground/background class; excludes barycenter",
                            "input_dimension": int(pooled.shape[1]), "explained_variance_ratio": variance.tolist(),
                            "coordinate_order": "sorted datasets, foreground then background; matches samples.json within each group",
                            "artifact": "cloud_projection.npz"}


def alignment_triptych(path: Path, image, mask, occupancy, grid_shape, *, normalization="training",
                      tau_fg=.8, tau_bg=.2, title="", selected=None) -> dict:
    """Display the actual processed input, mask overlay, and patch assignments."""
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from dino_peft.utils.transforms import IMAGENET_MEAN, IMAGENET_STD, OPENCLIP_MEAN, OPENCLIP_STD
    mean, std = (OPENCLIP_MEAN, OPENCLIP_STD) if normalization == "openclip_native" else (IMAGENET_MEAN, IMAGENET_STD)
    rgb = np.clip(image.numpy().transpose(1, 2, 0) * np.asarray(std) + np.asarray(mean), 0, 1)
    foreground = np.asarray(mask)
    q = np.asarray(occupancy).reshape(grid_shape)
    classes = np.where(q >= tau_fg, 2, np.where(q <= tau_bg, 0, 1))
    colors = ["#3182bd", "#e6ad39", "#31a354"]
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
    axes[2].imshow(classes, cmap=ListedColormap(colors), vmin=0, vmax=2, interpolation="nearest", alpha=.5, extent=extent)
    handles = [Patch(color=color, label=label) for color, label in zip(colors, (f"BG: q ≤ {tau_bg:g}", "Mixed (excluded)", f"FG: q ≥ {tau_fg:g}"))]
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
    rows = result["per_dataset"]
    names = [r["dataset"] for r in rows]
    projection = output.parent / "cloud_projection.npz"
    if projection.is_file():
        with np.load(projection, allow_pickle=False) as data:
            xy, dataset, kind, bary, variance = (data[k] for k in ("coordinates", "dataset", "kind", "barycenter", "explained_variance_ratio"))
        points = np.concatenate([xy, bary])
        margin = np.maximum(np.ptp(points, axis=0), 1e-3) * .08
        limits = np.array([points.min(axis=0) - margin, points.max(axis=0) + margin]).T
        palette = {name: plt.get_cmap("tab10")(i % 10) for i, name in enumerate(names)}
        fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
        for name in names:
            points = xy[(dataset == name) & (kind == "foreground")]
            ax.scatter(*points.T, s=12, alpha=.5, color=palette[name], label=name, rasterized=True)
        ax.scatter(*bary.T, s=65, marker="*", c="black", label=f"Barycenter ({len(bary)} points)", zorder=3)
        ax.set(title="Foreground clouds and fitted barycenter", xlabel=f"PC1 ({variance[0]:.1%} of variance)", ylabel=f"PC2 ({variance[1]:.1%} of variance)")
        ax.set(xlim=limits[0], ylim=limits[1], aspect="equal")
        ax.legend(fontsize=9)
        for suffix in ("png", "svg"):
            fig.savefig(output / f"foreground_clouds.{suffix}", dpi=160, bbox_inches="tight")
        plt.close(fig)
        fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.5), sharex=True, sharey=True, squeeze=False, layout="constrained")
        for name, ax in zip(names, axes[0]):
            for label, marker in (("background", "x"), ("foreground", "o")):
                points = xy[(dataset == name) & (kind == label)]
                ax.scatter(*points.T, s=13, alpha=.4, marker=marker, color=palette[name], label=label, rasterized=True)
            ax.scatter(*bary.T, s=45, marker="*", c="black", label="barycenter", zorder=3)
            ax.set(title=name, xlabel=f"PC1 ({variance[0]:.1%})", xlim=limits[0], ylim=limits[1], aspect="equal")
            ax.legend(fontsize=8)
        axes[0, 0].set_ylabel(f"PC2 ({variance[1]:.1%})")
        fig.suptitle("Foreground vs background · identical PCA axes and limits")
        for suffix in ("png", "svg"):
            fig.savefig(output / f"foreground_background.{suffix}", dpi=160, bbox_inches="tight")
        plt.close(fig)
    bary = result.get("barycenter", {})
    if bary.get("objective_history") and bary.get("gradient_rms_history"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
        axes[0].plot(bary["objective_history"], color="#146b82")
        axes[0].set(title="Shared-reference fit", xlabel="Iteration", ylabel="Mean foreground distance")
        axes[1].semilogy(np.maximum(bary["gradient_rms_history"], np.finfo(float).tiny), color="#146b82")
        axes[1].axhline(bary["tolerance"], color="#b87800", linestyle="--", label="Required tolerance")
        axes[1].set(title="Convergence check", xlabel="Iteration", ylabel="Gradient RMS (log scale)")
        axes[1].legend()
        for suffix in ("png", "svg"):
            fig.savefig(output / f"convergence.{suffix}", dpi=160, bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(max(11, len(rows) * 2), 4.4), constrained_layout=True)
    for ax, key, label, color in zip(axes, ["C_i", "S_i", "r_i"], ["Consensus deviation C (lower = closer)", "FG/BG separation S (larger = clearer)", "Candidate risk r"], ["#317a9b", "#4e8a67", "#b06b35"]):
        values = [r[key] for r in rows]
        bars = ax.bar(names, values, color=color)
        ax.set_title(label, fontsize=10)
        ax.bar_label(bars, labels=[number(v) for v in values], padding=3, fontsize=8)
        ax.margins(y=.22)
        ax.tick_params(axis="x", rotation=25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f'{result["backbone"]} · {result["collection_id"]} · {result["run_id"]}', fontsize=11)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"components.{suffix}", dpi=170)
    plt.close(fig)
    if result.get("pairwise_foreground") is not None:
        values = np.asarray(result["pairwise_foreground"])
        fig, ax = plt.subplots(figsize=(max(5, len(names) * 1.2), max(4, len(names))))
        m = ax.imshow(values, cmap="Blues", vmin=0)
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
        ax.set_yticks(range(len(names)), names)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, number(values[i, j]), ha="center", va="center", color="white" if values[i, j] > values.max() * .65 else "#172f43", fontsize=9)
        ax.set_title("Pairwise foreground Sinkhorn divergence")
        fig.colorbar(m, ax=ax)
        fig.tight_layout()
        for suffix in ("png", "svg"):
            fig.savefig(output / f"pairwise.{suffix}", dpi=170)
        plt.close(fig)
    matched = [r for r in rows if "delta_i" in r]
    if matched:
        fig, ax = plt.subplots(figsize=(6, 4))
        for row in matched:
            ax.scatter(row["r_i"], row["delta_i"], color="#146b82")
            ax.annotate(row["dataset"], (row["r_i"], row["delta_i"]), xytext=(5, 5), textcoords="offset points")
        ax.axhline(0, color="#82909a", linewidth=1)
        ax.set(xlabel="Candidate risk r", ylabel="Observed ΔIoU (joint − individual)", title="Hypothesis check against observed training")
        fig.tight_layout()
        for suffix in ("png", "svg"):
            fig.savefig(output / f"observed_effect.{suffix}", dpi=170)
        plt.close(fig)


def collection_row(result: dict) -> dict:
    row = {"backbone": result["backbone"], "collection_id": result["collection_id"], "run_id": result["run_id"], **result["collection"]}
    row.update(result.get("config", {}))
    row["cache_hash"] = result.get("cache_hash", "")
    row["barycenter_converged"] = result.get("barycenter", {}).get("converged", result.get("barycenter", {}).get("diagnostics", {}).get("converged"))
    row["barycenter_gradient_rms"] = result.get("barycenter", {}).get("final_mass_scaled_gradient_rms")
    row["warning_count"] = len(result.get("warnings", []))
    return row


def write_report(result: dict, output: Path, *, make_plots: bool = True) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if make_plots:
        make_figures(result, output / "figs")
    rows = result["per_dataset"]
    for row in rows:
        row.update({"backbone": result["backbone"], "collection_id": result["collection_id"], "run_id": result["run_id"]})
    write_json(output / "components.json", result)
    write_csv(output / "per_dataset.csv", rows)
    write_csv(output / "collection.csv", [collection_row(result)])
    cfg = result.get("config", {})
    c = result["collection"]
    warnings = result.get("warnings", [])
    fields = [("dataset", "Dataset"), ("C_i", "C: consensus deviation"), ("S_i", "S: FG/BG separation"), ("r_i", "r: candidate risk"), ("C_pairwise", "Pairwise consensus"), ("r_pairwise", "Pairwise risk")]
    if any("delta_i" in r for r in rows):
        fields += [("iou_individual", "Individual IoU"), ("iou_joint", "Joint IoU"), ("delta_i", "ΔIoU")]
    counts = [("dataset", "Dataset"), ("n_available", "All patches"), ("n_foreground", "Clear FG"), ("n_background", "Clear BG"), ("n_ambiguous", "Discarded mixed"), ("n_sampled_foreground", "Sampled FG"), ("n_sampled_background", "Sampled BG")]
    title = f'{result["collection_id"]} · {result["backbone"]}'
    body = '<div class="tag">Joint-training compatibility · experimental diagnostic</div>'
    body += f'<h1>{esc(title)}</h1><p class="muted">Run {esc(result["run_id"])} · {len(rows)} equally weighted datasets · seed {esc(cfg.get("seed"))}</p>'
    body += '<nav>'
    if result.get("index_relative"):
        body += f'<a href="{esc(result["index_relative"])}">All experiments</a>'
    if result.get("projection") and make_plots:
        body += '<a href="#clouds">2D feature clouds</a>'
    if result.get("alignment"):
        body += '<a href="#alignment">Alignment examples</a>'
    body += '<a href="#diagnostics">Convergence</a>'
    body += '<a href="per_dataset.csv">Dataset CSV</a><a href="collection.csv">Collection CSV</a><a href="components.json">Raw components &amp; provenance</a><a href="config_used.yaml">Resolved config</a></nav>'
    if (output / "samples.json").is_file():
        body += '<p><a href="samples.json">Sample identities</a></p>'
    body += '<div class="cards">' + ''.join(f'<div class="card">{esc(label)}<b>{number(c[key])}</b><small>{esc(desc)}</small></div>' for key, label, desc in [("G", "Global risk G", "Mean candidate risk"), ("L", "Selective risk L", "Population standard deviation"), ("L_rel", "Relative dispersion", "L / (G + ε)")]) + '</div>'
    body += f'<p>Risk formula: <code>{esc(cfg.get("risk_formula", "ratio"))}</code>; consensus for risk: <code>{esc(cfg.get("risk_consensus", "barycenter"))}</code>. C below always records the barycenter deviation.</p>'
    body += f'<p>Sinkhorn scaling: <b>{esc(cfg.get("sinkhorn_scaling"))}</b> · OT regularization ε: {esc(cfg.get("ot_epsilon"))} · seed: {esc(cfg.get("seed"))}. Scaling controls numerical accuracy; compare convergence and scores with the original 0.9 run using the same seed.</p>'
    body += '<p>These are continuous research diagnostics. Compare runs with matching settings; no universal safe/unsafe thresholds have been calibrated. A large L means uneven risk, while G measures its average level. Inspect C and S before interpreting the ratio.</p>'
    if warnings:
        body += '<div class="warn"><b>Checks requiring attention</b><ul>' + ''.join(f'<li>{esc(w)}</li>' for w in warnings) + '</ul></div>'
    body += '<h2>Per-dataset components</h2><p>Click a column heading to sort. C measures disagreement with the shared foreground distribution; S measures separation from each dataset’s own background.</p><div class="panel">' + table(rows, fields, "components") + '</div>'
    if make_plots:
        body += '<figure class="panel">' + _image(output, "figs/components.png", "Separate consensus, saliency and candidate risk plots") + '<figcaption><a href="figs/components.svg">Download vector figure</a></figcaption></figure>'
        if result.get("pairwise_foreground") is not None:
            body += '<figure class="panel">' + _image(output, "figs/pairwise.png", "Pairwise foreground distance matrix") + '<figcaption><a href="figs/pairwise.svg">Download vector matrix</a></figcaption></figure>'
        if any("delta_i" in r for r in rows):
            body += '<figure class="panel">' + _image(output, "figs/observed_effect.png", "Candidate risk versus observed joint-training effect") + '</figure>'
        if result.get("projection") and (output / "cloud_projection.npz").is_file():
            projection = result["projection"]
            retained = sum(projection["explained_variance_ratio"])
            body += f'<h2 id="clouds">Feature clouds and shared foreground reference</h2><p>Each dot is an embedding actually sampled for this run. Black stars are the fitted barycenter support points. One shared PCA projects foreground, background and the fitted barycenter onto the same axes, retaining <b>{retained:.1%}</b> of sampled feature variance.</p>'
            body += f'<p>This is a display in 2D: distances, barycenter fitting and risk scores were calculated in {projection["input_dimension"]} dimensions. Overlap in this projection does not prove overlap in the full feature space. The barycenter is projected after fitting; it is not refitted in 2D.</p>'
            for stem in ("foreground_clouds", "foreground_background"):
                body += '<figure class="panel">' + _image(output, f"figs/{stem}.png", stem.replace("_", " ")) + f'<figcaption><a href="figs/{stem}.svg">Download vector figure</a></figcaption></figure>'
            body += '<p><a href="cloud_projection.npz">Projection coordinates and shared PCA axes</a> · Point order is recorded in the raw components and corresponds to sample identities.</p>'
    body += '<h2>Sampling and filtering</h2><div class="panel">' + table(rows, counts, "counts") + '</div>'
    alignment = result.get("alignment", {})
    if alignment:
        body += '<h2 id="alignment">Image / mask / patch alignment</h2><p>Up to three reproducible random training images per dataset, reconstructed using the cache’s original preprocessing. Left: actual processed input. Middle: foreground mask in green. Right: foreground (green), background (blue), mixed patches (amber). Circles and crosses mark foreground and background patches sampled for this metric run.</p>'
        body += '<p>Check that the green mask follows the target structures and that patch classes follow mask coverage. Black surroundings remain visible. “Near-black” means all processed RGB channels ≤ 2/255; this is a visual diagnostic, not a tissue detector or an exclusion rule. Percentages describe these example images only.</p>'
        for preview in alignment.get("previews", []):
            caption = f'{preview["dataset"]} · {preview["sample_id"]} · cached occupancy verified · crop: {preview["crop_box_xyxy"]} · near-black pixels: {preview["near_black_pixel_fraction"]:.1%}'
            body += '<figure class="panel">' + _image(output, preview["figure"], caption) + f'<figcaption>{esc(caption)} · <a href="{esc(preview["figure"])}">Download triptych</a></figcaption></figure>'
    bary = result.get("barycenter", {})
    body += '<h2 id="diagnostics">Method and numerical diagnostics</h2><div class="panel">'
    body += f'<p>Barycenter: <b>{"converged" if bary.get("converged") else "check convergence"}</b> · support atoms: {esc(bary.get("support_size"))} · iterations: {esc(bary.get("iterations"))} · final gradient RMS: {esc(number(bary.get("final_mass_scaled_gradient_rms")))}</p>'
    if make_plots and (output / "figs/convergence.png").is_file():
        body += _image(output, "figs/convergence.png", "Barycenter objective and convergence threshold")
    if (output / "barycenter.npz").is_file():
        body += '<p><a href="barycenter.npz">Download fitted barycenter support</a></p>'
    if (output / "pca.npz").is_file():
        body += '<p><a href="pca.npz">Download shared PCA transform</a></p>'
    body += f'<details><summary>Full settings, convergence history and preprocessing diagnostics</summary><pre>{esc(json.dumps({"metric": cfg, "distance": result.get("distance"), "barycenter": bary, "transform": {k:v for k,v in result.get("transform", {}).items() if k not in ("pca_components", "pca_mean")}}, indent=2))}</pre></details></div>'

    (output / "index.html").write_text(document(title, body))
    md = [f"# {title}", "", f"Run `{result['run_id']}` · {len(rows)} datasets", "", f"G = {number(c['G'])} · L = {number(c['L'])} · L_rel = {number(c['L_rel'])}", "", "Continuous experimental diagnostics; no calibrated decision thresholds.", "", "| Dataset | C | S | r | Pairwise C |", "|---|---:|---:|---:|---:|"]
    md += [f"| {r['dataset']} | {number(r['C_i'])} | {number(r['S_i'])} | {number(r['r_i'])} | {number(r.get('C_pairwise'))} |" for r in rows]
    md += ["", "Warnings:"] + [f"- {w}" for w in warnings] if warnings else []
    md += ["", "Open `index.html` for plots, counts, links and numerical diagnostics.", ""]
    (output / "README.md").write_text("\n".join(md))


def summarize_results(root: Path) -> list[dict]:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows, datasets, invalid = [], [], []
    for path in sorted(root.rglob("components.json")):
        if any(p.startswith(".") for p in path.relative_to(root).parts):
            continue
        try:
            result = json.loads(path.read_text())
            row = collection_row(result)
            row["report"] = quote(str(path.parent.relative_to(root) / "index.html"), safe="/")
            rows.append(row)
            datasets.extend(result["per_dataset"])
        except (KeyError, ValueError, TypeError) as exc:
            invalid.append(f"{path.relative_to(root)}: {exc}")
    write_csv(root / "all_collections.csv", rows)
    write_csv(root / "all_datasets.csv", datasets)
    fields = [("collection_id", "Collection"), ("backbone", "Backbone"), ("run_id", "Run"), ("G", "Global G"), ("L", "Selective L"), ("pairwise_G", "Pairwise G"), ("seed", "Seed"), ("sinkhorn_scaling", "Sinkhorn scaling"), ("barycenter_converged", "Converged"), ("barycenter_gradient_rms", "Gradient RMS"), ("tau_fg", "τ FG"), ("tau_bg", "τ BG"), ("ot_epsilon", "OT ε"), ("risk_formula", "Risk formula"), ("warning_count", "Warnings")]
    body = '<div class="tag">Experiment navigator</div><h1>Joint-training compatibility</h1>'
    body += f'<p>{len(rows)} completed metric runs. Filter by collection, backbone, seed or run ID, then open the run for components, plots and numerical checks.</p><nav><a href="all_collections.csv">All collection results (CSV)</a><a href="all_datasets.csv">All dataset components (CSV)</a></nav>'
    body += '<p>Compare matching threshold, normalization, OT and sampling settings. G is average risk; L is its unevenness across datasets. These values do not yet establish safe/unsafe categories.</p><div class="panel">' + table(rows, fields, "runs", {"run_id": lambda row: row["report"]}) + '</div>'
    if invalid:
        body += '<div class="warn">Unreadable result files:<ul>' + ''.join(f'<li>{esc(v)}</li>' for v in invalid) + '</ul></div>'
    if not rows:
        body += '<p>No completed runs found. Check the submission manifest and SLURM logs.</p>'
    (root / "index.html").write_text(document("Joint-training compatibility experiments", body))
    (root / "README.md").write_text(f"# Joint-training compatibility\n\n{len(rows)} completed runs. Open `index.html` to navigate.\n\n- `all_datasets.csv`: C, S and risk per dataset.\n- `all_collections.csv`: G, L and full comparison settings.\n- `logs/`: cluster jobs and submission manifests, when run via SLURM.\n")
    return rows
