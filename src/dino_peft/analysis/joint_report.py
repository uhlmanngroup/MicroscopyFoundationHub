"""Portable, offline reports for joint-training compatibility experiments."""
from __future__ import annotations

import csv
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


def make_figures(result: dict, output: Path) -> None:
    plt = _pyplot()
    output.mkdir(parents=True, exist_ok=True)
    rows = result["per_dataset"]
    names = [r["dataset"] for r in rows]
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
    body += '<a href="per_dataset.csv">Dataset CSV</a><a href="collection.csv">Collection CSV</a><a href="components.json">Raw components &amp; provenance</a><a href="config_used.yaml">Resolved config</a></nav>'
    if (output / "samples.json").is_file():
        body += '<p><a href="samples.json">Sample identities</a></p>'
    body += '<div class="cards">' + ''.join(f'<div class="card">{esc(label)}<b>{number(c[key])}</b><small>{esc(desc)}</small></div>' for key, label, desc in [("G", "Global risk G", "Mean candidate risk"), ("L", "Selective risk L", "Population standard deviation"), ("L_rel", "Relative dispersion", "L / (G + ε)")]) + '</div>'
    body += f'<p>Risk formula: <code>{esc(cfg.get("risk_formula", "ratio"))}</code>; consensus for risk: <code>{esc(cfg.get("risk_consensus", "barycenter"))}</code>. C below always records the barycenter deviation.</p>'
    body += '<p>These are continuous research diagnostics. Compare runs with matching settings; no universal safe/unsafe thresholds have been calibrated. A large L means uneven risk, while G measures its average level. Inspect C and S before interpreting the ratio.</p>'
    if warnings:
        body += '<div class="warn"><b>Checks requiring attention</b><ul>' + ''.join(f'<li>{esc(w)}</li>' for w in warnings) + '</ul></div>'
    body += '<h2>Per-dataset components</h2><p>Click a column heading to sort. C measures disagreement with the shared foreground distribution; S measures separation from each dataset’s own background.</p><div class="panel">' + table(rows, fields, "components") + '</div>'
    if make_plots:
        body += '<figure class="panel"><img src="figs/components.png" alt="Separate consensus, saliency and candidate risk plots"><figcaption><a href="figs/components.svg">Download vector figure</a></figcaption></figure>'
        if result.get("pairwise_foreground") is not None:
            body += '<figure class="panel"><img src="figs/pairwise.png" alt="Pairwise foreground distance matrix"><figcaption><a href="figs/pairwise.svg">Download vector matrix</a></figcaption></figure>'
        if any("delta_i" in r for r in rows):
            body += '<figure class="panel"><img src="figs/observed_effect.png" alt="Candidate risk versus observed joint-training effect"></figure>'
    body += '<h2>Sampling and filtering</h2><div class="panel">' + table(rows, counts, "counts") + '</div>'
    bary = result.get("barycenter", {})
    body += '<h2>Method and numerical diagnostics</h2><div class="panel">'
    body += f'<p>Barycenter: <b>{"converged" if bary.get("converged") else "check convergence"}</b> · support atoms: {esc(bary.get("support_size"))} · iterations: {esc(bary.get("iterations"))} · final gradient RMS: {esc(number(bary.get("final_mass_scaled_gradient_rms")))}</p>'
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
    fields = [("collection_id", "Collection"), ("backbone", "Backbone"), ("run_id", "Run"), ("G", "Global G"), ("L", "Selective L"), ("pairwise_G", "Pairwise G"), ("seed", "Seed"), ("tau_fg", "τ FG"), ("tau_bg", "τ BG"), ("ot_epsilon", "OT ε"), ("risk_formula", "Risk formula"), ("warning_count", "Warnings")]
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
