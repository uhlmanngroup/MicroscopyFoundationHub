"""Joint-training compatibility: cache, compute, risk, summary, or SLURM submit."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dino_peft.config import load_config, repo_root
from dino_peft.analysis.joint_run import resolve_experiment, run_from_config, recompute, expand_runs
from dino_peft.analysis.joint_metric import MetricConfig
from dino_peft.analysis.joint_report import summarize_results


def submit(args, experiments):
    """One job template, different resources for cache and metric stages."""
    root = Path(experiments[0][2]["results_root"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
    log_dir = root / "logs" / stamp
    records, metric_ids = [], []
    if not args.dry_run:
        log_dir.mkdir(parents=True)
    env = dict(os.environ, PY=sys.executable)
    common = ["sbatch", "--parsable", f"--chdir={repo_root()}", "--export=ALL"]
    if env.get("JOINT_METRIC_PARTITION"):
        common += [f"--partition={env['JOINT_METRIC_PARTITION']}"]

    def queue(stage, collection, backbone, options, command):
        name = f"{collection}-{backbone}-{stage}"
        call = common + [f"--job-name=joint-{stage}", f"--output={log_dir}/{name}-%j.out"]
        call += options + [str(repo_root() / "slurm/joint_metric.sbatch"), *command]
        print(shlex.join(call), flush=True)
        job_id = str(900000 + len(records)) if args.dry_run else subprocess.check_output(call, env=env, cwd=repo_root(), text=True).strip().split(";")[0]
        if not job_id.isdigit():
            raise ValueError(f"Unexpected sbatch job ID: {job_id!r}")
        records.append([stage, collection, backbone, job_id])
        if not args.dry_run:
            with (log_dir / "jobs.tsv").open("w", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["stage", "collection", "backbone", "job_id"])
                writer.writerows(records)
        return job_id

    for collection, backbone, cfg in experiments:
        selectors = ["--cfg", str(args.cfg.resolve()), "--collection", collection, "--backbone", backbone]
        dependencies = []
        print(f"Cache: {cfg['cache_dir']}")
        if not args.metrics_only:
            gpu = [f"--gpus={env.get('JOINT_METRIC_GPU', 'H100:1')}", "--time=12:00:00"]
            if env.get("JOINT_METRIC_CONSTRAINT", "H100"):
                gpu += [f"--constraint={env.get('JOINT_METRIC_CONSTRAINT', 'H100')}"]
            job = queue("cache", collection, backbone, gpu, ["cache", *selectors])
            dependencies = [f"--dependency=afterok:{job}", "--kill-on-invalid-dep=yes"]
        command = ["compute", *selectors, "--no-index"]
        if args.sensitivity:
            command += ["--sensitivity"]
        if args.sweep:
            command += ["--sweep", str(args.sweep.resolve())]
        metric_ids.append(queue("compute", collection, backbone, dependencies, command))
    queue("summary", "all", "all", [f"--dependency=afterany:{':'.join(metric_ids)}", "--cpus-per-task=1", "--mem=4G", "--time=00:20:00"], ["summary", "--results-root", str(root)])
    print(f"Reports: {root / 'index.html'}")
    print("Dry run: no jobs or directories created." if args.dry_run else f"Jobs and logs: {log_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("cache", "compute", "submit"):
        p = commands.add_parser(action)
        p.add_argument("--cfg", type=Path, default=repo_root() / "configs/cluster/joint_metric.yaml")
        p.add_argument("--collection", default="em", help="Collection key in config, or all")
        p.add_argument("--backbone", default="dinov3", help="Backbone key in config, or all")
        if action == "cache":
            p.add_argument("--device", help="Override extraction device, e.g. cpu")
        else:
            p.add_argument("--sensitivity", action="store_true", help="Apply sweep from the shared config")
            p.add_argument("--sweep", type=Path, help="Optional external sweep YAML")
        if action == "compute":
            p.add_argument("--no-index", action="store_true")
            p.add_argument("--datasets", nargs="+", help="Compute a subset from the same cache")
        if action == "submit":
            p.add_argument("--metrics-only", action="store_true")
            p.add_argument("--dry-run", action="store_true")
    p = commands.add_parser("risk")
    p.add_argument("--components", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--formula", choices=["ratio", "consensus"], default="ratio")
    p.add_argument("--consensus", choices=["barycenter", "pairwise"], default="barycenter")
    p.add_argument("--epsilon", type=float, default=1e-8)
    p.add_argument("--performance-csv")
    p = commands.add_parser("summary")
    p.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == "risk":
            recompute(args.components, args.out_dir, formula=args.formula, epsilon=args.epsilon, consensus=args.consensus, performance_csv=args.performance_csv)
        elif args.action == "summary":
            summarize_results(args.results_root)
        else:
            if not args.cfg.is_file():
                raise FileNotFoundError(args.cfg)
            cfg = load_config(args.cfg)
            collections = list(cfg["collections"]) if args.collection == "all" else [args.collection]
            backbones = list(cfg["backbones"]) if args.backbone == "all" else [args.backbone]
            experiments = [(c, b, resolve_experiment(cfg, c, b)) for c in collections for b in backbones]
            if args.action != "cache":
                sweep = yaml.safe_load(args.sweep.read_text()) if args.sweep else cfg.get("sweep", {}) if args.sensitivity else {}
                for _, _, experiment in experiments:
                    for run in expand_runs(experiment, sweep):
                        MetricConfig(**run["metric"])
            if args.action == "submit":
                submit(args, experiments)
            elif args.action == "cache":
                from dino_peft.analysis.joint_cache import extract_joint_cache
                for _, _, experiment in experiments:
                    if args.device:
                        experiment["extraction"] = {**experiment["extraction"], "device": args.device}
                    extract_joint_cache(experiment)
            else:
                for _, _, experiment in experiments:
                    if args.datasets:
                        experiment["dataset_ids"] = args.datasets
                        experiment["collection_id"] += "__" + "+".join(sorted(args.datasets))
                    run_from_config(experiment, sweep=sweep, update_index=not args.no_index)
    except (ValueError, TypeError, KeyError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"Joint metric: {exc}\n")


if __name__ == "__main__":
    main()
