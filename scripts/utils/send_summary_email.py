#!/usr/bin/env python3
"""Turn a summarize_seg_results.py report into a plain-text digest, and email it.

Reads ``<summary-dir>/summary.csv`` (the aggregate one row per dataset x regime) and
writes a short table a person can read in a mail client, with the per-source
foreground IoU columns kept for the joint runs. Cells with fewer repeats than
expected are called out, so a sweep that half-failed does not look complete.

Sending is best-effort by design: this runs at the tail of a SLURM dependency chain,
and a missing MTA should not turn a finished sweep into a failed job. Without one, or
with --no-send, the digest just goes to stdout.

Example:
    python scripts/utils/send_summary_email.py \
      --summary-dir /scratch/DINO-LoRA/openclip/summary \
      --to me@example.org --subject "OpenCLIP sweep finished"
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, List, Optional

EXPECTED_REPEATS = 5

MODE_LABEL = {
    "head_only": "frozen",
    "lora": "LoRA",
    "full_finetune": "end-to-end",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary-dir", type=Path, required=True,
                    help="Directory holding summary.csv, as written by summarize_seg_results.py.")
    ap.add_argument("--to", action="append", default=[],
                    help="Recipient address. Repeatable. Falls back to $NOTIFY_EMAIL.")
    ap.add_argument("--subject", default="Segmentation sweep summary")
    ap.add_argument("--from-addr", default=None,
                    help="Sender address; defaults to the first recipient.")
    ap.add_argument("--expected-repeats", type=int, default=EXPECTED_REPEATS,
                    help="Repeats a complete cell should have (default: 5).")
    ap.add_argument("--note", action="append", default=[],
                    help="Extra line to put above the table. Repeatable.")
    ap.add_argument("--no-send", action="store_true",
                    help="Only print the digest.")
    return ap.parse_args()


def _f(value: Optional[str], digits: int = 3) -> str:
    if value in (None, "", "None"):
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except ValueError:
        return str(value)


def _pair(row: Dict[str, str], stem: str) -> str:
    mean, std = row.get(f"{stem}_mean"), row.get(f"{stem}_std")
    if mean in (None, "", "None"):
        return "--"
    return f"{_f(mean)} ± {_f(std)}"


def read_rows(summary_csv: Path) -> List[Dict[str, str]]:
    with summary_csv.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def per_dataset_stems(rows: List[Dict[str, str]]) -> List[str]:
    """Column stems for the per-source foreground IoU, in a stable order."""
    prefix = "foreground_iou_per_dataset__"
    stems = {
        key[: -len("_mean")]
        for row in rows
        for key in row
        if key.startswith(prefix) and key.endswith("_mean")
    }
    return sorted(stems)


def build_digest(rows: List[Dict[str, str]], expected_repeats: int,
                 notes: List[str]) -> tuple[str, int]:
    stems = per_dataset_stems(rows)
    short = [s[len("foreground_iou_per_dataset__"):] for s in stems]

    header = ["modality", "task", "regime", "n", "fg IoU", "fg Dice"] + short
    widths = [len(h) for h in header]
    body: List[List[str]] = []
    incomplete = 0

    def sort_key(row: Dict[str, str]):
        order = {"head_only": 0, "lora": 1, "full_finetune": 2}
        return (row.get("modality", ""), row.get("task_type", ""),
                order.get(row.get("training_mode", ""), 9))

    for row in sorted(rows, key=sort_key):
        n = row.get("num_runs", "0")
        try:
            short_run = int(n) < expected_repeats
        except ValueError:
            short_run = True
        if short_run:
            incomplete += 1
        cells = [
            row.get("modality", "?"),
            row.get("task_type", "?"),
            MODE_LABEL.get(row.get("training_mode", ""), row.get("training_mode", "?")),
            f"{n}!" if short_run else str(n),
            _pair(row, "foreground_iou"),
            _pair(row, "foreground_dice"),
        ] + [_pair(row, stem) for stem in stems]
        widths = [max(w, len(c)) for w, c in zip(widths, cells)]
        body.append(cells)

    def line(cells: List[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()

    out = list(notes)
    if notes:
        out.append("")
    out.append(line(header))
    out.append("  ".join("-" * w for w in widths))
    out.extend(line(c) for c in body)
    out.append("")
    out.append(f"{len(body)} cells, foreground IoU as mean ± std over repeats.")
    if incomplete:
        out.append(
            f"WARNING: {incomplete} cell(s) marked '!' have fewer than {expected_repeats} "
            "repeats — those runs failed or are still pending."
        )
    else:
        out.append(f"All cells have {expected_repeats} repeats.")
    return "\n".join(out), incomplete


def send(message: EmailMessage) -> bool:
    """Hand the message to a local MTA. Returns False when there is none."""
    sendmail = shutil.which("sendmail") or shutil.which("/usr/sbin/sendmail")
    if sendmail:
        proc = subprocess.run([sendmail, "-t", "-oi"], input=message.as_bytes())
        return proc.returncode == 0
    mail = shutil.which("mail") or shutil.which("mailx")
    if mail:
        proc = subprocess.run(
            [mail, "-s", message["Subject"], *message["To"].split(", ")],
            input=message.get_content().encode("utf-8"),
        )
        return proc.returncode == 0
    return False


def main() -> int:
    args = parse_args()
    import os

    recipients = list(args.to)
    if not recipients:
        env_to = os.environ.get("NOTIFY_EMAIL", "").strip()
        if env_to:
            recipients = [addr.strip() for addr in env_to.split(",") if addr.strip()]

    summary_csv = args.summary_dir.expanduser() / "summary.csv"
    if not summary_csv.is_file():
        print(f"[error] no summary at {summary_csv}", file=sys.stderr)
        return 1

    rows = read_rows(summary_csv)
    if not rows:
        print(f"[warn] {summary_csv} has no rows; nothing to report.")
        return 0

    notes = list(args.note) + [f"Source: {summary_csv}"]
    digest, incomplete = build_digest(rows, args.expected_repeats, notes)
    print(digest)

    if args.no_send:
        return 0
    if not recipients:
        print("\n[warn] no recipient (--to or $NOTIFY_EMAIL); digest printed only.",
              file=sys.stderr)
        return 0

    subject = args.subject
    if incomplete:
        subject = f"{subject} — {incomplete} incomplete cell(s)"

    message = EmailMessage()
    message["Subject"] = subject
    message["To"] = ", ".join(recipients)
    message["From"] = args.from_addr or recipients[0]
    message.set_content(digest)

    if send(message):
        print(f"\n[info] emailed digest to {message['To']}")
    else:
        print("\n[warn] no sendmail/mail on this host; digest printed only. "
              "SLURM's own --mail-type notification still fires.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
