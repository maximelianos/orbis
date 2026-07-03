#!/usr/bin/env python3
"""
Export TensorBoard logs to a PDF report.

Usage:
    python tensorboard_to_pdf.py --logdir ./logs --output report.pdf
    python tensorboard_to_pdf.py --logdir ./logs --output report.pdf --max_steps 1000
    python tensorboard_to_pdf.py --logdir ./logs --output report.pdf --tags loss accuracy
    
    # good
    python tensorboard_to_pdf.py --logdir ./logs_nuplan/2026-04-30T00-39-28_nuplan --output report.pdf
    
    python tensorboard_to_pdf.py --logdir /work/dlclarge1/velikanm-max/orbis/logs_nuplan/2026-05-15T10-32-24_nuplan --output report.pdf
    
    python tensorboard_to_pdf.py --logdir ./logs_nuplan --last --output report.pdf
    
    python tensorboard_to_pdf.py --logdir ./logs_exp/2026-06-18T18-50-32_nuplan_encoder --output report.pdf

Dependencies:
    pip install tensorboard matplotlib reportlab
"""

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak, HRFlowable,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_events(
    logdir: str,
    tags_filter: list[str] | None = None,
) -> tuple[dict, dict]:
    """
    Walk logdir, load all event files, and return scalar data grouped by run.
    Returns:
        runs: { run_name: { tag: [(step, value), ...] } }
        step_times: { run_name: (start_datetime, end_datetime) }
    """
    logdir = Path(logdir)
    runs: dict[str, dict] = {}
    step_times: dict[str, tuple] = {}

    # Each sub-directory (or the root itself) is treated as a run
    event_paths = list(logdir.rglob("events.out.tfevents.*"))
    if not event_paths:
        sys.exit(f"No TensorBoard event files found under: {logdir}")

    run_dirs = {p.parent for p in event_paths}

    for run_dir in sorted(run_dirs):
        run_name = str(run_dir.relative_to(logdir)) if run_dir != logdir else "."
        ea = EventAccumulator(str(run_dir))
        ea.Reload()

        available_tags = ea.Tags().get("scalars", [])
        selected_tags = (
            [t for t in available_tags if t in tags_filter]
            if tags_filter else available_tags
        )

        if not selected_tags:
            continue

        run_data: dict[str, list] = {}
        for tag in selected_tags:
            events = ea.Scalars(tag)
            run_data[tag] = [(e.step, e.value) for e in events]

        if run_data:
            runs[run_name] = run_data
            min_step = None
            max_step = None
            min_step_time = None
            max_step_time = None

            for tag in selected_tags:
                for event in ea.Scalars(tag):
                    step = event.step
                    event_time = datetime.fromtimestamp(event.wall_time)

                    if min_step is None or step < min_step:
                        min_step = step
                        min_step_time = event_time
                    elif step == min_step and event_time < min_step_time:
                        min_step_time = event_time

                    if max_step is None or step > max_step:
                        max_step = step
                        max_step_time = event_time
                    elif step == max_step and event_time > max_step_time:
                        max_step_time = event_time

            if min_step_time is not None and max_step_time is not None:
                step_times[run_name] = (min_step_time, max_step_time)

    if not runs:
        sys.exit("No scalar data found. Check --tags or --logdir.")

    return runs, step_times


def trim_steps(series: list[tuple], max_steps: int | None) -> list[tuple]:
    if max_steps is None:
        return series
    return [(s, v) for s, v in series if s <= max_steps]


def slugify(text: str) -> str:
    return text.replace("/", "_").replace(" ", "_")


# ── Chart generation ─────────────────────────────────────────────────────────

def make_chart(
    tag: str,
    run_series: dict[str, list[tuple]],
    tmp_dir: Path,
    max_steps: int | None,
) -> Path:
    """Render one metric (all runs) to a PNG and return its path."""
    fig, ax = plt.subplots(figsize=(9, 4), dpi=120)

    for run_name, series in run_series.items():
        series = trim_steps(series, max_steps)
        if not series:
            continue
        steps, values = zip(*series)
        label = run_name if run_name != "." else "run"
        ax.plot(steps, values, linewidth=1.5, label=label)

    ax.set_title(tag, fontsize=11, pad=8)
    ax.set_xlabel("Step", fontsize=9)
    ax.set_ylabel("Value", fontsize=9)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"
    ))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=8)

    if len(run_series) > 1:
        ax.legend(fontsize=7, loc="best", framealpha=0.7)

    fig.tight_layout()
    path = tmp_dir / f"{slugify(tag)}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Summary table ─────────────────────────────────────────────────────────────

def make_summary_table(runs: dict, max_steps: int | None) -> list[list]:
    """Build rows for a summary table: run | tag | min | max | final | steps."""
    header = ["Run", "Tag", "Min", "Max", "Final", "Steps"]
    rows = [header]
    for run_name, tag_data in sorted(runs.items()):
        for tag, series in sorted(tag_data.items()):
            series = trim_steps(series, max_steps)
            if not series:
                continue
            values = [v for _, v in series]
            rows.append([
                run_name,
                tag,
                f"{min(values):.4g}",
                f"{max(values):.4g}",
                f"{values[-1]:.4g}",
                str(len(series)),
            ])
    return rows


# ── PDF assembly ──────────────────────────────────────────────────────────────

def build_pdf(
    runs: dict,
    step_times: dict,
    output_path: str,
    logdir: str,
    max_steps: int | None,
    tmp_dir: Path,
) -> None:
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "MyTitle", parent=styles["Title"], fontSize=18, spaceAfter=6
    )
    h1 = ParagraphStyle(
        "H1", parent=styles["Heading1"], fontSize=13, spaceBefore=14, spaceAfter=4
    )
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontSize=10, spaceBefore=10, spaceAfter=2
    )
    small = ParagraphStyle(
        "Small", parent=styles["Normal"], fontSize=8, textColor=colors.grey
    )

    story = []

    # ── Title page ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph("TensorBoard Export Report", title_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#4A90D9")))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(f"<b>Log directory:</b> {logdir}", styles["Normal"]))

    run_count = len(runs)
    tag_count = sum(len(v) for v in runs.values())
    story.append(Paragraph(
        f"<b>Runs:</b> {run_count} &nbsp;&nbsp; <b>Scalar tags:</b> {tag_count}",
        styles["Normal"],
    ))
    if max_steps:
        story.append(Paragraph(f"<b>Step limit:</b> {max_steps:,}", styles["Normal"]))

    if step_times:
        story.append(Spacer(1, 0.2 * cm))
        for run_name, time_range in sorted(step_times.items()):
            if not time_range:
                continue
            start_time, end_time = time_range
            start_time = start_time.isoformat(sep=" ", timespec="seconds")
            end_time = end_time.isoformat(sep=" ", timespec="seconds")
            label = run_name if run_name != "." else "run"
            story.append(Paragraph(
                f"<b>Run {label}:</b> {start_time} &rarr; {end_time}",
                styles["Normal"],
            ))

    story.append(Spacer(1, 0.6 * cm))

    # ── Summary table ─────────────────────────────────────────────────────────
    story.append(Paragraph("Summary Statistics", h1))
    rows = make_summary_table(runs, max_steps)

    col_widths = [3.5 * cm, 5 * cm, 2 * cm, 2 * cm, 2 * cm, 1.8 * cm]
    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4A90D9")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F4FA")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(tbl)
    story.append(PageBreak())

    # ── Charts grouped by run ─────────────────────────────────────────────────
    # Collect per-tag data across all runs for side-by-side comparison
    tag_to_runs: dict[str, dict] = defaultdict(dict)
    for run_name, tag_data in runs.items():
        for tag, series in tag_data.items():
            tag_to_runs[tag][run_name] = series

    story.append(Paragraph("Scalar Metrics", h1))
    story.append(Spacer(1, 0.2 * cm))

    for tag, run_series in sorted(tag_to_runs.items()):
        story.append(Paragraph(tag, h2))
        chart_path = make_chart(tag, run_series, tmp_dir, max_steps)

        img_width = 15 * cm
        img_height = img_width * (4 / 9)
        story.append(Image(str(chart_path), width=img_width, height=img_height))
        story.append(Spacer(1, 0.4 * cm))

    doc.build(story)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export TensorBoard scalar logs to a PDF report."
    )
    parser.add_argument(
        "--logdir", required=True,
        help="Path to the TensorBoard log directory."
    )
    parser.add_argument(
        "--last", action="store_true",
        help="Use the most recent run subdirectory under --logdir."
    )
    parser.add_argument(
        "--output", default="tensorboard_report.pdf",
        help="Output PDF file path (default: tensorboard_report.pdf)."
    )
    parser.add_argument(
        "--tags", nargs="+", default=None,
        help="Whitelist of scalar tags to include (default: all tags)."
    )
    parser.add_argument(
        "--max_steps", type=int, default=None,
        help="Only plot steps up to this value."
    )
    args = parser.parse_args()

    if args.last:
        if not os.path.exists(args.logdir):
            raise ValueError(f"Logdir {args.logdir} does not exist")

        subdirs = [
            d
            for d in os.listdir(args.logdir)
            if os.path.isdir(os.path.join(args.logdir, d))
        ]
        if not subdirs:
            raise ValueError(f"No subdirectories found in {args.logdir}")

        subdirs.sort()
        args.logdir = os.path.join(args.logdir, subdirs[-1])
        print(f"Using last logdir: {args.logdir}")

    print(f"Loading events from: {args.logdir}")
    runs, step_times = load_events(args.logdir, args.tags)

    tmp_dir = Path("/tmp/tb_pdf_charts")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating PDF: {args.output}")
    build_pdf(runs, step_times, args.output, args.logdir, args.max_steps, tmp_dir)
    print(f"Done! Report saved to: {args.output}")


if __name__ == "__main__":
    main()