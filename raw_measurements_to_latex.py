#!/usr/bin/env python3
"""Convert benchmark raw measurements into LaTeX summary tables.

By default, this script reads ``results/raw_measurements.csv`` and writes a
prefill table and a decode table under ``results/latex_tables/``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
from typing import Iterable


# Default CSV produced by benchmark_dense_vs_moe.py.
DEFAULT_INPUT_CSV = Path("results/raw_measurements.csv")

# Default directory for the generated LaTeX table files.
DEFAULT_OUTPUT_DIR = Path("results/latex_tables")

# Required CSV columns used to group and summarize measurements.
REQUIRED_COLUMNS = {
    "label",
    "workload",
    "requested_input_tokens",
    "parallel_generations",
    "tokens_per_second",
}

# Metadata for each workload's row variable, caption, label, and output file.
TABLE_SPECS = {
    "prefill": {
        "x_column": "requested_input_tokens",
        "x_heading": "Input context length",
        "caption": (
            "Prefill-dominated aggregate throughput in input tokens per second. "
            "Each cell reports the mean and sample standard deviation across "
            "timed repetitions."
        ),
        "latex_label": "tab:prefill-throughput",
        "filename": "prefill_throughput_table.tex",
    },
    "decode": {
        "x_column": "parallel_generations",
        "x_heading": "Parallel generations",
        "caption": (
            "Decode-dominated aggregate throughput in output tokens per second. "
            "Each cell reports the mean and sample standard deviation across "
            "timed repetitions."
        ),
        "latex_label": "tab:decode-throughput",
        "filename": "decode_throughput_table.tex",
    },
}


# Parse the input path, output directory, and displayed decimal precision.
def parse_args() -> argparse.Namespace:
    # Command-line parser for the CSV-to-LaTeX converter.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="raw_measurements.csv to read (default: results/raw_measurements.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for generated .tex files (default: results/latex_tables)",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=1,
        help="digits after the decimal point in throughput values (default: 1)",
    )
    return parser.parse_args()


# Escape text that will be placed in a normal LaTeX table cell.
def escape_latex(text: str) -> str:
    # Character substitutions needed by model labels and headings.
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    return "".join(replacements.get(character, character) for character in text)


# Load the benchmark CSV and validate the columns needed for both tables.
def read_measurements(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise SystemExit(f"Input CSV does not exist: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        # CSV reader retaining the benchmark's named columns.
        reader = csv.DictReader(handle)

        # Column names found in the supplied CSV header.
        fieldnames = set(reader.fieldnames or [])

        # Required columns absent from the supplied CSV.
        missing_columns = REQUIRED_COLUMNS - fieldnames
        if missing_columns:
            raise SystemExit(
                "Input CSV is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        # All raw measurement rows in their original order.
        rows = list(reader)

    if not rows:
        raise SystemExit(f"Input CSV contains no measurements: {csv_path}")
    return rows


# Preserve the model order in which labels first appear in the raw CSV.
def ordered_model_labels(rows: Iterable[dict[str, str]]) -> list[str]:
    # Model labels already added to the ordered result.
    seen: set[str] = set()

    # Model labels in their first-appearance order.
    labels: list[str] = []
    for row in rows:
        # Readable benchmark label for this measurement.
        label = row["label"]
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


# Format a mean and sample standard deviation as one LaTeX math cell.
def format_summary(values: Iterable[float], precision: int) -> str:
    # Materialized repetitions for this model and X-axis value.
    samples = list(values)
    if not samples:
        return "--"

    # Arithmetic mean across timed repetitions.
    mean_value = statistics.mean(samples)

    # Sample standard deviation, defined as zero for a single repetition.
    standard_deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0

    # Number formatting with LaTeX-safe thousands separators.
    mean_text = f"{mean_value:,.{precision}f}".replace(",", "{,}")

    # Number formatting for the sample standard deviation.
    deviation_text = f"{standard_deviation:,.{precision}f}".replace(",", "{,}")
    return rf"\({mean_text} \pm {deviation_text}\)"


# Build one complete booktabs table for a prefill or decode workload.
def build_table(
    rows: list[dict[str, str]],
    workload: str,
    model_labels: list[str],
    precision: int,
) -> str:
    # Table metadata associated with this workload.
    specification = TABLE_SPECS[workload]

    # CSV column whose unique values become table rows.
    x_column = specification["x_column"]

    # Measurements belonging to this workload only.
    workload_rows = [row for row in rows if row["workload"] == workload]
    if not workload_rows:
        raise SystemExit(f"No {workload!r} measurements found in the input CSV")

    # Sorted input lengths or parallel-generation counts.
    x_values = sorted({int(row[x_column]) for row in workload_rows})

    # Right-aligned numeric column declaration for each model.
    column_format = "r" + "r" * len(model_labels)

    # Header cells containing the row variable and model labels.
    header_cells = [specification["x_heading"], *model_labels]

    # Complete LaTeX source accumulated line by line.
    latex_lines = [
        "% Generated by raw_measurements_to_latex.py; do not edit by hand.",
        "% Requires \\usepackage{booktabs} in the document preamble.",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{specification['caption']}}}",
        rf"\label{{{specification['latex_label']}}}",
        rf"\begin{{tabular}}{{{column_format}}}",
        r"\toprule",
        " & ".join(escape_latex(cell) for cell in header_cells) + r" \\",
        r"\midrule",
    ]

    for x_value in x_values:
        # Cells for this context length or concurrency level.
        row_cells = [str(x_value)]
        for model_label in model_labels:
            # Throughput repetitions matching the current table cell.
            cell_values = (
                float(row["tokens_per_second"])
                for row in workload_rows
                if row["label"] == model_label and int(row[x_column]) == x_value
            )
            row_cells.append(format_summary(cell_values, precision))
        latex_lines.append(" & ".join(row_cells) + r" \\")

    latex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(latex_lines)


# Read the raw measurements and write one LaTeX file per workload.
def main() -> None:
    # User-selected converter settings.
    args = parse_args()
    if args.precision < 0:
        raise SystemExit("--precision must be zero or greater")

    # Validated raw benchmark measurements.
    rows = read_measurements(args.input_csv)

    # Stable model column order shared by both tables.
    model_labels = ordered_model_labels(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for workload, specification in TABLE_SPECS.items():
        # Rendered LaTeX source for the current workload.
        latex_source = build_table(rows, workload, model_labels, args.precision)

        # Destination file named by the workload specification.
        output_path = args.output_dir / specification["filename"]
        output_path.write_text(latex_source, encoding="utf-8")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
