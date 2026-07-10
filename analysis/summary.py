from pathlib import Path

import pandas as pd


def generate_summary(raw_results_path: Path) -> pd.DataFrame:
    """
    Create a data_summary.csv by averaging all repeats for each
    (model, backend, sample_size).
    """

    summary_path = raw_results_path.parent / "data_summary.csv"

    if summary_path.exists():
        return pd.read_csv(summary_path)

    df = pd.read_csv(raw_results_path)

    group_columns = [
        "model",
        "backend",
        "sample_size",
    ]

    # Average every numeric column except repeat
    numeric_columns = (
        df.select_dtypes(include="number")
        .columns
        .drop("repeat")
        .tolist()
    )

    summary = (
        df.groupby(group_columns, as_index=False)[numeric_columns]
        .mean()
    )

    output_path = raw_results_path.parent / "data_summary.csv"
    summary.to_csv(output_path, index=False)

    print(f"Created {output_path}")

    return summary


def generate_all_summaries(root: Path):
    """
    Search recursively for every raw_results.csv and create a
    data_summary.csv beside it.
    """

    raw_files = sorted(root.rglob("raw_results.csv"))

    if not raw_files:
        raise FileNotFoundError(
            f"No raw_results.csv found under {root}"
        )

    print(f"Found {len(raw_files)} benchmark folders")

    for raw_file in raw_files:
        generate_summary(raw_file)