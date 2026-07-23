from pathlib import Path

import pandas as pd


TRAIN_RATIO = 0.8
TEST_RATIO = 0.2


def generate_summary(raw_results_path: Path) -> pd.DataFrame:
    """
    Create a data_summary.csv by averaging all repeats for each
    (model, backend, sample_size).

    Also stores the derived TrainRows and TestRows based on the
    configured train/test split.
    """

    summary_path = raw_results_path.parent / "data_summary.csv"

    if summary_path.exists():
        return pd.read_csv(summary_path)

    df = pd.read_csv(raw_results_path)

    # Add train/test row counts
    df["TrainRows"] = (df["sample_size"] * TRAIN_RATIO).round().astype(int)
    df["TestRows"] = (df["sample_size"] * TEST_RATIO).round().astype(int)

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

    # Convert row counts back to integers
    summary["TrainRows"] = summary["TrainRows"].round().astype(int)
    summary["TestRows"] = summary["TestRows"].round().astype(int)

    summary.to_csv(summary_path, index=False)

    print(f"Created {summary_path}")

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