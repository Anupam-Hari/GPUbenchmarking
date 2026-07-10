from pathlib import Path

import pandas as pd


def load_all_summaries(root: Path) -> pd.DataFrame:
    """
    Load every data_summary.csv under the benchmark root
    and combine them into a single DataFrame.
    """

    summary_files = sorted(root.rglob("data_summary.csv"))

    if not summary_files:
        raise FileNotFoundError(
            f"No data_summary.csv files found under {root}"
        )

    tables = []

    for summary_file in summary_files:
        df = pd.read_csv(summary_file)

        # Store the benchmark folder name for debugging/reference
        df["benchmark"] = summary_file.parent.name

        tables.append(df)

    combined = pd.concat(tables, ignore_index=True)

    print(f"Loaded {len(summary_files)} summary files")
    print(f"Total rows: {len(combined)}")

    return combined