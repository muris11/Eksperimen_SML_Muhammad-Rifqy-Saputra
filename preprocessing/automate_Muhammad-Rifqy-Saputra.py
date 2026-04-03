import argparse
from pathlib import Path

import pandas as pd


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = ["age", "income", "loan_amount", "employment_years", "late_payments"]
    categorical_cols = ["home_ownership", "marital_status"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("unknown").str.strip().str.lower()
            mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else "unknown"
            df[col] = df[col].replace({"<NA>": mode_value}).fillna(mode_value)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated preprocessing untuk submission SMSML.")
    parser.add_argument("--input_path", type=str, required=True, help="Path dataset raw CSV")
    parser.add_argument("--output_path", type=str, required=True, help="Path output dataset preprocessing CSV")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    processed = preprocess_dataframe(df)
    processed.to_csv(output_path, index=False)

    print(f"Preprocessing selesai. File disimpan di: {output_path}")


if __name__ == "__main__":
    main()
