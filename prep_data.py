mport pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/main.csv")
OUT_PATH = Path("data/processed/main.csv")

def main():
    df = pd.read_csv(RAW_PATH)

    # Basic cleanup: standardize column names
    df.columns = [c.strip() for c in df.columns]

    # Ensure numeric columns are numeric (coerce errors to NaN)
    numeric_cols = [
        "Age",
        "Sleep Duration",
        "Quality of Sleep",
        "Physical Activity Level",
        "Stress Level",
        "Heart Rate",
        "Daily Steps",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing key fields for analysis
    required = ["Quality of Sleep", "Physical Activity Level", "Gender", "Occupation"]
    keep = [c for c in required if c in df.columns]
    df = df.dropna(subset=keep)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved cleaned data to: {OUT_PATH} ({len(df)} rows)")

if __name__ == "__main__":
    main()