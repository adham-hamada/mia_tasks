import pandas as pd

DATA_PATH = "Task2/data/kl.csv"

# The 8 skill attributes named in the task, plus player Value.
SKILL_FEATURES = [
    "Finishing", "ShortPassing", "Dribbling", "SprintSpeed",
    "Strength", "Stamina", "Interceptions", "StandingTackle",
]
ALL_FEATURES = SKILL_FEATURES + ["Value"]


def parse_value(value_str):
    """
    Convert a FIFA value string like '€110.5M', '€600K', or '€ 0' into a
    plain float number of euros. This has to be done manually because
    pandas reads the column as text ("part of the job", per the task).
    """
    s = str(value_str).replace("€", "").strip()
    if s.endswith("M"):
        return float(s[:-1]) * 1_000_000
    if s.endswith("K"):
        return float(s[:-1]) * 1_000
    return float(s)  # plain number, e.g. "0"


def load_fifa_data(path=DATA_PATH):
    """
    Load the raw FIFA 19 CSV. The file is cp1252-encoded (it contains
    characters like accented names — Modrić, Müller — that aren't valid
    UTF-8), so a plain pd.read_csv() would raise a UnicodeDecodeError.
    """
    return pd.read_csv(path, encoding="cp1252")


def clean_and_select(df):
    """
    Produce a clean, analysis-ready DataFrame with:
      - Name, Position           (kept for labeling / color-coding)
      - the 8 skill columns + a numeric Value column

    Rows missing a Position or any of the skill columns are dropped —
    in this dataset that's exactly the same 48 rows (players with no
    club/incomplete records), so nothing extra is lost by handling both
    at once.
    """
    df = df.copy()
    df["Value"] = df["Value"].apply(parse_value)

    keep_cols = ["Name", "Position"] + ALL_FEATURES
    df = df[keep_cols]

    before = len(df)
    df = df.dropna(subset=["Position"] + SKILL_FEATURES).reset_index(drop=True)
    print(f"[clean] dropped {before - len(df)} rows with missing Position/skill data "
          f"({len(df)} players remain)")

    return df


def get_feature_matrix(df):
    """Return the raw (unstandardized) feature matrix as a plain NumPy array."""
    return df[ALL_FEATURES].to_numpy(dtype=float)


def standardize(X):
    """
    Z-score standardization, computed manually with NumPy:
        z = (x - mean) / std
    for every column (feature) independently. Returns the standardized
    matrix plus the per-column mean/std, in case they're needed later
    (e.g. to standardize a new row using the same scale).
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    Z = (X - mean) / std
    return Z, mean, std