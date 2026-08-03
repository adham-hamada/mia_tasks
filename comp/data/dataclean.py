"""
Data cleaning & feature prep for the Player Rating regression task.

TRAIN cleaning:
1. Drop rows where minutes_played == 0 (unused subs -> rating forced to 0; not a
   real performance-based label, and keeping them lets the model "cheat" by
   learning minutes==0 -> 0 instead of learning real performance patterns).
2. Drop composite/derived score columns that leak the target (see note below).
3. Drop ID/metadata columns.
4. One-hot encode categoricals.

TEST cleaning:
- Same column drops as train, but rows are NEVER dropped -- every row in test.csv
  needs a prediction for the submission file.
- Rows with minutes_played == 0 are flagged (zero_mask) so the training script can
  force their prediction to 0.0 directly, rather than asking a model that never saw
  such rows during training to extrapolate to them.
- One-hot columns are reindexed against the train columns so train/test always end
  up with identical, aligned feature columns.

NOTE on performance_score / possession_impact:
Both are present in train AND test, so using them is not technically leakage (the
info IS available at prediction time). But performance_score correlates 0.997 with
player_rating and is a derived summary of the same stats you already have -- using
it turns the task into fitting one column instead of predicting from raw match
stats, which is what the brief asks for. Tested empirically: including it drops MLP
val RMSE from 0.633 to 0.279 (R^2 0.265 -> 0.857) -- a huge leaderboard win, at the
cost of the model mostly just decoding one column rather than learning real
performance patterns. performance_score is now KEPT below per request;
possession_impact remains dropped (much weaker correlation, less justified to keep).
"""

import pandas as pd

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

TARGET = "player_rating"
ID_COL = "Id"

ID_METADATA_COLS = [
    "Id", "player_id", "player_name", "team", "jersey_number",
    "nationality", "club_name", "match_id", "match_date",
    "stadium", "city", "opponent_team",
]

DROP_COMPOSITE_SCORES = ["possession_impact"]  # performance_score kept intentionally

CATEGORICAL_COLS = ["position", "preferred_foot", "tournament_stage", "match_result"]


def _base_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=DROP_COMPOSITE_SCORES)
    df = df.drop(columns=ID_METADATA_COLS)
    return df


def prepare_train(path: str = TRAIN_PATH):
    df = pd.read_csv(path)
    df = df[df["minutes_played"] > 0].copy()  # drop unused-sub rows
    y = df[TARGET].copy()
    X = _base_clean(df.drop(columns=[TARGET]))
    X = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=True)
    return X.reset_index(drop=True), y.reset_index(drop=True)


def prepare_test(path: str = TEST_PATH, train_columns=None):
    df = pd.read_csv(path)
    ids = df[ID_COL].copy()
    zero_mask = (df["minutes_played"] == 0).values  # rule-based override, applied later

    X = _base_clean(df)
    X = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=True)

    if train_columns is not None:
        X = X.reindex(columns=train_columns, fill_value=0)  # align to train's columns

    return X.reset_index(drop=True), ids.reset_index(drop=True), zero_mask


if __name__ == "__main__":
    X_train, y_train = prepare_train()
    X_test, test_ids, zero_mask = prepare_test(train_columns=X_train.columns)

    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Test:  X={X_test.shape}, ids={test_ids.shape}, zero-minute rows={zero_mask.sum()}")
    assert list(X_train.columns) == list(X_test.columns), "train/test columns misaligned!"

    X_train.to_csv("data/X_train.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    test_ids.to_csv("data/test_ids.csv", index=False)
    pd.Series(zero_mask, name="zero_minutes").to_csv(
        "data/test_zero_mask.csv", index=False
    )