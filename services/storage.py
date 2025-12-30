import json
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

LABELS_PATH = Path("storage/labels.csv")
RUN_LOG_PATH = Path("storage/run_log.json")
EXPECTED_COLUMNS = ["RowID", "Label_Anomalie", "Commentaire"]


def ensure_storage():
    Path("storage").mkdir(exist_ok=True)


def empty_labels_df() -> pd.DataFrame:
    return pd.DataFrame(columns=EXPECTED_COLUMNS)


def load_labels() -> pd.DataFrame:
    ensure_storage()

    if not LABELS_PATH.exists() or LABELS_PATH.stat().st_size == 0:
        return empty_labels_df()

    try:
        df = pd.read_csv(LABELS_PATH, sep=";")
    except EmptyDataError:
        return empty_labels_df()

    # Fichier sans en-tête : pandas crée des colonnes 0,1,2...
    if not any(col in df.columns for col in EXPECTED_COLUMNS):
        return empty_labels_df()

    for c in EXPECTED_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA

    return df[EXPECTED_COLUMNS]


def save_labels(labels_df: pd.DataFrame) -> None:
    ensure_storage()

    out = labels_df.copy()
    for c in EXPECTED_COLUMNS:
        if c not in out.columns:
            out[c] = pd.NA

    out = out[EXPECTED_COLUMNS]
    out.to_csv(LABELS_PATH, index=False, sep=";")


def read_run_log() -> list:
    ensure_storage()

    if not RUN_LOG_PATH.exists() or RUN_LOG_PATH.stat().st_size == 0:
        return []

    try:
        with RUN_LOG_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def append_run_log(entry: dict) -> None:
    ensure_storage()
    log = read_run_log()
    log.append(entry)
    with RUN_LOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
