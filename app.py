from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from services.features import build_features
from services.model import RF_PARAMS, save_model, score, train
from services.storage import (
    append_run_log,
    ensure_storage,
    load_labels,
    save_labels,
)


st.set_page_config(page_title="RF Audit Tool", layout="wide")
st.title("RF Audit Tool DEC 2025")

# FATF/GAFI list (simplified placeholder)
FATF_DEFAULT = {"Iran", "Myanmar", "Coree du Nord", "Syrie", "Yemen", "Afrique du Sud"}


def normalize_label_value(val) -> int:
    if pd.isna(val):
        return 0
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return 1 if float(val) == 1 else 0
    if isinstance(val, str):
        cleaned = val.strip().lower()
        if cleaned == "":
            return 0
        if cleaned in {"1", "true", "vrai", "oui"}:
            return 1
        try:
            return 1 if float(cleaned) == 1 else 0
        except ValueError:
            return 0
    return 0


def normalize_labels(series: pd.Series) -> pd.Series:
    return series.apply(normalize_label_value).astype(int)


def label_distribution(labels_df: pd.DataFrame) -> tuple[int, int]:
    if labels_df.empty or "Label_Anomalie" not in labels_df.columns:
        return 0, 0
    normalized = normalize_labels(labels_df["Label_Anomalie"])
    return int((normalized == 1).sum()), int((normalized == 0).sum())


ensure_storage()

uploaded = st.file_uploader("Importer votre Excel (.xlsx)", type=["xlsx"])

if uploaded is None:
    st.info("Importez un fichier Excel pour demarrer le workflow complet.")
else:
    df = pd.read_excel(uploaded, engine="openpyxl")
    st.success(f"Fichier charge : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    df = df.reset_index(drop=True)
    df["RowID"] = df.index.astype(int)

    with st.spinner("Generation des controles et features..."):
        feat = build_features(df, FATF_DEFAULT)

    st.subheader("Apercu des features")
    st.dataframe(feat.head(20), use_container_width=True)

    st.subheader("Labeling (anomalie confirmee / OK)")
    labels = load_labels()
    if not labels.empty:
        labels["Label_Anomalie"] = normalize_labels(labels["Label_Anomalie"])
        labels["RowID"] = pd.to_numeric(labels["RowID"], errors="coerce").astype("Int64")
    nb_pos, nb_neg = label_distribution(labels)
    st.caption(f"Labels charges : {len(labels)} ligne(s)")
    st.caption(f"Distribution labels 1={nb_pos} | 0={nb_neg}")

    top_n = st.slider("Nombre de lignes a labelliser (priorisees)", 50, 1000, 200, step=50)
    to_label = feat.sort_values(["RiskRule"], ascending=False).head(top_n).copy()
    to_label = to_label.merge(labels, on="RowID", how="left")

    edited = st.data_editor(
        to_label[
            [
                "RowID",
                "NumRF",
                "DateDon",
                "Montant",
                "MontantRF",
                "AdressePays",
                "NomDonateur",
                "RiskRule",
                "Label_Anomalie",
                "Commentaire",
            ]
        ],
        use_container_width=True,
        num_rows="dynamic",
        key="label_editor",
    )

    if st.button("💾 Sauvegarder les labels"):
        out = edited[["RowID", "Label_Anomalie", "Commentaire"]].copy()
        out = out.dropna(subset=["RowID"])
        out["RowID"] = pd.to_numeric(out["RowID"], errors="coerce").astype("Int64")
        out["Label_Anomalie"] = normalize_labels(out["Label_Anomalie"])
        out["Commentaire"] = out["Commentaire"].fillna("")

        save_labels(out)
        nb_pos_save = int((out["Label_Anomalie"] == 1).sum())
        nb_neg_save = int((out["Label_Anomalie"] == 0).sum())
        st.success(
            f"Labels sauvegardes dans storage/labels.csv (1={nb_pos_save}, 0={nb_neg_save})"
        )

    st.subheader("Entrainer la Random Forest")
    if st.button("🚀 Entrainer le modele (avec labels sauvegardes)"):
        labels2 = load_labels()
        if labels2.empty:
            st.error("Aucun label sauvegarde. Ajoutez des labels avant d'entrainer.")
        else:
            labels2["Label_Anomalie"] = normalize_labels(labels2["Label_Anomalie"])
            labels2["RowID"] = pd.to_numeric(labels2["RowID"], errors="coerce").astype("Int64")

            value_counts = labels2["Label_Anomalie"].value_counts().sort_index()
            st.write("value_counts() des labels :", value_counts)

            pos = int(value_counts.get(1, 0))
            neg = int(value_counts.get(0, 0))
            if pos < 10 or neg < 10:
                st.error(
                    f"Pas assez de labels : 1={pos}, 0={neg}. Il faut au moins 10 labels de chaque classe."
                )
            else:
                train_df = feat.merge(
                    labels2[["RowID", "Label_Anomalie"]], on="RowID", how="inner"
                )
                if train_df.empty:
                    st.error("Aucune correspondance RowID entre features et labels.")
                else:
                    model, auc, report = train(train_df, "Label_Anomalie")
                    save_model(model, "storage/model.joblib")
                    st.success(f"Modele entraine et sauvegarde. AUC={auc:.3f}")
                    st.text(report)

                    append_run_log(
                        {
                            "type": "train",
                            "timestamp": datetime.utcnow().isoformat(),
                            "rows": int(train_df.shape[0]),
                            "labels_1": pos,
                            "labels_0": neg,
                            "rf_params": RF_PARAMS,
                            "auc": auc,
                            "threshold": 0.5,
                        }
                    )

    st.subheader("Scorer le fichier complet")
    threshold = st.slider("Seuil haut risque (ML)", 0.1, 0.9, 0.6, 0.05)

    if st.button("🧮 Generer les scores (probabilite de risque)"):
        model_path = Path("storage/model.joblib")
        if not model_path.exists():
            st.error("Entrainer le modele avant de scorer (storage/model.joblib manquant).")
        else:
            model = joblib.load(model_path)
            feat["Prob_Risque"] = score(model, feat)
            feat["Classe_ML"] = feat["Prob_Risque"].ge(threshold).map(
                {True: "Haut risque", False: "Risque faible"}
            )

            out_path = "storage/scored.csv"
            feat.to_csv(out_path, index=False, sep=";")
            st.success(f"Export genere : {out_path}")
            st.dataframe(
                feat.sort_values("Prob_Risque", ascending=False).head(50),
                use_container_width=True,
            )

            labels_for_log = load_labels()
            pos_log, neg_log = label_distribution(labels_for_log)
            append_run_log(
                {
                    "type": "score",
                    "timestamp": datetime.utcnow().isoformat(),
                    "rows": int(feat.shape[0]),
                    "labels_1": pos_log,
                    "labels_0": neg_log,
                    "rf_params": RF_PARAMS,
                    "auc": None,
                    "threshold": threshold,
                }
            )
