from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from services.features import build_features
from services.model import FEATURES, RF_PARAMS, save_model, score, train
from services.storage import append_run_log, ensure_storage, load_labels, save_labels

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

with st.sidebar.expander("Guide rapide (toujours visible)"):
    st.markdown(
        """
        1) Importer l'Excel (colonnes NumRF, Montant, MontantRF, DateDon, AdressePays, NomDonateur).
        2) Les controles et priorites sont calcules automatiquement; labellez les lignes haut risque en premier.
        3) Sauvegardez les labels, puis entrainez le modele quand vous avez >=10 oui et >=10 non.
        4) Generez les scores, ajustez le seuil et exportez/scrollez les alertes.
        5) Chaque scoring cree un rapport dans storage/ pour la trace d'audit.
        """
    )

uploaded = st.file_uploader("Importer votre Excel (.xlsx)", type=["xlsx"])

if uploaded is None:
    st.info("Importez un fichier Excel pour demarrer le workflow complet.")
else:
    df = pd.read_excel(uploaded, engine="openpyxl")
    source_name = getattr(uploaded, "name", "fichier_source")
    st.success(f"Fichier chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes (source : {source_name})")

    df = df.reset_index(drop=True)
    df["RowID"] = df.index.astype(int)

    st.subheader("1. Import et controles")
    with st.spinner("Génération des contrôles et features..."):
        feat = build_features(df, FATF_DEFAULT)

    st.subheader("Apercu des features")
    st.dataframe(feat.head(20), use_container_width=True)

    st.subheader("2. Priorisation et labeling")
    labels = load_labels()
    if not labels.empty:
        labels["Label_Anomalie"] = normalize_labels(labels["Label_Anomalie"])
        labels["RowID"] = pd.to_numeric(labels["RowID"], errors="coerce").astype("Int64")
    nb_pos, nb_neg = label_distribution(labels)
    st.caption(f"Labels chargés : {len(labels)} ligne(s)")
    st.caption(f"Distribution labels 1={nb_pos} | 0={nb_neg}")

    col_pos, col_neg, col_total = st.columns(3)
    col_pos.metric("Labels positifs (1)", nb_pos)
    col_neg.metric("Labels negatifs (0)", nb_neg)
    col_total.metric("Total labels", int(len(labels)))

    st.subheader("Synthese rapide")
    col_metrics = st.columns(3)
    col_metrics[0].metric("Lignes source", f"{len(df):,}".replace(",", " "))
    col_metrics[1].metric("Labels 1 (anomalies)", f"{nb_pos}")
    col_metrics[2].metric("Labels 0 (OK)", f"{nb_neg}")

    st.divider()
    top_n = st.slider("Nombre de lignes à labelliser (priorisées)", 50, 1000, 200, step=50)
    to_label = feat.sort_values(["RiskRule"], ascending=False).head(top_n).copy()
    to_label = to_label.merge(labels, on="RowID", how="left")

    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.caption("Vue rapide des lignes prioritaires (RiskRule en rouge quand la priorité est haute).")
        priority_cols = ["RowID", "NumRF", "Montant", "MontantRF", "AdressePays", "RiskRule"]
        priority_view = to_label[priority_cols].copy()

        def highlight_risk_rule(col):
            return [
                "background-color: #ffd6d6" if (pd.notna(v) and float(v) >= 1) else ""
                for v in col
            ]

        styled_priority = priority_view.style.apply(highlight_risk_rule, subset=["RiskRule"])
        st.dataframe(styled_priority, width="stretch")
    with col_right:
        st.caption("Saisie des labels (1=anomalie, 0=OK) + commentaire.")
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
            width="stretch",
            num_rows="dynamic",
            key="label_editor",
        )

    st.divider()
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
            f"Labels sauvegardés dans storage/labels.csv (1={nb_pos_save}, 0={nb_neg_save})"
        )

    st.subheader("Entrainer la Random Forest")
    if st.button("🚀 Entrainer le modele (avec labels sauvegardes)"):
        labels2 = load_labels()
        if labels2.empty:
            st.error("Aucun label sauvegardé. Ajoutez des labels avant d'entraîner.")
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
                    model, metrics, feat_importances = train(train_df, "Label_Anomalie")
                    auc = metrics.get("auc", 0.0)
                    report_dict = metrics.get("report_dict", {})
                    pos_metrics = report_dict.get("1", {})
                    pos_precision = pos_metrics.get("precision", 0.0)
                    pos_recall = pos_metrics.get("recall", 0.0)

                    save_model(model, "storage/model.joblib")
                    st.success(f"Modele entraine et sauvegarde. AUC={auc:.3f}")

                    col_m = st.columns(3)
                    col_m[0].metric("AUC (test)", f"{auc:.3f}")
                    col_m[1].metric(
                        "Rappel anomalies",
                        f"{pos_recall:.2f}",
                        help="Capacite a reperer les anomalies labellisees (classe 1) sur l'echantillon test.",
                    )
                    col_m[2].metric(
                        "Precision anomalies",
                        f"{pos_precision:.2f}",
                        help="Part des alertes qui sont vraiment des anomalies dans l'echantillon test.",
                    )

                    conf = metrics.get("confusion", {})
                    st.caption(
                        f"Confusion (seuil 0.5 sur l'echantillon test) : "
                        f"TP={conf.get('tp', 0)}, FN={conf.get('fn', 0)}, "
                        f"FP={conf.get('fp', 0)}, TN={conf.get('tn', 0)}. "
                        "TP=true positives (anomalies bien trouvees), FN=anomalies manquees."
                    )

                    st.text(metrics.get("report_text", ""))

                    st.subheader("Features les plus influentes (importance moyenne)")
                    st.dataframe(feat_importances.head(8), width="stretch")

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

    if st.button("📊 Générer les scores (probabilité de risque)"):
        model_path = Path("storage/model.joblib")
        if not model_path.exists():
            st.error("Entraîner le modèle avant de scorer (storage/model.joblib manquant).")
        else:
            model = joblib.load(model_path)
            feat["Prob_Risque"] = score(model, feat)
            feat["Classe_ML"] = feat["Prob_Risque"].ge(threshold).map(
                {True: "Haut risque", False: "Risque faible"}
            )

            out_path = "storage/scored.csv"
            feat.to_csv(out_path, index=False, sep=";")
            st.success(f"Export généré : {out_path}")
            nb_alertes = int((feat["Prob_Risque"] >= threshold).sum())
            st.caption(f"Alertes >= seuil {threshold:.2f} : {nb_alertes} / {len(feat)} lignes")

            top_scores = feat.sort_values("Prob_Risque", ascending=False).head(50)

            def highlight_prob(col):
                return [
                    "background-color: #ffd6d6" if (pd.notna(v) and float(v) >= threshold) else ""
                    for v in col
                ]

            def highlight_status(col):
                return [
                    "background-color: #ffd6d6" if str(v) == "Haut risque" else "background-color: #e6f4ea"
                    for v in col
                ]

            styled_scores = (
                top_scores.style.apply(highlight_prob, subset=["Prob_Risque"])
                .apply(highlight_status, subset=["Classe_ML"])
            )
            st.dataframe(styled_scores, width="stretch")

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

            # Rapport synthétique par fichier
            report_lines = [
                "# Rapport scoring RF",
                f"- Source : {source_name}",
                f"- Lignes traitées : {len(feat)}",
                f"- Seuil haut risque : {threshold:.2f}",
                f"- Alertes (>= seuil) : {nb_alertes}",
                f"- Labels présents au moment du scoring : 1={pos_log}, 0={neg_log}",
                "",
            ]
            try:
                importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
                report_lines.append("## Top features (importance moyenne)")
                for name, val in importances.head(5).items():
                    report_lines.append(f"- {name}: {val:.3f}")
                report_lines.append("")
            except Exception:
                report_lines.append("## Top features (non disponibles)")
                report_lines.append("")

            report_name = f"storage/report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
            Path(report_name).write_text("\n".join(report_lines), encoding="utf-8")
            st.info(f"Rapport sauvegardé : {report_name}")
