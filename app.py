from datetime import datetime
from pathlib import Path
import io
import os
import re

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from services.audit import generate_audit_report, log_audit_event, read_audit_events, reset_audit_log
from services.features import CONTROL_RULES, REQUIRED, build_features
from services.model import FEATURES, RF_PARAMS, save_model, score, train
from services.storage import append_run_log, ensure_storage, load_labels, read_run_log, save_labels

st.set_page_config(page_title="RF Audit Tool", layout="wide")
st.markdown(
    """
<style>
:root {
  --bg: #f5f7fb;
  --card: #ffffff;
  --text: #101318;
  --muted: #5b677a;
  --accent: #0b5ed7;
  --danger: #b42318;
  --ok: #107f3e;
}
.block-container { padding-top: 2rem; }
.hero {
  background: linear-gradient(135deg, #0b5ed7 0%, #1f3a8a 100%);
  color: #ffffff;
  padding: 22px 24px;
  border-radius: 16px;
  margin-bottom: 16px;
}
.hero-title { font-size: 24px; font-weight: 700; }
.hero-sub { opacity: 0.9; margin-top: 4px; }
.section-title { font-size: 20px; font-weight: 700; margin: 18px 0 8px; }
.card {
  background: var(--card);
  border: 1px solid #e6e9ef;
  border-radius: 12px;
  padding: 14px 16px;
  box-shadow: 0 1px 2px rgba(16,24,40,.06);
}
.card-info { border-left: 4px solid #1f3a8a; background: #eef2ff; }
.card-safe { border-left: 4px solid #0f766e; background: #ecfeff; }
.card-file { border-left: 4px solid #b45309; background: #fff7ed; }
.card-legal { border-left: 4px solid #6d28d9; background: #f5f3ff; }
.section-title {
  background: linear-gradient(90deg, #eef2ff 0%, #ffffff 70%);
  padding: 8px 12px;
  border-left: 4px solid var(--accent);
  border-radius: 8px;
}
.card-accent {
  border-left: 4px solid var(--accent);
}
.card-danger {
  border-left: 4px solid var(--danger);
}
.card-ok {
  border-left: 4px solid var(--ok);
}
.kpi { font-size: 22px; font-weight: 700; }
.kpi-label { color: var(--muted); font-size: 12px; letter-spacing: .02em; text-transform: uppercase; }
.badge { display: inline-block; padding: 4px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; }
.badge-danger { background: #ffe4e6; color: #b42318; }
.badge-ok { background: #dcfce7; color: #166534; }
.badge-info { background: #e0e7ff; color: #1f3a8a; }
.note {
  background: #f8fafc;
  border: 1px dashed #d0d5dd;
  padding: 10px 12px;
  border-radius: 10px;
  color: var(--muted);
}
.modal-wrap { max-width: 900px; margin: 0 auto; }
.modal-note { color: var(--muted); font-size: 12px; }
.section-gap { height: 18px; }
.section-gap-lg { height: 28px; }
</style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
<div class="hero">
  <div class="hero-title">RF Audit Tool</div>
  <div class="hero-sub">Audit RF, priorisation et scoring local, clair et trace.</div>
</div>
    """,
    unsafe_allow_html=True,
)

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


if "audit_flags" not in st.session_state:
    st.session_state["audit_flags"] = {"file_signature": None, "features_signature": None}
if "export_format" not in st.session_state:
    st.session_state["export_format"] = "CSV (;) - Power BI"
if "last_action" not in st.session_state:
    st.session_state["last_action"] = None
if "column_map" not in st.session_state:
    st.session_state["column_map"] = {}


ensure_storage()

uploaded = st.file_uploader("Importer votre Excel (.xlsx)", type=["xlsx"])

if uploaded is None:
    st.info("Importez un fichier Excel pour demarrer le workflow complet.")
else:
    df = pd.read_excel(uploaded, engine="openpyxl")
    source_name = getattr(uploaded, "name", "fichier_source")
    st.success(f"Fichier chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes (source : {source_name})")
    safe_source = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in Path(source_name).stem).strip("_") or "source"
    st.session_state["default_export_name"] = f"scored_{safe_source}_{datetime.utcnow().strftime('%Y%m%d')}"

    file_sig = (source_name, df.shape[0], df.shape[1])
    if st.session_state["audit_flags"].get("file_signature") != file_sig:
        reset_audit_log()
        log_audit_event(
            "fichier_lu",
            {"file": source_name, "rows": int(df.shape[0]), "columns": list(df.columns)},
        )
        st.session_state["audit_flags"]["file_signature"] = file_sig

    def clean_col_name(col: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(col).lower())

    proposals = {}
    cleaned_cols = {c: clean_col_name(c) for c in df.columns}
    for target in REQUIRED:
        target_clean = clean_col_name(target)
        match = next((c for c, cc in cleaned_cols.items() if cc == target_clean), None)
        if not match:
            match = next((c for c, cc in cleaned_cols.items() if target_clean in cc), None)
        proposals[target] = match

    st.markdown("#### Correspondance des colonnes (adapter aux en-têtes de votre fichier)")
    column_map = {}
    for target in REQUIRED:
        options = ["(aucune)"] + list(df.columns)
        default = st.session_state["column_map"].get(target) or proposals.get(target) or "(aucune)"
        try:
            default_index = options.index(default)
        except ValueError:
            default_index = 0
        column_map[target] = st.selectbox(
            f"Colonne pour {target}",
            options,
            index=default_index,
            key=f"map_{target}",
            help="Choisissez la colonne de votre fichier qui correspond à ce champ attendu.",
        )
    st.session_state["column_map"] = column_map

    df = df.reset_index(drop=True)
    df["RowID"] = df.index.astype(int)
    for target, source in column_map.items():
        if source != "(aucune)" and source in df.columns:
            df[target] = df[source]

    with st.spinner("Génération des contrôles et features..."):
        feat, missing_cols = build_features(df, FATF_DEFAULT, allow_missing=True)
    applicable_rules = [
        r for r in CONTROL_RULES if not set(r.get("depends_on", [])).intersection(missing_cols)
    ]
    feature_sig = (file_sig, tuple(sorted(feat.columns)))
    if st.session_state["audit_flags"].get("features_signature") != feature_sig:
        log_audit_event(
            "features_calculees",
            {
                "file": source_name,
                "used_columns": list(REQUIRED),
                "missing_columns": missing_cols,
                "column_map": column_map,
                "derived_columns": [c for c in feat.columns if c not in df.columns and c not in missing_cols],
                "rules": [r["id"] for r in applicable_rules],
                "model_features": list(FEATURES),
                "fatf_countries": sorted(FATF_DEFAULT),
            },
        )
        st.session_state["audit_flags"]["features_signature"] = feature_sig

    labels = load_labels()
    if not labels.empty:
        labels["Label_Anomalie"] = normalize_labels(labels["Label_Anomalie"])
        labels["RowID"] = pd.to_numeric(labels["RowID"], errors="coerce").astype("Int64")
    nb_pos, nb_neg = label_distribution(labels)

    st.sidebar.markdown("### Etat du fichier")
    st.sidebar.caption(f"Source : {source_name}")
    st.sidebar.metric("Lignes", f"{len(df):,}".replace(",", " "))
    st.sidebar.metric("Anomalies (labels 1)", f"{nb_pos}")
    st.sidebar.metric("OK (labels 0)", f"{nb_neg}")

    view = "Tout"

    if view in ("Tout", "1 Import"):
        with st.expander("1. Import et controles", expanded=True):
            st.markdown('<div class="section-title">1. Import et controles</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Apercu des features</div>', unsafe_allow_html=True)
            if missing_cols:
                st.warning(
                    "Colonnes manquantes dans le fichier source : "
                    + ", ".join(missing_cols)
                    + ". Elles ont été ajoutées avec des valeurs vides, ce qui peut dégrader le scoring."
                )
                disabled_rules = [r["id"] for r in CONTROL_RULES if set(r.get("depends_on", [])).intersection(missing_cols)]
                if disabled_rules:
                    st.caption("Contrôles non appliqués faute de colonnes : " + ", ".join(disabled_rules))
            st.caption("Colonnes présentes : " + ", ".join(df.columns))
            st.dataframe(feat.head(20), width="stretch")
        st.markdown('<div class="section-gap-lg"></div>', unsafe_allow_html=True)

    if view in ("Tout", "2 Labeling"):
        with st.expander("2. Priorisation et labeling", expanded=True):
            st.markdown('<div class="section-title">2. Priorisation et labeling</div>', unsafe_allow_html=True)
            st.markdown('<div class="card">Aide rapide : utilisez <b>Priorite</b> pour comprendre la mise en avant, cochez <b>Signaler</b> pour marquer une anomalie.</div>', unsafe_allow_html=True)
            st.caption(f"Labels charges : {len(labels)} ligne(s)")
            st.caption(f"Distribution labels 1={nb_pos} | 0={nb_neg}")

            col_pos, col_neg, col_total = st.columns(3)
            col_pos.metric("Labels positifs (1)", nb_pos)
            col_neg.metric("Labels negatifs (0)", nb_neg)
            col_total.metric("Total labels", int(len(labels)))

            st.markdown('<div class="section-title">Synthese rapide</div>', unsafe_allow_html=True)
            col_metrics = st.columns(3)
            with col_metrics[0]:
                st.markdown(
                    f'<div class="card card-accent"><div class="kpi">{len(df):,}'.replace(",", " ")
                    + '</div><div class="kpi-label">Lignes source</div></div>',
                    unsafe_allow_html=True,
                )
            with col_metrics[1]:
                st.markdown(
                    f'<div class="card card-danger"><div class="kpi">{nb_pos}</div><div class="kpi-label">Anomalies (labels 1)</div></div>',
                    unsafe_allow_html=True,
                )
            with col_metrics[2]:
                st.markdown(
                    f'<div class="card card-ok"><div class="kpi">{nb_neg}</div><div class="kpi-label">OK (labels 0)</div></div>',
                    unsafe_allow_html=True,
                )

            st.divider()
            top_n = st.slider(
                "Nombre de lignes a labelliser (priorisees)",
                50,
                1000,
                200,
                step=50,
                help="Fixe combien de lignes prioritaires sont proposées pour le labeling. Augmentez si vous voulez traiter plus de cas en une fois.",
            )
            to_label = feat.sort_values(["RiskRule"], ascending=False).head(top_n).copy()
            to_label = to_label.merge(labels, on="RowID", how="left")

            work_df = to_label.copy()
            work_df["Priorite"] = work_df["RiskRule"].map({1: "Haute", 0: "Normale"}).fillna("Normale")
            work_df["Signaler"] = work_df["Label_Anomalie"].apply(lambda v: normalize_label_value(v) == 1)
            high_count = int((work_df["RiskRule"] == 1).sum())
            st.caption(f"Lignes haute priorite : {high_count} / {len(work_df)}")

            filter_col1, filter_col2, filter_col3 = st.columns(3)
            with filter_col1:
                show_high_only = st.checkbox("Priorite haute seulement", value=False)
            with filter_col2:
                hide_flagged = st.checkbox("Masquer deja signalees", value=False)
            with filter_col3:
                rowid_filter = st.text_input("Filtrer par RowID", value="")

            filtered = work_df.copy()
            if show_high_only:
                filtered = filtered[filtered["Priorite"] == "Haute"]
            if hide_flagged:
                filtered = filtered[~filtered["Signaler"]]
            if rowid_filter.strip().isdigit():
                filtered = filtered[filtered["RowID"] == int(rowid_filter.strip())]

            active_filters = []
            if show_high_only:
                active_filters.append("Priorite haute")
            if hide_flagged:
                active_filters.append("Masquer signalees")
            if rowid_filter.strip().isdigit():
                active_filters.append(f"RowID={int(rowid_filter.strip())}")
            if active_filters:
                st.caption("Filtres actifs : " + " | ".join(active_filters))

            edited = st.data_editor(
                filtered[
                    [
                        "Priorite",
                        "Signaler",
                        "RowID",
                        "NumRF",
                        "DateDon",
                        "Montant",
                        "MontantRF",
                        "AdressePays",
                        "NomDonateur",
                        "RiskRule",
                        "Commentaire",
                    ]
                ],
                width="stretch",
                num_rows="dynamic",
                key="label_editor",
                column_config={
                    "Signaler": st.column_config.CheckboxColumn(
                        "Signaler comme anomalie", help="Cochez pour signaler une anomalie."
                    ),
                },
            )

            st.divider()
            if st.button("Sauvegarder les labels"):
                out = edited[["RowID", "Signaler", "Commentaire"]].copy()
                out = out.dropna(subset=["RowID"])
                out["RowID"] = pd.to_numeric(out["RowID"], errors="coerce").astype("Int64")
                out["Label_Anomalie"] = out["Signaler"].fillna(False).astype(bool).astype(int)
                out["Commentaire"] = out["Commentaire"].fillna("")

                base = labels.copy()
                if base.empty:
                    base = pd.DataFrame(columns=["RowID", "Label_Anomalie", "Commentaire"])
                base["RowID"] = pd.to_numeric(base["RowID"], errors="coerce").astype("Int64")
                base["Label_Anomalie"] = normalize_labels(base["Label_Anomalie"])
                base["Commentaire"] = base["Commentaire"].fillna("")

                merged = pd.concat(
                    [base[["RowID", "Label_Anomalie", "Commentaire"]], out[["RowID", "Label_Anomalie", "Commentaire"]]],
                    ignore_index=True,
                )
                merged = merged.drop_duplicates(subset=["RowID"], keep="last")

                save_labels(merged)
                nb_pos_save = int((merged["Label_Anomalie"] == 1).sum())
                nb_neg_save = int((merged["Label_Anomalie"] == 0).sum())
                st.success(
                    f"Labels sauvegardes dans storage/labels.csv (1={nb_pos_save}, 0={nb_neg_save})"
                )
                log_audit_event(
                    "labels_sauvegardes",
                    {
                        "file": source_name,
                        "rows": int(len(merged)),
                        "labels_1": nb_pos_save,
                        "labels_0": nb_neg_save,
                    },
                )
                st.info(
                    f"Ce qui vient de se passer : labels sauvegardés ({len(merged)} lignes, 1={nb_pos_save}, 0={nb_neg_save}).\n"
                    "Fichier : storage/labels.csv"
                )
        st.markdown('<div class="section-gap-lg"></div>', unsafe_allow_html=True)
    if view in ("Tout", "3 Entrainement"):
        with st.expander("3. Entrainement du modele", expanded=True):
            st.markdown('<div class="section-title">3. Entrainement du modele</div>', unsafe_allow_html=True)
            last_train = next((e for e in reversed(read_run_log()) if e.get("type") == "train"), None)
            if last_train:
                st.caption(
                    f"Modèle entraîné à {last_train.get('timestamp','')} sur {last_train.get('rows','?')} lignes (1={last_train.get('labels_1','?')}, 0={last_train.get('labels_0','?')}), AUC={last_train.get('auc','?')}"
                )
            else:
                st.caption("Modèle non entraîné ou journal absent.")
            if st.button("Entrainer le modele (avec labels sauvegardes)"):
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

                            st.markdown(
                                '<div class="section-title">Features les plus influentes (importance moyenne)</div>',
                                unsafe_allow_html=True,
                            )
                            st.dataframe(feat_importances.head(8), width="stretch")
                            top_chart = (
                                alt.Chart(feat_importances.head(10))
                                .mark_bar(color="#0b5ed7")
                                .encode(
                                    x=alt.X("importance:Q", title="Importance moyenne"),
                                    y=alt.Y("feature:N", sort="-x", title="Feature"),
                                    tooltip=["feature", "importance"],
                                )
                                .properties(height=320)
                            )
                            st.altair_chart(top_chart, use_container_width=True)

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
                            log_audit_event(
                                "entrainement_modele",
                                {
                                    "rows": int(train_df.shape[0]),
                                    "labels_1": pos,
                                    "labels_0": neg,
                                    "auc": auc,
                                    "rf_params": RF_PARAMS,
                                },
                            )
                            st.info(
                                f"Ce qui vient de se passer : modèle entraîné sur {train_df.shape[0]} lignes (1={pos}, 0={neg}), AUC={auc:.3f}. "
                                "Fichier modèle : storage/model.joblib"
                            )
        st.markdown('<div class="section-gap-lg"></div>', unsafe_allow_html=True)

    if view in ("Tout", "4 Scoring"):
        with st.expander("4. Scoring et rapport", expanded=True):
            st.markdown('<div class="section-title">4. Scoring et rapport</div>', unsafe_allow_html=True)
            threshold = st.slider(
                "Seuil haut risque (ML)",
                0.1,
                0.9,
                0.6,
                0.05,
                help="Plus le seuil est élevé, moins il y aura d'alertes mais elles seront plus ciblées.",
            )
            export_name = st.text_input(
                "Nom du fichier de sortie",
                value=st.session_state.get("default_export_name", "scored"),
                help="Nom utilisé pour le fichier exporté (extension ajoutée automatiquement).",
            )
            export_format = st.selectbox(
                "Format du fichier",
                ["CSV (;) - Power BI", "Excel (.xlsx)"],
                index=["CSV (;) - Power BI", "Excel (.xlsx)"].index(st.session_state.get("export_format", "CSV (;) - Power BI")),
                help="Format d'export. CSV ; est pratique pour Power BI / Excel avec séparateur point-virgule.",
            )
            st.session_state["export_format"] = export_format

            if st.button("Generer les scores (probabilite de risque)"):
                model_path = Path("storage/model.joblib")
                if not model_path.exists():
                    st.error("Entraîner le modèle avant de scorer (storage/model.joblib manquant).")
                else:
                    model = joblib.load(model_path)
                    feat["Prob_Risque"] = score(model, feat)
                    feat["Classe_ML"] = feat["Prob_Risque"].ge(threshold).map(
                        {True: "Haut risque", False: "Risque faible"}
                    )

                    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in export_name).strip("_")
                    safe_name = safe_name or "scored"

                    if export_format.startswith("CSV"):
                        out_path = f"storage/{safe_name}.csv"
                        feat.to_csv(out_path, index=False, sep=";", encoding="utf-8-sig")
                        file_data = feat.to_csv(index=False, sep=";", encoding="utf-8-sig")
                        file_name = f"{safe_name}.csv"
                        mime = "text/csv"
                    else:
                        out_path = f"storage/{safe_name}.xlsx"
                        buffer = io.BytesIO()
                        feat.to_excel(buffer, index=False, engine="openpyxl")
                        file_data = buffer.getvalue()
                        file_name = f"{safe_name}.xlsx"
                        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    st.success(f"Export généré : {out_path}")
                    st.download_button(
                        label=f"Telecharger {file_name}",
                        data=file_data,
                        file_name=file_name,
                        mime=mime,
                    )
                    nb_alertes = int((feat["Prob_Risque"] >= threshold).sum())
                    st.caption(f"Alertes >= seuil {threshold:.2f} : {nb_alertes} / {len(feat)} lignes")
                    st.markdown(
                        '<div class="card"><span class="badge badge-danger">Haut risque</span> '
                        '<span class="badge badge-ok">Risque faible</span></div>',
                        unsafe_allow_html=True,
                    )

                    # Histogramme des probabilites
                    proba_df = feat[["Prob_Risque", "Classe_ML"]].copy()
                    prob_hist = (
                        alt.Chart(proba_df)
                        .mark_bar(opacity=0.7)
                        .encode(
                            x=alt.X("Prob_Risque:Q", bin=alt.Bin(maxbins=30), title="Probabilité de risque"),
                            y=alt.Y("count():Q", title="Nombre de lignes"),
                            color=alt.Color("Classe_ML:N", title="Classe", scale=alt.Scale(scheme="redyellowgreen")),
                            tooltip=["count()"],
                        )
                        .properties(height=300)
                    )
                    seuil_rule = alt.Chart(pd.DataFrame({"threshold": [threshold]})).mark_rule(color="#0b5ed7").encode(
                        x="threshold:Q"
                    )
                    st.altair_chart(prob_hist + seuil_rule, use_container_width=True)

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
                    log_audit_event(
                        "scoring_effectue",
                        {
                            "file": source_name,
                            "rows": int(feat.shape[0]),
                            "threshold": threshold,
                            "alerts": nb_alertes,
                            "export": out_path,
                            "labels_1": pos_log,
                            "labels_0": neg_log,
                        },
                    )
                    st.info(
                        f"Ce qui vient de se passer : scoring terminé ({len(feat)} lignes, seuil={threshold:.2f}, alertes={nb_alertes}). "
                        f"Export : {out_path}"
                    )

                    # Rapport synthétique par fichier (format visuel en markdown)
                    risk_ratio = nb_alertes / max(len(feat), 1)
                    bar_blocks = int(round(risk_ratio * 20))
                    bar = "[" + ("#" * bar_blocks).ljust(20, ".") + "]"
                    importances = None
                    top_feats_text = ""
                    feature_importance_error = None
                    if hasattr(model, "feature_importances_"):
                        try:
                            importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
                            top_feats = importances.head(3).index.tolist()
                            top_feats_text = "Principaux signaux: " + ", ".join(top_feats)
                        except Exception as exc:
                            feature_importance_error = exc
                            top_feats_text = f"Importances non disponibles (erreur: {exc.__class__.__name__}). Réentraînez le modèle si besoin."
                    else:
                        top_feats_text = "Le modèle chargé ne fournit pas d'importances. Réentraînez avec le RandomForest."

                    report_lines = [
                        "# Rapport scoring RF",
                        "",
                        "## Synthese",
                        "| Indicateur | Valeur |",
                        "| --- | --- |",
                        f"| Source | {source_name} |",
                        f"| Lignes traitees | {len(feat)} |",
                        f"| Seuil haut risque | {threshold:.2f} |",
                        f"| Alertes (>= seuil) | {nb_alertes} |",
                        f"| Labels disponibles | 1={pos_log} / 0={neg_log} |",
                        "",
                        f"### Taux d'alertes (visuel)\n{bar} {risk_ratio:.1%}",
                        "",
                    ]
                    if importances is not None:
                        report_lines.append("## Top features (importance moyenne)")
                        report_lines.append("| Feature | Importance |")
                        report_lines.append("| --- | --- |")
                        for name, val in importances.head(5).items():
                            report_lines.append(f"| {name} | {val:.3f} |")
                        report_lines.append("")
                    else:
                        report_lines.append("## Top features (non disponibles)")
                        if feature_importance_error:
                            report_lines.append(f"_Raison: {feature_importance_error.__class__.__name__}_")
                        report_lines.append("")

                    report_name = f"storage/report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
                    Path(report_name).write_text("\n".join(report_lines), encoding="utf-8")
                    st.info(f"Rapport sauvegardé : {report_name}")
                    log_audit_event(
                        "rapport_markdown_genere",
                        {"file": source_name, "path": report_name, "threshold": threshold, "alerts": nb_alertes},
                    )

                    # Interpretation courte
                    risk_pct = risk_ratio * 100
                    if risk_pct >= 20:
                        rating = "élevé"
                    elif risk_pct >= 10:
                        rating = "modéré"
                    else:
                        rating = "faible"
                    bareme = "Barème : <10% faible | 10-20% modéré | >20% élevé."
                    summary = (
                        f"Avec un seuil à {threshold:.2f}, {nb_alertes} ligne(s) sur {len(feat)} "
                        f"sont classées haut risque ({risk_pct:.1f}%, niveau {rating})."
                    )
                    if not top_feats_text:
                        top_feats_text = "Le modèle ne fournit pas d'importances. Réentraînez si nécessaire."
                    st.markdown(f"**Interpretation rapide :** {summary} {top_feats_text} {bareme}")

    st.markdown('<div class="section-gap-lg"></div>', unsafe_allow_html=True)
