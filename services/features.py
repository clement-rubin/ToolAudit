import pandas as pd

REQUIRED = ["NumRF", "Montant", "MontantRF", "DateDon", "AdressePays", "NomDonateur"]

def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

def build_features(df: pd.DataFrame, fatf_countries: set[str]) -> pd.DataFrame:
    d = df.copy()
    validate_columns(d)

    # Types
    d["DateDon"] = pd.to_datetime(d["DateDon"], errors="coerce")
    d["Montant"] = pd.to_numeric(d["Montant"], errors="coerce").fillna(0.0)
    d["MontantRF"] = pd.to_numeric(d["MontantRF"], errors="coerce").fillna(0.0)

    # NumRF numeric (pour séquentialité)
    d["NumRF_numeric"] = pd.to_numeric(d["NumRF"].astype(str), errors="coerce")

    # Doublons
    d["Nb_Doublons_NumRF"] = d.groupby("NumRF")["NumRF"].transform("count")
    d["Ctrl_DoublonRF"] = (d["Nb_Doublons_NumRF"] > 1).astype(int)

    # Rupture de séquentialité (sur NumRF_numeric trié)
    d_sorted = d.sort_values("NumRF_numeric")
    prev = d_sorted["NumRF_numeric"].shift(1)
    gap = (d_sorted["NumRF_numeric"] - prev).fillna(1)
    d_sorted["Taille_RuptureSeq"] = gap.clip(lower=1).astype(int)
    d_sorted["Ctrl_RuptureSeq"] = (d_sorted["Taille_RuptureSeq"] > 1).astype(int)

    # Recolle dans l’ordre d’origine via index
    d = d.join(d_sorted[["Taille_RuptureSeq", "Ctrl_RuptureSeq"]])

    # Pays GAFI
    d["Ctrl_PaysGAFI"] = d["AdressePays"].astype(str).isin(fatf_countries).astype(int)

    # Montant négatif / nul / identité vide
    d["Ctrl_MontantNegatif"] = (d["MontantRF"] < 0).astype(int)
    d["Ctrl_MontantNul"] = (d["Montant"] == 0).astype(int)
    d["Ctrl_IdentiteVide"] = (
        d["NomDonateur"].isna() | (d["NomDonateur"].astype(str).str.strip() == "")
    ).astype(int)

    # Contrôles cumulatifs
    d["Ctrl_Montant500"] = (d["Montant"] >= 500).astype(int)
    d["Ctrl_DateApresNov"] = (d["DateDon"].dt.month.ge(11)).fillna(False).astype(int)
    d["Ctrl_Etranger"] = (d["AdressePays"].astype(str).str.lower() != "france").astype(int)

    # Features calendrier
    d["Mois"] = d["DateDon"].dt.month.fillna(0).astype(int)
    d["JourSemaine"] = d["DateDon"].dt.dayofweek.fillna(0).astype(int)
    d["EstFinAnnee"] = d["Mois"].isin([11, 12]).astype(int)

    # Heuristique “audit” (pour prioriser le labeling)
    # (règle hybride : éliminatoires OU cumuls>=2)
    auto = (
        d["Ctrl_RuptureSeq"] + d["Ctrl_DoublonRF"] + d["Ctrl_PaysGAFI"]
        + d["Ctrl_MontantNegatif"] + d["Ctrl_IdentiteVide"] + d["Ctrl_MontantNul"]
    )
    cumul = d["Ctrl_Montant500"] + d["Ctrl_DateApresNov"] + d["Ctrl_Etranger"]
    d["RiskRule"] = ((auto >= 1) | (cumul >= 2)).astype(int)

    return d
