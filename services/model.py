import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

FEATURES = [
    "Ctrl_RuptureSeq",
    "Ctrl_DoublonRF",
    "Ctrl_PaysGAFI",
    "Ctrl_MontantNegatif",
    "Ctrl_IdentiteVide",
    "Ctrl_MontantNul",
    "Ctrl_Montant500",
    "Ctrl_DateApresNov",
    "Ctrl_Etranger",
    "Taille_RuptureSeq",
    "Nb_Doublons_NumRF",
    "Montant",
    "MontantRF",
    "Mois",
    "JourSemaine",
    "EstFinAnnee",
]

RF_PARAMS = {
    "n_estimators": 500,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


def train(df: pd.DataFrame, label_col: str = "Label_Anomalie"):
    X = df[FEATURES].fillna(0)
    y = df[label_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    pred = (proba >= 0.5).astype(int)
    report = classification_report(y_test, pred)

    return model, auc, report


def score(model, df: pd.DataFrame):
    X = df[FEATURES].fillna(0)
    return model.predict_proba(X)[:, 1]


def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
