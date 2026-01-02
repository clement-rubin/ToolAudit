# RF Audit Tool — Transparence & Fonctionnement

Outil Streamlit d'audit RF 100% local et open source (streamlit, pandas, numpy, scikit-learn, joblib, pyyaml, openpyxl). Aucun service externe : les donnees restent sur la machine.

## Vue audit (de l'Excel au score)
- **Source Excel** : registre RF lu en lecture seule. Colonnes minimales : `NumRF, Montant, MontantRF, DateDon, AdressePays, NomDonateur`. Un `RowID` interne est ajoute pour suivre chaque ligne.
- **Construction des controles** (`services/features.py`) : flags explicites dans le code (rupture de sequence NumRF, doublons, montants nuls/negatifs, pays GAFI simplifie, identite vide, saisonnalite, etc.). Tous calculs reproduisibles.
- **Priorisation RiskRule** : combinaison de flags pour classer les lignes a examiner d'abord (audit-first).
- **Labeling humain** : saisie `Label_Anomalie` (0/1 ou True/False) + `Commentaire`. Normalisation robuste : True/1 -> 1 ; vide/False/autre -> 0. Sauvegarde dans `storage/labels.csv` (separateur `;`). Recap des 1/0 affiche.
- **Garde-fous avant ML** : valeur minimale 10 labels positifs et 10 negatifs ; blocage si condition non remplie ou si aucun `RowID` ne matche.
- **Entrainement** (`services/model.py`) : RandomForest avec hyperparametres fixes (`RF_PARAMS`) et `random_state=42` pour reproductibilite. Rapport AUC + classification report affiches. Modele persiste dans `storage/model.joblib`.
- **Scoring** : generation `Prob_Risque` (0-1) et `Classe_ML` ("Haut risque"/"Risque faible") selon seuil reglable. Export `storage/scored.csv` en `;`, directement lisible (Power BI, Excel).
- **Traçabilite** : chaque train/score ecrit une entree dans `storage/run_log.json` (timestamp, volumes, nb labels, params RF, AUC, seuil).

## Comment fonctionne la Random Forest ici ?
- **Features alimentees** : seule la colonne `RowID` est ignoree par le modele ; toutes les colonnes derivees par `services/features.py` (flags de controles metier + montants normalises + derivees de date) sont injectees dans l'algorithme.
- **Hyperparametrage verrouille** : les parametres `RF_PARAMS` sont fixes dans `services/model.py` (profondeur max, nombre d'arbres, min_samples_leaf…) pour eviter le sur-ajustement et garantir la repetabilite avec `random_state=42`.
- **Equilibre des classes** : l'utilisateur fournit les labels. L'entrainement est bloque s'il y a moins de 10 exemples dans chaque classe afin d'avoir un signal fiable.
- **Metric suivie** : l'AUC est calculee a l'entrainement et affichee dans l'interface. Un `classification_report` complet est imprime pour auditer precision/rappel par classe.
- **Seuil de decision explicite** : la foret retourne une probabilite `Prob_Risque`; le seuil choisi dans l'UI pilote la classe finale `Classe_ML` (Haut risque/Risque faible) et apparait dans `run_log.json` pour audit posteriori.
- **Traçabilite du modele** : le fichier `storage/model.joblib` contient le modele entrainne avec ses hyperparametres ; il peut etre recharge pour reproduire un scoring ou investiguer une decision.

## Transparence des artefacts
- `storage/labels.csv` : labels saisis, schema fixe `RowID;Label_Anomalie;Commentaire`.
- `storage/model.joblib` : RandomForest seriealise, rechargable pour audit posteriori.
- `storage/scored.csv` : dataset source + features + `Prob_Risque` + `Classe_ML`.
- `storage/run_log.json` : journal des actions pour reconstituer l'historique.

## Controles techniques et reproductibilite
- Normalisation des labels et schema enforced avant ecriture (evite NaN/colonnes manquantes).
- Hyperparametres fixes et `random_state` impose pour entrainements repetables.
- Verrous fonctionnels : minimum de labels par classe, presence du modele avant scoring.
- Aucune dependance reseau ou service tiers pendant l'usage.

## Architecture (lecture rapide)
- `app.py` : flux Streamlit, messages, blocages, exports.
- `services/features.py` : calcul des controles/features metier.
- `services/model.py` : definition des features d'entree + entrainement/scoring RF.
- `services/storage.py` : IO securisee (labels, run_log) avec schemas de secours.
