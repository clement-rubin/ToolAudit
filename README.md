# Guide terrain - Outil d'audit RF

Ce document est concu pour un auditeur. Il contient trois parties : 1) comment utiliser l'outil, 2) comment la Random Forest est utilisee, 3) mentions legales et infos complementaires.

## 1. Utiliser l'outil (sans jargon)
- Preparer le fichier Excel avec au minimum les colonnes : `NumRF`, `Montant`, `MontantRF`, `DateDon`, `AdressePays`, `NomDonateur`.
- Lancer l'interface : ouvrir un terminal dans le dossier du projet puis taper `streamlit run app.py`. Une page s'ouvre dans le navigateur.
- Importer l'Excel. L'outil indique le nombre de lignes/colonnes et ajoute un identifiant interne `RowID`.
- Voir les controles automatiques (doublons, ruptures de sequence, montants negatifs/nuls, pays sensibles, identite vide, effets de fin d'annee). Ils servent a trier les lignes a verifier.
- Dans "Labeling", choisir le nombre de lignes prioritaires (tri sur `RiskRule`), puis saisir : `1` = anomalie, `0` ou vide = OK. Ajouter un commentaire court (pourquoi c'est douteux / pourquoi c'est OK).
- Cliquer "Sauvegarder les labels". Les decisions sont stockees dans `storage/labels.csv`.
- Quand il y a au moins 10 anomalies et 10 OK, cliquer "Entrainer le modele". L'outil affiche AUC, rappel, precision, confusion, et les controles qui comptent le plus.
- Regler le seuil haut risque (curseur), puis "Generer les scores". Chaque ligne recoit une probabilite (`Prob_Risque`) et une classe ("Haut risque" / "Risque faible"). Export dans `storage/scored.csv`.
- Un rapport synthese est genere automatiquement dans `storage/report_YYYYMMDD_HHMMSS.md` (source, volumes, seuil, nb d'alertes, top features).
- Conseils pratiques : conserver la copie source et les exports par date, remplir le champ commentaire, surveiller `run_log.json` qui trace chaque entrainement/scoring.

## 2. Comment l'algo Random Forest est utilise ici
- Idee simple : une foret de petits arbres de decision. Chaque arbre examine les controles metiers (doublons, ruptures, montants, calendrier, pays, identite) et propose un avis. La foret vote pour donner une probabilite de risque.
- Reglages fixes pour rester reproductible : 500 arbres, feuilles >= 5 observations, profondeur limitee, `class_weight="balanced"` pour ne pas ignorer les anomalies rares, `random_state=42`.
- Schema resumant le flux :
```
Excel source -> controles metiers -> foret d'arbres (vote) -> Prob_Risque (0-1) -> seuil -> Haut risque / Risque faible
```
- Evaluation integree : l'outil montre AUC (qualite globale), rappel/precision sur les anomalies, matrice de confusion simplifiee (TP/FN/FP/TN) et les controles les plus influents. Cela aide a expliquer pourquoi une alerte sort.
- Sorties stockees pour audit : le modele est enregistre dans `storage/model.joblib`, les scores complets dans `storage/scored.csv`, les rapports dans `storage/report_*.md`.

## 3. Mentions legales et infos complementaires
- Donnees locales : aucune donnee n'est envoyee en ligne pendant l'utilisation; tout reste sur la machine.
- Fichiers de travail : `storage/labels.csv` (labels et commentaires), `storage/scored.csv` (scores avec probabilites et classes), `storage/run_log.json` (journal des operations), `storage/report_*.md` (rapports de scoring), `storage/model.joblib` (modele).
- Confidentialite : proteger les fichiers sources et exports. Supprimer ou chiffrer les copies apres usage selon la politique interne.
- Responsabilite : les labels saisis par l'auditeur guident le modele; verifier les alertes critiques avant decision finale.
- Licence : voir le fichier `LICENSE` du projet.
