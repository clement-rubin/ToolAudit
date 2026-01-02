# Guide terrain  
## Outil d’aide à la priorisation des reçus fiscaux

> **Document interne – à destination des auditeurs**  
> Outil d’assistance à l’analyse des risques sur les reçus fiscaux  
> *(ne se substitue pas au jugement professionnel)*

---

## 🎯 Objet de l’outil

Cet outil a pour objectif de **faciliter l’identification et la priorisation des reçus fiscaux à examiner**, à partir :
- de contrôles automatisés fondés sur des règles explicites,
- des qualifications réalisées par l’auditeur,
- d’un traitement statistique destiné à ordonner les lignes selon leur niveau de risque.

Il s’agit d’un **outil d’aide**, utilisé en complément des diligences d’audit habituelles.

---

## 1. Données utilisées

### 1.1 Source
- Registre des reçus fiscaux transmis sous format Excel.
- Fichier utilisé **en lecture seule**.

### 1.2 Informations minimales attendues
- Numéro du reçu fiscal  
- Montant du don  
- Montant du reçu fiscal  
- Date du don  
- Pays ou localisation du donateur  
- Nom du donateur  

Chaque ligne est associée à un identifiant interne permettant un **suivi ligne à ligne**.

---

## 2. Contrôles réalisés automatiquement

Les contrôles reposent sur des **règles définies à l’avance**, identiques pour tous les dossiers, notamment :
- cohérence et continuité des numéros de reçus,
- recherche de doublons,
- détection de montants atypiques (nuls ou négatifs),
- identification de pays présentant un risque particulier,
- absence ou insuffisance d’informations sur le donateur,
- variations inhabituelles dans le temps.

> ℹ️ Les règles sont **documentées, reproductibles et traçables**.

---

## 3. Classement initial des lignes

À partir des contrôles réalisés, l’outil établit un **premier classement des lignes**, destiné à :
- faire ressortir celles cumulant plusieurs signaux d’alerte,
- orienter en priorité les travaux de l’auditeur.

Ce classement est effectué **avant toute analyse statistique**.

---

## 4. Qualification par l’auditeur

L’auditeur examine les lignes proposées et renseigne :
- l’existence ou non d’une anomalie,
- un commentaire justifiant son appréciation.

Ces informations constituent :
- une **trace du jugement professionnel**,
- une base d’analyse progressive pour les traitements ultérieurs.

Un seuil minimal de qualifications est requis avant toute exploitation statistique.

---

## 5. Score de risque

Lorsque les conditions sont réunies :
- chaque ligne est associée à un **score de risque compris entre 0 et 1**,
- un niveau de priorité est proposé :
  - *Risque faible*
  - *Risque élevé*

Le seuil de classement peut être ajusté en fonction du contexte de la mission.

---

## 6. Restitution des résultats

Les résultats sont restitués sous la forme :
- d’un fichier Excel reprenant le registre initial,
- enrichi d’un score de risque et d’un niveau de priorité.

Ce support est directement exploitable pour :
- la planification des tests,
- la documentation du dossier d’audit,
- des analyses complémentaires.

---

## 7. Traçabilité et historique

Chaque utilisation de l’outil génère un historique précisant notamment :
- la date d’utilisation,
- le volume de données traitées,
- le nombre de qualifications disponibles,
- les paramètres de classement retenus.

Cette traçabilité permet de **justifier a posteriori les choix opérés**.

---

## 8. Principes de fiabilité

- Règles fixes et connues.
- Résultats reproductibles à données identiques.
- Blocages empêchant toute utilisation dans des conditions non fiables.
- Fonctionnement sans recours à des services externes.

---

## 9. Positionnement au regard des normes d’audit

### NEP 315 — Identification et évaluation des risques
L’outil contribue à l’identification des risques d’anomalies significatives en facilitant :
- l’analyse de populations complètes,
- la détection de situations atypiques,
- l’orientation des travaux vers les zones à risque.

---

### NEP 520 — Procédures analytiques
Le score constitue une **procédure analytique d’aide**, utilisée pour :
- orienter les diligences,
- apprécier la cohérence globale des données.

Toute anomalie identifiée doit faire l’objet :
- d’investigations complémentaires,
- d’éléments probants,
- d’une conclusion fondée sur le jugement professionnel.

---

## 10. Rappel déont[0;36m[0;36mologique

L’outil n’émet **aucune conclusion d’audit**.  
Les décisions, analyses et conclusions demeurent **sous la responsabilité exclusive de l’auditeur**.

---
