# Phase 5 — critères de réussite, fixés AVANT toute mesure

Ce fichier est commité avant de lancer le moindre modèle : l'historique Git en donne la date. Les seuils ne seront pas modifiés
après avoir vu un résultat. Si un critère n'est pas atteint, le résultat est publié tel quel, avec le défaut identifié.

## Ce qu'on mesure, et avec quelle vérité de référence

CUAD ne contient aucun verdict de conformité. Il n'y a donc pas de « bonne réponse » officielle pour « conforme / non conforme ».

| Étape | Mesure | Vérité de référence |
| --- | --- | --- |
| 1 | Dire qu'une clause attendue est **manquante** | annotations des juristes CUAD (30 contrats jamais utilisés pour le diagnostic) |
| 2 | Verdict conforme / non conforme / à vérifier | annotations de l'**équipe** (environ 65 cas) : un accord avec nous, **pas une validation juridique** |
| 3 | **Citer des sources qui existent** | contrôle automatique : toute source citée doit figurer dans le prompt |
| 4 | **S'abstenir** quand la clause ne concerne pas la règle | paires construites exprès (clause + règle sans rapport, ou paragraphe neutre) : le bon verdict est `a_verifier` |

## Étape 1 — verdict « manquante » (sans modèle de langage)

Catégories dont l'absence est à signaler : celles dont le champ `absence_a_signaler` vaut vrai dans le référentiel de règles.
Réglage : une catégorie est présente si un segment la reçoit avec une confiance de 0,5 au moins (réponse la plus probable), ou si
l'extracteur d'identité trouve l'information. Aucun autre réglage n'est essayé.

* **rappel de « manquante » : 60 % ou plus**
* **précision de « manquante » : 50 % ou plus** (à titre de repère, répondre toujours « manquante » donnerait une précision égale au taux de base)

## Étapes 2 à 4 — juge à modèle ouvert (Qwen2.5, en précision réduite fp16 sur Colab)

Sur les réponses avec règle fournie (condition `with_rule`), par modèle :

* **JSON valide : 95 % ou plus** des réponses
* **sources inventées : 5 % ou moins** des réponses (une source citée qui n'était pas dans le prompt)
* **abstention sur les cas sans rapport : 70 % ou plus** de `a_verifier`
* **accord avec l'équipe : 60 % ou plus, ET au moins 10 points au-dessus de la réponse toujours majoritaire** sur les cas annotés
* **fausse abstention : 30 % ou moins** (`a_verifier` alors que l'équipe a tranché)

Mesurés mais sans seuil : l'apport de la règle (même modèle, avec ou sans règle fournie), et l'accord entre les deux annotateurs
(coefficient kappa sur les cas annotés par les deux ; en dessous de 0,4, les conclusions sur l'accord sont déclarées peu fiables).

## Limites assumées d'avance

* Les verdicts de référence sont ceux de deux étudiantes, pas de juristes.
* Le corpus de lois ne contient que le RGPD et la loi tunisienne 2004-63 : seule la famille « données personnelles » peut citer un
  article de loi ; pour les autres catégories, la source citable est la règle de contrôle de l'équipe.
* Les contrats sont anglais (CUAD) ; les règles sont en français.
