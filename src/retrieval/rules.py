"""Référentiel de règles : lecture, recherche par catégorie, clauses manquantes, déclencheur RGPD.

Deux voies pour trouver la règle d'une clause :
  1. directe : le classifieur a dit "Cap On Liability" -> règle R06 (rule_for)
  2. par recherche : on cherche dans l'index (BM25 / dense / hybride) -> c'est ce qu'on mesure
     avec Recall@5 et MRR, la vérité terrain étant la catégorie CUAD de chaque clause de test.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .corpus import Article, hash_text

RULES_SOURCE_ID = "rules"
NEUTRAL_LABEL = "Autre"     # classe des paragraphes qui ne sont aucune des 41 catégories


@dataclass
class RuleBook:
    version: str
    validated: bool
    rules: list[dict]
    rgpd_checks: list[dict]
    rgpd_triggers: list[str]

    def by_category(self) -> dict[str, dict]:
        return {r["category"]: r for r in self.rules}


def load_rules(path: Path | str) -> RuleBook:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return RuleBook(data["version"], data.get("validated", False), data["rules"],
                    data.get("rgpd_checks", []), data.get("rgpd_triggers", []))


def rule_for(book: RuleBook, category: str) -> dict | None:
    return book.by_category().get(category)


def missing_categories(book: RuleBook, found: set[str]) -> list[dict]:
    """Clauses attendues mais absentes du contrat : les règles `absence_a_signaler`
    dont la catégorie n'a été prédite sur aucune clause. C'est l'erreur la plus fréquente
    en audit manuel (cf. slide 2)."""
    return [r for r in book.rules if r["absence_a_signaler"] and r["category"] not in found]


def rgpd_candidate(book: RuleBook, clause_text: str) -> bool:
    """Le classifieur n'a pas de catégorie "données personnelles". On repère donc ces clauses
    par mots-clés ; la recherche dans le corpus RGPD prend ensuite le relais."""
    low = clause_text.lower()
    return any(re.search(rf"\b{re.escape(t)}", low) for t in book.rgpd_triggers)


def validate_against_labels(book: RuleBook, labels: list[str]) -> dict[str, list[str]]:
    """Vérifie que chaque catégorie du classifieur a exactement une règle (et inversement)."""
    cats = [r["category"] for r in book.rules]
    labels = [l for l in labels if l != NEUTRAL_LABEL]      # la classe "Autre" n'a pas de règle : c'est voulu
    return {
        "sans_regle": sorted(set(labels) - set(cats)),
        "regle_orpheline": sorted(set(cats) - set(labels)),
        "doublons": sorted({c for c in cats if cats.count(c) > 1}),
    }


def rules_to_articles(book: RuleBook) -> list[Article]:
    """Les règles sont indexées comme des articles : la même recherche retrouve règles et lois.
    `keywords_en` rend BM25 utilisable face à des clauses en anglais (sinon la comparaison
    BM25 / dense serait faussée par la langue, pas par la méthode)."""
    out = []
    for r in book.rules:
        text = f"{r['titre']}. {r['regle']}"
        provenance = r.get("provenance", "")
        if r.get("definition_cuad"):
            provenance += f" : « {r['definition_cuad']} »"
        out.append(Article(
            article_id=f"{RULES_SOURCE_ID}:{r['id']}",
            source_id=RULES_SOURCE_ID,
            law_title="Référentiel de règles contractuelles",
            number=r["id"],
            text=text,
            lang="fr",
            jurisdiction="playbook",
            content_hash=hash_text(text + r["keywords_en"] + r["category"] + provenance),
            validated=book.validated,
            keywords=f"{r['category']} {r['keywords_en']}",
            provenance=provenance,
        ))
    return out
