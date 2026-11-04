"""Fabrication des cas d'évaluation du juge, à partir de contrats de test JAMAIS utilisés pour les mesures précédentes.

Quatre familles (voir metrics.py). Le verdict de référence des familles `playbook` et `rgpd` est donné par l'équipe
(feuille d'annotation) : c'est un accord avec nous, pas une validation juridique. Celui des familles `mismatch` et
`neutral` est connu par construction : le juge doit s'abstenir.

On donne au juge la clause EXACTE annotée par les juristes (pas le segment issu du découpage), pour mesurer le juge seul,
sans mélanger les erreurs du découpage et du classifieur.
"""

from __future__ import annotations

import random
import re

from ..data.autre import neutral_segments
from ..data.header import HEADER_CATEGORIES
from ..data.segment import segment

IDENTITY = set(HEADER_CATEGORIES)

# Familles de catégories proches : une clause ne doit pas être appariée à une règle de sa propre famille
# (une clause de non-concurrence face à la règle d'exclusivité n'est pas "sans rapport").
FAMILIES = {
    "responsabilité": {"Cap On Liability", "Uncapped Liability", "Liquidated Damages", "Insurance", "Warranty Duration"},
    "concurrence": {"Non-Compete", "Exclusivity", "No-Solicit Of Customers", "No-Solicit Of Employees",
                    "Competitive Restriction Exception", "Most Favored Nation", "Price Restrictions", "Non-Disparagement"},
    "licence": {"License Grant", "Non-Transferable License", "Affiliate License-Licensee", "Affiliate License-Licensor",
                "Irrevocable Or Perpetual License", "Unlimited/All-You-Can-Eat-License", "Ip Ownership Assignment",
                "Joint Ip Ownership", "Source Code Escrow", "Covenant Not To Sue"},
    "durée": {"Expiration Date", "Renewal Term", "Notice Period To Terminate Renewal", "Termination For Convenience",
              "Post-Termination Services", "Change Of Control", "Anti-Assignment"},
    "commercial": {"Minimum Commitment", "Volume Restriction", "Revenue/Profit Sharing", "Rofr/Rofo/Rofn", "Audit Rights",
                   "Third Party Beneficiary", "Governing Law"},
}
_FAMILY_OF = {c: f for f, cats in FAMILIES.items() for c in cats}

_PERSONAL = re.compile(r"personal\s+(?:data|information)|data\s+protection|data\s+privacy|gdpr|data\s+subject|"
                       r"identified\s+or\s+identifiable|privacy\s+(?:law|policy|exhibit|regulation)|nonpublic\s+personal", re.I)
MIN_PERSONAL_MENTIONS = 2     # une seule mention est souvent incidente (définition, liste d'obligations) : pas une clause de données


def is_personal_data_clause(t: str) -> bool:
    return len(_PERSONAL.findall(t)) >= MIN_PERSONAL_MENTIONS and "[Redacted" not in t and not looks_like_junk(t)


_CHECK_KEYWORDS = [   # clause sur les données personnelles -> contrôle RGPD le plus proche (G01 par défaut)
    ("G05", r"breach|incident|unauthori[sz]ed\s+(?:access|disclosure)|notif"),
    ("G03", r"sub-?processor|subcontract|third\s+part(?:y|ies)\s+(?:processor|service)"),
    ("G08", r"transfer|outside\s+(?:the\s+)?(?:eea|european)|cross-border|international"),
    ("G06", r"retain|retention|delet|destr|return\s+(?:all|of)\s+(?:personal|data)|erase"),
    ("G04", r"secur|safeguard|encrypt|technical\s+and\s+organi[sz]ational"),
    ("G09", r"audit|inspect"),
    ("G07", r"access\s+request|data\s+subject.{0,40}(?:right|request)|rectif|portab"),
    ("G02", r"instruction|on\s+behalf\s+of|only\s+to\s+the\s+extent\s+necessary"),
]


def _clean(t: str) -> str:
    return " ".join(t.split())


def looks_like_junk(t: str) -> bool:
    """Table des matières, liste de numéros de page, bloc de chiffres : pas une clause."""
    words = t.split()
    if not words:
        return True
    numeric = sum(1 for w in words if re.fullmatch(r"\(?\d{1,3}[.)]?", w)) / len(words)
    return bool(re.match(r"\s*(table of )?contents\b", t, re.I)) or numeric > 0.12


def _rule_dict(r: dict) -> dict:
    return {"id": r["id"], "titre": r["titre"], "regle": r["regle"]}


def _rule_source(r: dict) -> dict:
    return {"id": r["id"], "citation": f"Règle de contrôle {r['id']}", "text": r["regle"]}


def _articles_from_refs(refs: list[str], articles: dict) -> list[dict]:
    """Articles du corpus cités par un contrôle RGPD (« RGPD art. 28(3)(a) » -> article 28). Une plage ne garde que son début."""
    sources, seen = [], set()
    for ref in refs:
        m = re.search(r"art\.\s*(\d+)", ref)
        if not m or f"rgpd:{m.group(1)}" in seen:
            continue
        a = articles.get(f"rgpd:{m.group(1)}")
        if a is not None:
            seen.add(a.article_id)
            sources.append({"id": a.article_id, "citation": a.citation(), "text": a.text})
    return sources


def _pick_check(clause: str, checks: dict[str, dict]) -> dict:
    for gid, pattern in _CHECK_KEYWORDS:
        if gid in checks and re.search(pattern, clause, re.I):
            return checks[gid]
    return checks["G01"]


def build_cases(docs: list[dict], book, articles: dict, *, rgpd_docs: list[dict] | None = None, seed: int = 42, n_matched: int = 50,
                n_rgpd: int = 15, n_mismatch: int = 25, n_neutral: int = 25, rgpd_per_contract: int = 2) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    rules = {r["category"]: r for r in book.rules if r["category"] not in IDENTITY}
    stats = {}

    # 1) clauses réelles + règle adaptée, réparties sur les catégories (les plus rares d'abord), un contrat par tour
    pool: dict[str, list[tuple[str, str]]] = {}
    for d in docs:
        for cat, a, b in d["spans"]:
            t = _clean(d["text"][a:b])
            if cat in rules and b > a and 80 <= len(t) <= 700:
                pool.setdefault(cat, []).append((d["title"], t))
    for cat in pool:
        rng.shuffle(pool[cat])
    matched, used_contracts = [], {c: set() for c in pool}
    order = sorted(pool, key=lambda c: len(pool[c]))
    while len(matched) < n_matched and any(pool.values()):
        for cat in order:
            while pool[cat] and pool[cat][0][0] in used_contracts[cat]:
                pool[cat].pop(0)
            if pool[cat] and len(matched) < n_matched:
                title, t = pool[cat].pop(0)
                used_contracts[cat].add(title)
                matched.append((cat, title, t))
    cases = []
    for cat, title, t in matched:
        r = rules[cat]
        cases.append({"family": "playbook", "category": cat, "clause": t, "rule": _rule_dict(r),
                      "sources": [_rule_source(r)], "allowed_sources": [r["id"]], "expected": None, "contract": title})
    stats["playbook"] = len(cases)

    # 2) clauses sur les données personnelles + contrôle RGPD + articles du corpus.
    #    Ces clauses sont rares dans CUAD : on pioche dans `rgpd_docs` (tous les contrats si fourni), 2 par contrat au plus.
    #    Le juge n'est pas entraîné sur CUAD : aucun risque de fuite à utiliser aussi les contrats d'entraînement.
    checks = {c["id"]: c for c in book.rgpd_checks}
    cand = []
    for d in (rgpd_docs if rgpd_docs is not None else docs):
        for s in segment(d["text"]):
            if 150 <= len(s.text) <= 900 and is_personal_data_clause(s.text):
                cand.append((d["title"], s.text))
    rng.shuffle(cand)
    per_contract: dict[str, int] = {}
    n_added = 0
    for title, t in cand:
        if n_added >= n_rgpd or not checks or per_contract.get(title, 0) >= rgpd_per_contract:
            continue
        chk = _pick_check(t, checks)
        src = _articles_from_refs(chk["refs"], articles)
        if not src:
            continue
        per_contract[title] = per_contract.get(title, 0) + 1
        n_added += 1
        rule = {"id": chk["id"], "titre": chk["theme"], "regle": chk["regle"]}
        cases.append({"family": "rgpd", "category": None, "clause": t, "rule": rule,
                      "sources": [{"id": chk["id"], "citation": f"Contrôle RGPD {chk['id']}", "text": chk["regle"]}] + src,
                      "allowed_sources": [chk["id"]] + [x["id"] for x in src], "expected": None, "contract": title})
    stats["rgpd"] = n_added
    stats["rgpd_contrats"] = len(per_contract)

    # 3) clause + règle SANS RAPPORT (autre famille) : le juge doit s'abstenir
    base = [c for c in cases if c["family"] == "playbook"]
    rng.shuffle(base)
    for c in base[:n_mismatch]:
        fam = _FAMILY_OF.get(c["category"])
        others = [r for cat, r in rules.items() if _FAMILY_OF.get(cat) != fam]
        r = rng.choice(others)
        cases.append({"family": "mismatch", "category": c["category"], "clause": c["clause"], "rule": _rule_dict(r),
                      "sources": [_rule_source(r)], "allowed_sources": [r["id"]], "expected": "a_verifier", "contract": c["contract"]})
    stats["mismatch"] = min(n_mismatch, len(base))

    # 4) paragraphe neutre (aucune des 37 catégories) + règle quelconque : le juge doit s'abstenir
    neutral = []
    for d in docs:
        neutral += [(d["title"], _clean(s.text)) for s in neutral_segments(d)
                    if 80 <= len(_clean(s.text)) <= 700 and not looks_like_junk(s.text)]
    rng.shuffle(neutral)
    for title, t in neutral[:n_neutral]:
        r = rng.choice(list(rules.values()))
        cases.append({"family": "neutral", "category": None, "clause": t, "rule": _rule_dict(r),
                      "sources": [_rule_source(r)], "allowed_sources": [r["id"]], "expected": "a_verifier", "contract": title})
    stats["neutral"] = min(n_neutral, len(neutral))

    rng.shuffle(cases)
    for i, c in enumerate(cases, 1):
        c["id"] = f"J{i:03d}"
        c["a_annoter"] = c["family"] in ("playbook", "rgpd")
    return cases, stats
