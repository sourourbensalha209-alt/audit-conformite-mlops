"""Corpus de lois, article par article, avec historique des versions.

Un article = une unite de recherche et de citation. Chaque article porte une
empreinte (hash) de son texte : c'est elle qui permet de detecter qu'une loi a
change sans rien relire, et de ne recalculer que ce qui a bouge.

Fichiers ecrits dans le dossier du corpus (a versionner avec DVC) :
    articles.jsonl   etat courant, un article par ligne
    changelog.jsonl  journal ajoute-seulement : ajoute / modifie / abroge / signale
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# Modele
# --------------------------------------------------------------------------- #


@dataclass
class Article:
    article_id: str                 # "tn-2004-63:15"
    source_id: str                  # "tn-2004-63"
    law_title: str
    number: str                     # "15", "5-bis"
    text: str
    lang: str = "fr"
    jurisdiction: str = ""
    content_hash: str = ""
    version: int = 1
    status: str = "in_force"        # in_force | repealed
    validated: bool = True          # False : texte recent, pas encore relu par un juriste
    under_review: bool = False      # un texte recent annonce modifier cet article
    first_seen: str = ""
    last_changed: str = ""
    publication_date: str | None = None
    effective_from: str | None = None
    source_url: str | None = None
    keywords: str = ""              # mots-cles ajoutes a l'indexation (utile pour les regles)

    def citation(self) -> str:
        """Ex. "Loi organique n° 2004-63 du 27 juillet 2004, art. 15" (titre coupé avant "portant...")."""
        short = re.split(r"\s+(?:portant|relative?|relatif|modifiant|fixant)\b", self.law_title)[0].rstrip(" ,")
        return f"{short}, art. {self.number}"


@dataclass
class Change:
    ts: str
    source_id: str
    article_id: str
    change: str                     # added | modified | repealed | flagged
    old_hash: str | None = None
    new_hash: str | None = None
    old_text: str | None = None     # permet de reconstituer la version precedente
    note: str | None = None


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def collapse(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def hash_text(text: str) -> str:
    """Empreinte insensible aux retours a la ligne et aux espaces multiples
    (un PDF re-mis en page ne doit pas ressembler a une modification de loi)."""
    return hashlib.sha256(collapse(text).encode("utf-8")).hexdigest()


def corpus_version(articles: dict[str, Article]) -> str:
    """Identifiant court de l'etat du corpus. Change si, et seulement si, le contenu change.
    A inscrire dans chaque rapport d'audit : on sait ainsi sur quelle version des lois il s'appuie."""
    lines = sorted(f"{a.article_id}|{a.content_hash}|{a.status}" for a in articles.values())
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Decoupage d'un texte de loi en articles
# --------------------------------------------------------------------------- #

_ART_RE = re.compile(
    r"^\s*(?:Article|Art\.)\s+(premier|1er|\d+)"
    r"(?:\s*[-–]?\s*\(?(bis|ter|quater|quinquies|nouveau)\)?)?"
    r"(?:\s*[:.\-–—)]\s*(?P<rest>.*)|\s*$)"
)
_HEADING_RE = re.compile(
    r"^\s*(?:CHAPITRE|CHAPTER|SECTION|TITRE|TITLE|LIVRE|Chapitre|Chapter|Section|Titre|Title|Livre)"
    r"\s+(?:premier|first|[IVXLC]+|\d+)\b.*$"
)


def _norm_number(num: str, suffix: str | None) -> str:
    n = num.lower()
    if n in ("premier", "1er"):
        n = "1"
    return f"{n}-{suffix.lower()}" if suffix else n


def parse_articles(
    text: str,
    source_id: str,
    law_title: str,
    *,
    lang: str = "fr",
    jurisdiction: str = "",
    validated: bool = True,
    publication_date: str | None = None,
    effective_from: str | None = None,
    source_url: str | None = None,
) -> list[Article]:
    """Decoupe un texte en articles.

    Reconnait : "Article 5", "Article 5 - ...", "Art. 2. - ...", "Article premier",
    "Article 5 bis", "Article 12 (nouveau)". Une ligne qui commence par "Article 5 de la loi..."
    (renvoi au milieu d'une phrase) n'est PAS un debut d'article.
    """
    blocks: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        m = _ART_RE.match(line)
        if m:
            number = _norm_number(m.group(1), m.group(2))
            rest = (m.group("rest") or "").lstrip(" .-–—:)").strip()
            blocks.append((number, [rest] if rest else []))
        elif _HEADING_RE.match(line):
            continue
        elif blocks:
            blocks[-1][1].append(line.rstrip())

    articles: list[Article] = []
    seen: dict[str, int] = {}
    for number, lines in blocks:
        body = "\n".join(lines).strip()
        if not body:
            continue
        seen[number] = seen.get(number, 0) + 1
        suffix = "" if seen[number] == 1 else f"#{seen[number]}"
        articles.append(Article(
            article_id=f"{source_id}:{number}{suffix}",
            source_id=source_id,
            law_title=law_title,
            number=number + suffix,
            text=body,
            lang=lang,
            jurisdiction=jurisdiction,
            content_hash=hash_text(body),
            validated=validated,
            publication_date=publication_date,
            effective_from=effective_from,
            source_url=source_url,
        ))
    return articles


# --------------------------------------------------------------------------- #
# Stockage et comparaison
# --------------------------------------------------------------------------- #


class CorpusStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.articles_path = self.root / "articles.jsonl"
        self.changelog_path = self.root / "changelog.jsonl"

    # -- lecture / ecriture
    def load(self) -> dict[str, Article]:
        if not self.articles_path.exists():
            return {}
        known = {f.name for f in fields(Article)}
        out: dict[str, Article] = {}
        with open(self.articles_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    a = Article(**{k: v for k, v in row.items() if k in known})
                    out[a.article_id] = a
        return out

    def _save(self, articles: dict[str, Article]) -> None:
        tmp = self.articles_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for a in sorted(articles.values(), key=lambda x: x.article_id):
                f.write(json.dumps(asdict(a), ensure_ascii=False) + "\n")
        tmp.replace(self.articles_path)

    def _log(self, changes: list[Change]) -> None:
        if not changes:
            return
        with open(self.changelog_path, "a", encoding="utf-8") as f:
            for c in changes:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    def version(self) -> str:
        return corpus_version(self.load())

    def history(self, article_id: str) -> list[dict]:
        if not self.changelog_path.exists():
            return []
        with open(self.changelog_path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return [r for r in rows if r["article_id"] == article_id]

    # -- comparaison
    def apply_snapshot(
        self,
        source_id: str,
        incoming: list[Article],
        *,
        full_snapshot: bool,
        now: str | None = None,
    ) -> list[Change]:
        """Compare ce qu'on vient de lire avec ce qu'on connait, et met a jour.

        full_snapshot=True  : `incoming` est le texte COMPLET de la source (ex. RGPD consolide).
                              Un article connu mais absent est considere abroge.
        full_snapshot=False : `incoming` est un texte nouveau (ex. une loi du JORT) ;
                              on ajoute ou on modifie, on n'abroge rien.
        """
        now = now or now_iso()
        current = self.load()
        existing = {i: a for i, a in current.items() if a.source_id == source_id}
        changes: list[Change] = []
        incoming_ids = set()

        for new in incoming:
            incoming_ids.add(new.article_id)
            old = existing.get(new.article_id)
            if old is None:
                new.first_seen = new.last_changed = now
                current[new.article_id] = new
                changes.append(Change(now, source_id, new.article_id, "added", None, new.content_hash))
            elif old.content_hash != new.content_hash or old.status == "repealed":
                new.version = old.version + 1
                new.first_seen = old.first_seen
                new.last_changed = now
                current[new.article_id] = new          # under_review repart a False : le texte a ete mis a jour
                changes.append(Change(
                    now, source_id, new.article_id, "modified",
                    old.content_hash, new.content_hash, old_text=old.text,
                    note="reintroduit" if old.status == "repealed" else None))
            # sinon : identique, on ne touche a rien

        if full_snapshot:
            for aid, old in existing.items():
                if aid not in incoming_ids and old.status == "in_force":
                    old.status = "repealed"
                    old.last_changed = now
                    changes.append(Change(now, source_id, aid, "repealed",
                                          old.content_hash, None, old_text=old.text))

        if changes:
            self._save(current)
            self._log(changes)
        return changes

    def flag_under_review(self, source_id: str, reference: str, now: str | None = None) -> list[Change]:
        """Un texte recent annonce modifier cette loi. On ne sait pas encore QUEL article
        est touche (la consolidation exacte demande un juriste) : on signale toute la loi."""
        now = now or now_iso()
        current = self.load()
        changes: list[Change] = []
        for aid, a in current.items():
            if a.source_id == source_id and a.status == "in_force" and not a.under_review:
                a.under_review = True
                changes.append(Change(now, source_id, aid, "flagged", a.content_hash, a.content_hash,
                                      note=f"modifiée par {reference}"))
        if changes:
            self._save(current)
            self._log(changes)
        return changes
