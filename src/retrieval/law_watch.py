"""Veille juridique : le système prend en compte les nouvelles lois sans intervention.

Ce que fait un passage de la veille (`python -m src.retrieval.law_watch`) :

  1. relit chaque source configurée (configs/law_sources.yaml) ;
  2. découpe les textes en articles et compare leur empreinte à celle déjà connue ;
  3. enregistre ce qui est AJOUTÉ, MODIFIÉ, ABROGÉ dans le journal (changelog.jsonl) ;
  4. repère les textes qui en modifient d'autres ("loi n° X modifiant la loi n° Y")
     et signale les articles de la loi visée comme "à vérifier" ;
  5. ne ré-encode que les articles nouveaux ou modifiés, et rafraîchit l'index ;
  6. écrit un rapport et le trace dans MLflow.

Aucun réentraînement : la connaissance juridique vit dans le corpus et l'index, pas dans les
poids des modèles. Legal-BERT (qui classe le TYPE de clause) n'est pas touché par une nouvelle loi.

Types de sources
  file          un texte consolidé local (RGPD, loi 2004-63...) ; un article disparu = abrogé
  url           idem, téléchargé depuis une adresse
  rules         le référentiel de règles (JSON)
  inbox         un dossier où l'on dépose des textes nouveaux (un .txt par texte)
  jort_listing  une page qui liste les derniers textes publiés (Journal officiel) ; les nouveautés
                pertinentes (mots-clés) sont téléchargées et ingérées

Limites assumées
  * Les textes détectés automatiquement entrent avec `validated: false` : le système peut les citer
    mais le rapport les signale comme "non validés par un juriste". Consolider précisément une loi
    modifiée (quel alinéa remplace quoi) est un travail de juriste : on le détecte et on le signale,
    on ne le devine pas.
  * La partie réseau (url, jort_listing) n'a pas pu être testée sur les sites réels : les sélecteurs
    de configs/law_sources.yaml sont à ajuster après inspection de la page. Tout le reste
    (découpage, comparaison, index, signalement) est couvert par tests/test_law_watch.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin

import yaml

from .corpus import Change, CorpusStore, now_iso, parse_articles
from .index import HybridIndex, strip_accents
from .rules import load_rules, rules_to_articles
from .textclean import strip_page_noise

USER_AGENT = "audit-conformite-veille/1.0 (projet etudiant TEK-UP)"

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class Source:
    id: str
    title: str
    kind: str
    jurisdiction: str = ""
    lang: str = "fr"
    path: str | None = None
    url: str | None = None
    auto_validate: bool = True
    keywords: list[str] = field(default_factory=list)
    min_score: int = 2
    link_pattern: str = ".*"
    max_new_per_run: int = 20
    aliases: list[str] = field(default_factory=list)   # numéros par lesquels d'autres textes la citent : "2004-63"


def load_sources(path: Path | str) -> list[Source]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [Source(**s) for s in data["sources"]]


# --------------------------------------------------------------------------- #
# Lecture des documents
# --------------------------------------------------------------------------- #


def http_get(url: str, timeout: int = 30, retries: int = 3) -> tuple[bytes, str]:
    """Téléchargement poli : identifie le robot, réessaie avec pause croissante.
    Garder une fréquence faible (1 passage par jour suffit) et respecter les conditions du site."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read(), r.headers.get("Content-Type", "")
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"échec du téléchargement de {url} : {last}")


class _TextExtractor(HTMLParser):
    BLOCKS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        if tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        if tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    p = _TextExtractor()
    p.feed(html)
    text = "".join(p.parts)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


class _LinkExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self._href = dict(attrs).get("href")
            self._buf = []

    def handle_data(self, data):
        if self._href is not None:
            self._buf.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._href:
            self.links.append((self._href, " ".join("".join(self._buf).split())))
            self._href = None


@dataclass
class Link:
    url: str
    title: str


def extract_links(html: str, base_url: str, pattern: str = ".*") -> list[Link]:
    p = _LinkExtractor()
    p.feed(html)
    rx = re.compile(pattern)
    out, seen = [], set()
    for href, title in p.links:
        url = urljoin(base_url, href)
        if rx.search(url) and url not in seen and title:
            seen.add(url)
            out.append(Link(url, title))
    return out


def bytes_to_text(data: bytes, content_type: str, url: str) -> str:
    if "pdf" in content_type.lower() or url.lower().endswith(".pdf"):
        import io

        import pdfplumber

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "\n".join((page.extract_text() or "") for page in pdf.pages)
    m = re.search(r"charset=([\w-]+)", content_type, re.I)
    html = data.decode(m.group(1) if m else "utf-8", errors="replace")
    return html_to_text(html)


def read_pdf_text(path: Path) -> str:
    """Texte d'un PDF de loi (par exemple un texte du Journal officiel), sans en-têtes ni numéros de page."""
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        pages = [(page.extract_text() or "").splitlines() for page in pdf.pages]
    kept, _, _ = strip_page_noise(pages)
    return "\n".join(kept)


_TITLE_RE = re.compile(r"^\s*(?:loi organique|loi|décret-loi|decret-loi|décret|decret|arrêté|arrete|ordonnance)\b", re.I)


def guess_title(body: str, fallback: str) -> str:
    """Titre d'un texte : la première ligne qui commence par Loi / Décret / Arrêté, sinon la première ligne."""
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    for ln in lines[:20]:
        if _TITLE_RE.match(ln):
            return ln
    return lines[0] if lines else fallback


def split_header(text: str) -> tuple[dict[str, str], str]:
    """En-tête optionnel d'un fichier déposé :
        # id: tn-loi-2026-12
        # title: Loi n° 2026-12 du ... portant ...
        # publication_date: 2026-09-30
        # validated: false
    """
    meta: dict[str, str] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines) and (lines[i].startswith("#") or not lines[i].strip()):
        m = re.match(r"#\s*([a-z_]+)\s*:\s*(.+)", lines[i])
        if m:
            meta[m.group(1)] = m.group(2).strip()
        i += 1
    return meta, "\n".join(lines[i:])


# --------------------------------------------------------------------------- #
# Détection : pertinence et modifications d'autres lois
# --------------------------------------------------------------------------- #

# "loi n° 2004-63", mais aussi "loi no 2004-63", "loi nº 2004‐63" (tiret spécial U+2010 vu dans de vrais PDF tunisiens)
_DASH = "[-‐‑‒–—−]"
_NUM = r"(?:n\s*[°º˚]|no\.?|n\.)"
_REF_RE = re.compile(
    rf"(?:loi organique|loi|décret-loi|decret-loi|décret|decret|arrêté|arrete|ordonnance)\s+{_NUM}\s*(\d{{4}})\s*{_DASH}\s*(\d+)", re.I)
_AMEND_RE = re.compile(r"modifi|compl[eé]t|abrog|remplac|rectifi", re.I)


def detect_amendment_targets(head: str) -> list[str]:
    """Numéros ("2004-63") des textes que celui-ci modifie, complète ou abroge.
    Ex. : "Loi n° 2026-12 portant modification de la loi n° 2004-63" -> ["2004-63"]."""
    if not _AMEND_RE.search(head):
        return []
    refs = [f"{y}-{n}" for y, n in _REF_RE.findall(head)]   # toujours avec un tiret ASCII
    if refs and _REF_RE.match(head.strip()):      # la 1re référence est le numéro du texte lui-même
        refs = refs[1:]
    return list(dict.fromkeys(refs))


def relevance(title: str, body: str, keywords: list[str]) -> tuple[int, list[str]]:
    """Score simple : 2 points par mot-clé dans le titre, 1 point dans le début du texte."""
    t = strip_accents(title.lower())
    b = strip_accents(body[:4000].lower())
    score, matched = 0, []
    for k in keywords:
        kk = strip_accents(k.lower())
        if kk in t:
            score += 2
            matched.append(k)
        elif kk in b:
            score += 1
            matched.append(k)
    return score, matched


def _effective_from(publication_date: str | None) -> str | None:
    """Les textes sont exécutoires le lendemain de leur publication dans l'édition électronique
    du JORT, sauf disposition contraire du texte (décret-loi n° 2020-1 du 14 avril 2020).
    Valeur par défaut, à corriger à la main si le texte fixe une autre date."""
    try:
        return (date.fromisoformat(publication_date) + timedelta(days=1)).isoformat() if publication_date else None
    except ValueError:
        return None


def _slug(title: str, url: str = "") -> str:
    m = re.search(rf"(loi organique|loi|décret-loi|décret|arrêté)\s+{_NUM}\s*(\d{{4}})\s*{_DASH}\s*(\d+)", title, re.I)
    if m:
        kind = strip_accents(m.group(1).lower()).replace(" ", "-")
        return f"tn-{kind}-{m.group(2)}-{m.group(3)}"
    return "jort-" + hashlib.sha1((url or title).encode()).hexdigest()[:8]


# --------------------------------------------------------------------------- #
# La veille
# --------------------------------------------------------------------------- #

Fetcher = Callable[[str], tuple[bytes, str]]


class Watcher:
    def __init__(
        self,
        sources: list[Source],
        store: CorpusStore,
        *,
        root: Path | str = ".",
        state_path: Path | str | None = None,
        index: HybridIndex | None = None,
        fetch: Fetcher = http_get,
    ):
        self.sources = sources
        self.store = store
        self.root = Path(root)
        self.index = index
        self.fetch = fetch
        self.state_path = Path(state_path) if state_path else store.root / "state.json"
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {}
        self.alias_map = {a: s.id for s in sources for a in s.aliases}

    # -- point d'entrée
    def run_once(self) -> dict:
        before = self.store.version()
        report: dict = {"ts": now_iso(), "corpus_version_before": before, "sources": {}}
        for src in self.sources:
            try:
                report["sources"][src.id] = self._sync(src)
            except Exception as e:  # noqa: BLE001 — une source en panne ne doit pas bloquer les autres
                report["sources"][src.id] = {"erreur": str(e)}
        after = self.store.version()
        report["corpus_version_after"] = after
        report["changed"] = before != after
        report["embedded"] = 0
        if self.index is not None and (report["changed"] or self.index.version != after):
            report["embedded"] = self.index.refresh(self.store.load())
        self._save_state()
        return report

    # -- une source
    def _sync(self, src: Source) -> dict:
        if src.kind == "file":
            text = self._read_local(src)
            return self._summarize(self._ingest_snapshot(src, text))
        if src.kind == "url":
            data, ctype = self.fetch(src.url)
            return self._summarize(self._ingest_snapshot(src, bytes_to_text(data, ctype, src.url)))
        if src.kind == "rules":
            book = load_rules(self.root / src.path)
            changes = self.store.apply_snapshot(src.id, rules_to_articles(book), full_snapshot=True)
            return self._summarize(changes)
        if src.kind == "inbox":
            return self._sync_inbox(src)
        if src.kind == "jort_listing":
            return self._sync_listing(src)
        raise ValueError(f"type de source inconnu : {src.kind}")

    def _read_local(self, src: Source) -> str:
        p = self.root / src.path
        if not p.exists():
            raise FileNotFoundError(f"{p} introuvable (télécharger le texte et l'enregistrer à cet endroit)")
        return p.read_text(encoding="utf-8")

    def _ingest_snapshot(self, src: Source, text: str) -> list[Change]:
        meta, body = split_header(text)
        pub = meta.get("publication_date")
        articles = parse_articles(
            body, src.id, meta.get("title", src.title), lang=src.lang, jurisdiction=src.jurisdiction,
            validated=src.auto_validate, publication_date=pub, effective_from=_effective_from(pub),
            source_url=src.url)
        if not articles:
            raise ValueError("aucun article reconnu : format du texte inattendu, rien n'a été modifié")
        return self.store.apply_snapshot(src.id, articles, full_snapshot=True)

    def _sync_inbox(self, src: Source) -> dict:
        """Dossier de dépôt : un .txt ou un .pdf par texte (par exemple un texte du Journal officiel)."""
        folder = self.root / src.path
        folder.mkdir(parents=True, exist_ok=True)
        st = self.state.setdefault(src.id, {"files": {}})
        total: list[Change] = []
        no_article: list[str] = []
        for f in sorted(list(folder.glob("*.txt")) + list(folder.glob("*.pdf"))):
            sig = f"{f.stat().st_size}-{int(f.stat().st_mtime)}"
            if f.suffix.lower() == ".pdf":
                if st["files"].get(f.name) == sig:
                    continue                                    # PDF déjà lu, inchangé : inutile de le relire
                text = read_pdf_text(f)
                doc_id = re.sub(r"[^a-z0-9]+", "-", f.stem.lower()).strip("-")
            else:
                text, doc_id = f.read_text(encoding="utf-8"), f.stem
            changes, n_articles = self._ingest_document(src, text, doc_id=doc_id)
            st["files"][f.name] = sig
            total += changes
            if n_articles == 0:
                no_article.append(f.name)
        out = self._summarize(total)
        if no_article:
            out["sans_article"] = no_article          # lu mais illisible : le dire plutôt que se taire
        return out

    def _sync_listing(self, src: Source) -> dict:
        st = self.state.setdefault(src.id, {"seen": [], "ignored": {}})
        data, ctype = self.fetch(src.url)
        html = data.decode("utf-8", errors="replace")
        new = [lk for lk in extract_links(html, src.url, src.link_pattern) if lk.url not in st["seen"]]
        total: list[Change] = []
        ignored = 0
        for link in new[: src.max_new_per_run]:
            body_bytes, body_type = self.fetch(link.url)
            body = bytes_to_text(body_bytes, body_type, link.url)
            st["seen"].append(link.url)
            score, matched = relevance(link.title, body, src.keywords)
            if score < src.min_score:
                st["ignored"][link.url] = {"title": link.title, "score": score}
                ignored += 1
                continue
            text = f"# title: {link.title}\n# url: {link.url}\n" + body
            total += self._ingest_document(src, text, doc_id=_slug(link.title, link.url))[0]
        out = self._summarize(total)
        out["ignores_non_pertinents"] = ignored
        return out

    def _ingest_document(self, src: Source, text: str, doc_id: str) -> tuple[list[Change], int]:
        meta, body = split_header(text)
        sid = meta.get("id", doc_id)
        title = meta.get("title") or guess_title(body, sid)
        pub = meta.get("publication_date")
        validated = meta["validated"].lower() == "true" if "validated" in meta else src.auto_validate
        articles = parse_articles(
            body, sid, title, lang=src.lang, jurisdiction=src.jurisdiction, validated=validated,
            publication_date=pub, effective_from=meta.get("effective_from") or _effective_from(pub),
            source_url=meta.get("url"))
        changes = self.store.apply_snapshot(sid, articles, full_snapshot=False)
        if changes:  # seulement si le texte est nouveau ou modifié : sinon on re-signalerait à chaque passage
            head = f"{title}\n{body[:800]}"
            for ref in detect_amendment_targets(head):
                base = self.alias_map.get(ref)
                if base and base != sid:
                    changes += self.store.flag_under_review(base, reference=title)
        return changes, len(articles)

    # -- utilitaires
    @staticmethod
    def _summarize(changes: list[Change]) -> dict:
        c = Counter(ch.change for ch in changes)
        return {"ajoutés": c["added"], "modifiés": c["modified"], "abrogés": c["repealed"], "signalés": c["flagged"]}

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Rapport et traçabilité
# --------------------------------------------------------------------------- #


def print_report(report: dict) -> None:
    print(f"\n=== Veille juridique {report['ts']} ===")
    for sid, res in report["sources"].items():
        if "erreur" in res:
            print(f"  {sid:14s} ERREUR : {res['erreur']}")
        else:
            extra = f", {res['ignores_non_pertinents']} ignoré(s)" if res.get("ignores_non_pertinents") else ""
            print(f"  {sid:14s} +{res['ajoutés']} ajouté(s)  ~{res['modifiés']} modifié(s)  "
                  f"-{res['abrogés']} abrogé(s)  !{res['signalés']} à vérifier{extra}")
            if res.get("sans_article"):
                print(f"  {'':14s} ATTENTION, aucun article reconnu dans : {', '.join(res['sans_article'])}")
    if report["changed"]:
        extra = f" ({report['embedded']} article(s) ré-encodé(s))" if report["embedded"] else ""
        print(f"  Corpus {report['corpus_version_before']} -> {report['corpus_version_after']}{extra}")
    else:
        print(f"  Aucun changement (corpus {report['corpus_version_after']}).")


def log_mlflow(report: dict, tracking_uri: str | None) -> None:
    """Trace chaque passage dans MLflow (best effort : la veille ne doit pas tomber si MLflow est absent)."""
    try:
        import mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("04-veille-juridique")
        with mlflow.start_run(run_name=f"veille-{report['ts']}"):
            mlflow.set_tag("etage", "veille")
            mlflow.set_tag("corpus_version", report["corpus_version_after"])
            totals = Counter()
            for res in report["sources"].values():
                for k, v in res.items():
                    if isinstance(v, int):
                        totals[k] += v
            mlflow.log_metrics({
                "ajoutes": totals["ajoutés"], "modifies": totals["modifiés"], "abroges": totals["abrogés"],
                "signales": totals["signalés"], "articles_reencodes": report["embedded"],
            })
    except Exception as e:  # noqa: BLE001
        print(f"(MLflow non renseigné : {e})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Veille juridique : met à jour le corpus de lois et l'index.")
    ap.add_argument("--config", default="configs/law_sources.yaml")
    ap.add_argument("--corpus-dir", default="data/corpus")
    ap.add_argument("--dense", action="store_true", help="encodeur dense (sinon BM25 seul)")
    ap.add_argument("--model", default="BAAI/bge-m3", help="encodeur dense retenu : BGE-M3")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--loop", type=int, default=0, metavar="SECONDES",
                    help="repasse toutes les N secondes (en production : cron ou tâche planifiée Kubeflow)")
    args = ap.parse_args()

    try:
        from src.config import MLFLOW_TRACKING_URI, ROOT
    except Exception:  # config absente ou illisible : on travaille depuis le dossier courant
        ROOT, MLFLOW_TRACKING_URI = Path.cwd(), None

    root = Path(ROOT)
    store = CorpusStore(root / args.corpus_dir)
    embedder = None
    if args.dense:
        from .index import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(args.model)
    index = HybridIndex(embedder, cache_path=store.root / "embeddings.npz")
    watcher = Watcher(load_sources(root / args.config), store, root=root, index=index)

    while True:
        report = watcher.run_once()
        print_report(report)
        out = root / "reports" / "veille"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"veille_{report['ts'].replace(':', '-')}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if not args.no_mlflow:
            log_mlflow(report, MLFLOW_TRACKING_URI)
        if not args.loop:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
