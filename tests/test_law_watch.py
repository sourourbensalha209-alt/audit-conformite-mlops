"""Tests de la veille juridique. Aucun réseau, aucun modèle téléchargé :
    pytest tests/test_law_watch.py -v
Les textes de loi utilisés ici sont FICTIFS (numéros en 9xxx) : ils servent à vérifier la mécanique."""

import json
from pathlib import Path

import pytest

from src.retrieval.corpus import CorpusStore, hash_text, parse_articles
from src.retrieval.eval_retrieval import build_rule_index, evaluate
from src.retrieval.index import HashingEmbedder, HybridIndex
from src.retrieval.law_watch import (Source, Watcher, detect_amendment_targets, extract_links,
                                     relevance, split_header)
from src.retrieval.rules import (load_rules, missing_categories, rgpd_candidate, rule_for,
                                 rules_to_articles, validate_against_labels)

ROOT = Path(__file__).resolve().parents[1]

LOI_V1 = """LOI FICTIVE n° 9001-01 sur la conservation des données (démonstration)
CHAPITRE I
Dispositions générales
Article premier. - La présente loi fixe les règles de conservation des données.
Art. 2. - Les données sont conservées pendant cinq ans au plus.
Article 3
Le responsable informe la personne concernée de la durée de conservation.
Il tient un registre des traitements.
Article 3 bis - Le registre est tenu à la disposition de l'autorité.
Les dispositions de l'article 2 de la présente loi s'appliquent aux sous-traitants.
Article 4 : Toute violation est notifiée à l'autorité sous 72 heures.
"""

LOI_V2 = LOI_V1.replace("cinq ans au plus", "trois ans au plus").replace(
    "Article 4 : Toute violation est notifiée à l'autorité sous 72 heures.",
    "Article 4 : Toute violation est notifiée à l'autorité sous 48 heures.\n"
    "Article 5 : Une amende est prévue en cas de manquement.")


# --------------------------------------------------------------------------- #
# Découpage en articles
# --------------------------------------------------------------------------- #

def test_parse_formats_et_renvois():
    arts = parse_articles(LOI_V1, "loi-test", "Loi fictive")
    assert [a.number for a in arts] == ["1", "2", "3", "3-bis", "4"]
    a3 = next(a for a in arts if a.number == "3")
    assert "registre des traitements" in a3.text                # article sur plusieurs lignes
    a3bis = next(a for a in arts if a.number == "3-bis")
    assert "article 2 de la présente loi" in a3bis.text         # un renvoi n'ouvre PAS un nouvel article
    assert not any("CHAPITRE" in a.text for a in arts)           # les titres de chapitre sont écartés


def test_hash_insensible_a_la_mise_en_page():
    assert hash_text("Les données  sont\nconservées.") == hash_text("Les données sont conservées.")


# --------------------------------------------------------------------------- #
# Comparaison : ajouté / modifié / abrogé / inchangé
# --------------------------------------------------------------------------- #

def test_snapshot_ajout_modification_abrogation(tmp_path):
    store = CorpusStore(tmp_path)
    first = store.apply_snapshot("loi-test", parse_articles(LOI_V1, "loi-test", "Loi fictive"), full_snapshot=True)
    assert {c.change for c in first} == {"added"} and len(first) == 5
    v1 = store.version()

    # même texte relu : rien ne change, version du corpus identique
    again = store.apply_snapshot("loi-test", parse_articles(LOI_V1, "loi-test", "Loi fictive"), full_snapshot=True)
    assert again == [] and store.version() == v1

    # nouvelle version : art. 2 et 4 modifiés, art. 5 ajouté, art. 1, 3, 3-bis intacts
    changes = store.apply_snapshot("loi-test", parse_articles(LOI_V2, "loi-test", "Loi fictive"), full_snapshot=True)
    by = {c.article_id: c.change for c in changes}
    assert by == {"loi-test:2": "modified", "loi-test:4": "modified", "loi-test:5": "added"}
    assert store.version() != v1
    art2 = store.load()["loi-test:2"]
    assert art2.version == 2 and "trois ans" in art2.text
    old = store.history("loi-test:2")[-1]
    assert "cinq ans" in old["old_text"]                         # l'ancienne version est conservée

    # un article qui disparaît d'un texte consolidé est abrogé, pas supprimé
    store.apply_snapshot("loi-test", [a for a in parse_articles(LOI_V2, "loi-test", "Loi fictive")
                                      if a.number != "5"], full_snapshot=True)
    assert store.load()["loi-test:5"].status == "repealed"


def test_texte_nouveau_n_abroge_rien(tmp_path):
    store = CorpusStore(tmp_path)
    store.apply_snapshot("a", parse_articles("Article 1 - un.\nArticle 2 - deux.", "a", "A"), full_snapshot=False)
    store.apply_snapshot("a", parse_articles("Article 1 - un.", "a", "A"), full_snapshot=False)
    assert store.load()["a:2"].status == "in_force"


# --------------------------------------------------------------------------- #
# Détection des textes modificatifs
# --------------------------------------------------------------------------- #

def test_detection_modification():
    head = "Loi n° 9002-05 du 1 octobre 9002 portant modification de la loi n° 2004-63 du 27 juillet 2004"
    assert detect_amendment_targets(head) == ["2004-63"]
    assert detect_amendment_targets("Loi n° 9002-05 portant ouverture d'un crédit") == []
    assert detect_amendment_targets("Décret n° 9002-7 portant application de la loi n° 2004-63") == []


def test_relevance_et_en_tete():
    score, matched = relevance("Loi relative à la protection des données", "texte sur le contrat", ["protection des données", "contrat", "mines"])
    assert score == 3 and matched == ["protection des données", "contrat"]
    meta, body = split_header("# id: x\n# title: Mon titre\n\nArticle 1 - a.")
    assert meta == {"id": "x", "title": "Mon titre"} and body.startswith("Article 1")


def test_extraction_liens():
    html = '<a href="/textes/loi-9003.html">Loi n° 9003-02 relative aux contrats</a><a href="x.css">style</a>'
    links = extract_links(html, "https://exemple.tn/liste", r"/textes/")
    assert [(l.url, l.title) for l in links] == [("https://exemple.tn/textes/loi-9003.html", "Loi n° 9003-02 relative aux contrats")]


# --------------------------------------------------------------------------- #
# Scénario complet : une nouvelle loi arrive, l'application la prend en compte
# --------------------------------------------------------------------------- #

def _watcher(tmp_path, with_dense=False):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "base.txt").write_text(LOI_V1, encoding="utf-8")
    sources = [
        Source(id="loi-base", title="Loi fictive de base", kind="file", path="raw/base.txt",
               jurisdiction="TN", aliases=["9001-01"]),
        Source(id="inbox", title="Dépôt", kind="inbox", path="inbox", jurisdiction="TN", auto_validate=False),
    ]
    store = CorpusStore(tmp_path / "corpus")
    index = HybridIndex(HashingEmbedder() if with_dense else None,
                        cache_path=tmp_path / "corpus" / "emb.npz" if with_dense else None)
    return Watcher(sources, store, root=tmp_path, index=index), store, index


def test_scenario_nouvelle_loi(tmp_path):
    w, store, index = _watcher(tmp_path)

    r1 = w.run_once()
    assert r1["changed"] and r1["sources"]["loi-base"]["ajoutés"] == 5
    assert index.search("durée de conservation des données", k=3)[0].article.source_id == "loi-base"

    r2 = w.run_once()                                   # rien de neuf : rien ne bouge
    assert not r2["changed"] and r2["corpus_version_after"] == r1["corpus_version_after"]

    # arrive un texte modificatif dans le dossier de dépôt
    (tmp_path / "inbox").mkdir(exist_ok=True)
    (tmp_path / "inbox" / "loi-9002-05.txt").write_text(
        "# publication_date: 2026-09-30\n"
        "Loi fictive n° 9002-05 portant modification de la loi n° 9001-01 sur la conservation\n"
        "Article premier. - Les sous-traitants tiennent un registre des violations de données.\n",
        encoding="utf-8")
    r3 = w.run_once()
    assert r3["changed"]
    assert r3["sources"]["inbox"]["ajoutés"] == 1
    assert r3["sources"]["inbox"]["signalés"] == 5            # les 5 articles de la loi visée sont à vérifier

    arts = store.load()
    new = next(a for a in arts.values() if a.source_id.startswith("loi-9002") or a.source_id == "loi-9002-05")
    assert new.validated is False                             # texte automatique : pas encore validé
    assert new.effective_from == "2026-10-01"                 # exécutoire le lendemain de la publication
    assert all(a.under_review for a in arts.values() if a.source_id == "loi-base")

    # la nouvelle loi est immédiatement retrouvable, et filtrable "validés seulement"
    hit = index.search("registre des violations de données des sous-traitants", k=1)[0]
    assert hit.article.article_id == new.article_id
    assert all(h.article.validated for h in index.search("registre", k=5, validated_only=True))

    r4 = w.run_once()                                         # re-passage : pas de re-signalement
    assert not r4["changed"]


def test_source_en_panne_ne_bloque_pas_les_autres(tmp_path):
    w, _, _ = _watcher(tmp_path)
    w.sources.append(Source(id="absente", title="X", kind="file", path="raw/inexistant.txt"))
    r = w.run_once()
    assert "erreur" in r["sources"]["absente"]
    assert r["sources"]["loi-base"]["ajoutés"] == 5


def test_texte_au_mauvais_format_ne_modifie_rien(tmp_path):
    w, store, _ = _watcher(tmp_path)
    w.run_once()
    v = store.version()
    (tmp_path / "raw" / "base.txt").write_text("Page d'erreur 403 — accès refusé", encoding="utf-8")
    r = w.run_once()
    assert "erreur" in r["sources"]["loi-base"] and store.version() == v    # on n'abroge pas tout par accident


# --------------------------------------------------------------------------- #
# Index : mise à jour incrémentale
# --------------------------------------------------------------------------- #

def test_seuls_les_articles_modifies_sont_reencodes(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "base.txt").write_text(LOI_V1, encoding="utf-8")
    src = Source(id="loi-base", title="Loi fictive", kind="file", path="raw/base.txt")
    store = CorpusStore(tmp_path / "c")
    index = HybridIndex(HashingEmbedder(), cache_path=tmp_path / "c" / "emb.npz")
    w = Watcher([src], store, root=tmp_path, index=index)
    assert w.run_once()["embedded"] == 5

    (tmp_path / "raw" / "base.txt").write_text(LOI_V2, encoding="utf-8")   # 2 modifiés + 1 ajouté
    assert w.run_once()["embedded"] == 3

    # un nouvel index (redémarrage de l'API) repart du cache : rien à ré-encoder
    fresh = HybridIndex(HashingEmbedder(), cache_path=tmp_path / "c" / "emb.npz")
    assert fresh.ensure_fresh(store) is True and fresh.n_embedded_last_refresh == 0
    assert fresh.ensure_fresh(store) is False


def test_index_recharge_automatiquement(tmp_path):
    w, store, index = _watcher(tmp_path)
    w.run_once()
    other = HybridIndex()                    # autre processus (ex. l'API) avec son propre index
    other.ensure_fresh(store)
    (tmp_path / "raw" / "base.txt").write_text(LOI_V2, encoding="utf-8")
    w.run_once()
    assert other.ensure_fresh(store) is True
    assert "48 heures" in other.search("notifiée à l'autorité heures violation", k=1)[0].article.text


# --------------------------------------------------------------------------- #
# Référentiel de règles
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def book():
    path = ROOT / "data" / "rules" / "rules_contractuelles.json"
    if not path.exists():
        pytest.skip("lancer d'abord : python scripts/build_rules.py")
    return load_rules(path)


def _find_labels():
    for p in [ROOT / "models" / "best_model_v2" / "labels.json", ROOT / "labels.json",
              Path("/mnt/user-data/uploads/labels.json")]:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def test_une_regle_par_categorie(book):
    labels = _find_labels()
    if labels is None:
        pytest.skip("labels.json introuvable")
    assert len(labels) == 41
    assert validate_against_labels(book, labels) == {"sans_regle": [], "regle_orpheline": [], "doublons": []}


def test_regles_utilitaires(book):
    assert rule_for(book, "Cap On Liability")["id"] == "R06"
    manquantes = {r["category"] for r in missing_categories(book, {"Parties", "Governing Law"})}
    assert "Cap On Liability" in manquantes and "Governing Law" not in manquantes
    assert rgpd_candidate(book, "The Processor shall process Personal Data only on documented instructions.")
    assert rgpd_candidate(book, "Le sous-traitant notifie toute violation de données.")
    assert not rgpd_candidate(book, "This Agreement shall be governed by the laws of Delaware.")
    assert all(r["refs"] for r in book.rgpd_checks)


def test_evaluation_recherche_de_regle(book):
    index = build_rule_index(book)                       # BM25 seul
    clauses = ["The aggregate liability of either party shall not exceed the fees paid in the twelve months preceding the claim.",
               "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware.",
               "Neither party may assign this Agreement without the prior written consent of the other party."]
    truth = ["rules:R06", "rules:R14", "rules:R04"]
    res = evaluate(index, clauses, truth, "bm25")
    assert res["recall@5"] == 1.0 and res["mrr"] > 0.6
    hybrid = build_rule_index(book, HashingEmbedder())
    assert evaluate(hybrid, clauses, truth, "hybrid")["recall@5"] == 1.0
    assert len(rules_to_articles(book)) == 41


def test_cache_embeddings_separe_par_modele(tmp_path):
    """Deux encodeurs différents sur le même cache ne doivent jamais se partager un vecteur."""
    def encodeur(nom):
        e = HashingEmbedder()
        e.name = nom
        return e

    arts = {a.article_id: a for a in parse_articles(LOI_V1, "x", "Loi")}
    cache = tmp_path / "emb.npz"
    assert HybridIndex(encodeur("modele-A"), cache_path=cache).refresh(arts) == 5
    assert HybridIndex(encodeur("modele-A"), cache_path=cache).refresh(arts) == 0   # même modèle : cache réutilisé
    assert HybridIndex(encodeur("modele-B"), cache_path=cache).refresh(arts) == 5   # autre modèle : tout recalculé


def test_titres_de_chapitre_ne_polluent_pas_l_article_precedent():
    texte = """Article 23
Limitations
1. Le droit de l'Union peut limiter la portée des obligations.
CHAPITRE IV
Responsable du traitement et sous-traitant
Section 1
Obligations générales
Article 24
Responsabilité du responsable du traitement
1. Le responsable met en œuvre des mesures appropriées.
Section 2 du présent chapitre : cette ligne est du texte, pas un titre.
Chapitre V : Des sanctions
Article 25
Protection dès la conception
Le responsable agit.
"""
    arts = {a.number: a.text for a in parse_articles(texte, "x", "Loi")}
    assert "Responsable du traitement et sous-traitant" not in arts["23"]     # titre de chapitre écarté
    assert "Obligations générales" not in arts["23"]                          # titre de section écarté
    assert arts["23"].endswith("obligations.")                               # le vrai texte est intact
    assert "Section 2 du présent chapitre" in arts["24"]                      # une phrase n'est pas un titre
    assert "Des sanctions" not in arts["24"]                                  # titre sur une seule ligne écarté
    assert arts["25"].startswith("Protection dès la conception")             # le titre propre de l'article reste


def test_nettoyage_pdf_retire_l_entete_mais_jamais_le_corps():
    from src.retrieval.textclean import strip_page_noise

    pages = []
    for p in range(1, 7):                      # 6 pages : même bandeau en haut, corps TRÈS répétitif
        pages.append([f"Page {p} sur 6", "Base de données", "LA LEGISLATION DU SECTEUR DE LA SECURITE",
                      f"Art. {p} :", "Le texte de l'article impose des obligations précises au responsable.",
                      "a) premier point de l'article", "b) second point de l'article",
                      "Le responsable informe la personne concernée.", "Il tient un registre à jour.", str(p)])
    kept, dropped, found = strip_page_noise(pages)
    texte = "\n".join(kept)
    assert "LEGISLATION DU SECTEUR" not in texte and "Base de données" not in texte   # bandeau retiré
    assert texte.count("Le texte de l'article impose") == 6                            # corps intact, même répété
    assert texte.count("a) premier point") == 6
    assert any("LEGISLATION" in n for n, _ in found)                                   # et signalé à l'utilisateur
