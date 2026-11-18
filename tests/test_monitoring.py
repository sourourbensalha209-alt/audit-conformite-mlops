"""Monitoring : indicateurs du modèle, journal des audits, alertes de dérive, format Prometheus."""

from src import monitoring as m

LABELS = ["Governing Law", "Autre", "Cap On Liability"]


def test_indicateurs_du_modele():
    probs = [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.4, 0.35, 0.25]]
    s = m.model_stats(probs, LABELS)
    assert s["paragraphes"] == 3
    assert s["part_autre"] == round(1 / 3, 4)
    assert s["part_abstention"] == round(1 / 3, 4)            # 0.4 < 0.5
    assert s["confiance_moyenne"] == round((0.9 + 0.8 + 0.4) / 3, 4)


def test_document_vide():
    assert m.model_stats([], LABELS)["confiance_moyenne"] is None


def test_pas_d_alerte_quand_tout_ressemble_a_la_reference(tmp_path):
    log = tmp_path / "audits.jsonl"
    for _ in range(5):
        m.record({"statut": "termine", "duree_s": 10, "confiance_moyenne": 0.9, "part_abstention": 0.02,
                  "part_autre": 0.80}, log)
    s = m.summary(log, tmp_path / "aucune.jsonl")
    assert s["indicateurs"]["audits"] == 5 and s["alertes"] == []


def test_alertes_de_derive_et_de_service(tmp_path):
    log = tmp_path / "audits.jsonl"
    for _ in range(5):        # contrats très différents de l'entraînement : presque tout « Autre », modèle hésitant
        m.record({"statut": "termine", "duree_s": 10, "confiance_moyenne": 0.55, "part_abstention": 0.30,
                  "part_autre": 0.98}, log)
    m.record({"statut": "erreur", "duree_s": 1, "erreur": "PDF illisible"}, log)
    s = m.summary(log, tmp_path / "aucune.jsonl")
    niveaux = [a["niveau"] for a in s["alertes"]]
    assert niveaux.count("derive") == 2 and "service" in niveaux       # Autre, abstention, taux d'erreur 17 %
    assert s["indicateurs"]["erreurs"] == 1


def test_taux_de_correction(tmp_path):
    log, fb = tmp_path / "audits.jsonl", tmp_path / "corrections.jsonl"
    for j in ("a", "b", "c", "d"):
        m.record({"job_id": j, "statut": "termine", "duree_s": 5, "part_autre": 0.8, "part_abstention": 0.0}, log)
    m.record({"job_id": "a", "verdict_corrige": "presente"}, fb)
    m.record({"job_id": "a", "verdict_corrige": "manquante"}, fb)    # deux corrections, un seul audit corrigé
    s = m.summary(log, fb)
    assert s["indicateurs"]["corrections"] == 2 and s["indicateurs"]["taux_correction"] == 0.25
    assert s["alertes"] == []                                         # 4 audits : trop peu pour juger
    m.record({"job_id": "e", "statut": "termine", "duree_s": 5, "part_autre": 0.8, "part_abstention": 0.0}, log)
    m.record({"job_id": "b", "verdict_corrige": "presente"}, fb)
    s = m.summary(log, fb)                                            # 5 audits, 2 corrigés : 40 % > 30 %
    assert s["indicateurs"]["taux_correction"] == 0.4 and any(a["niveau"] == "qualite" for a in s["alertes"])


def test_ligne_tronquee_ignoree(tmp_path):
    log = tmp_path / "audits.jsonl"
    m.record({"statut": "termine", "duree_s": 3}, log)
    with log.open("a", encoding="utf-8") as f:
        f.write('{"statut": "term')                                 # arrêt brutal pendant l'écriture
    assert m.summary(log, tmp_path / "x.jsonl")["indicateurs"]["audits"] == 1


def test_format_prometheus(tmp_path):
    log = tmp_path / "audits.jsonl"
    m.record({"statut": "termine", "duree_s": 12.5, "confiance_moyenne": 0.9, "part_abstention": 0.0,
              "part_autre": 0.8}, log)
    txt = m.prometheus(log, tmp_path / "x.jsonl")
    assert 'audit_total{statut="termine"} 1' in txt
    assert "audit_duree_moyenne_secondes 12.5" in txt
    assert "# TYPE modele_part_autre gauge" in txt


def test_pas_de_derive_sur_un_seul_contrat(tmp_path):
    """Un contrat atypique isolé n'est pas une dérive : il faut au moins MIN_AUDITS_DERIVE audits."""
    log = tmp_path / "audits.jsonl"
    m.record({"statut": "termine", "duree_s": 19, "part_autre": 0.41, "part_abstention": 0.06}, log)
    s = m.summary(log, tmp_path / "x.jsonl")
    assert s["alertes"] == [] and s["derive"] == {"evaluable": False, "audits": 1, "minimum": m.MIN_AUDITS_DERIVE}
