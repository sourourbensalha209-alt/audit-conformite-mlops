"""Service d'audit de conformite.

Lancement :
    uvicorn src.api.main:app --reload
Documentation interactive : http://localhost:8000/docs
"""

import json
import shutil
import tempfile
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import monitoring
from src.orchestration import retrain
from src.orchestration.registry import production_model_dir

ROOT = Path(__file__).resolve().parents[2]
FEEDBACK_FILE = ROOT / "data" / "feedback" / "corrections.jsonl"
FRONTEND_DIR = ROOT / "frontend"

app = FastAPI(
    title="Agent d'audit de conformite documentaire",
    description="Analyse un document et retourne un rapport de conformite clause par clause.",
    version="0.1.0",
)

# Stockage en memoire, a remplacer par une base lors de l'etape de deploiement.
JOBS: dict[str, dict] = {}
REPORTS: dict[str, str] = {}      # job_id -> page HTML du rapport


class Verdict(str, Enum):
    conforme = "conforme"
    non_conforme = "non_conforme"
    a_verifier = "a_verifier"
    manquante = "manquante"
    presente = "presente"         # clause attendue trouvee (aucun verdict de conformite automatique)


class ClauseFinding(BaseModel):
    clause_ref: str | None = Field(None, description="Numero de la clause dans le document")
    clause_type: str = Field(..., description="Type d'obligation detecte")
    verdict: Verdict
    article: str | None = Field(None, description="Article reglementaire applicable")
    justification: str
    confidence: float = Field(..., ge=0, le=1)


class AuditReport(BaseModel):
    job_id: str
    status: str
    created_at: datetime
    document: str | None = None
    erreur: str | None = None
    duree_s: float | None = Field(None, description="Duree de l'analyse en secondes")
    findings: list[ClauseFinding] = []
    resultat: dict | None = Field(None, description="Analyse detaillee : identite, clauses attendues, clauses trouvees")


class Feedback(BaseModel):
    job_id: str
    clause_ref: str
    verdict_corrige: Verdict | None = None
    commentaire: str | None = None
    texte: str | None = Field(None, description="Passage du contrat concerné : avec categorie_corrigee, devient un exemple d'entraînement")
    categorie_corrigee: str | None = Field(None, description="Bonne catégorie de ce passage, selon le juriste")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "modele": production_model_dir().name}


@app.get("/categories")
def categories() -> list[str]:
    """Catégories du modèle en production (pour corriger la catégorie d'un passage)."""
    path = production_model_dir() / "labels.json"
    if not path.exists():
        raise HTTPException(503, "Modèle absent : labels.json introuvable")
    return json.loads(path.read_text(encoding="utf-8"))


def _findings(result: dict) -> list[dict]:
    """Clauses attendues du rapport -> format ClauseFinding de l'API."""
    out = []
    for a in result["attendues"]:
        preuve = a["preuve"] or {}
        out.append({
            "clause_ref": None,
            "clause_type": a["categorie"],
            "verdict": a["statut"],
            "article": None,
            "justification": a["regle"] if a["statut"] != "presente" else preuve.get("texte", "")[:300],
            "confidence": preuve.get("confiance") or (1.0 if a["statut"] == "presente" else 0.0),
        })
    return out


def _run_audit(job_id: str, path: Path, title: str) -> None:
    """Tache de fond : execute la chaine d'audit et range le resultat."""
    from src.pipeline import audit_file      # import ici : l'API demarre sans charger le modele

    JOBS[job_id]["status"] = "en_cours"
    debut = datetime.now(timezone.utc)
    taille_ko = round(path.stat().st_size / 1024, 1)
    trace = {"job_id": job_id, "document": JOBS[job_id]["document"], "taille_ko": taille_ko}
    try:
        result, html = audit_file(path, title)
        duree = round((datetime.now(timezone.utc) - debut).total_seconds(), 1)
        REPORTS[job_id] = html
        JOBS[job_id]["findings"] = _findings(result)
        JOBS[job_id]["resultat"] = result
        JOBS[job_id]["duree_s"] = duree
        JOBS[job_id]["status"] = "termine"
        monitoring.record({**trace, "statut": "termine", "duree_s": duree, **result.get("monitoring", {})})
    except Exception as exc:  # noqa: BLE001
        JOBS[job_id]["status"] = "erreur"
        JOBS[job_id]["erreur"] = str(exc)
        duree = round((datetime.now(timezone.utc) - debut).total_seconds(), 1)
        monitoring.record({**trace, "statut": "erreur", "duree_s": duree, "erreur": str(exc)[:300]})
    finally:
        shutil.rmtree(path.parent, ignore_errors=True)


@app.post("/audit", response_model=AuditReport, status_code=202)
async def submit_audit(background: BackgroundTasks, file: UploadFile = File(...)) -> AuditReport:
    """Soumet un document a l'audit. Retourne un identifiant de tache."""
    if not file.filename.lower().endswith((".pdf", ".docx", ".txt")):
        raise HTTPException(400, "Formats acceptes : .pdf, .docx, .txt")

    job_id = str(uuid4())
    tmp = Path(tempfile.mkdtemp()) / Path(file.filename).name
    tmp.write_bytes(await file.read())

    report = AuditReport(
        job_id=job_id,
        status="en_attente",
        created_at=datetime.now(timezone.utc),
        document=file.filename,
    )
    JOBS[job_id] = report.model_dump()
    background.add_task(_run_audit, job_id, tmp, Path(file.filename).stem)
    return report


@app.get("/audits")
def list_audits() -> list[dict]:
    """Historique des audits de la session, du plus recent au plus ancien."""
    jobs = sorted(JOBS.values(), key=lambda j: j["created_at"], reverse=True)
    return [{k: j[k] for k in ("job_id", "status", "created_at", "document")} for j in jobs]


@app.get("/audit/{job_id}", response_model=AuditReport)
def get_audit(job_id: str) -> AuditReport:
    """Recupere le rapport d'audit."""
    if job_id not in JOBS:
        raise HTTPException(404, "Tache inconnue")
    return AuditReport(**JOBS[job_id])


@app.get("/audit/{job_id}/rapport", response_class=HTMLResponse)
def get_report_html(job_id: str) -> str:
    """Page HTML du rapport, une fois l'audit termine."""
    if job_id not in JOBS:
        raise HTTPException(404, "Tache inconnue")
    if job_id not in REPORTS:
        raise HTTPException(409, f"Rapport pas encore pret (statut : {JOBS[job_id]['status']})")
    return REPORTS[job_id]


@app.get("/monitoring")
def get_monitoring() -> dict:
    """Indicateurs du service et du modèle, alertes de dérive, série des derniers audits."""
    return monitoring.summary(feedback_path=FEEDBACK_FILE)


@app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
def get_metrics() -> str:
    """Mêmes indicateurs au format Prometheus."""
    return monitoring.prometheus(feedback_path=FEEDBACK_FILE)


@app.get("/reentrainement")
def get_reentrainement() -> dict:
    """Dernière décision de l'orchestrateur, et exemples de juristes disponibles."""
    exemples, _ = retrain.collect(FEEDBACK_FILE)
    return {"derniere_decision": retrain.state(), "exemples_disponibles": len(exemples), "seuil": retrain.SEUIL_EXEMPLES}


@app.post("/reentrainement/verifier")
def run_reentrainement(force: bool = False) -> dict:
    """Lance l'orchestrateur : collecte des corrections, décision, paquet Colab si besoin."""
    return retrain.run(force=force, feedback_path=FEEDBACK_FILE)


@app.get("/reentrainement/paquet", include_in_schema=False)
def download_paquet() -> FileResponse:
    etat = retrain.state()
    if not etat or not etat.get("paquet"):
        raise HTTPException(404, "Aucun paquet de réentraînement préparé")
    return FileResponse(retrain.WORK_DIR / etat["paquet"], filename=etat["paquet"])


@app.post("/feedback", status_code=201)
def submit_feedback(feedback: Feedback) -> dict:
    """Enregistre la correction d'un juriste. Alimente le reentrainement."""
    if feedback.job_id not in JOBS:
        raise HTTPException(404, "Tache inconnue")
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    line = feedback.model_dump(mode="json") | {"date": datetime.now(timezone.utc).isoformat()}
    with FEEDBACK_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return {"status": "enregistre", "clause_ref": feedback.clause_ref}


# Interface web (code source dans frontend/) : http://localhost:8000/
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

    @app.get("/", include_in_schema=False)
    def accueil() -> RedirectResponse:
        return RedirectResponse("/app/")
