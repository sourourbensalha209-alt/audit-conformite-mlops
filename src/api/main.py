"""Service d'audit de conformite.

Lancement :
    uvicorn src.api.main:app --reload
Documentation interactive : http://localhost:8000/docs
"""

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Agent d'audit de conformite documentaire",
    description="Analyse un document et retourne un rapport de conformite clause par clause.",
    version="0.1.0",
)

# Stockage en memoire, a remplacer par une base lors de l'etape de deploiement.
JOBS: dict[str, dict] = {}


class Verdict(str, Enum):
    conforme = "conforme"
    non_conforme = "non_conforme"
    a_verifier = "a_verifier"
    manquante = "manquante"


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
    findings: list[ClauseFinding] = []


class Feedback(BaseModel):
    job_id: str
    clause_ref: str
    verdict_corrige: Verdict
    commentaire: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/audit", response_model=AuditReport, status_code=202)
async def submit_audit(file: UploadFile = File(...)) -> AuditReport:
    """Soumet un document a l'audit. Retourne un identifiant de tache."""
    if not file.filename.lower().endswith((".pdf", ".docx")):
        raise HTTPException(400, "Formats acceptes : .pdf, .docx")

    job_id = str(uuid4())
    report = AuditReport(
        job_id=job_id,
        status="en_attente",
        created_at=datetime.now(timezone.utc),
        document=file.filename,
    )
    JOBS[job_id] = report.model_dump()
    # TODO : declencher le pipeline d'audit en tache de fond
    return report


@app.get("/audit/{job_id}", response_model=AuditReport)
def get_audit(job_id: str) -> AuditReport:
    """Recupere le rapport d'audit."""
    if job_id not in JOBS:
        raise HTTPException(404, "Tache inconnue")
    return AuditReport(**JOBS[job_id])


@app.post("/feedback", status_code=201)
def submit_feedback(feedback: Feedback) -> dict:
    """Enregistre la correction d'un juriste. Alimente le reentrainement."""
    if feedback.job_id not in JOBS:
        raise HTTPException(404, "Tache inconnue")
    # TODO : persister la correction dans le jeu de donnees de reentrainement
    return {"status": "enregistre", "clause_ref": feedback.clause_ref}
