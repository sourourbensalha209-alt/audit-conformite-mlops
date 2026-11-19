"""Générateur de contrats synthétiques de la démonstration Big Data : reproductible et découpable en clauses."""

import json
import subprocess
import sys

from src.data.segment import segment


def test_generation_reproductible(tmp_path):
    for dossier in ("a", "b"):
        subprocess.run([sys.executable, "scripts/generer_contrats_demo.py", "--n", "5", "--out", str(tmp_path / dossier)],
                       check=True, capture_output=True)
    fa = sorted(p.name for p in (tmp_path / "a").glob("*.txt"))
    assert len(fa) == 5
    assert all((tmp_path / "a" / f).read_text(encoding="utf-8") == (tmp_path / "b" / f).read_text(encoding="utf-8") for f in fa)
    verite = json.loads((tmp_path / "a" / "_verite.json").read_text(encoding="utf-8"))
    assert set(verite) == set(fa)
    texte = (tmp_path / "a" / fa[0]).read_text(encoding="utf-8")
    assert len(segment(texte)) >= 3                               # titre/parties + clauses numérotées
