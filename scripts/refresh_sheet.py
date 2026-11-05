"""Régénère une feuille d'annotation PROPRE et PROTÉGÉE à partir de data/judge/cases.jsonl.

    python scripts/refresh_sheet.py                       # feuille vierge (pour Ghayth, ou pour tout refaire)
    python scripts/refresh_sheet.py --keep a_remplir.csv  # reprend les verdicts déjà donnés (.xlsx ou .csv)

À utiliser si le texte des clauses a été abîmé dans Excel : cases.jsonl, lui, n'est jamais modifié par Excel.
Les cas, leur ordre et la répartition Sourour / Ghayth / Les deux sont identiques à ceux de la première feuille.
La feuille produite est verrouillée : seules les cases jaunes sont modifiables.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, ROOT.as_posix())

from src.judge.sheet import merge_sheets, read_sheet, write_sheet  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/judge/cases.jsonl")
    ap.add_argument("--out", default="data/judge/a_remplir_propre.xlsx")
    ap.add_argument("--keep", nargs="*", default=[], help="fichiers (.xlsx ou .csv) dont on reprend les verdicts")
    args = ap.parse_args()

    path = Path(args.cases) if Path(args.cases).is_absolute() else ROOT / args.cases
    if not path.exists():
        sys.exit(f"{path} introuvable. Lancer d'abord : python scripts/build_judge_cases.py")
    cases = [json.loads(line) for line in open(path, encoding="utf-8")]
    # La famille « mismatch » réutilise VOLONTAIREMENT la clause d'un autre cas (avec une règle sans rapport) : on ne la compte pas.
    own = [c for c in cases if c["family"] != "mismatch"]
    distinct = len({c["clause"] for c in own})
    print(f"{len(cases)} cas lus dans {path.name} ({distinct} clauses distinctes sur {len(own)} cas qui ont chacun la leur)")
    if distinct < 0.9 * len(own):
        sys.exit("cases.jsonl semble abîmé (beaucoup de clauses identiques). Relancer : python scripts/build_judge_cases.py")

    keep = {}
    if args.keep:
        keep = {r["id"]: r for r in merge_sheets([read_sheet(p if Path(p).is_absolute() else ROOT / p) for p in args.keep])}
        n = sum(1 for r in keep.values() if r["verdict_sourour"] or r["verdict_ghayth"])
        print(f"{n} cas avec un verdict repris de : {', '.join(args.keep)}")

    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    who = write_sheet(out, cases, keep=keep)
    shown = out.relative_to(ROOT) if ROOT in out.parents else out
    print(f"Feuille propre et protégée écrite : {shown} ({len(who)} cas à annoter)")


if __name__ == "__main__":
    main()
