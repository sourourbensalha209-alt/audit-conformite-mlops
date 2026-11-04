"""Feuille Excel d'annotation : l'équipe y écrit son verdict (menu déroulant), le programme la relit.

Deux onglets : « Consignes » (définitions, répartition, exemple) et « Cas ». Les cellules à remplir sont en jaune.
"""

from __future__ import annotations

from .prompt import VERDICTS

COLUMNS = ["id", "famille", "catégorie", "règle de contrôle", "sources juridiques fournies", "clause du contrat",
           "à annoter par", "verdict_sourour", "verdict_ghayth", "commentaire"]
WIDTHS = [7, 11, 22, 52, 46, 70, 14, 16, 16, 34]


def _height(values, widths, line=13.0, minimum=30.0) -> float:
    """Hauteur de ligne (en points) pour que tout le texte à retour automatique soit visible."""
    import math

    lines = 1
    for v, w in zip(values, widths):
        if v:
            lines = max(lines, sum(max(1, math.ceil(len(part) / max(w * 1.05, 1))) for part in str(v).split("\n")))
    return min(409.0, max(minimum, lines * line + 6))


def assign(cases: list[dict], n_shared: int = 10) -> dict[str, str]:
    """Répartit les cas à annoter entre les deux personnes ; `n_shared` cas sont annotés par les deux (mesure de l'accord)."""
    todo = [c["id"] for c in cases if c["a_annoter"]]
    shared = set(todo[:: max(1, len(todo) // n_shared)][:n_shared])
    out, flip = {}, 0
    for cid in todo:
        if cid in shared:
            out[cid] = "Les deux"
        else:
            out[cid] = "Sourour" if flip % 2 == 0 else "Ghayth"
            flip += 1
    return out


def write_sheet(path, cases: list[dict]) -> dict[str, str]:
    from openpyxl import Workbook
    from openpyxl.worksheet.properties import PageSetupProperties
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.worksheet.datavalidation import DataValidation

    who = assign(cases)
    todo = [c for c in cases if c["a_annoter"]]
    wb = Workbook()
    arial = lambda **kw: Font(name="Arial", size=kw.pop("size", 10), **kw)
    yellow = PatternFill("solid", fgColor="FFFF99")
    head = PatternFill("solid", fgColor="14213D")
    thin = Side(style="thin", color="D9D3C6")

    # ---- onglet Consignes
    ws = wb.active
    ws.title = "Consignes"
    ws.sheet_view.showGridLines = False
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 100
    rows = [
        ("Annotation des cas du juge", None, "title"),
        ("But", "Donner pour chaque cas le verdict qu'un relecteur raisonnable donnerait en comparant la CLAUSE à la RÈGLE. "
                "Ces verdicts servent de référence pour mesurer le juge automatique : c'est un accord avec l'équipe, "
                "pas une validation juridique.", None),
        ("Les trois verdicts", None, "sub"),
        ("conforme", "La clause satisfait tous les points de la règle.", None),
        ("non_conforme", "La clause contredit ou omet clairement un point exigé par la règle.", None),
        ("a_verifier", "La clause ne traite pas le sujet de cette règle, ou ne permet pas de trancher (information insuffisante).", None),
        ("Qui remplit quoi", None, "sub"),
        ("Colonne G", "Chaque cas est attribué à Sourour, à Ghayth, ou aux DEUX (10 cas : ils servent à mesurer votre accord, remplissez-les chacun de votre côté, sans vous concerter).", None),
        ("Cellules jaunes", "Colonnes H (verdict_sourour), I (verdict_ghayth) et J (commentaire facultatif). Choisissez le verdict dans le menu déroulant. Ne modifiez aucune autre cellule.", None),
        ("Règle d'or", "Ne regardez PAS les réponses du modèle avant d'avoir fini, et ne changez pas un verdict après coup.", None),
        ("Exemple (à ne pas recopier)", None, "sub"),
        ("Règle", "[R06] Plafonnement de la responsabilité : la responsabilité de chaque partie est plafonnée ; le plafond s'applique aux deux parties ; les exceptions sont listées.", None),
        ("Clause", "Supplier's total liability under this Agreement shall not exceed the fees paid by Customer in the twelve months preceding the claim.", None),
        ("Verdict raisonnable", "a_verifier : le plafond existe, mais la clause ne dit pas s'il s'applique aux deux parties ni quelles exceptions s'appliquent.", None),
        ("Enregistrer", "Enregistrez en .xlsx (Fichier > Enregistrer). Gardez le nom du fichier : a_remplir.xlsx.", None),
        ("Avancement", None, "sub"),
    ]
    r = 1
    for k, v, kind in rows:
        if kind == "title":
            ws.cell(r, 1, k).font = arial(size=14, bold=True)
        elif kind == "sub":
            ws.cell(r, 1, k).font = arial(bold=True, color="8A5412")
        else:
            ws.cell(r, 1, k).font = arial(bold=True)
            c = ws.cell(r, 2, v)
            c.font = arial()
            c.alignment = Alignment(wrap_text=True, vertical="top")
            ws.cell(r, 1).alignment = Alignment(vertical="top")
            ws.row_dimensions[r].height = _height([v], [100], minimum=16)
        r += 1
    n = len(todo)
    last = n + 1
    for label, col in (("Cas remplis par Sourour", "H"), ("Cas remplis par Ghayth", "I")):
        ws.cell(r, 1, label).font = arial(bold=True)
        ws.cell(r, 2, f"=COUNTA(Cas!{col}2:{col}{last})").font = arial()
        ws.cell(r, 2).alignment = Alignment(horizontal="left")
        r += 1
    ws.cell(r, 1, "Nombre de cas à annoter").font = arial(bold=True)
    ws.cell(r, 2, n).font = arial(color="0000FF")
    ws.cell(r, 2).alignment = Alignment(horizontal="left")
    ws.cell(r + 1, 2, "(valeur saisie : nombre de lignes de l'onglet Cas)").font = arial(italic=True, color="5E6672")

    # ---- onglet Cas
    cs = wb.create_sheet("Cas")
    for j, (h, w) in enumerate(zip(COLUMNS, WIDTHS), 1):
        c = cs.cell(1, j, h)
        c.font = arial(bold=True, color="FFFFFF")
        c.fill = head
        c.alignment = Alignment(wrap_text=True, vertical="center")
        cs.column_dimensions[c.column_letter].width = w
    for i, c in enumerate(todo, 2):
        src = "\n".join(f"[{s['id']}] {s['citation']} : {' '.join(s['text'].split())[:350]}" for s in c["sources"]
                        if s["id"] != c["rule"]["id"]) or "aucune"
        vals = [c["id"], c["family"], c["category"] or "", f"[{c['rule']['id']}] {c['rule']['titre']} : {c['rule']['regle']}",
                src, c["clause"], who[c["id"]], None, None, None]
        for j, v in enumerate(vals, 1):
            cell = cs.cell(i, j, v)
            cell.font = arial()
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.border = Border(top=thin, bottom=thin, left=thin, right=thin)
            if j in (8, 9, 10):
                cell.fill = yellow
        cs.row_dimensions[i].height = _height(vals, WIDTHS)
    dv = DataValidation(type="list", formula1='"' + ",".join(VERDICTS) + '"', allow_blank=True, showErrorMessage=True,
                        errorTitle="Verdict inconnu", error="Choisissez conforme, non_conforme ou a_verifier.")
    cs.add_data_validation(dv)
    dv.add(f"H2:I{last}")
    cs.freeze_panes = "B2"
    for sheet in (ws, cs):                       # impression : paysage, une page de large
        sheet.page_setup.orientation = "landscape"
        sheet.page_setup.fitToWidth = 1
        sheet.page_setup.fitToHeight = 0
        sheet.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)
    cs.print_title_rows = "1:1"
    wb.save(path)
    return who


def read_sheet(path) -> list[dict]:
    from openpyxl import load_workbook

    ws = load_workbook(path, data_only=True)["Cas"]
    head = [c.value for c in ws[1]]
    ix = {h: i for i, h in enumerate(head)}
    out = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[ix["id"]]:
            out.append({"id": row[ix["id"]], "verdict_sourour": (row[ix["verdict_sourour"]] or "").strip(),
                        "verdict_ghayth": (row[ix["verdict_ghayth"]] or "").strip(),
                        "commentaire": row[ix["commentaire"]] or ""})
    return out
