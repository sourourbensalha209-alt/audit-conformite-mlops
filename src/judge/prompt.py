"""Construction du prompt du juge : une clause, une règle de contrôle, des sources juridiques, un verdict en JSON.

Trois verdicts seulement : "conforme", "non_conforme", "a_verifier". Le troisième est une vraie réponse : le juge DOIT
s'abstenir quand la clause ne traite pas le sujet de la règle, ou quand l'information manque. Il ne peut citer que des
sources dont l'identifiant figure dans le prompt (contrôlé automatiquement : src/judge/parse.py).

Deux conditions, pour mesurer l'apport de la règle fournie :
  with_rule=True   la règle de contrôle et les sources juridiques sont dans le prompt (RAG)
  with_rule=False  le même modèle, sans règle ni source : il juge de mémoire, et ne doit rien citer
"""

from __future__ import annotations

VERDICTS = ("conforme", "non_conforme", "a_verifier")

SYSTEM = (
    "Tu es un assistant d'audit de contrats commerciaux. Tu compares UNE clause à UNE règle de contrôle. "
    "Tu es prudent : tu ne tranches que si la clause permet de le faire, sinon tu t'abstiens. "
    "Tu ne cites jamais une source qui ne t'a pas été fournie."
)

_CONSIGNES = """CONSIGNES :
- "conforme" : la clause satisfait tous les points de la règle.
- "non_conforme" : la clause contredit ou omet clairement un point exigé par la règle.
- "a_verifier" : la clause ne traite pas le sujet de cette règle, ou l'information est insuffisante pour trancher.
{sources_rule}
- Réponds UNIQUEMENT par un objet JSON, sans autre texte :
{{"verdict": "conforme" | "non_conforme" | "a_verifier", "justification": "<deux phrases au plus, en français>", "sources": [<identifiants cités>]}}"""


def build_messages(case: dict, with_rule: bool = True) -> list[dict]:
    clause = " ".join(case["clause"].split())
    if with_rule:
        rule = case["rule"]
        blocs = [f"RÈGLE DE CONTRÔLE [{rule['id']}] — {rule['titre']} : {rule['regle']}"]
        extra = [s for s in case.get("sources", []) if s["id"] != rule["id"]]
        if extra:
            blocs.append("SOURCES JURIDIQUES FOURNIES :")
            blocs += [f"[{s['id']}] {s['citation']} : {' '.join(s['text'].split())[:700]}" for s in extra]
        else:
            blocs.append("SOURCES JURIDIQUES FOURNIES : aucune.")
        sources_rule = ('- Dans "sources", ne mets que des identifiants figurant entre crochets ci-dessus '
                        '(la règle, ou une source juridique). Si tu ne t\'appuies sur aucun, mets [].')
    else:
        blocs = [f"TYPE DE CLAUSE : {case.get('category') or 'non précisé'}",
                 "Aucune règle de contrôle ni source juridique n'est fournie : juge d'après les bonnes pratiques usuelles."]
        sources_rule = '- Aucune source ne t\'est fournie : "sources" doit être [].'
    blocs += ["", 'CLAUSE DU CONTRAT À JUGER :', f'"""{clause}"""', "", _CONSIGNES.format(sources_rule=sources_rule)]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(blocs)}]
