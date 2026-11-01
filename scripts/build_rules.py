"""Génère data/rules/rules_contractuelles.json.

Les 41 règles suivent exactement les 41 catégories du classifieur (labels.json) :
une clause classée "Cap On Liability" est jugée avec la règle R06.

IMPORTANT : ce sont des règles de bonne pratique contractuelle (type "playbook" de
revue de contrat), PAS des articles de loi. Elles sont à faire valider par un juriste
avant de prétendre à une valeur juridique : le fichier porte `validated: false`.
Les citations d'articles viennent du corpus de lois (RGPD, droit tunisien), pas d'ici.

Usage : python scripts/build_rules.py
"""

import json
from pathlib import Path

# (catégorie, titre, règle, gravité, absence_à_signaler, mots_clés_anglais)
RULES = [
    ("Affiliate License-Licensee", "Licence étendue aux affiliés du licencié",
     "Si la licence s'étend aux sociétés affiliées du licencié, le contrat définit la notion d'affilié "
     "(seuil de contrôle) et précise que le licencié reste responsable des actes de ses affiliés.",
     "medium", False, "affiliate licensee license extends subsidiaries controlled entities"),
    ("Affiliate License-Licensor", "Licence consentie par les affiliés du concédant",
     "Si des affiliés du concédant concèdent des droits, le contrat identifie les droits concernés, "
     "garantit que le concédant peut les concéder et précise qui répond des garanties.",
     "medium", False, "affiliate licensor license granted by affiliates of licensor"),
    ("Agreement Date", "Date de conclusion",
     "Le contrat indique sa date de conclusion (signature) de façon non ambiguë.",
     "medium", True, "agreement dated made and entered into as of date"),
    ("Anti-Assignment", "Cession du contrat",
     "La cession du contrat ou des droits par une partie est subordonnée à l'accord écrit de l'autre. "
     "Vérifier que la règle est réciproque et que les exceptions (fusion, société affiliée, successeur) sont explicites.",
     "medium", False, "assign assignment transfer consent prior written consent delegate"),
    ("Audit Rights", "Droit d'audit",
     "Tout droit d'audit est encadré : fréquence, préavis, périmètre, confidentialité des informations "
     "consultées et répartition des coûts.",
     "medium", False, "audit inspect records books examine upon notice"),
    ("Cap On Liability", "Plafonnement de la responsabilité",
     "La responsabilité de chaque partie est plafonnée (montant fixe ou référence, par exemple les sommes payées "
     "sur les 12 derniers mois). Le plafond s'applique aux deux parties et les exceptions (dol, faute lourde, "
     "confidentialité, propriété intellectuelle) sont listées expressément.",
     "high", True, "limitation of liability aggregate liability shall not exceed cap fees paid twelve months"),
    ("Change Of Control", "Changement de contrôle",
     "Si un changement de contrôle donne un droit de résiliation ou exige un consentement, le déclencheur est "
     "défini (seuil, opération visée), le délai d'exercice est précisé et la clause est réciproque.",
     "medium", False, "change of control merger acquisition ownership terminate consent"),
    ("Competitive Restriction Exception", "Exceptions aux restrictions de concurrence",
     "Les exceptions aux clauses d'exclusivité ou de non-concurrence (activités existantes, produits exclus, "
     "participations minoritaires) sont énumérées de façon limitative.",
     "medium", False, "exception exclusivity non-compete notwithstanding provided that shall not restrict"),
    ("Covenant Not To Sue", "Engagement de ne pas poursuivre",
     "Un engagement de ne pas agir en justice précise sa portée (droits ou produits visés), sa durée et "
     "s'il est réciproque.",
     "medium", False, "covenant not to sue shall not bring claim assert patent against"),
    ("Document Name", "Intitulé du contrat",
     "Le contrat porte un intitulé qui permet de l'identifier et d'en connaître la nature.",
     "low", True, "agreement contract title name of document"),
    ("Effective Date", "Date d'entrée en vigueur",
     "La date d'entrée en vigueur est précisée ; si elle diffère de la date de signature, la différence "
     "et sa condition (événement, formalité) sont expliquées.",
     "medium", True, "effective date commence take effect as of"),
    ("Exclusivity", "Exclusivité",
     "Toute exclusivité précise son périmètre (produits, territoire, clients), sa durée, ses contreparties "
     "et ses exceptions. Vérifier sa compatibilité avec le droit de la concurrence applicable.",
     "high", False, "exclusive exclusivity sole and exclusive territory only"),
    ("Expiration Date", "Durée et échéance",
     "La durée initiale du contrat ou sa date d'échéance est indiquée clairement, avec son point de départ.",
     "high", True, "term expires expiration initial term years from the effective date"),
    ("Governing Law", "Droit applicable",
     "Le droit applicable au contrat est désigné expressément. Vérifier la cohérence avec la juridiction "
     "ou le mode de règlement des litiges prévus, et la validité du choix pour ce type de contrat.",
     "high", True, "governed by laws of jurisdiction construed in accordance with"),
    ("Insurance", "Assurances",
     "Les assurances exigées sont précisées : types de couverture, montants minimaux, durée, et obligation "
     "de fournir une attestation.",
     "medium", False, "insurance coverage certificate policy maintain insurer"),
    ("Ip Ownership Assignment", "Cession de propriété intellectuelle",
     "La cession de droits de propriété intellectuelle désigne précisément les œuvres ou inventions visées, "
     "l'étendue des droits cédés, la contrepartie et les formalités. Vérifier les droits qui ne peuvent pas "
     "être cédés selon le droit applicable.",
     "high", False, "assign ownership intellectual property work product title and interest invention"),
    ("Irrevocable Or Perpetual License", "Licence irrévocable ou perpétuelle",
     "Une licence irrévocable ou perpétuelle précise ses seules causes de fin (par exemple manquement grave "
     "non réparé) et la contrepartie versée pour une durée sans limite.",
     "medium", False, "irrevocable perpetual license in perpetuity paid-up"),
    ("Joint Ip Ownership", "Copropriété intellectuelle",
     "En cas de copropriété, le contrat fixe les quotes-parts, le droit d'exploiter, les conditions pour "
     "concéder une licence ou céder sa part, et qui supporte les frais de protection.",
     "medium", False, "jointly owned joint ownership co-own intellectual property both parties"),
    ("License Grant", "Concession de licence",
     "La licence précise son objet, son étendue, son territoire, sa durée, son caractère exclusif ou non, "
     "et le droit ou non de sous-licencier.",
     "high", False, "grants license non-exclusive right to use sublicense territory licensed"),
    ("Liquidated Damages", "Clause pénale",
     "Les pénalités forfaitaires correspondent à un déclencheur précis et à un montant proportionné au préjudice "
     "prévisible. Vérifier le pouvoir du juge de les réviser selon le droit applicable.",
     "medium", False, "liquidated damages penalty shall pay per day delay"),
    ("Minimum Commitment", "Engagement minimum",
     "Un engagement minimum (volume ou montant) est chiffré, avec sa période de mesure et les conséquences "
     "d'un engagement non atteint.",
     "medium", False, "minimum commitment minimum purchase at least shortfall take or pay"),
    ("Most Favored Nation", "Clause du client le plus favorisé",
     "La clause précise le périmètre comparé, le mécanisme de vérification, le délai d'alignement "
     "et la sanction en cas de non-respect.",
     "medium", False, "most favored nation lowest price no less favorable terms best price"),
    ("No-Solicit Of Customers", "Non-sollicitation de clientèle",
     "L'interdiction de démarcher les clients est limitée dans le temps, dans son périmètre et dans son objet.",
     "medium", False, "not solicit customers clients accounts business of"),
    ("No-Solicit Of Employees", "Non-sollicitation du personnel",
     "La non-sollicitation du personnel a une durée limitée et excepte les candidatures spontanées "
     "et les offres publiques d'emploi.",
     "low", False, "not solicit employees hire personnel recruit"),
    ("Non-Compete", "Non-concurrence",
     "La non-concurrence est limitée dans le temps, dans l'espace et dans son objet, proportionnée "
     "à l'intérêt protégé, et vérifiée au regard du droit applicable.",
     "high", False, "non-compete shall not compete competing business restricted period"),
    ("Non-Disparagement", "Non-dénigrement",
     "L'obligation de non-dénigrement est réciproque, de portée précise, et réserve les déclarations "
     "exigées par la loi ou une autorité.",
     "low", False, "disparage non-disparagement negative statements reputation"),
    ("Non-Transferable License", "Licence non transférable",
     "La non-transférabilité de la licence est exprimée clairement, avec ses exceptions éventuelles "
     "(affiliés, successeur en cas de fusion).",
     "medium", False, "non-transferable not transfer license shall not be assigned"),
    ("Notice Period To Terminate Renewal", "Préavis de non-renouvellement",
     "Le préavis pour empêcher le renouvellement est explicite, d'une durée raisonnable, avec la forme "
     "de notification et la partie qui peut l'exercer.",
     "medium", False, "notice of non-renewal days prior written notice before expiration renew"),
    ("Parties", "Identification des parties",
     "Chaque partie est identifiée complètement : dénomination, forme juridique, siège ou adresse, "
     "et représentant ayant pouvoir d'engager.",
     "high", True, "by and between party parties company inc corporation address"),
    ("Post-Termination Services", "Obligations après la fin du contrat",
     "Les obligations après la fin du contrat (assistance à la transition, restitution des données, "
     "survie de certaines clauses) sont précisées.",
     "medium", False, "upon termination transition assistance return data survive following termination"),
    ("Price Restrictions", "Restrictions de prix",
     "Toute restriction portant sur les prix pratiqués par une partie est vérifiée au regard du droit "
     "de la concurrence applicable ; un prix imposé est signalé.",
     "high", False, "price resale price shall not charge pricing restriction minimum price"),
    ("Renewal Term", "Renouvellement",
     "Les modalités de renouvellement (tacite ou expresse) et la durée de chaque période de renouvellement "
     "sont indiquées.",
     "medium", False, "renew renewal automatically successive additional term"),
    ("Revenue/Profit Sharing", "Partage de revenus",
     "Le partage de revenus ou de profits précise l'assiette, le taux, le calendrier de versement "
     "et le droit de vérifier les calculs.",
     "medium", False, "revenue share profit sharing royalty percentage of net revenue"),
    ("Rofr/Rofo/Rofn", "Droit de préemption ou de première offre",
     "Un droit de préemption, de première offre ou de première négociation précise sa procédure, "
     "ses délais et les conditions de la transaction visée.",
     "medium", False, "right of first refusal first offer first negotiation"),
    ("Source Code Escrow", "Séquestre du code source",
     "Le séquestre du code source désigne le tiers séquestre, les cas de libération, la mise à jour "
     "des versions déposées et la prise en charge des frais.",
     "medium", False, "source code escrow escrow agent release conditions deposit"),
    ("Termination For Convenience", "Résiliation sans motif",
     "Une résiliation sans motif précise la partie qui en bénéficie, le préavis et les conséquences "
     "financières. Vérifier la réciprocité.",
     "medium", False, "terminate for convenience without cause upon days notice"),
    ("Third Party Beneficiary", "Bénéficiaires tiers",
     "Les tiers bénéficiaires sont identifiés avec la portée de leurs droits ; à défaut, le contrat "
     "exclut expressément tout droit des tiers.",
     "medium", False, "third party beneficiary no third party rights beneficiaries"),
    ("Uncapped Liability", "Responsabilité illimitée",
     "Les cas de responsabilité illimitée sont limités à des situations justifiées (dol, faute lourde, "
     "dommage corporel...) et vérifiés selon le droit applicable. Une extension large est signalée.",
     "high", False, "unlimited liability not subject to cap uncapped exclusions gross negligence willful"),
    ("Unlimited/All-You-Can-Eat-License", "Licence d'usage illimité",
     "Une licence d'usage illimité définit tout de même le périmètre d'utilisation (entités, finalité) "
     "et ses limites contre les abus.",
     "medium", False, "unlimited use enterprise license any number of users all you can eat"),
    ("Volume Restriction", "Restriction de volume",
     "Les plafonds ou seuils de volume sont définis, avec leur période de calcul et les conséquences "
     "en cas de dépassement.",
     "medium", False, "volume limit not exceed maximum quantity units per year"),
    ("Warranty Duration", "Durée de garantie",
     "La durée de garantie est précisée avec son point de départ, ainsi que le recours en cas de défaut.",
     "medium", False, "warranty period warranted for months remedy defect repair replace"),
]

# Le classifieur n'a PAS de catégorie "données personnelles" ni "sous-traitance" (CUAD n'en contient pas).
# Ces contrôles sont donc déclenchés par mots-clés + recherche dans le corpus RGPD, sur n'importe quelle clause.
# Articles cités : RGPD (UE) 2016/679.
RGPD_CHECKS = [
    {"id": "G01", "theme": "Contrat de sous-traitance des données",
     "regle": "Le traitement par un sous-traitant est encadré par un contrat précisant objet, durée, nature, finalité, "
              "type de données, catégories de personnes et obligations des parties.",
     "refs": ["RGPD art. 28(3)"], "absence_a_signaler": True},
    {"id": "G02", "theme": "Instructions documentées",
     "regle": "Le sous-traitant ne traite les données que sur instruction documentée du responsable du traitement.",
     "refs": ["RGPD art. 28(3)(a)"], "absence_a_signaler": True},
    {"id": "G03", "theme": "Sous-traitants ultérieurs",
     "regle": "Le recours à un autre sous-traitant exige l'autorisation du responsable et lui impose les mêmes obligations.",
     "refs": ["RGPD art. 28(2)", "RGPD art. 28(4)"], "absence_a_signaler": True},
    {"id": "G04", "theme": "Sécurité des données",
     "regle": "Des mesures techniques et organisationnelles de sécurité appropriées sont prévues.",
     "refs": ["RGPD art. 32", "RGPD art. 28(3)(c)"], "absence_a_signaler": True},
    {"id": "G05", "theme": "Notification des violations",
     "regle": "Le sous-traitant notifie au responsable toute violation de données dans les meilleurs délais.",
     "refs": ["RGPD art. 33(2)"], "absence_a_signaler": True},
    {"id": "G06", "theme": "Conservation et restitution",
     "regle": "La durée de conservation est précisée ; à la fin du contrat, les données sont supprimées ou restituées.",
     "refs": ["RGPD art. 5(1)(e)", "RGPD art. 13(2)(a)", "RGPD art. 28(3)(g)"], "absence_a_signaler": True},
    {"id": "G07", "theme": "Droits des personnes",
     "regle": "Le sous-traitant aide le responsable à répondre aux demandes d'exercice des droits des personnes.",
     "refs": ["RGPD art. 28(3)(e)", "RGPD art. 12 à 22"], "absence_a_signaler": False},
    {"id": "G08", "theme": "Transferts hors UE",
     "regle": "Tout transfert de données hors de l'Union européenne repose sur un mécanisme prévu par le RGPD "
              "(décision d'adéquation, clauses types, garanties appropriées).",
     "refs": ["RGPD art. 44 à 49"], "absence_a_signaler": False},
    {"id": "G09", "theme": "Audit",
     "regle": "Le sous-traitant fournit les informations nécessaires et permet les audits du responsable.",
     "refs": ["RGPD art. 28(3)(h)"], "absence_a_signaler": False},
]

RGPD_TRIGGERS = [
    "personal data", "data protection", "gdpr", "data subject", "processor", "controller",
    "sub-processor", "subprocessor", "privacy", "data breach", "personally identifiable",
    "données personnelles", "données à caractère personnel", "sous-traitant", "responsable du traitement",
    "rgpd", "protection des données", "violation de données",
]


def main() -> None:
    rules = []
    for i, (cat, titre, regle, gravite, absence, kw) in enumerate(RULES, start=1):
        rules.append({
            "id": f"R{i:02d}", "category": cat, "titre": titre, "regle": regle,
            "gravite": gravite, "absence_a_signaler": absence, "keywords_en": kw,
        })
    out = {
        "version": "0.1",
        "validated": False,
        "avertissement": "Règles de bonne pratique contractuelle proposées par l'équipe, à valider par un juriste. "
                         "Elles ne sont pas des articles de loi.",
        "rules": rules,
        "rgpd_checks": RGPD_CHECKS,
        "rgpd_triggers": RGPD_TRIGGERS,
    }
    path = Path(__file__).resolve().parents[1] / "data" / "rules" / "rules_contractuelles.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(rules)} règles + {len(RGPD_CHECKS)} contrôles RGPD -> {path}")


if __name__ == "__main__":
    main()
