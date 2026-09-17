from src.data.prepare import split_clauses


def test_decoupe_sur_numerotation():
    texte = (
        "1. Objet du contrat. Le present contrat a pour objet la fourniture de services "
        "de conseil en transformation numerique pour une duree determinee.\n\n"
        "2. Duree. Le contrat prend effet a la date de signature et court pour une periode "
        "de vingt-quatre mois renouvelable par tacite reconduction.\n\n"
        "3. Donnees personnelles. Le prestataire traite les donnees pour le compte du client."
    )
    clauses = split_clauses(texte, min_chars=40, max_chars=2000)
    assert len(clauses) == 3
    assert clauses[0].startswith("1.")
    assert clauses[2].startswith("3.")


def test_ignore_les_segments_trop_courts():
    texte = "1. Ok.\n\n2. Cette clause est suffisamment longue pour etre conservee dans le resultat."
    clauses = split_clauses(texte, min_chars=40, max_chars=2000)
    assert len(clauses) == 1
