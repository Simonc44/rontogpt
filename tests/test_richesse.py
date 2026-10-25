"""Nettoyage du raisonnement (CoT) : ce que le modele PENSE ne sort pas.

Le mode `riche` (dissertation multi-pass) a ete supprime : ces tests ne
couvrent donc plus que ce qui reste vrai et qui compte — la separation
entre la REFLEXION du modele (balises  thinking ou <thinking>) et la REPONSE
qu'il livre, et le fait que la reflexion n'est jamais jetee silencieusement :
elle est detournee vers le flux typé et la memoire du tour.
"""
from aura import llama_cerveau

# Les balises sont assemblees a partir de morceaux : ecrites d'un seul bloc
# dans ce fichier, certains outils de rendu les prennent pour du HTML et les
# suppriment — le test ne testerait alors plus la bonne balise.
OUV = "<" "think>"
FER = "<" "/think>"


class TestSeparationPenseeReponse:
    def test_reflexion_native_gemma_extracte(self):
        """Format d'un cerveau a reflexion native (Gemma Heretic, R1...)."""
        brut = (f"{OUV}2 plus 2 font 4. Je verifie : oui.{FER}"
                "2 plus 2 font 4.")
        propre, reflexion = llama_cerveau.separer_reflexion(brut)
        assert propre == "2 plus 2 font 4."
        assert "je verifie" in reflexion.lower()
        assert "2 plus 2" in reflexion

    def test_consigne_thinking_longue_extractee(self):
        """Autre famille de balise : <thinking>, posee par _INSTRUCTION_THINK."""
        brut = ("<thinking>1. Analyse : ok. 2. Plan : A, B.</thinking>"
                "Reponse finale claire.")
        propre, reflexion = llama_cerveau.separer_reflexion(brut)
        assert propre == "Reponse finale claire."
        assert "Analyse" in reflexion

    def test_balise_orpheline(self):
        """Generation coupee : le bloc de reflexion brut ne doit JAMAIS sortir.

        Le matching question/reponse du banc comparerait la VERITE a la
        reflexion du modele — un faux negatif garanti.
        """
        brut = "<thinking>reflexion coupee... Reponse partielle."
        propre, reflexion = llama_cerveau.separer_reflexion(brut)
        assert "<thinking>" not in propre
        assert propre == ""          # rien avant la balise ouvrante
        # La reflexion non fermee n'est PAS jetee : elle est detournee (elle
        # part dans le flux et la memoire du tour), mais jamais livree.
        assert "reflexion coupee" in reflexion

    def test_balise_orpheline_apres_prefixe(self):
        """Le modele commence a repondre, puis ouvre une pensee qu'il ne ferme pas."""
        brut = f"Voici le debut. {OUV}pensee inachevee"
        propre, reflexion = llama_cerveau.separer_reflexion(brut)
        assert propre == "Voici le debut."
        assert "pensee inachevee" in reflexion
        assert OUV not in propre

    def test_jamais_de_balise_dans_la_reponse(self):
        """Quel que soit l'echantillon, aucune balise ne doit fuir cote reponse."""
        for brut in (f"{OUV}coupee", f"prefixe {OUV}coupee",
                     f"{OUV}fermee{FER}reponse", "texte brut"):
            propre, _ = llama_cerveau.separer_reflexion(brut)
            assert OUV not in propre
            assert FER not in propre
            assert "think" not in propre.lower()

    def test_sans_balise_texte_intact(self):
        propre, reflexion = llama_cerveau.separer_reflexion("Reponse simple.")
        assert propre == "Reponse simple."
        assert reflexion == ""

    def test_deux_blocs_de_reflexion(self):
        """Le modele relance parfois un bloc de verification apres sa reponse."""
        brut = (f"{OUV}premier doute{FER}Debut. "
                f"{OUV}verification finale{FER}Fin.")
        propre, reflexion = llama_cerveau.separer_reflexion(brut)
        assert "Debut." in propre and "Fin." in propre
        assert "doute" not in propre and "verification" not in propre
        assert "doute" in reflexion and "verification" in reflexion
