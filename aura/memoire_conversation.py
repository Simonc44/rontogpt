"""Memoire de conversation : RLM (MIT), fenetre glissante, etat evenementiel.

Trois mecanismes complémentaires pour tenir un 1B dans sa fenetre :

1. RLM (Recursive Language Models, MIT) : l'historique complet vit dans
   un FICHIER externe, jamais injecte en entier. L'IA y accede par un
   OUTIL symbolique `rechercher_dans_historique(mot_cle)` — la fenetre
   du prompt reste minuscule.

2. Fenetre glissante avec ANCRAGE : les ancres fixes (message systeme,
   consignes, dissertation TAS) ne bougent jamais ; en bas, seuls les
   4 derniers tours bruts survivent. Au-dela : effet « lost in the
   middle » — un 1B n'exploite plus les vieux messages, ils ne font
   que polluer sa fenetre.

3. Etat pilote par EVENEMENTS (event-driven state) : un script pur
   extrait l'etat civil de la conversation (nom, preferences, faits
   numeriques de l'utilisateur) dans un mini-dictionnaire JSON de
   quelques octets, injecte dans le systeme. Plus besoin des phrases
   « je m'appelle Pierre » dans l'historique.

Usage orchestrateur :
    memoire = MemoireConversation()
    memoire.ajouter("user", q)
    memoire.ajouter("assistant", r)
    tours = memoire.fenetre()          # 4 derniers tours bruts
    etat  = memoire.etat_json()        # {"utilisateur": "Pierre", ...}
    hit   = memoire.rechercher("Pierre")  # outil RLM (recherche fichier)
"""
import json
import os
import re
import threading
import time
from pathlib import Path

# ── 1. RLM : historique externalise en fichier ────────────────────────

_TAMPON = 4096                # lecture arriere par blocs de 4 Ko
_MAX_RESULTATS = 3            # resultats max renvoyes a l'IA
_MAX_EXTRAIT = 280            # taille max d'un extrait
_TAILLE_ROTATION = 32 * 1024 * 1024   # rotation : archive au-dela de 32 Mo
_MAX_ARCHIVES = 3                     # archives d'historique conservees
_MAX_FICHIER = 20_000                 # signes livres par la lecture fichier
_MAX_OCTETS_FICHIER = 80_000          # octets lus au plus (fichier tronque)


def _sans_accents(texte: str) -> str:
    import unicodedata
    n = unicodedata.normalize("NFKD", texte.lower())
    return "".join(c for c in n if not unicodedata.combining(c))


class MemoireConversation:
    """Historique externalise (RLM) + fenetre glissante + etat evenementiel.

    Thread-safe : l'orchestrateur est mono-thread par question, mais le
    fan-out web || PGS tourne en parallele et le RLSS/selfplay peuvent
    partager une instance — un verrou coute 0 et evite les surprises.
    """

    def __init__(self, chemin: str | Path | None = None,
                 fenetre: int = 4):
        self._verrou = threading.Lock()
        self._chemin = (Path(chemin) if chemin
                        else Path(__file__).parent / ".conversation.jsonl")
        self._fenetre = max(2, fenetre)
        # etat civil : dictionnaire plat {cle: valeur} (JSON, quelques octets)
        # — PERSISTANT : recharge depuis .etat.json au demarrage, sauvegarde
        # a chaque extraction. Le nom de l'utilisateur survit aux sessions.
        self._chemin_etat = self._chemin.with_suffix(".etat.json")
        self._etat: dict[str, str] = self._charger_etat()
        # Une session commence neuve : l'historique complet reste disponible
        # via l'outil RLM, mais il ne doit pas reapparaitre dans le chat apres
        # chaque redemarrage. AURA_REPRENDRE_CONVERSATION=1 restaure le
        # comportement historique pour les personnes qui le souhaitent.
        reprendre = os.environ.get("AURA_REPRENDRE_CONVERSATION", "").strip() == "1"
        self._tours: list[dict[str, str]] = self._charger_fenetre() if reprendre else []
        # outil RLM declare pour le prompt systeme
        self.outil_disponible = True

    def _charger_fenetre(self) -> list[dict[str, str]]:
        """Restaure au demarrage les derniers tours valides du fichier RLM."""
        tours: list[dict[str, str]] = []
        try:
            with self._chemin.open("rb") as f:
                f.seek(0, 2)
                position = f.tell()
                debut = max(0, position - _TAMPON * self._fenetre * 2 * 4)
                f.seek(debut)
                bloc = f.read().decode("utf-8", errors="ignore")
            lignes = bloc.splitlines()
            if debut:
                lignes = lignes[1:]
            for ligne in lignes:
                try:
                    tour = json.loads(ligne)
                except (json.JSONDecodeError, TypeError):
                    continue
                role = tour.get("role")
                contenu = tour.get("content")
                if role in ("user", "assistant") and isinstance(contenu, str):
                    tours.append({"role": role, "content": contenu})
        except OSError:
            return []
        return tours[-self._fenetre * 2:]

    # -- RLM : le fichier externe ------------------------------------------

    def _charger_etat(self) -> dict[str, str]:
        """Etat civil precedent session (JSON plat), {} si absent/corrompu."""
        try:
            brut = json.loads(self._chemin_etat.read_text(encoding="utf-8"))
            if isinstance(brut, dict):
                return {str(k): str(v) for k, v in brut.items()}
        except (OSError, ValueError):
            pass
        return {}

    def _sauver_etat(self) -> None:
        """Ecrit l'etat civil (quelques octets) — jamais bloquant."""
        try:
            self._chemin_etat.write_text(
                json.dumps(self._etat, ensure_ascii=False, sort_keys=True),
                encoding="utf-8")
        except OSError:
            pass

    def _ecrire_tour(self, role: str, contenu: str) -> None:
        ligne = json.dumps({"role": role, "content": contenu},
                           ensure_ascii=False)
        with self._chemin.open("a", encoding="utf-8") as f:
            f.write(ligne + "\n")

    def _archives(self) -> list[Path]:
        """Archives de rotation de l'historique, les plus recentes d'abord."""
        try:
            return sorted(self._chemin.parent.glob(
                f"{self._chemin.stem}-*{self._chemin.suffix}"), reverse=True)
        except OSError:
            return []

    def _rotation_si_necessaire(self) -> None:
        """Archive l'historique quand le fichier RLM depasse la taille max.

        L'historique complet est le support de la memoire longue (RLM) :
        il grandit sans limite avec l'usage. Au-dela de _TAILLE_ROTATION,
        le fichier est bascule dans une archive datee
        (.conversation-AAAA-MM-JJ-HHMMSS.jsonl) et la recherche scrute le
        fichier courant PUIS les archives les plus recentes — l'ancien
        reste accessible, le fichier actif redevient petit. Les
        _MAX_ARCHIVES archives les plus recentes sont conservees.
        """
        try:
            taille = self._chemin.stat().st_size
        except OSError:
            return
        if taille < _TAILLE_ROTATION:
            return
        horodatage = time.strftime("%Y-%m-%d-%H%M%S")
        archive = self._chemin.with_name(
            f"{self._chemin.stem}-{horodatage}{self._chemin.suffix}")
        try:
            self._chemin.replace(archive)
        except OSError:
            return
        for vieille in self._archives()[_MAX_ARCHIVES:]:
            try:
                vieille.unlink()
            except OSError:
                pass

    def rechercher(self, mot_cle: str) -> str:
        """Outil RLM : recherche arriere dans le FICHIER d'historique.

        Lecture par blocs depuis la fin (le fichier et ses archives
        peuvent etre gros sans jamais etre charges entierement) ; le
        fichier courant est scrute d'abord, puis les archives de
        rotation les plus recentes. Une ligne a cheval sur deux blocs
        est RECONSTRUITE (queue gardee en attente, completee par la
        tete du bloc precedent) — aucune ligne n'est perdue. Renvoie
        les extraits (role + contexte) ou une chaine vide.
        """
        if not mot_cle or not mot_cle.strip():
            return ""
        aiguille = _sans_accents(mot_cle.strip())
        resultats: list[str] = []
        # fichier courant d'abord (memoire la plus fraiche), puis les
        # archives de rotation, les plus recentes d'abord
        chemins: list[Path] = [self._chemin, *self._archives()]
        try:
            for chemin in chemins:
                if len(resultats) >= _MAX_RESULTATS:
                    break
                if not chemin.exists():
                    continue              # fichier absent (apres rotation)
                with chemin.open("rb") as f:
                    position = chemin.stat().st_size
                    reste = ""    # queue d'une ligne coupee, en attente
                    while position > 0 and len(resultats) < _MAX_RESULTATS:
                        debut = max(0, position - _TAMPON)
                        f.seek(debut)
                        texte = f.read(position - debut).decode(
                            "utf-8", errors="ignore")
                        if debut > 0:
                            # le bloc commence-t-il sur une frontiere de
                            # ligne ? (l'octet d'avant est-il un \n ?)
                            f.seek(debut - 1)
                            commence_ligne = f.read(1) == b"\n"
                        else:
                            commence_ligne = True
                        morceaux = (texte + reste).split("\n")
                        reste = ""
                        if not commence_ligne:
                            # la 1re ligne du bloc est coupee : sa queue
                            # est ici, sa TETE arrive au bloc precedent —
                            # on la met de cote au lieu de la perdre
                            reste = morceaux[0]
                            morceaux = morceaux[1:]
                        for ligne in reversed(morceaux):
                            if len(resultats) >= _MAX_RESULTATS:
                                break
                            ligne = ligne.strip()
                            if not ligne:
                                continue
                            try:
                                tour = json.loads(ligne)
                            except json.JSONDecodeError:
                                continue    # fragment non reconstructible
                            contenu = str(tour.get("content", ""))
                            if aiguille in _sans_accents(contenu):
                                resultats.append(
                                    f"[{tour.get('role', '?')}] "
                                    f"{contenu[:_MAX_EXTRAIT]}")
                        position = debut
        except OSError:
            return ""
        return "\n---\n".join(resultats)

    # -- fenetre glissante ---------------------------------------------------

    def ajouter(self, role: str, contenu: str) -> None:
        """Enregistre un tour : rotation eventuelle, fichier RLM, fenetre,
        extraction d'etat. La rotation passe AVANT l'ecriture : le tour
        frais demarre le nouveau fichier, l'archive garde l'ancien."""
        contenu = (contenu or "").strip()
        if not contenu:
            return
        with self._verrou:
            self._rotation_si_necessaire()
            self._ecrire_tour(role, contenu)
            self._tours.append({"role": role, "content": contenu})
            if len(self._tours) > self._fenetre * 2:   # (paire user/assistant)
                self._tours = self._tours[-self._fenetre * 2:]
            if role == "user":
                self._extraire_evenements(contenu)

    def fenetre(self) -> list[dict[str, str]]:
        """Les N derniers tours bruts (ancres en haut, fenetre en bas)."""
        with self._verrou:
            return list(self._tours)

    def remplacer_derniere_reponse(self, contenu: str) -> bool:
        """Corrige la derniere reponse de l'assistant (repli vers un autre cerveau).

        Quand une reponse de repli remplace une reponse locale faible, la
        fenetre ne doit pas garder l'ancienne : la question suivante (« et
        lui ? ») s'appuierait dessus. Si la fenetre se termine par une question
        SANS reponse (la reponse locale etait vide, et `ajouter` ignore le
        vide), la reponse est ajoutee. Le fichier RLM reste en ajout seul : une
        reponse remplacee y garde sa trace, seul le cas « ajoutee » y ecrit.
        Renvoie False s'il n'y a rien a corriger.
        """
        contenu = (contenu or "").strip()
        if not contenu:
            return False
        with self._verrou:
            if not self._tours:
                return False
            if self._tours[-1]["role"] == "assistant":
                self._tours[-1]["content"] = contenu
                return True
            if self._tours[-1]["role"] != "user":
                return False
        self.ajouter("assistant", contenu)      # hors verrou : ajouter() le prend
        return True

    # -- 3. etat pilote par evenements --------------------------------------

    # declarations sur soi : « je m'appelle X », « je m appelle X », « je suis X »
    # (la normalisation sans accents conserve l'apostrophe ; le texte oral
    # sans apostrophe doit matcher aussi)
    _RE_NOM = re.compile(
        r"\b(?:je\s+m\s*'?\s*appelle|moi\s+c\s*'?\s*est|je\s+suis)\s+"
        r"([A-Z][\w-]{1,20})", re.IGNORECASE)
    # possessions chiffrees : « j'ai achete 4 serveurs »
    _RE_POSS = re.compile(
        r"\bj\s*'?\s*ai (?:achete[sr]?|acquis|recu|ajoute[sr]?|commande)\s+"
        r"(\d+)\s+([a-z\w-]{2,20})", re.IGNORECASE)
    _RE_POSS2 = re.compile(
        r"\bj\s*'?\s*ai\s+(\d+)\s+([a-z\w-]{2,20})", re.IGNORECASE)

    def _extraire_evenements(self, texte: str) -> None:
        """Extraction symbolique (regex) de l'etat civil — aucun LLM.

        Les regex tournent sur le texte normalise (sans accents), mais
        les extraits sont repris dans le texte ORIGINAL (span identique :
        la normalisation NFKD ne change pas les longueurs) pour garder
        la casse reelle (« Pierre », pas « pierre »).
        """
        original = texte.strip()
        t = _sans_accents(original)
        avant = dict(self._etat)
        m = self._RE_NOM.search(t)
        if m and m.group(1).lower() not in (
                "un", "une", "la", "le", "content", "desole", "sure",
                "au courant"):
            self._etat["utilisateur"] = original[m.start(1):m.end(1)]
        m = self._RE_POSS.search(t) or self._RE_POSS2.search(t)
        if m:
            n, objet = m.group(1), m.group(2).lower()
            # eviter "j'ai 3 pommes et..." pris comme etat durable :
            # on garde tout, mais l'etat reste ecrasable (dernier etat gagne)
            if objet not in ("pas", "plus", "peu", "trop", "besoin"):
                self._etat[objet] = n
        if self._etat != avant:          # changement -> persistance disque
            self._sauver_etat()

    def etat_json(self) -> str:
        """L'etat civil compact (JSON une ligne), '' si vide."""
        with self._verrou:
            if not self._etat:
                return ""
            return json.dumps(self._etat, ensure_ascii=False, sort_keys=True)

    def etat(self) -> dict[str, str]:
        with self._verrou:
            return dict(self._etat)

    # -- memoire VISIBLE et EDITABLE ----------------------------------------
    # L'etat civil est injecte dans le prompt a chaque tour : l'utilisateur doit
    # pouvoir le lire ET le corriger (une regex peut se tromper : « je suis
    # fatigue » ne fait pas de lui « fatigue »). Cles et valeurs sont bornees :
    # elles partent dans le prompt du 1B.
    _MAX_CLE = 40
    _MAX_VALEUR = 120

    def definir_etat(self, cle: str, valeur: str) -> tuple[str, str]:
        """Ajoute ou corrige une entree de l'etat civil. Rend (cle, valeur).

        ValueError si la cle ou la valeur est vide apres nettoyage.
        """
        cle = re.sub(r"\s+", " ", str(cle or "")).strip()[:self._MAX_CLE]
        valeur = re.sub(r"\s+", " ", str(valeur or "")).strip()[:self._MAX_VALEUR]
        if not cle or not valeur:
            raise ValueError("une cle et une valeur non vides sont requises")
        with self._verrou:
            self._etat[cle] = valeur
            self._sauver_etat()
        return cle, valeur

    def oublier_etat(self, cle: str) -> bool:
        """Retire une entree (insensible a la casse). False si elle n'existe pas."""
        cible = str(cle or "").strip().lower()
        with self._verrou:
            for existante in list(self._etat):
                if existante.lower() == cible:
                    del self._etat[existante]
                    self._sauver_etat()
                    return True
        return False

    # -- integration ancres + fenetre ----------------------------------------

    def bloc_systeme(self, systeme: str | None) -> str | None:
        """Injecte l'etat + l'annonce de l'outil RLM dans le message systeme.

        Les ANCRAGES (consignes, dissertation TAS) restent en haut ;
        l'etat evenementiel est ajoute en bas, compact.
        """
        morceaux = []
        if systeme:
            morceaux.append(systeme)
        etat = self.etat_json()
        if etat:
            morceaux.append(f"ETAT UTILISATEUR (fiable, a jour) : {etat}")
        if self.outil_disponible:
            morceaux.append(
                "Pour retrouver ce qui a ete dit plus tot dans la "
                "conversation, demande : « cherche dans l'historique : "
                "<mot> » — l'historique complet est consultable, il n'est "
                "pas dans ta memoire immediate.")
            morceaux.append(
                "Pour lire un fichier texte de l'ordinateur, demande : "
                "« lis le fichier <chemin> » (texte, "
                f"{_MAX_FICHIER} signes max).")
        return "\n".join(morceaux) if morceaux else None

    def vider(self) -> None:
        """Nouvelle conversation : fenetre + etat remis a zero (le fichier
        RLM est preserve : la memoire longue reste accessible par outil)."""
        with self._verrou:
            self._tours.clear()
            self._etat.clear()
            self._sauver_etat()

    def compacter(self, garder: int = 0) -> int:
        """Compacte la fenetre : oublie les tours bruts, GARDE l'etat civil.

        Difference avec `vider` : le nom, les preferences et les faits
        durables (etat evenementiel) survivent — seule la matiere brute
        (questions/reponses) est jetee. L'historique fichier RLM reste
        intact, donc l'outil de recherche retrouve toujours l'ancien.

        `garder` : nombre de messages RECENTS a conserver (0 = tout jeter,
        comportement de `/compact`). La compaction automatique garde le
        dernier echange pour que « et lui ? » ait encore un referent.
        Renvoie le nombre de messages retires.
        """
        with self._verrou:
            garder = max(0, int(garder))
            if garder >= len(self._tours):
                return 0
            n = len(self._tours) - garder
            self._tours = (self._tours[-garder:] if garder else [])
            return n

    def compacter_en_gardant(self, indices: list[int]) -> int:
        """Compacte en ne gardant que CERTAINS messages (ordre chronologique).

        Le mode `importance` (aura/contexte.py) choisit les echanges a
        conserver ; contrairement a `compacter(garder=N)`, la selection n'est
        pas forcement un suffixe contigu (un vieil echange utile peut survivre
        a des echanges recents banals). Le fichier RLM reste intact.
        Renvoie le nombre de messages retires.
        """
        with self._verrou:
            propres = sorted({int(i) for i in indices
                              if 0 <= int(i) < len(self._tours)})
            if len(propres) == len(self._tours):
                return 0
            n = len(self._tours) - len(propres)
            self._tours = [self._tours[i] for i in propres]
            return n

    def tourner(self, question: str) -> str | None:
        """Applique l'outil RLM si la question demande l'historique.

        Motif : « cherche dans l'historique : X » (ou l'IA demande
        elle-meme l'acces). Renvoie le resultat de recherche a injecter,
        None sinon. C'est l'appel symbolique : zero token de prompt,
        zero LLM.
        """
        m = re.search(r"cherche dans l.historique\s*:?\s*(.+)",
                      question, re.IGNORECASE)
        if not m:
            return None
        return self.rechercher(m.group(1).strip()) or "(aucun resultat)"

    # -- outil local : lecture d'un fichier texte ---------------------------

    _RE_FICHIER = re.compile(
        r"\b(?:lis|lire|montre|ouvre)(?:\s*-\s*moi)?\s+(?:le\s+)?"
        r"fichier\s*:?\s*(.+)", re.IGNORECASE)

    def lire_fichier(self, question: str) -> str | None:
        """Outil local : « lis le fichier <chemin> » -> contenu (ou motif).

        Resolution symbolique, zero LLM, zero token injecte inutilement :
        comme pour l'historique, seul le CONTENU est rendu au modele.
        Garde-fous : fichier textuel uniquement (pas de nul octet),
        lecture bornee a _MAX_OCTETS_FICHIER puis tronque a
        _MAX_FICHIER signes — un gros binaire ne peut pas etre avale.
        Renvoie None si la question ne demande aucun fichier.
        """
        m = self._RE_FICHIER.search(question)
        if not m:
            return None
        brut = m.group(1).strip().strip("\"'«»<> ").split("\n")[0].strip()
        if not brut:
            return "(chemin manquant : lis le fichier <chemin>)"
        try:
            chemin = Path(brut).expanduser()
            if not chemin.is_absolute():
                chemin = Path.cwd() / chemin
            if not chemin.exists():
                return f"(fichier introuvable : {chemin})"
            if not chemin.is_file():
                return f"(ce n'est pas un fichier : {chemin})"
            donnees = chemin.read_bytes()[:_MAX_OCTETS_FICHIER]
            if b"\x00" in donnees[:2048]:
                return f"(fichier binaire, lecture refusee : {chemin.name})"
            texte = donnees.decode("utf-8", errors="replace")
            if not texte.strip():
                return f"(fichier vide : {chemin})"
            if len(texte) > _MAX_FICHIER:
                texte = texte[:_MAX_FICHIER] + "\n[...tronque]"
            return texte
        except OSError as e:
            return f"(lecture impossible : {e})"
