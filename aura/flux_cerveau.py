"""Bus de flux de RontoGPT : qui regarde generer le cerveau, et quoi.

SORTI DE `aura/llama_cerveau.py` (16 oct. 2026), exactement comme le
post-traitement l'avait ete : ce module ne connait NI llama.cpp, NI le
chargement du modele, NI les budgets de tokens, NI les balises de fin de tour
(celles de `stops_famille`). Il ne fait qu'une chose — repartir les morceaux de
texte d'un tour entre la PENSEE et la REPONSE, et les remettre aux abonnes qui
les ont demandes. Les seules balises qu'il connait sont celles de la reflexion
(`<think>`, `<thinking>`), et c'est justement de quoi il doit savoir parler.

CONTRAT INCHANGE. `llama_cerveau` reexporte tout ce qui suit, donc
`llama_cerveau.publier_flux(...)`, `llama_cerveau.PHASE_PENSEE`,
`llama_cerveau._abonnes_flux` et `llama_cerveau._SILENCIEUX` continuent de
marcher — tests compris, qui lisent cet etat par ce chemin.

Le seul etat porte ici est celui du bus : la table des abonnes, le verrou qui
la protege, et les deux ContextVar du tour courant. Rien d'autre n'est partage
avec le reste du cerveau.
"""
import contextvars
import threading
from collections.abc import Callable


# ── FLUX (streaming) : DEUX phases, et un abonnement PAR phase ────────
# La reponse s'ecrit a l'ecran PENDANT que le modele (ou le serveur Vulkan)
# la produit — latence percue divisee. Zero cout quand personne n'est
# abonne (test sous verrou, ~100 ns).
#
# LE POINT QUI A ETE CORRIGE ICI
# -----------------------------
# Le modele ecrit sa REFLEXION dans le meme flux de tokens que sa reponse
# (balises <think>...</think>). Jusqu'ici, `publier_flux` envoyait tout, en
# vrac, a TOUS les abonnes — donc :
#   * l'utilisateur ne pouvait pas distinguer la reflexion de la reponse
#     (elles arrivaient melangees dans le meme fil de texte) ;
#   * un abonne qui attend le RESULTAT (cache, ancrage, verificateur,
#     graphe de faits) pouvait memoriser la reflexion comme si c'en etait
#     un — un modele qui ecrit « je ne suis pas sur, peut-etre 51 » dans sa
#     pensee puis « la reponse est 47 » avait les deux dans le meme genre
#     de canal.
#
# Desormais chaque morceau porte sa PHASE, et l'abonnement DEFAUT ne prend
# que la reponse : qui veut voir penser le modele doit le DEMANDER
# (`phases=LES_DEUX_PHASES`), ce que fait la GUI. Le defaut est le choix
# sur : un module qui n'a pas exprime de besoin de pensee n'en recoit pas.

PHASE_PENSEE = "pensee"
PHASE_REPONSE = "reponse"
#: Les deux phases, dans l'ordre ou l'utilisateur les lit.
LES_DEUX_PHASES = (PHASE_PENSEE, PHASE_REPONSE)
#: Etapes de traitement (« Recherche sur le web… ») : canal SEPARE des
#: morceaux de texte, jamais melange a la reponse. Publie via publier_flux.
PHASE_ETAPE = "etape"
#: Vrai sur le fil d'une generation INTERNE (reformulation de requete) : ses
#: morceaux ne doivent jamais atteindre l'utilisateur.
_SILENCIEUX: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "silencieux", default=False)

_verrou_flux = threading.Lock()
#: phase -> liste d'abonnes. Un abonne qui prend les deux phases figure
#: dans les deux listes : `flux_actif` reste donc un test par liste.
_abonnes_flux: dict[str, list] = {PHASE_PENSEE: [], PHASE_REPONSE: []}


#: Balises ouvrantes / fermantes de reflexion, dans l'ordre de recherche.
#: <think> est celle des cerveaux a raisonnement natif (Gemma-3 Heretic
#: Thinking, MiniCPM5, R1, QwQ) ; <thinking> celle de la consigne
#: _INSTRUCTION_THINK. Les deux formes longues sont testees AVANT les
#: courtes, sinon `<thinking>` serait lu comme `<think>` suivi de « ing> ».
_BALISES = (
    ("<thinking>", "</thinking>"),
    ("<think>", "</think>"),
)
_MAX_BALISE = max(len(b) for paire in _BALISES for b in paire) - 1


def abonner_flux(fn: Callable[[str], None],
                 phases: tuple[str, ...] = (PHASE_REPONSE,),
                 ) -> Callable[[], None]:
    """Abonne `fn(morceau: str)` aux morceaux des `phases` demandees.

    PAR DEFAUT : la REPONSE seulement. C'est le contrat que reclame tout
    module qui n'a besoin que du resultat final (cache semantique, ancrage
    extractif, graphe de faits, verificateurs) : la reflexion du modele ne
    doit jamais lui parvenir, sinon elle est prise pour la reponse.
    `phases=LES_DEUX_PHASES` pour la voir passer (GUI, tableaux de bord).

    Renvoie la fonction de desabonnement (toujours a appeler en finally).
    Thread-safe : un client (API HTTP, TUI) s'abonne depuis son propre fil
    pendant que le cerveau publie depuis le thread d'inference.
    """
    with _verrou_flux:
        for phase in phases:
            _abonnes_flux.setdefault(phase, []).append(fn)

    def _desabonner() -> None:
        with _verrou_flux:
            for phase in phases:
                try:
                    _abonnes_flux.setdefault(phase, []).remove(fn)
                except ValueError:
                    pass
    return _desabonner


def abonner_flux_phase(fn: Callable[[str, str], None],
                       phases: tuple[str, ...] = LES_DEUX_PHASES,
                       ) -> Callable[[], None]:
    """Abonne `fn(morceau, phase)` aux morceaux des `phases`.

    Variante de `abonner_flux` pour les clients qui DOIVENT distinguer la
    reflexion de la reponse — l'API SSE et la GUI, c'est-a-dire ceux qui
    portent la demande produit « l'utilisateur voit la pensee ET la
    reponse ». Le contrat de `abonner_flux` (un seul argument, reponse par
    defaut) reste inchange : un module qui n'a besoin que du resultat final
    ne change pas une ligne et ne recoit toujours aucune pensee.

    Defaut corrige le 9 oct. 2026 : l'API et la GUI appelaient
    `abonner_flux` SANS `phases`. Elles s'abonnaient donc a la reponse
    seule, et la reflexion — pourtant publiee, phase par phase — n'avait
    AUCUN abonne : elle partait a la poubelle dans les deux interfaces,
    alors que le commentaire de la section FLUX ci-dessus decrivait deja ce
    comportement comme acquis (« ce que fait la GUI »).
    """
    def _morceau_pour(phase: str) -> Callable[[str], None]:
        """Fixe la phase dans une fermeture (pas de lambda a defaut : mypy
        ne sait pas typer `lambda m, _p=phase: ...`)."""
        def _recevoir(morceau: str) -> None:
            fn(morceau, phase)
        return _recevoir

    desabonnes = [abonner_flux(_morceau_pour(phase), phases=(phase,))
                  for phase in phases]

    def _desabonner() -> None:
        for une in desabonnes:
            une()
    return _desabonner


def flux_actif(phase: str = PHASE_REPONSE) -> bool:
    """Au moins un abonne A CETTE PHASE -> le serveur HTTP peut streamer."""
    with _verrou_flux:
        return bool(_abonnes_flux.get(phase))


def flux_demande() -> bool:
    """Quelqu'un regarde-t-il le flux, dans N'IMPORTE QUELLE phase ?"""
    return flux_actif(PHASE_REPONSE) or flux_actif(PHASE_PENSEE)


def publier_flux(morceau: str, phase: str = PHASE_REPONSE) -> None:
    """Diffuse un morceau de texte aux abonnes DE SA PHASE (jamais
    bloquant : un abonne foireux ne tue pas la generation en cours)."""
    if not morceau:
        return
    with _verrou_flux:
        abonnes = list(_abonnes_flux.get(phase, ()))
    for fn in abonnes:
        try:
            fn(morceau)
        except Exception:      # noqa: BLE001
            pass


class _FluxTour:
    """Repartiteur : separe le flux BRUT d'un tour en pensee et reponse.

    Le generateur ne sait pas ou il en est : il emet des morceaux, et les
    balises peuvent etre COUPEES entre deux morceaux (« <th » + « ink> »).
    Ce repartiteur garde donc les derniers caracteres en attente tant qu'ils
    peuvent etre le debut d'une balise, et ne publie le reste que lorsqu'il
    sait a quelle phase il appartient.

    Sans aucune balise, tout part en PHASE_REPONSE : le comportement est
    strictement celui d'avant.
    """

    def __init__(self, retro=None) -> None:
        self.pensee = ""
        self.reponse = ""
        self._attente = ""
        self._dans_pensee = False
        #: appele sur chaque fragment de PENSEE (Self-Correction Stream) :
        #: c'est le seul endroit qui voit la reflexion AVANT la reponse.
        self._retro = retro

    def _balise(self, texte: str):
        """(index, longueur, vers_pensee) de la 1re balise TROUVEE, sinon None."""
        trouve = None
        for ouv, fer in _BALISES:
            for balise, vers in ((ouv, True), (fer, False)):
                i = texte.find(balise)
                if i >= 0 and (trouve is None or i < trouve[0]):
                    trouve = (i, len(balise), vers)
        return trouve

    def pousser(self, morceau: str) -> None:
        """Un morceau brut du modele. Publie ce qui est PUBLIABLE."""
        if not morceau:
            return
        self._attente += morceau
        while True:
            b = self._balise(self._attente)
            if b is None:
                # rien de sur : on ne publie que ce qui ne peut plus etre
                # le debut d'une balise coupee, et on garde le reste.
                garde = self._queue_balise(self._attente)
                sur = self._attente[:len(self._attente) - garde]
                self._attente = self._attente[len(self._attente) - garde:]
                self._emettre(sur)
                return
            i, taille, vers = b
            self._emettre(self._attente[:i])
            self._attente = self._attente[i + taille:]
            self._dans_pensee = vers

    def terminer(self) -> None:
        """Fin de generation : plus rien a attendre, on vide le tampon."""
        self._emettre(self._attente)
        self._attente = ""

    def _emettre(self, texte: str) -> None:
        if not texte:
            return
        if self._dans_pensee:
            self.pensee += texte
            if self._retro is not None:
                self._retro(texte)
        else:
            self.reponse += texte
        if _SILENCIEUX.get():
            return                  # generation interne (ex. reformulation)
        publier_flux(texte, PHASE_PENSEE if self._dans_pensee
                     else PHASE_REPONSE)

    @staticmethod
    def _queue_balise(texte: str) -> int:
        """Longueur du suffixe de `texte` qui peut etre un debut de balise."""
        borne = min(len(texte), _MAX_BALISE)
        for n in range(borne, 0, -1):
            fin = texte[len(texte) - n:]
            if any(b.startswith(fin) for paire in _BALISES for b in paire):
                return n
        return 0


# ── PENSEE DU TOUR COURANT : memoire EXPLICITE, jamais dans la reponse ──
# Le flux permet de VOIR la reflexion ; il ne permet pas de la RELIRE apres
# coup (un abonne peut s'etre desabonne, ou ne pas avoir existe). Or le
# resultat d'un tour doit pouvoir porter sa reflexion — pour l'API, le
# debogage et le Self-Correction Stream. D'ou ce ContextVar : par fil, donc
# deux requetes concurrentes ne se marchent pas dessus, et explicitement
# separe de la reponse.
_REFLEXION: contextvars.ContextVar[str] = contextvars.ContextVar(
    "aura_reflexion", default="")
#: Alerte du Self-Correction Stream pour le tour en cours (cf.
#: aura/auto_correction.py) : la correction qui a ete injectee, ou "".
_ALERTE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "aura_alerte", default="")


def _retenir_reflexion(texte: str) -> None:
    """Retient la reflexion du tour en cours (jamais vide n'ecrase)."""
    if texte:
        _REFLEXION.set(texte)


def _retenir_alerte(texte: str) -> None:
    """Retient la correction exacte qui a declenche une relance."""
    if texte:
        _ALERTE.set(texte)


def reflexion_du_tour() -> str:
    """La reflexion du tour en cours, ou "" (None converti en chaine vide).

    A appeler DEPUIS le fil qui a lance la generation : c'est a ce moment
    que la valeur du ContextVar est celle du tour.
    """
    return _REFLEXION.get()


def alerte_du_tour() -> str:
    """La correction d'auto-correction du tour ("" si aucune)."""
    return _ALERTE.get()


def effacer_reflexion() -> None:
    """Remet la reflexion et l'alerte a vide (debut de tour)."""
    _REFLEXION.set("")
    _ALERTE.set("")
