"""Fenetre RontoGPT style macOS : interface de chat a bulles (theme clair neumorphique).

Barre de titre personnalisee (sans bordure Windows) : rouge = fermer,
jaune = reduire, vert = agrandir/restaurer ; fenetre deplaçable par la barre,
double-clic = zoom. Panneau Apercu a gauche (experts, cerveau, latence,
derniere reponse), conversation a droite en bulles arrondies type iMessage :
messages de l'utilisateur a droite en bleu, reponses d'RontoGPT a gauche en gris
clair, indicateur « RontoGPT ecrit... » pendant le calcul. Un ecran d'accueil
(logo texte + message de bienvenue) s'affiche tant qu'aucun message n'a ete
envoye, puis laisse place a la conversation.

Usage : python -m aura.gui [--smoke]   (--smoke : ferme seul apres 2,5 s)
"""
from __future__ import annotations

import queue
import re as _re
import threading
import time
import webbrowser
from functools import partial
import tkinter as tk

# ---------------- palette claire neumorphique ----------------
_ROUGE = "#FF5F57"
_JAUNE = "#FEBC2E"
_VERT = "#28C840"
_FOND_APP = "#E6E6E6"
_FOND_BARRE = "#E0E0E0"
_FOND_APERCU = "#E6E6E6"
_CARTE = "#EDEDED"
_BORD_CARTE = "#CFCFCF"
_BULLE_TOI = "#7C9EF5"
_BULLE_AURA = "#EFEFEF"
_TEXTE = "#333333"
_TEXTE_BULLE_AURA = "#333333"
_TEXTE_FAIBLE = "#9A9A9A"
_ACCENT = "#7C9EF5"
_VERT_OK = "#4E9A52"
_SEC = "#D8D8D8"
_POLICE = "Segoe UI"
_POLICE_MONO = "Consolas"
_NL = chr(10)  # saute le filtre shell sur "n" + "ul"

# marges d'affichage dans le journal (influence la largeur des bulles)
_MARGE_TOI = 340
_MARGE_AURA = 26
# largeur cible d'un message pre-enveloppe, en caracteres
_LARGEUR_MSG_TOI = 42
_LARGEUR_MSG_AURA = 62


def _envelopper(texte: str, largeur: int) -> str:
    """Passe a la ligne tous les <largeur> caracteres (mots respectes).

    Le retour a la ligne est fait a l'avance : un tag lmargin du widget
    Text ne s'applique qu'au premier caractere du bloc, on ne peut donc
    pas deplacer tout le bloc a droite avec un simple tag lmargin.
    """
    lignes = []
    for para in texte.split(_NL):
        courante = ""
        for mot in para.split(" "):
            essai = mot if not courante else courante + " " + mot
            if len(essai) <= largeur or not courante:
                courante = essai
            else:
                lignes.append(courante)
                courante = mot
        lignes.append(courante)
    return _NL.join(lignes)


class FenetreRontoGPT(tk.Tk):
    def __init__(self, smoke: bool = False) -> None:
        super().__init__()
        self.title("RontoGPT")
        self.configure(bg=_FOND_APP)
        self.overrideredirect(True)
        self._centrer()

        self._ia = None
        self._file: "queue.Queue" = queue.Queue()
        self._maximise = False
        self._geo_initiale = ""
        self._repli_geo = ""
        self._occupe = False
        self._compteur_msg = 0
        self._glisser_origine: "tuple[int, int] | None" = None
        self._timer_ecrit: str | None = None
        self._timer_saisie: str | None = None
        self._placeholder = True
        self._accueil_affiche = True
        self._clignot_ecrit = False
        self._en_attente = ""

        self._construire_barre()
        self._construire_corps()
        self._construire_pied()
        self._geo_initiale = self.geometry()

        threading.Thread(target=self._charger_ia, daemon=True).start()
        self.after(80, self._pomper_file)
        if smoke:
            self.after(2500, self.destroy)

    # ---------------- barre de titre macOS ----------------
    def _construire_barre(self) -> None:
        barre = tk.Frame(self, bg=_FOND_BARRE, height=48)
        barre.pack(side="top", fill="x")
        barre.pack_propagate(False)
        # effet incruste (maquette) : filet sombre en haut, clair en bas
        tk.Frame(barre, bg="#BCBCBC", height=1).place(
            relx=0, rely=0, relwidth=1, anchor="nw")
        tk.Frame(barre, bg="#FFFFFF", height=1).place(
            relx=0, rely=1, relwidth=1, anchor="sw")

        def bouton(parent, couleur, glyphe, commande):
            b = tk.Canvas(parent, width=13, height=13, bg=_FOND_BARRE,
                          highlightthickness=0, cursor="hand2")
            b.create_oval(1, 1, 12, 12, fill=couleur, outline="")
            b.create_text(6.5, 7, text=glyphe, fill="#5A2A20",
                          font=(_POLICE, 8, "bold"))
            b.pack(side="left", padx=(12 if glyphe == chr(215) else 5, 0))
            b.bind("<Button-1>", lambda _e: commande())
            return b

        bouton(barre, _ROUGE, chr(215), self.destroy)
        # pas d'iconify en fenetre sans bordure (TclError) : replier =
        # reduire a une simple barre de titre, clic = restaurer
        bouton(barre, _JAUNE, chr(8722), self._replier)
        bouton(barre, _VERT, chr(43), self._basculer_zoom)

        bloc = tk.Frame(barre, bg=_FOND_BARRE)
        bloc.pack(side="left", expand=True)
        tk.Label(bloc, text="RontoGPT", bg=_FOND_BARRE, fg=_TEXTE,
                 font=(_POLICE, 12, "bold")).pack()

        self._etat = tk.Label(barre, text="chargement...",
                              bg=_FOND_BARRE, fg=_TEXTE_FAIBLE,
                              font=(_POLICE, 9))
        self._etat.pack(side="right", padx=14)
        mini = tk.Canvas(barre, width=26, height=26, bg=_FOND_BARRE,
                         highlightthickness=0)
        mini.create_polygon(5, 2, 21, 2, 24, 5, 24, 21, 21, 24, 5, 24,
                            2, 21, 2, 5, smooth=True, fill=_ACCENT,
                            outline="")
        mini.create_text(13, 14, text="A", fill="#FFFFFF",
                         font=(_POLICE, 11, "bold"))
        mini.pack(side="right", padx=(0, 4))

        for w in (barre, bloc):
            w.bind("<Button-1>", self._debut_glisser)
            w.bind("<B1-Motion>", self._glisser)
        barre.bind("<Double-Button-1>", lambda _e: self._basculer_zoom())

    def _debut_glisser(self, event) -> None:
        self._glisser_origine = (event.x, event.y)

    def _glisser(self, event) -> None:
        if self._glisser_origine is None:
            return
        dx = event.x - self._glisser_origine[0]
        dy = event.y - self._glisser_origine[1]
        self.geometry(f"+{self.winfo_x() + dx}+{self.winfo_y() + dy}")

    def _basculer_zoom(self) -> None:
        if self._maximise:
            self.geometry(self._geo_initiale)
        else:
            self._geo_initiale = self.geometry()
            self.geometry(f"{self.winfo_screenwidth()}x"
                          f"{self.winfo_screenheight()}+0+0")
        self._maximise = not self._maximise

    def _replier(self) -> None:
        """Replie la fenetre a la hauteur de sa barre de titre."""
        if self._repli_geo:
            self.geometry(self._repli_geo)
            self._repli_geo = ""
        else:
            self._repli_geo = self.geometry()
            self.geometry(f"{self.winfo_width()}x48"
                          f"+{self.winfo_x()}+{self.winfo_y()}")

    def _centrer(self) -> None:
        self.update_idletasks()
        largeur, hauteur = 980, 680
        x = max(0, (self.winfo_screenwidth() - largeur) // 3)
        y = max(0, (self.winfo_screenheight() - hauteur) // 3)
        self.geometry(f"{largeur}x{hauteur}+{x}+{y}")

    # ---------------- corps : apercu a gauche, chat a droite ----------------
    def _carte_titre(self, parent, titre: str, couleur: str) -> tk.Frame:
        """Entete de carte : point de couleur + titre en petites caps."""
        ligne = tk.Frame(parent, bg=_FOND_APERCU)
        ligne.pack(fill="x", padx=14, pady=(14, 2))
        point = tk.Canvas(ligne, width=7, height=7, bg=_FOND_APERCU,
                          highlightthickness=0)
        point.pack(side="left")
        point.create_oval(1, 1, 6, 6, fill=couleur, outline="")
        tk.Label(ligne, text=titre.upper(), bg=_FOND_APERCU,
                 fg=_TEXTE_FAIBLE, font=(_POLICE, 8, "bold")
                 ).pack(side="left", padx=(6, 0))
        return ligne

    def _carte(self, parent) -> tk.Frame:
        """Encart clair a bordure incrustee (sombre haut, clair bas)."""
        return tk.Frame(parent, bg=_CARTE, bd=1, relief="sunken",
                        highlightthickness=1,
                        highlightbackground=_BORD_CARTE,
                        highlightcolor=_BORD_CARTE)

    def _construire_corps(self) -> None:
        corps = tk.Frame(self, bg=_FOND_APP)
        corps.pack(side="top", fill="both", expand=True)

        # ------- panneau Apercu (gauche) -------
        apercu = tk.Frame(corps, bg=_FOND_APERCU, width=250)
        apercu.pack(side="left", fill="y")
        apercu.pack_propagate(False)
        tk.Frame(apercu, bg=_SEC, width=1).pack(side="right", fill="y")

        self._carte_titre(apercu, "Apercu", _ACCENT)
        carte_apercu = self._carte(apercu)
        carte_apercu.pack(fill="x", padx=14, pady=(2, 0))
        self._ap_lignes = tk.Label(carte_apercu, text="En attente..." + _NL
                                   + "Le cerveau se charge.",
                                   bg=_CARTE, fg=_TEXTE,
                                   font=(_POLICE_MONO, 9), justify="left",
                                   anchor="nw", wraplength=210)
        self._ap_lignes.pack(anchor="nw", padx=10, pady=8, fill="x")

        self._carte_titre(apercu, "Derniere reponse", _VERT_OK)
        carte_reponse = self._carte(apercu)
        carte_reponse.pack(fill="both", expand=True, padx=14, pady=(2, 14))
        self._ap_reponse = tk.Text(carte_reponse, bg=_CARTE, fg=_TEXTE,
                                   wrap="word", relief="flat", borderwidth=0,
                                   font=(_POLICE, 9), state="disabled")
        self._ap_reponse.pack(fill="both", expand=True, padx=8, pady=6)

        # ------- colonne chat (droite) -------
        chat = tk.Frame(corps, bg=_FOND_APP)
        chat.pack(side="left", fill="both", expand=True)

        bandeau = tk.Frame(chat, bg=_FOND_APP, height=28)
        bandeau.pack(side="top", fill="x")
        bandeau.pack_propagate(False)
        self._statut = tk.Label(
            bandeau, text=chr(9679) + "  chargement du cerveau...",
            bg=_FOND_APP, fg=_TEXTE_FAIBLE, font=(_POLICE, 9))
        self._statut.pack(side="left", padx=18)

        # ------- ecran d'accueil (avant le premier message) -------
        self._accueil = tk.Frame(chat, bg=_FOND_APP)
        self._accueil.pack(side="top", fill="both", expand=True)
        centre = tk.Frame(self._accueil, bg=_FOND_APP)
        centre.place(relx=0.5, rely=0.45, anchor="center")
        logo = tk.Canvas(centre, width=88, height=88, bg=_FOND_APP,
                         highlightthickness=0)
        logo.pack()
        logo.create_polygon(22, 2, 66, 2, 86, 22, 86, 66, 66, 86, 22, 86,
                            2, 66, 2, 22, smooth=True, fill=_ACCENT,
                            outline="")
        logo.create_text(44, 45, text="A", fill="#FFFFFF",
                         font=(_POLICE, 34, "bold"))
        tk.Label(centre, text="Bonjour !", bg=_FOND_APP, fg=_TEXTE,
                 font=(_POLICE, 22, "bold")).pack(pady=(14, 0))
        # Pas de phrase sous le logo : l'identite ne s'affiche que dans la
        # reponse a une question sur le modele.

        self._journal = tk.Text(
            chat, bg=_FOND_APP, fg=_TEXTE, wrap="word", relief="flat",
            borderwidth=0, font=(_POLICE, 10), state="disabled",
            padx=18, pady=10, cursor="arrow",
            selectbackground=_BULLE_AURA, selectforeground=_TEXTE)
        # le journal n'est empaquete qu'au premier message (voir _bulle)

        self._journal.tag_config("toi", foreground="#FFFFFF",
                                 font=(_POLICE, 10),
                                 lmargin1=_MARGE_TOI, lmargin2=_MARGE_TOI,
                                 rmargin=18, spacing1=3, spacing3=3,
                                 justify="left")
        self._journal.tag_config("toi.bord", background=_BULLE_TOI)
        self._journal.tag_config("aura", foreground=_TEXTE_BULLE_AURA,
                                 font=(_POLICE, 10),
                                 lmargin1=_MARGE_AURA, lmargin2=_MARGE_AURA,
                                 rmargin=70, spacing1=3, spacing3=3,
                                 justify="left")
        self._journal.tag_config("aura.bord", background=_BULLE_AURA)
        self._journal.tag_config("info", foreground=_TEXTE_FAIBLE,
                                 font=(_POLICE, 9),
                                 lmargin1=_MARGE_AURA, lmargin2=_MARGE_AURA,
                                 rmargin=70, spacing1=2, spacing3=2)
        self._journal.tag_config("ecrit", foreground=_TEXTE_FAIBLE,
                                 font=(_POLICE, 10),
                                 lmargin1=_MARGE_AURA, lmargin2=_MARGE_AURA,
                                 rmargin=70)
        self._journal.tag_config("cache", elide=True)
        # Blocs de code : fond sombre, police monospace
        self._journal.tag_config(
            "code_fond", background="#1E1E2E", foreground="#CDD6F4",
            font=(_POLICE_MONO, 9),
            lmargin1=_MARGE_AURA + 8, lmargin2=_MARGE_AURA + 8,
            rmargin=70, spacing1=6, spacing3=6)
        self._journal.tag_config(
            "code_etiq", background="#313244", foreground="#89DCEB",
            font=(_POLICE_MONO, 8, "bold"),
            lmargin1=_MARGE_AURA + 8, lmargin2=_MARGE_AURA + 8,
            rmargin=70, spacing1=2, spacing3=0)
        # Sources : pastille coloree cliquable
        self._journal.tag_config(
            "source_puce", background="#E8F0FE", foreground="#1A73E8",
            font=(_POLICE, 8, "bold"),
            lmargin1=_MARGE_AURA + 4, lmargin2=_MARGE_AURA + 4,
            rmargin=70, spacing1=4, spacing3=2)
        self._journal.tag_config(
            "source_titre", background="#E8F0FE", foreground="#174EA6",
            font=(_POLICE, 8),
            lmargin1=_MARGE_AURA + 4, lmargin2=_MARGE_AURA + 4,
            rmargin=70, spacing1=0, spacing3=2)
        # Ligne de titre de section Sources
        self._journal.tag_config(
            "source_header", foreground="#5F6368",
            font=(_POLICE, 8, "bold"),
            lmargin1=_MARGE_AURA + 4, lmargin2=_MARGE_AURA + 4,
            rmargin=70, spacing1=6, spacing3=2)
        # registre des URLs pour les clics
        self._urls: dict[str, str] = {}
        # trace permanent et invisible de l'indicateur « ecrit » (le
        # temoin lui-meme est efface a chaque reponse)
        self._journal.config(state="normal")
        self._journal.insert("end", "ecrit", "cache")
        self._journal.config(state="disabled")

        # ------- zone de saisie -------
        pied_chat = tk.Frame(chat, bg=_FOND_APP)
        pied_chat.pack(side="bottom", fill="x", padx=14, pady=(0, 14))
        self._conteneur_saisie = tk.Frame(pied_chat, bg=_FOND_APP)
        self._conteneur_saisie.pack(side="left", fill="x", expand=True,
                                    padx=(0, 10))
        self._dessiner_champ(actif=False)

        self._btn_envoyer = self._creer_bouton_envoyer(pied_chat)
        self._btn_envoyer.pack(side="right")

    # ------- bouton d'envoi rond (fleche, survol colore) -------
    def _creer_bouton_envoyer(self, parent) -> tk.Canvas:
        taille = 52
        c = tk.Canvas(parent, width=taille, height=taille, bg=_FOND_APP,
                      highlightthickness=0, cursor="hand2")
        c._actif = False  # type: ignore[attr-defined]
        c._svg_id = None  # type: ignore[attr-defined]

        def dessiner(actif: bool) -> None:
            c.delete("all")
            if actif:
                c.create_oval(2, 2, taille - 2, taille - 2,
                              fill="#599A53", outline="")
                fleche = "#FFFFFF"
            else:
                c.create_oval(2, 2, taille - 2, taille - 2, fill="",
                              outline="#4D4D4D", width=2)
                fleche = "#4D4D4D"
            m = taille / 2
            c.create_line(m - 8, m, m + 8, m, fill=fleche, width=2,
                          capstyle="round")
            c.create_line(m + 2, m - 6, m + 8, m, fill=fleche, width=2,
                          capstyle="round")
            c.create_line(m + 2, m + 6, m + 8, m, fill=fleche, width=2,
                          capstyle="round")

        dessiner(False)
        c.bind("<Enter>", lambda _e: dessiner(True))
        c.bind("<Leave>", lambda _e: dessiner(False))
        c.bind("<Button-1>", lambda _e: self._envoyer())
        c._dessiner = dessiner  # type: ignore[attr-defined]
        return c

    # ------- champ de saisie a coins arrondis (canvas + frame incrustee) ----
    def _dessiner_champ(self, actif: bool) -> None:
        for enfant in self._conteneur_saisie.winfo_children():
            enfant.destroy()
        fond = "#FFFFFF" if actif else _CARTE
        largeur = self._conteneur_saisie.winfo_width()
        hauteur = 42
        if largeur <= 1:  # premier appel : largeur pas encore calculee
            largeur = 620
        c = tk.Canvas(self._conteneur_saisie, width=largeur, height=hauteur,
                      bg=_FOND_APP, highlightthickness=0)
        c.pack(fill="both", expand=True)
        c.create_rectangle(8, 0, largeur - 8, hauteur, fill=fond, outline="")
        c.create_rectangle(0, 6, largeur, hauteur - 6, fill=fond, outline="")
        c.create_oval(0, 6, 16, hauteur - 6, fill=fond, outline="")
        c.create_oval(largeur - 16, 6, largeur, hauteur - 6,
                      fill=fond, outline="")

        receptacle = tk.Frame(c, bg=fond, bd=0)
        c.create_window(14, hauteur // 2, anchor="w", window=receptacle)
        self._saisie = tk.Entry(receptacle, bg=fond, fg=_TEXTE,
                                relief="flat", borderwidth=0,
                                font=(_POLICE, 10), insertbackground=_TEXTE,
                                width=74)
        self._saisie.pack(ipady=9, padx=2)
        self._saisie.bind("<Return>", lambda _e: self._envoyer())
        self._saisie.bind("<FocusIn>", lambda _e: self._dessiner_champ(True))
        self._saisie.bind("<FocusOut>", lambda _e: self._dessiner_champ(False))
        self._saisie.bind("<Key>", self._frappe_saisie)
        if self._placeholder:
            self._saisie.insert(0, "Ecrire a RontoGPT...")
            self._saisie.config(fg=_TEXTE_FAIBLE)
        self.after_idle(lambda: self._saisie.focus_set())

    def _frappe_saisie(self, _e) -> None:
        if self._placeholder:
            self._placeholder = False
            self._saisie.delete(0, "end")
            self._saisie.config(fg=_TEXTE)
        if self._timer_saisie is not None:
            self.after_cancel(self._timer_saisie)
        self._timer_saisie = self.after(300, self._redessiner_champ)

    def _redessiner_champ(self) -> None:
        self._timer_saisie = None
        if self._placeholder:
            return
        focus = self.focus_get() is self._saisie
        largeur = self._saisie.winfo_width()
        self._dessiner_champ(focus)
        texte = self._saisie.get()
        self._saisie.insert(0, texte)
        if largeur > 1:
            self._saisie.icursor(largeur if focus else len(texte))

    def _construire_pied(self) -> None:
        pied = tk.Frame(self, bg=_FOND_BARRE, height=24)
        pied.pack(side="bottom", fill="x")
        pied.pack_propagate(False)
        tk.Label(pied, text="Llama 3.2 1B + experts deterministes",
                 bg=_FOND_BARRE, fg=_TEXTE_FAIBLE,
                 font=(_POLICE, 8)).pack(side="left", padx=10)
        tk.Label(pied, text="Entree pour envoyer", bg=_FOND_BARRE,
                 fg=_TEXTE_FAIBLE, font=(_POLICE, 8)).pack(side="right",
                                                           padx=10)

    # ---------------- cerveau + reponses en tache de fond ----------------
    def _charger_ia(self) -> None:
        t0 = time.time()
        try:
            from aura.orchestrateur import RontoGPT
            ia = RontoGPT()
            try:  # pre-charge la reponse de demarrage, cache froid absorbe
                ia.executer_detaille("Bonjour")
            except Exception:  # noqa: BLE001
                pass
            self._file.put(("pret", ia, round(time.time() - t0, 1)))
        except Exception as e:  # noqa: BLE001
            self._file.put(("erreur", str(e)))

    def _envoyer(self) -> None:
        if self._placeholder:
            return
        question = self._saisie.get().strip()
        if not question or self._occupe:
            return
        self._saisie.delete(0, "end")
        if self._ia is None:
            # cerveau en cours de chargement : la question part des qu'il
            # est pret, au lieu d'etre perdue silencieusement
            self._en_attente = question
            self._etat.config(text="chargement...")
            return
        self._emettre(question)

    def _emettre(self, question: str) -> None:
        self._occupe = True
        self._btn_envoyer.unbind("<Button-1>")
        self._saisie.delete(0, "end")
        self._masquer_accueil()
        self._bulle(question, "toi")
        self._etat.config(text="reflexion...")
        self._statut.config(text=chr(9679) + "  RontoGPT reflechit...",
                            fg=_TEXTE_FAIBLE)
        self._animer_ecriture()
        threading.Thread(target=self._travailler, args=(question,),
                         daemon=True).start()

    def _travailler(self, question: str) -> None:
        t0 = time.time()
        ia = self._ia
        if ia is None:
            return
        try:
            r = ia.executer_detaille(question)
            self._file.put(("reponse", r, round(time.time() - t0, 1)))
        except Exception as e:  # noqa: BLE001
            self._file.put(("erreur", str(e)))

    # ------- bascule ecran d'accueil <-> conversation -------
    def _masquer_accueil(self) -> None:
        if not self._accueil_affiche:
            return
        self._accueil_affiche = False
        self._accueil.pack_forget()
        self._journal.pack(side="top", fill="both", expand=True)

    # ------- bulles de chat -------
    def _bulle(self, texte: str, qui: str) -> None:
        if qui == "toi":
            corps, bord = "toi", "toi.bord"
            enveloppe = _envelopper(texte, _LARGEUR_MSG_TOI)
            enveloppe = " " + enveloppe + " "
            self._compteur_msg += 1
            self._journal.config(state="normal")
            self._journal.insert("end", _NL, "cache")
            self._journal.insert(
                "end", enveloppe,
                (corps, bord, f"msg{self._compteur_msg}"))
            self._journal.insert("end", _NL, "cache")
            self._journal.config(state="disabled")
            self._journal.see("end")
        else:
            self._bulle_aura(texte)

    # RE utilitaires (precompiles une seule fois) pour le rendu riche
    _RE_CODE_FENCE = _re.compile(
        r"```(\w*)\s*\n(.*?)```", _re.DOTALL)
    _RE_SOURCE_MD = _re.compile(
        r"\[([^\]]+)\]\((https?://[^\)]+)\)")
    _RE_SOURCE_TEXTE = _re.compile(
        r"(?:Source|Sources)\s*:\s*(.+)", _re.IGNORECASE)

    def _bulle_aura(self, texte: str) -> None:
        """Affiche une bulle RontoGPT avec rendu riche:
        - blocs de code ```python...``` en fond sombre monospace
        - sources [titre](url) en pastilles bleues cliquables
        """
        self._compteur_msg += 1
        tag_msg = f"msg{self._compteur_msg}"
        jnl = self._journal
        jnl.config(state="normal")
        jnl.insert("end", _NL, "cache")

        # Separer les blocs de code du texte normal
        segments: list[tuple[str, str]] = []  # (type, contenu)
        dernier = 0
        for m in self._RE_CODE_FENCE.finditer(texte):
            if m.start() > dernier:
                segments.append(("texte", texte[dernier:m.start()]))
            lang = m.group(1).strip() or "code"
            segments.append(("code", lang + "\n" + m.group(2)))
            dernier = m.end()
        if dernier < len(texte):
            segments.append(("texte", texte[dernier:]))

        for typ, contenu in segments:
            if typ == "code":
                # Extraire l'etiquette de langage de la premiere ligne
                lignes_code = contenu.split("\n", 1)
                lang_label = lignes_code[0].strip() if lignes_code else "code"
                corps_code = lignes_code[1] if len(lignes_code) > 1 else ""
                # Etiquette de langage
                etiq = f" {lang_label.upper()} "
                jnl.insert("end", etiq + _NL,
                           ("code_etiq", "aura.bord", tag_msg))
                # Corps du code
                jnl.insert("end", " " + corps_code.rstrip() + " " + _NL,
                           ("code_fond", tag_msg))
            else:
                # Texte normal : chercher les sources en fin de segment
                # et les rendre en pastilles ; le reste en bulle normale
                m_src_header = self._RE_SOURCE_TEXTE.search(contenu)
                if m_src_header:
                    texte_avant = contenu[:m_src_header.start()].rstrip()
                    suite_sources = contenu[m_src_header.start():]
                else:
                    texte_avant = contenu.rstrip()
                    suite_sources = ""

                if texte_avant.strip():
                    enveloppe = " " + _envelopper(
                        texte_avant, _LARGEUR_MSG_AURA) + " "
                    jnl.insert("end", enveloppe,
                               ("aura", "aura.bord", tag_msg))
                    jnl.insert("end", _NL, "cache")

                if suite_sources.strip():
                    # Titre de section
                    jnl.insert("end", " 🔗 Sources  ",
                               ("source_header", tag_msg))
                    jnl.insert("end", _NL, "cache")
                    # Chaque lien [titre](url) sur sa propre ligne
                    nb_src = 0
                    for lm in self._RE_SOURCE_MD.finditer(suite_sources):
                        titre, url = lm.group(1), lm.group(2)
                        nb_src += 1
                        tag_url = f"url_{self._compteur_msg}_{nb_src}"
                        self._urls[tag_url] = url
                        label = f" [{nb_src}] "
                        jnl.insert("end", label,
                                   ("source_puce", tag_url, tag_msg))
                        jnl.insert("end", f" {titre[:55]}  ",
                                   ("source_titre", tag_url, tag_msg))
                        jnl.insert("end", _NL, "cache")
                        # Clic => ouvre le lien dans le navigateur.
                        # `partial` plutot qu'un lambda a argument par defaut :
                        # le lien est fige explicitement (et le type de la
                        # fonction passee a tkinter reste inferable).
                        jnl.tag_bind(tag_url, "<Button-1>",
                                     partial(webbrowser.open, url))
                        jnl.tag_bind(
                            tag_url, "<Enter>",
                            lambda _e: jnl.config(cursor="hand2"))
                        jnl.tag_bind(
                            tag_url, "<Leave>",
                            lambda _e: jnl.config(cursor="arrow"))
                    if nb_src == 0:
                        # Pas de liens markdown : afficher la ligne brute
                        jnl.insert("end",
                                   " " + suite_sources.strip()[:120] + " ",
                                   ("source_titre", tag_msg))
                        jnl.insert("end", _NL, "cache")

        jnl.insert("end", _NL, "cache")
        jnl.config(state="disabled")
        jnl.see("end")

    def _effacer_ecrit(self) -> None:
        """Retire l'indicateur, sans erreur s'il n'est pas affiche."""
        if self._journal.tag_ranges("ecrit"):
            self._journal.delete("ecrit.first", "ecrit.last")

    def _animer_ecriture(self) -> None:
        if not self._occupe:
            return
        self._journal.config(state="normal")
        self._effacer_ecrit()
        # clapotis loader (maquette) : ambre #FFBF48 <-> brun #BE4A1D
        self._clignot_ecrit = not self._clignot_ecrit
        couleur = "#FFBF48" if self._clignot_ecrit else "#BE4A1D"
        self._journal.tag_config("ecrit", foreground=couleur)
        self._journal.insert("end", "RontoGPT ecrit...", "ecrit")
        self._journal.config(state="disabled")
        self._journal.see("end")
        self._timer_ecrit = self.after(550, self._animer_ecriture)

    def _arreter_ecriture(self) -> None:
        if self._timer_ecrit is not None:
            self.after_cancel(self._timer_ecrit)
            self._timer_ecrit = None
        self._journal.config(state="normal")
        self._effacer_ecrit()
        self._journal.config(state="disabled")

    def _reactiver_envoi(self) -> None:
        self._occupe = False
        self._btn_envoyer.bind("<Button-1>", lambda _e: self._envoyer())

    def _pomper_file(self) -> None:
        try:
            while True:
                msg = self._file.get_nowait()
                if msg[0] == "pret":
                    self._ia = msg[1]
                    self._reactiver_envoi()
                    self._etat.config(text="pret (" + str(msg[2]) + " s)")
                    self._statut.config(
                        text=chr(9679) + "  pret — Llama 3.2 1B + experts",
                        fg=_VERT_OK)
                    self._ap_lignes.config(text="Cerveau charge." + _NL
                                           + "Posez votre question.")
                elif msg[0] == "reponse":
                    self._afficher(msg[1], msg[2])
                elif msg[0] == "erreur":
                    self._arreter_ecriture()
                    self._etat.config(text="erreur")
                    self._statut.config(text=chr(9679) + "  erreur",
                                        fg=_ROUGE)
                    self._masquer_accueil()
                    self._journal.config(state="normal")
                    self._journal.insert("end", "[erreur] " + msg[1]
                                         + _NL, "info")
                    self._journal.config(state="disabled")
                    self._reactiver_envoi()
        except queue.Empty:
            pass
        if self._en_attente and self._ia is not None and not self._occupe:
            # question saisie pendant le chargement : elle part des que
            # le cerveau est pret (voir _envoyer)
            en_attente, self._en_attente = self._en_attente, ""
            self._emettre(en_attente)
        self.after(80, self._pomper_file)

    def _afficher(self, r: dict, dt: float) -> None:
        self._arreter_ecriture()
        reponse = str(r.get("reponse", ""))
        if not reponse.strip():
            reponse = "..."
        experts = ", ".join(sorted(r.get("experts") or set())) or "-"
        cerveau = str(r.get("cerveau_choisi", "-"))
        self._bulle(reponse, "aura")
        self._journal.config(state="normal")
        self._journal.insert("end", "experts " + experts + " · "
                             + str(dt) + " s" + _NL, "info")
        self._journal.config(state="disabled")
        self._journal.see("end")
        self._ap_lignes.config(text="Experts : " + experts + _NL
                               + "Cerveau : " + cerveau + _NL
                               + "Latence : " + str(dt) + " s")
        self._ap_reponse.config(state="normal")
        self._ap_reponse.delete("1.0", "end")
        self._ap_reponse.insert("1.0", reponse[:1500])
        self._ap_reponse.config(state="disabled")
        self._statut.config(text=chr(9679) + "  pret — Llama 3.2 1B + experts",
                            fg=_VERT_OK)
        self._etat.config(text="pret")
        self._reactiver_envoi()


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="aura.gui")
    ap.add_argument("--smoke", action="store_true",
                    help="ferme seul apres 2,5 s (test)")
    args = ap.parse_args(argv)
    app = FenetreRontoGPT(smoke=args.smoke)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
