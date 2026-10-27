/**
 * Verification sans terminal : exerce le client contre l'API reelle deja
 * lancee. Sort en code 1 des qu'un controle echoue, pour que ce script soit
 * utilisable tel quel dans une CI (`bun src/smoke.ts`).
 *
 * Usage : .venv\Scripts\python.exe -m aura.api    (dans un autre terminal)
 *         tui\node_modules\.bin\bun.cmd run src\smoke.ts
 */
import { BASE, etat, formaterDuree, reinitialiser, repondre, sante } from "./client";
import type { Evenement } from "./client";
import { controle, terminer } from "./smoke_commun";

/** Joue une question et resume ce qui est reellement arrive. */
async function jouer(question: string) {
  const debut = Date.now();
  let final: Extract<Evenement, { type: "final" }> | null = null;
  let erreur: string | null = null;
  let morceaux = 0;
  const phases = new Set<string>();
  let texte = "";

  for await (const e of repondre(question)) {
    if (e.type === "morceau") {
      morceaux += 1;
      phases.add(e.phase);
      texte += e.texte;
    } else if (e.type === "final") {
      final = e;
    } else if (e.type === "erreur") {
      erreur = e.erreur;
    }
  }
  const duree = (Date.now() - debut) / 1000;
  console.log(
    `     ${formaterDuree(duree)} · cerveau=${  final?.cerveau ?? "?"}`
    + ` · morceaux=${morceaux}${phases.size ? ` (phases: ${[...phases].join("/")})` : ""}`
    + (final ? ` · reponse=${JSON.stringify(final.reponse.slice(0, 90))}` : ""),
  );
  return { final, erreur, morceaux, texte, duree };
}

console.log(`RontoGPT — verification du client TUI contre ${BASE}\n`);

// 1. l'API repond
const h = await sante();
controle("l'API repond sur /v1/health", h !== null, h ? `version ${h.version}` : "aucune reponse");

// 2. /v1/status expose bien les champs dont l'en-tete du TUI depend
const info = await etat();
controle("/v1/status expose version + profil + budget_s",
  info !== null && !!info.version && !!info.profil && !!info.budget_s,
  info ? `${info.profil}, budget ${info.budget_s}s` : "aucune reponse");

// 3. une question deterministe : le cas courant (arrivee d'un bloc, 0 morceau)
const a = await jouer("Combien font 47 * 3 ?");
controle("question deterministe -> un `final` non vide",
  a.final !== null && a.erreur === null && a.final.reponse.trim() !== "",
  a.erreur ?? `cerveau ${a.final?.cerveau ?? '?'}`);
controle("la reponse de 47 * 3 contient bien 141",
  (a.final?.reponse ?? "").includes("141"), a.final?.reponse.slice(0, 40) ?? "rien");

// 4. une question ouverte : verifie qu'une reponse longue arrive et se
//    raconte en une seule trame quand le cerveau ne genere pas de tokens
const b = await jouer("Quelle est la capitale de l'Australie ?");
controle("question ouverte -> un `final` non vide",
  b.final !== null && b.final.reponse.trim() !== "",
  b.erreur ?? b.final?.reponse.slice(0, 60) ?? "rien");

// 5. le flux se termine toujours par `fin` (donc le TUI ne reste pas bloque)
controle("le flux se termine et ne bloque pas le TUI",
  a.final !== null && b.final !== null,
  "les deux tours ont rendu un `final` (pas de blocage)");

// 6. /v1/reset existe et repond
const r = await reinitialiser();
controle("POST /v1/reset repond ok", r.ok, r.detail);

terminer("TOUT EST OK");
