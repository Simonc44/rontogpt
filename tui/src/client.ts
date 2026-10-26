/**
 * Client HTTP du cerveau RontoGPT — `aura/api.py`, streaming SSE.
 *
 * VOLONTAIREMENT sans aucune dependance a OpenTUI : ce module se teste seul
 * (`bun src/smoke.ts`), donc la partie risquee (le protocole) est verifiee
 * sans avoir besoin d'un vrai terminal.
 *
 * Contrat reel de `/v1/ask/stream` (mesure sur cette machine) :
 *   - `morceau` {texte, phase} : uniquement quand le cerveau GENERE des
 *     tokens. La plupart des questions partent aux couches deterministes
 *     (niveau0-instantane, rag-binaire, extraction-web) et arrivent donc
 *     d'un bloc, sans aucun morceau ;
 *   - `final`  {reponse, experts, cerveau, latence, en_attente, clarification} ;
 *   - `erreur` {erreur} ;
 *   - `fin`, puis la ligne sentinelle `data: [DONE]`.
 */
export const BASE = process.env.AURA_API_URL ?? "http://127.0.0.1:8760";
const CLE = process.env.AURA_API_CLE ?? "";
/** Une clé d'API est-elle exigée ? (jamais la valeur : on ne l'affiche pas) */
export const CLE_ACTIVE = CLE !== "";

function entetes(json = false): Record<string, string> {
  const h: Record<string, string> = {};
  if (json) h["Content-Type"] = "application/json";
  if (CLE) h["X-RontoGPT-Cle"] = CLE;      // meme cle que la GUI (api.py)
  return h;
}

export type Phase = string;                // "reflexion" | "reponse" | ...

export interface Morceau { type: "morceau"; texte: string; phase: Phase }
/** Comment la reponse a ete obtenue (aura/provenance.py). */
export interface Provenance {
  niveau: string;
  badge: string;
  detail: string;
  verifie: boolean;
  quitte_la_machine: boolean;
}
/** Le modele local est juge insuffisant ; RIEN n'est parti. */
export interface RepliPropose { fournisseur: string; raison: string; moment?: string }

/** Remplissage du contexte et compaction du tour (aura/contexte.py). */
export interface EtatContexte {
  sature: boolean;
  pourcentage: number;
  tokens: number;
  budget: number;
  compacte?: number;
  gardes?: number;
  apres?: number;
  raison?: string;
  mode?: string;
  /** seuil de compaction, en % (90) */
  seuil?: number;
  systeme_tokens?: number;
  budget_conversation?: number;
  conversation_tokens?: number;
}

export interface Final {
  type: "final";
  provenance?: Provenance | null;
  repli_propose?: RepliPropose | null;
  contexte?: EtatContexte | null;
  ok: boolean;
  reponse: string;
  experts: string[];
  cerveau: string;
  latence: number;
  en_attente: boolean;
  clarification: string | null;
  erreur?: string;
}
export interface ErreurFlux { type: "erreur"; erreur: string }
export interface Fin { type: "fin" }
export interface Etape { type: "etape"; libelle: string }
export type Evenement = Morceau | Etape | Final | ErreurFlux | Fin;

export interface InfoEtat {
  version: string;
  profil: string;
  prete: boolean;
  chargement: boolean;
  erreur: string | null;
  serveur_llama: boolean;
  gpu: string;
  budget_s: string;
}

async function lire<T>(chemin: string, init: RequestInit = {},
                       delai = 5000): Promise<T | null> {
  try {
    const r = await fetch(`${BASE}${chemin}`, {
      ...init,
      headers: { ...entetes(), ...(init.headers ?? {}) },
      signal: AbortSignal.timeout(delai),
    });
    if (!r.ok) return null;
    return (await r.json()) as T;
  } catch {
    return null;                                    // API absente : cas normal au demarrage
  }
}

export const sante = () => lire<{ status: string; version: string; build?: string }>("/v1/health");
export const etat = () => lire<InfoEtat>("/v1/status");

/** Etat de l'API locale (GET /v1/api) : adresse, version, cle, routes, `/v1`. */
export interface InfoApi {
  ok?: boolean;
  actif: boolean;
  serveur: string;
  version: string;
  build?: string;
  cle_requise?: boolean;
  routes?: string[];
  /** Racine a donner aux clients OpenAI (Open WebUI, editeurs). */
  openai?: string;
  erreur?: string;
}

/**
 * `/api` du TUI. Contrairement a `lire`, garde le corps meme sur une erreur
 * HTTP (503…) : l'utilisateur lit la RAISON, pas un « API injoignable »
 * trompeur. `null` = vraiment injoignable.
 */
export async function api(func: "status"): Promise<InfoApi | null> {
  const chemin = func === "status" ? "/v1/api" : `/v1/api/${func}`;
  try {
    const r = await fetch(`${BASE}${chemin}`, {
      headers: entetes(),
      signal: AbortSignal.timeout(5000),
    });
    return (await r.json()) as InfoApi;
  } catch {
    return null;
  }
}

/** Ce que le cerveau garde en tete (GET /v1/memoire). */
export interface InfoMemoire {
  ok: boolean;
  messages: number;
  echanges: number;
  etat: Record<string, string>;
  fenetre: { role: string; contenu: string }[];
  contexte?: EtatContexte | null;
}

export const memoire = () => lire<InfoMemoire>("/v1/memoire");

export async function editerMemoire(cle: string, valeur?: string): Promise<{ ok: boolean; erreur?: string }> {
  const body = valeur === undefined ? { cle, supprimer: true } : { cle, valeur };
  const r = await lire<{ ok: boolean; erreur?: string }>("/v1/memoire", {
    method: "POST", headers: entetes(true), body: JSON.stringify(body),
  });
  return r ?? { ok: false, erreur: "API unreachable" };
}

export interface InfoDocuments { ok: boolean; fichiers: { nom: string; morceaux: number; caracteres: number }[]; erreur?: string }
export const documents = () => lire<InfoDocuments>("/v1/documents");
export async function reglerLangue(langue: "auto" | "fr" | "en"): Promise<{ ok: boolean; langue?: string; erreur?: string }> {
  const r = await lire<{ ok: boolean; langue?: string; erreur?: string }>("/v1/langue", {
    method: "POST", headers: entetes(true), body: JSON.stringify({ langue }),
  });
  return r ?? { ok: false, erreur: "API unreachable" };
}
export async function editerDocuments(body: { ajouter?: string[]; supprimer?: string }): Promise<{ ok: boolean; ajoutes?: number; supprimes?: number; erreurs?: { chemin: string; erreur: string }[]; erreur?: string }> {
  const r = await lire<{ ok: boolean; ajoutes?: number; supprimes?: number; erreurs?: { chemin: string; erreur: string }[]; erreur?: string }>("/v1/documents", {
    method: "POST", headers: entetes(true), body: JSON.stringify(body),
  }, 30000);
  return r ?? { ok: false, erreur: "API unreachable" };
}

/** Configuration Gemini vue par le cerveau (GET /v1/cloud). */
export interface InfoCloud {
  ok: boolean;
  configure: boolean;
  disponible: boolean;
  actif: boolean;
  cle_presente: boolean;
  cle_source: string;
  modele: string;
  mode: string;
  redaction: boolean;
  erreur: string;
  pause: boolean;
}

export const cloud = () => lire<InfoCloud>("/v1/cloud");

/**
 * Change la configuration Gemini dans le processus du cerveau
 * (POST /v1/cloud). `cle` est ecrite dans `.env` (ignore par git).
 */
export async function configurerCloud(champs: {
  cle?: string; actif?: boolean; mode?: string; modele?: string; redaction?: boolean;
}): Promise<{ ok: boolean; etat: InfoCloud | null; detail: string }> {
  try {
    const r = await fetch(`${BASE}/v1/cloud`, {
      method: "POST",
      headers: entetes(true),
      body: JSON.stringify(champs),
      signal: AbortSignal.timeout(10000),
    });
    const j = (await r.json()) as InfoCloud & { erreur?: string };
    if (!r.ok || j.ok !== true) {
      return { ok: false, etat: null, detail: j.erreur ?? `HTTP ${r.status}` };
    }
    return { ok: true, etat: j, detail: "applied" };
  } catch (e) {
    return { ok: false, etat: null,
             detail: `API unreachable (${BASE}) : ${(e as Error).message}` };
  }
}

/** Charge le cerveau maintenant, au lieu d'attendre la premiere question. */
export async function initialiser(): Promise<{ ok: boolean; deja: boolean; detail: string }> {
  const r = await lire<{ ok: boolean; prete?: boolean; deja?: boolean; erreur?: string }>(
    "/v1/init", { method: "POST", headers: entetes(true), body: "{}" }, 180000);
  if (r === null) return { ok: false, deja: false, detail: "API unreachable (or /v1/init missing)" };
  if (!r.ok) return { ok: false, deja: false, detail: r.erreur ?? "initialization failed" };
  return { ok: true, deja: r.deja === true, detail: r.deja ? "already loaded" : "loaded" };
}

/** Compacte la fenetre brute ; l'etat durable (nom, faits) est conserve. */
export async function compacter(): Promise<{ ok: boolean; compacte: number; detail: string }> {
  const r = await lire<{ ok: boolean; compacte?: number; raison?: string; erreur?: string }>(
    "/v1/compact", { method: "POST", headers: entetes(true), body: "{}" }, 20000);
  if (r === null) return { ok: false, compacte: 0, detail: "API unreachable (or /v1/compact missing)" };
  if (!r.ok) return { ok: false, compacte: 0, detail: r.erreur ?? "compaction failed" };
  return { ok: true, compacte: r.compacte ?? 0, detail: r.raison ?? "window compacted" };
}

/** Oublie la conversation cote cerveau (equivalent du `reset` en CLI). */
export async function reinitialiser(): Promise<{ ok: boolean; detail: string }> {
  const r = await lire<{ ok: boolean; oublie?: boolean; raison?: string; erreur?: string }>(
    "/v1/reset", { method: "POST", headers: entetes(true), body: "{}" }, 20000);
  if (r === null) return { ok: false, detail: "API injoignable (ou route /v1/reset absente)" };
  if (!r.ok) return { ok: false, detail: r.erreur ?? "echec" };
  return { ok: true, detail: r.raison ?? "conversation oubliee" };
}

/**
 * Envoi EXPLICITE a Gemini (POST /v1/ask/cloud). `confirme: true` est la trace
 * que l'utilisateur a vu l'avertissement : sans lui, l'API refuse (400).
 */
export async function demanderGemini(question: string): Promise<{
  ok: boolean; reponse: string; provenance: Provenance | null;
  latence: number; erreur: string | null;
}> {
  try {
    const r = await fetch(`${BASE}/v1/ask/cloud`, {
      method: "POST",
      headers: entetes(true),
      body: JSON.stringify({ question, confirme: true }),
      signal: AbortSignal.timeout(90000),
    });
    const j = (await r.json()) as {
      ok?: boolean; reponse?: string; provenance?: Provenance | null;
      latence?: number; erreur?: string; repli_echec?: string;
    };
    return {
      ok: r.ok && j.ok === true,
      reponse: j.reponse ?? "",
      provenance: j.provenance ?? null,
      latence: j.latence ?? 0,
      erreur: j.erreur ?? j.repli_echec ?? (r.ok ? null : `HTTP ${r.status}`),
    };
  } catch (e) {
    return { ok: false, reponse: "", provenance: null, latence: 0,
             erreur: `API injoignable : ${(e as Error).message}` };
  }
}

/** Repond en flux. Rend les `morceau` au fil de l'eau puis un `final`. */
export async function* repondre(
  question: string,
  opts: { signal?: AbortSignal } = {},
): AsyncGenerator<Evenement> {
  let r: Response;
  try {
    r = await fetch(`${BASE}/v1/ask/stream`, {
      method: "POST",
      headers: entetes(true),
      body: JSON.stringify({ question }),
      signal: opts.signal,
    });
  } catch (e) {
    yield { type: "erreur", erreur: `API injoignable (${BASE}) : ${(e as Error).message}` };
    return;
  }
  if (!r.ok || r.body === null) {
    let detail = `HTTP ${r.status}`;
    try {
      const corps = (await r.json()) as { erreur?: string };
      detail = corps.erreur ?? detail;          // ex. 409 « un flux est deja en cours »
    } catch { /* corps non JSON : on garde le code */ }
    yield { type: "erreur", erreur: detail };
    return;
  }

  const lecteur = r.body.getReader();
  const decodeur = new TextDecoder();
  let tampon = "";
  try {
    while (true) {
      const { done, value } = await lecteur.read();
      if (done) break;
      tampon += decodeur.decode(value, { stream: true });
      let coupe: number;
      while ((coupe = tampon.indexOf("\n\n")) !== -1) {
        const trame = tampon.slice(0, coupe);
        tampon = tampon.slice(coupe + 2);
        for (const ligne of trame.split("\n")) {
          if (!ligne.startsWith("data:")) continue;
          const brut = ligne.slice(5).trim();
          if (brut === "") continue;
          if (brut === "[DONE]") { yield { type: "fin" }; return; }
          try {
            yield JSON.parse(brut) as Evenement;
          } catch { /* trame tronquee : on l'ignore, le final arrive */ }
        }
      }
    }
  } finally {
    lecteur.releaseLock();
  }
}

export function formaterDuree(secondes: number): string {
  if (secondes < 60) return `${secondes.toFixed(1)} s`;
  const minutes = Math.floor(secondes / 60);
  return `${minutes} min ${Math.round(secondes - minutes * 60)} s`;
}
