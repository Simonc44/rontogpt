/**
 * Micro serveur Mock éphémère pour les endpoints demandés par smoke.ts.
 *
 * Pour l'essai avec le terminal (si besoin), utiliser directement avec Bun :
 *
 *   tui/node_modules/.bin/bun tui/src/mock-api.ts
 *
 * puis avec lancer le smoke client :
 *
 *   tui/node_modules/.bin/bun src/smoke.ts
 */
import { createServer } from 'node:http';

const HOST = '127.0.0.1';
const PORT_NUMBER = 8760;

type Event = Record<string, unknown>;

function streamSSE(res: any, events: Event[]): void {
  res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' });
  const enc = new TextEncoder();
  for (const ev of events) {
    res.write(enc.encode(`data: ${JSON.stringify(ev)}\n\n`));
  }
  res.write(enc.encode('data: [DONE]\n\n'));
  res.end();
}

const server = createServer((req, res) => {
  const host = req.headers.host ?? HOST;
  const url = new URL(req.url ?? '/', `http://${host}`);
  const path = url.pathname;
  const method = req.method;

  res.setHeader('Content-Type', 'application/json');

  try {
    if (method === 'GET' && path === '/v1/health') {
      res.writeHead(200);
      res.end(JSON.stringify({ status: 'ok', version: '0.1.0' }));
      return;
    }

    if (method === 'GET' && path === '/v1/models') {
      res.writeHead(200);
      res.end(JSON.stringify({ object: 'list', data: [{ id: 'rontogpt-local', object: 'model', owned_by: 'rontogpt' }] }));
      return;
    }

    if (method === 'GET' && path === '/v1/status') {
      res.writeHead(200);
      res.end(JSON.stringify({ version: '0.1.0', profil: 'qualite', budget_s: '180' }));
      return;
    }

    if (method === 'POST' && path === '/v1/reset') {
      res.writeHead(200);
      res.end(JSON.stringify({ ok: true, detail: 'conversation oubliee' }));
      return;
    }

    if (method === 'POST' && path === '/v1/ask/stream') {
      streamSSE(res, [
        { type: 'final', ok: true, reponse: '47 * 3 fait 141.',
          experts: [], cerveau: 'niveau0-instantane', latence: 0.1,
          en_attente: false, clarification: null },
      ]);
      return;
    }

    if (method === 'POST' && path === '/v1/ask/cloud') {  streamSSE(res, [
    { type: 'final', ok: true,
      reponse: 'Voici une reponse simulee de Gemini pour la question de test.',
      provenance: { niveau: 'cloud', badge: 'Gemini',
                    detail: 'simulate', verifie: false,
                    quitte_la_machine: false },
      experts: [], cerveau: 'gemma-cloud', latence: 1.2,
      en_attente: false, clarification: null },
  ]);
      return;
    }

    // Evite que le smoke client ne dsire un vrai Gemini.
    if (method === 'POST' && path === '/v1/ask/cloud') {
      // simulate un echec controllable, utile si le smoke veut tester Gemini.
      streamSSE(res, [
        { type: 'erreur', erreur: 'mock : pas de Gemini dans ce modele.' },
      ]);
      return;
    }

    if (method === 'POST' && path === '/v1/modeles') {
      res.writeHead(200);
      res.end(JSON.stringify({
        cerveau: { actif: true, modèle: 'RontoGPT-1B', statut: 'pret' },
        gemini: { actif: true, fournisseur: 'Google', statut: 'configure' },
      }));
      return;
    }

    res.writeHead(404);
    res.end(JSON.stringify({ error: `unknown endpoint ${method} ${path}` }));
  } catch (e) {
    res.writeHead(500);
    res.end(JSON.stringify({ error: String(e) }));
  }
});

server.listen(PORT_NUMBER, HOST, () => {
  console.log(`[mock-api] écoute sur http://${HOST}:${PORT_NUMBER}`);
});
