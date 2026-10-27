#!/usr/bin/env bash
# Démarre le mock éphémère, tourne le smoke client, puis coupe le mock.
#
# Usage (depuis la racine du projet) :
#   bash scripts/tester_client_avec_mock.sh
set -eo pipefail

echo "[mock-api] démarrage du serveur mock sur 127.0.0.1:8760"
env AURA_API_URL="http://127.0.0.1:8760" tui/node_modules/.bin/bun tui/src/mock-api.ts &
MOCK_PID=$!

# Laisser quelques millisecondes au serveur pour ouvrir le port.
sleep 1

cleanup() {
  if kill -0 "$MOCK_PID" >/dev/null 2>&1; then
    kill "$MOCK_PID" >/dev/null 2>&1 || true
  fi
  wait "$MOCK_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[mock-api] mock démarré, lancement du smoke client"
cd tui
./node_modules/.bin/bun src/smoke.ts
