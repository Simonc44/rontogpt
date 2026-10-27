"""
Client Nemotron 3 Ultra via NVIDIA NIM (gratuit, rate-limited).
Chaque utilisateur fournit sa propre clé API NIM (gratuite sur build.nvidia.com).
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

try:
    from openai import OpenAI  # type: ignore[import-not-found]
    from openai import RateLimitError, APITimeoutError, APIConnectionError  # type: ignore[import-not-found]
    OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None  # type: ignore
    RateLimitError = Exception  # type: ignore
    APITimeoutError = Exception  # type: ignore
    APIConnectionError = Exception  # type: ignore
    OPENAI_AVAILABLE = False

from .config import get_user_nvidia_key, set_user_nvidia_key

LOG = logging.getLogger("aura.nemotron")


@dataclass
class NemotronConfig:
    model: str = "nvidia/nemotron-3-ultra"
    base_url: str = "https://integrate.api.nvidia.com/v1"
    timeout: float = 60.0
    max_retries: int = 3
    base_delay: float = 2.0


class NemotronClient:
    """Client Nemotron 3 Ultra via NVIDIA NIM (clé utilisateur, gratuit + rate limit)."""

    def __init__(self, user_id: str, config: Optional[NemotronConfig] = None):
        self.user_id = user_id
        self.config = config or NemotronConfig()
        
        if not OPENAI_AVAILABLE or OpenAI is None:
            raise RuntimeError(
                "Le client Nemotron nécessite le paquet 'openai'. "
                "Installez-le avec: pip install openai"
            )
        
        api_key = get_user_nvidia_key(user_id)
        if not api_key:
            raise ValueError(
                f"Clé NVIDIA NIM non configurée pour l'utilisateur {user_id}. "
                "Va dans Settings > Nemotron 3 Ultra pour l'ajouter."
            )
        
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=api_key,
            timeout=self.config.timeout,
        )

    def repondre(
        self,
        question: str,
        contexte_web: str = "",
        formule: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Envoie la question à Nemotron 3 Ultra via NIM avec retry sur rate limit."""
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if contexte_web:
            messages.append({
                "role": "system", 
                "content": f"CONTEXTE WEB (vérifié) :\n{contexte_web}\n\nUtilise ces informations pour répondre."
            })
        
        if formule:
            messages.append({
                "role": "system",
                "content": f"FORMULE EXACTE : {formule}"
            })
        
        messages.append({"role": "user", "content": question})

        # Retry avec backoff exponentiel sur rate limit
        for attempt in range(self.config.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=False,
                )
                return resp.choices[0].message.content or ""
                
            except RateLimitError as e:
                if attempt < self.config.max_retries - 1:
                    delay = 2 ** attempt  # 1s, 2s, 4s...
                    LOG.warning(f"[Nemotron] Rate limit (tentative {attempt + 1}/{self.config.max_retries}), attente {delay}s")
                    time.sleep(delay)
                    continue
                raise NemotronRateLimitError(
                    "Limite de requêtes par minute atteinte. "
                    "Réessaie dans quelques secondes ou utilise le modèle local."
                ) from e
                
            except (APITimeoutError, APIConnectionError) as e:
                if attempt < self.config.max_retries - 1:
                    delay = self.config.base_delay * (2 ** attempt)
                    LOG.warning(f"[Nemotron] Erreur réseau (tentative {attempt + 1}), retry dans {delay}s: {e}")
                    time.sleep(delay)
                    continue
                raise NemotronConnectionError(f"Impossible de joindre NVIDIA NIM: {e}") from e

        # Fallback si max_retries == 0
        return ""

    def repondre_stream(self, question: str, **kwargs):
        """Version streaming (yield tokens au fur et à mesure)."""
        messages = [{"role": "user", "content": question}]
        stream = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.7),
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# ──────────────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────────────

class NemotronError(Exception):
    """Erreur de base Nemotron."""
    pass


class NemotronRateLimitError(NemotronError):
    """Rate limit dépassé (RPM)."""
    pass


class NemotronConnectionError(NemotronError):
    """Impossible de joindre NVIDIA NIM."""
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Helpers pour l'intégration pipeline
# ──────────────────────────────────────────────────────────────────────────────

def nemotron_disponible(user_id: str) -> bool:
    """Vérifie si l'utilisateur a configuré sa clé NVIDIA."""
    try:
        return bool(get_user_nvidia_key(user_id))
    except Exception:
        return False


def creer_client_nemotron(user_id: str) -> NemotronClient:
    """Factory pour créer un client Nemotron."""
    return NemotronClient(user_id)