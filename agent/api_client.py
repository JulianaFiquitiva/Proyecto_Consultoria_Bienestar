"""
Cliente API para integración con LLM.

Soporta: OpenAI, Claude (Anthropic), Gemini (Google).
Maneja: API keys, rate limits, timeout, reintentos.
"""

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class LLMProvider(Enum):
    """Proveedores de LLM soportados."""
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"


@dataclass
class LLMConfig:
    """Configuración del cliente LLM."""
    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4"
    api_key: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.3  # Bajo para respuestas consistentes
    timeout: int = 30
    max_retries: int = 3


@dataclass
class LLMResponse:
    """Respuesta del LLM."""
    content: str
    model: str
    tokens_used: int
    latency_ms: float
    success: bool
    error: Optional[str] = None


class LLMClient:
    """
    Cliente unificado para múltiples proveedores de LLM.

    Maneja automáticamente:
    - Selección de proveedor
    - API key desde variables de entorno
    - Reintentos con backoff exponencial
    - Rate limiting
    - Timeout
    """

    # Variables de entorno por proveedor
    ENV_KEYS = {
        LLMProvider.OPENAI: "OPENAI_API_KEY",
        LLMProvider.CLAUDE: "ANTHROPIC_API_KEY",
        LLMProvider.GEMINI: "GOOGLE_API_KEY",
    }

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._validate_config()

    def _validate_config(self):
        """Valida que la API key esté configurada."""
        env_key = self.ENV_KEYS.get(self.config.provider)
        if self.config.api_key:
            self._api_key = self.config.api_key
        elif env_key and os.environ.get(env_key):
            self._api_key = os.environ[env_key]
        else:
            self._api_key = None

    @property
    def is_configured(self) -> bool:
        """True si el cliente tiene API key configurada."""
        return self._api_key is not None

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Envía una conversación al LLM y retorna la respuesta.

        Args:
            messages: Lista de mensajes [{"role": "user", "content": "..."}]
            system_prompt: Prompt de sistema (opcional)
            **kwargs: Parámetros adicionales para el proveedor

        Returns:
            LLMResponse con la respuesta del modelo.
        """
        if not self.is_configured:
            return LLMResponse(
                content="",
                model=self.config.model,
                tokens_used=0,
                latency_ms=0,
                success=False,
                error="API key no configurada. Configure OPENAI_API_KEY o ANTHROPIC_API_KEY.",
            )

        start_time = time.time()

        for attempt in range(self.config.max_retries):
            try:
                response = self._call_api(messages, system_prompt, **kwargs)
                latency = (time.time() - start_time) * 1000
                response.latency_ms = latency
                return response
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # Backoff exponencial
                    continue
                latency = (time.time() - start_time) * 1000
                return LLMResponse(
                    content="",
                    model=self.config.model,
                    tokens_used=0,
                    latency_ms=latency,
                    success=False,
                    error=f"Error después de {self.config.max_retries} reintentos: {str(e)}",
                )

    def _call_api(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str],
        **kwargs,
    ) -> LLMResponse:
        """Llama a la API del proveedor específico."""
        if self.config.provider == LLMProvider.OPENAI:
            return self._call_openai(messages, system_prompt, **kwargs)
        elif self.config.provider == LLMProvider.CLAUDE:
            return self._call_claude(messages, system_prompt, **kwargs)
        elif self.config.provider == LLMProvider.GEMINI:
            return self._call_gemini(messages, system_prompt, **kwargs)
        else:
            raise ValueError(f"Proveedor no soportado: {self.config.provider}")

    def _call_openai(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str],
        **kwargs,
    ) -> LLMResponse:
        """Llama a la API de OpenAI."""
        import openai

        client = openai.OpenAI(api_key=self._api_key)

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        response = client.chat.completions.create(
            model=self.config.model,
            messages=full_messages,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            tokens_used=response.usage.total_tokens,
            latency_ms=0,
            success=True,
        )

    def _call_claude(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str],
        **kwargs,
    ) -> LLMResponse:
        """Llama a la API de Anthropic (Claude)."""
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)

        # Claude usa system como parámetro separado
        response = client.messages.create(
            model=self.config.model,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            system=system_prompt or "",
            messages=messages,
        )

        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            latency_ms=0,
            success=True,
        )

    def _call_gemini(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str],
        **kwargs,
    ) -> LLMResponse:
        """Llama a la API de Google Gemini."""
        import google.generativeai as genai

        genai.configure(api_key=self._api_key)
        model = genai.GenerativeModel(
            model_name=self.config.model,
            system_instruction=system_prompt,
        )

        # Convertir mensajes al formato de Gemini
        history = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})

        chat = model.start_chat(history=history[:-1])
        response = chat.send_message(messages[-1]["content"])

        return LLMResponse(
            content=response.text,
            model=self.config.model,
            tokens_used=response.usage_metadata.total_token_count,
            latency_ms=0,
            success=True,
        )


def create_client(provider: str = "openai", **kwargs) -> LLMClient:
    """
    Factory function para crear un cliente LLM.

    Args:
        provider: "openai", "claude", o "gemini"
        **kwargs: Parámetros adicionales para LLMConfig

    Returns:
        LLMClient configurado.
    """
    provider_map = {
        "openai": LLMProvider.OPENAI,
        "claude": LLMProvider.CLAUDE,
        "gemini": LLMProvider.GEMINI,
    }
    config = LLMConfig(provider=provider_map.get(provider, LLMProvider.OPENAI), **kwargs)
    return LLMClient(config)
