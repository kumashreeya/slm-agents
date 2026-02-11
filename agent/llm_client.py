from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class LLMClient:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2:latest"
    timeout_s: int = 120

    def chat(self, messages: list[dict[str, str]]) -> str:
        """
        Send chat messages to a local Ollama model and return the assistant text.
        """
        url = f"{self.base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        r = requests.post(url, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()

        # Ollama returns: {"message": {"role": "assistant", "content": "..."} ...}
        msg = data.get("message", {})
        return (msg.get("content") or "").strip()
