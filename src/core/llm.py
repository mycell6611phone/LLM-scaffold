import httpx, asyncio, json
from typing import List, Dict, Any
from .config import Config

class OpenAICompat:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = httpx.AsyncClient(base_url=cfg.base_url, timeout=120.0, headers={"Authorization": f"Bearer {cfg.api_key}"})

    async def chat(self, messages, temperature: float = 0.3, max_tokens: int = 512, model: str | None = None) -> str:
        body = {"model": model or self.cfg.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
 
        r = await self.client.post("/chat/completions", json=body)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def extract_json_block(text: str) -> Any:
        import re, json
        candidates = re.findall(r"\{[\s\S]*\}", text)
        for c in reversed(candidates):
            try:
                return json.loads(c)
            except Exception:
                continue
        return None
