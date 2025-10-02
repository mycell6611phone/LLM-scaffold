import httpx, asyncio, json
from typing import List, Dict, Any
from .config import Config, select_backend

class OpenAICompat:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        # do not fix base_url here; we’ll select it per call
        self.client = httpx.AsyncClient(timeout=120.0)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.5,
        max_tokens: int = 512,
        model: str | None = None
    ) -> str:
        # pick model (from argument or default in env)
        model_name = model or self.cfg.model

        # decide which backend to call
        base_url, api_key = select_backend(model_name, self.cfg)

        body = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        r = await self.client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=body
        )
        r.raise_for_status()
        data = r.json()

        # print raw response text for debugging / visibility
        text = data["choices"][0]["message"]["content"]
        print(f"\n[{model_name} response]: {text}\n")
        return text

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

