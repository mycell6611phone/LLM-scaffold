import os, openai
class OpenAIEngine:
    def __init__(self, model="gpt-3.5-turbo"):
        key = os.getenv("OPENAI_API_KEY")
        if not key: raise RuntimeError("OPENAI_API_KEY missing")
        openai.api_key = key
        self.model = model
    def complete(self, system_msg: str, user_msg: str, *, max_tokens=512, temperature=0.2):
        msgs = ([{"role":"system","content":system_msg}] if system_msg else []) + \
               [{"role":"user","content":user_msg}]
        r = openai.ChatCompletion.create(model=self.model, messages=msgs,
                                         max_tokens=max_tokens, temperature=temperature)
        return r["choices"][0]["message"]["content"].strip()

