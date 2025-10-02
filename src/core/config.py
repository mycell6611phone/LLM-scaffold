from pydantic import BaseModel
import os

class Config(BaseModel):
    # shared defaults
    model: str
    max_tool_calls_per_step: int = 3
    max_step_tokens: int = 1048
    workdir: str = "./workspace"

    # default endpoint (local)
    base_url: str
    api_key: str

    # secondary endpoint (e.g. OpenAI)
    openai_base_url: str | None = None
    openai_api_key: str | None = None

    # optional per-agent overrides
    model_plan_orchestrator: str | None = None
    model_orchestrator: str | None = None
    model_executor: str | None = None
    model_theorist: str | None = None
    model_critic: str | None = None
    model_refiner: str | None = None

    @classmethod
    def from_env(cls):
        return cls(
            model=os.getenv("MODEL", "Llama 3.1 8B Instruct 128k"),
            base_url=os.getenv("BASE_URL", "http://localhost:4891/v1"),
            api_key=os.getenv("API_KEY", "not-needed"),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tool_calls_per_step=int(os.getenv("MAX_TOOL_CALLS_PER_STEP", "3")),
            max_step_tokens=int(os.getenv("MAX_STEP_TOKENS", "1048")),
            workdir=os.getenv("WORKDIR", "./workspace"),
            model_plan_orchestrator=os.getenv("MODEL_PLAN_ORCHESTRATOR"),
            model_orchestrator=os.getenv("MODEL_ORCHESTRATOR"),
            model_executor=os.getenv("MODEL_EXECUTOR"),
            model_theorist=os.getenv("MODEL_THEORIST"),
            model_critic=os.getenv("MODEL_CRITIC"),
            model_refiner=os.getenv("MODEL_REFINER"),
        )


def select_backend(model: str, cfg: Config) -> tuple[str, str]:
    """
    Decide which backend to use based on model name.
    Routes GPT/o1/o4 models to OpenAI, everything else (Llama, Mistral, Reasoner, etc.)
    to the local endpoint.
    """
    m = model.lower()

    # OpenAI models
    if m.startswith(("gpt", "o1", "o4")):
        return cfg.openai_base_url, cfg.openai_api_key

    # Explicitly catch common local model families
    if "llama" in m or "mistral" in m or "reasoner" in m:
        return cfg.base_url, cfg.api_key

    # Default fallback → local
    return cfg.base_url, cfg.api_key

