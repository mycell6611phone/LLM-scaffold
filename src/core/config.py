from pydantic import BaseModel
import os

class Config(BaseModel):
    # shared defaults
    model: str
    base_url: str
    api_key: str
    max_tool_calls_per_step: int = 3
    max_step_tokens: int = 1048
    workdir: str = "./workspace"

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

