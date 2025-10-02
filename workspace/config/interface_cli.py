from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any

@dataclass
class Event:
    type: str
    payload: Dict[str, Any]
    ts: datetime = datetime.utcnow()

class CLIInterface:
    def __init__(self, prompt="VOIDE> "):
        self.prompt = prompt
    def fetch_event(self, *, timeout_s: int = None):
        try:
            text = input(self.prompt)
        except EOFError:
            return Event("ABORT", {})
        if text.strip().lower() in ("", "exit", "quit"):
            return Event("ABORT", {})
        return Event("MESSAGE", {"text": text})
    def send_output(self, message: str):
        print(message)

