from typing import Optional, Dict, Any
import re, json

def parse_action(text: str) -> Optional[Dict[str, Any]]:
    code_blocks = re.findall(r"```json\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    candidates = []
    if code_blocks:
        candidates.extend(code_blocks)
    candidates.append(text)
    for t in reversed(candidates):
        try:
            obj = json.loads(t.strip())
            if isinstance(obj, dict) and "action" in obj:
                return obj
        except Exception:
            continue
    return None
