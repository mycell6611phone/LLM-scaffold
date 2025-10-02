from pathlib import Path
import json

class Scratchpad:
    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, obj: dict):
        line = json.dumps(obj, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def read_all(self):
        if not self.path.exists():
            return []
        out = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
        return out

    def short_context(self, limit=8):
        items = self.read_all()[-limit:]
        return items
