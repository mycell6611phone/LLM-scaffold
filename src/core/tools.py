from __future__ import annotations
from pathlib import Path
import subprocess, json, shlex, tempfile, os, fnmatch
from typing import Any, Dict, List
from .config import Config
from .scratchpad import Scratchpad

def safe_join(root: Path, target: Path) -> Path:
    root = root.resolve()
    target = (root / target).resolve()
    if not str(target).startswith(str(root)):
        raise ValueError("path escapes workdir")
    return target

class Toolbelt:
    def __init__(self, cfg: Config, sp: Scratchpad, run_dir: Path):
        self.cfg = cfg
        self.sp = sp
        self.run_dir = run_dir
        self.root = Path(cfg.workdir).resolve()

    def fs_list(self, pattern: str = "**/*"):
        paths = [str(p) for p in self.root.glob(pattern)]
        return {"paths": paths}

    def fs_read(self, relpath: str, max_bytes: int = 200_000):
        p = safe_join(self.root, Path(relpath))
        data = p.read_bytes()[:max_bytes]
        return {"path": str(p), "text": data.decode("utf-8", errors="ignore")}

    def fs_write(self, relpath: str, text: str):
        p = safe_join(self.root, Path(relpath))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return {"path": str(p), "bytes": len(text.encode("utf-8"))}

    def fs_glob(self, pattern: str):
        base = self.root
        matches = []
        for p in base.rglob("*"):
            if fnmatch.fnmatch(str(p.relative_to(base)), pattern):
                matches.append(str(p))
        return {"matches": matches}

    def py_exec(self, code: str, timeout: int = 15):
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            script = tdir / "snippet.py"
            script.write_text(code, encoding="utf-8")
            try:
                res = subprocess.run(
                    [os.sys.executable, str(script)],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                return {"stdout": res.stdout, "stderr": res.stderr, "returncode": res.returncode}
            except subprocess.TimeoutExpired:
                return {"stdout": "", "stderr": "Timeout", "returncode": -1}

    def sh(self, cmd: str, timeout: int = 10):
        allowed = {"git", "grep", "ls", "wc", "awk", "sed", "rg"}
        prog = shlex.split(cmd)[0]
        if prog not in allowed:
            return {"error": f"command '{prog}' not allowed"}
        try:
            res = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return {"stdout": res.stdout, "stderr": res.stderr, "returncode": res.returncode}
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": "Timeout", "returncode": -1}

    def dispatch(self, call: Dict[str, Any]):
        name = call.get("tool")
        args = call.get("args", {})
        try:
            if name == "fs_list":
                return self.fs_list(**args)
            if name == "fs_read":
                # patch: accept both "path" and "relpath"
                if "path" in args and "relpath" not in args:
                    args["relpath"] = args.pop("path")
                return self.fs_read(**args)
            if name == "fs_write":
                return self.fs_write(**args)
            if name == "fs_glob":
                return self.fs_glob(**args)
            if name == "py_exec":
                return self.py_exec(**args)
            if name == "sh":
                return self.sh(**args)
            return {"error": f"unknown tool {name}"}
        except Exception as e:
            return {"error": str(e)}

