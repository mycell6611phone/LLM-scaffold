#!/usr/bin/env bash
set -euo pipefail

echo "== VOIDE environment check =="
ok=1

check() { echo -n "[check] $1 ... "; shift; if "$@"; then echo "ok"; else echo "FAIL"; ok=0; fi; }

check "Node >= 20.11.0" bash -lc 'v=$(node -v 2>/dev/null | sed s/^v//); test -n "$v" && python3 - <<PY
import sys
v=list(map(int,"$v".split(".")))
sys.exit(0 if (v[0]>20 or (v[0]==20 and v[1]>=11)) else 1)
PY'
check "npm present" bash -lc 'npm -v >/dev/null 2>&1'
check "pnpm present" bash -lc 'pnpm -v >/dev/null 2>&1'
check "gcc toolchain" bash -lc 'g++ --version >/dev/null 2>&1'
check "python3" bash -lc 'python3 --version >/dev/null 2>&1'
check "libsqlite3-dev" bash -lc 'dpkg -s libsqlite3-dev >/dev/null 2>&1'
check "electron runtime libs (gtk3,nss)" bash -lc 'dpkg -s libgtk-3-0 libnss3 >/dev/null 2>&1'
check "packaging tools (rpm,fakeroot,bsdtar)" bash -lc 'command -v rpm && command -v fakeroot && command -v bsdtar >/dev/null 2>&1'

if [ $ok -eq 1 ]; then
  echo "All checks passed."
  exit 0
else
  echo "Some checks failed."
  exit 1
fi
