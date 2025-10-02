# Offline preparation and install for Pop!_OS

This shows how to prepare all artifacts on an **online** machine, then move them to the **offline** dev box.

## A. On an online machine

```bash
# choose versions
export NODE_V=20.11.1
export PNPM_V=9.12.0

# 1) Download Node tarball
curl -LO https://nodejs.org/dist/v$NODE_V/node-v$NODE_V-linux-x64.tar.xz

# 2) Download pnpm package tarball (one file)
npm pack pnpm@$PNPM_V
# => creates pnpm-$PNPM_V.tgz

# 3) Prefetch project dependencies
git clone <YOUR_REPO_URL> voide
cd voide

# Corepack to use pnpm
corepack enable
corepack prepare pnpm@$PNPM_V --activate

# Fetch all deps per lockfile without linking node_modules
pnpm fetch

# Locate and archive the pnpm store (cache of tarballs)
STORE=$(pnpm store path)
tar -C "$STORE" -czf ../pnpm-store.tar.gz .

# Package the repository for transfer (optional if you use git bundle)
cd ..
tar -czf voide-repo.tar.gz voide
```

Transfer these files to the offline machine:
- `node-v$NODE_V-linux-x64.tar.xz`
- `pnpm-$PNPM_V.tgz`
- `pnpm-store.tar.gz`
- `voide-repo.tar.gz` (or use a git bundle/USB copy)
  
## B. On the offline Pop!_OS machine

```bash
# 1) Install Node locally without internet
mkdir -p $HOME/opt
tar -xJf node-v20.11.1-linux-x64.tar.xz -C $HOME/opt
echo 'export PATH=$HOME/opt/node-v20.11.1-linux-x64/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
node -v

# 2) Install pnpm from local tgz via npm (offline)
npm i -g ./pnpm-9.12.0.tgz
pnpm -v

# 3) Restore the pnpm store and configure location
mkdir -p $HOME/.local/share/pnpm-store
tar -C $HOME/.local/share/pnpm-store -xzf pnpm-store.tar.gz
pnpm config set store-dir $HOME/.local/share/pnpm-store

# 4) Unpack the repo and install offline
tar -xzf voide-repo.tar.gz
cd voide
export TURBO_TELEMETRY_DISABLED=1
export TURBO_API=
pnpm -w install --offline

# 5) Build and test
pnpm -w lint
pnpm -w build
pnpm -w test

# 6) Run dev
VOIDE_FREE=1 pnpm -w run dev
```

Notes:
- `--offline` succeeds only if all tarballs exist in the restored store.
- If some optional native dependency fails to build, ensure `build-essential`, `python3`, `pkg-config`, and `libsqlite3-dev` are installed.
