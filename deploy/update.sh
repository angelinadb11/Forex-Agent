#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/root/Forex-Agent}"
BRANCH="${BRANCH:-main}"

cd "$APP_DIR"
git fetch origin
git checkout "$BRANCH"
git pull origin "$BRANCH"

if [ -d .venv ]; then
  .venv/bin/pip install -r requirements.txt -q
else
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt -q
fi

sudo systemctl restart forex-agent
sudo systemctl status forex-agent --no-pager
