#!/usr/bin/env bash
# Install systemd unit for Forex Agent on VPS (default: /root/Forex-Agent).
# Run on the server as root:
#   cd ~/Forex-Agent && git pull && bash deploy/install-systemd.sh

set -euo pipefail

APP_DIR="${APP_DIR:-/root/Forex-Agent}"
UNIT_NAME="${UNIT_NAME:-forex-agent}"
SERVICE_SRC="${APP_DIR}/deploy/${UNIT_NAME}.service"
SERVICE_DST="/etc/systemd/system/${UNIT_NAME}.service"

if [ "$(id -u)" -ne 0 ]; then
  echo "Run as root (ssh root@your-vps)." >&2
  exit 1
fi

if [ ! -f "${APP_DIR}/main.py" ]; then
  echo "Not found: ${APP_DIR}/main.py" >&2
  exit 1
fi

if [ ! -x "${APP_DIR}/.venv/bin/python" ]; then
  echo "Not found: ${APP_DIR}/.venv/bin/python — create venv first." >&2
  exit 1
fi

echo "==> Stop legacy screen session (if any)"
pkill -f "SCREEN -dmS forex-bot" 2>/dev/null || true
screen -S forex-bot -X quit 2>/dev/null || true
pkill -f "${APP_DIR}/.venv/bin/python main.py" 2>/dev/null || true
sleep 1

echo "==> Disable old /opt/forex-agent service (if present)"
if systemctl is-enabled forex-agent &>/dev/null; then
  systemctl stop forex-agent 2>/dev/null || true
fi

echo "==> Install ${SERVICE_DST}"
cp "${SERVICE_SRC}" "${SERVICE_DST}"
systemctl daemon-reload
systemctl enable "${UNIT_NAME}"
systemctl restart "${UNIT_NAME}"

sleep 2
systemctl status "${UNIT_NAME}" --no-pager || true
echo
echo "Done. Useful commands:"
echo "  systemctl status ${UNIT_NAME}"
echo "  journalctl -u ${UNIT_NAME} -f"
echo "  tail -f ${APP_DIR}/logs/smc-ai-trading-agent.log"
