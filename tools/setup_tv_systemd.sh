#!/usr/bin/env bash
# Setup TradingView webhook + Cloudflare tunnel as systemd (24/7).
# Run on VPS as root: bash tools/setup_tv_systemd.sh
set -euo pipefail

REPO="${REPO:-/root/Forex-Agent}"
WEBHOOK_PORT="${TRADINGVIEW_WEBHOOK_PORT:-8788}"

echo "=== TradingView 24/7 setup ==="
echo "Repo: $REPO"
echo

if [[ ! -f "$REPO/.env" ]]; then
  echo "ERROR: $REPO/.env not found. Add TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TRADINGVIEW_WEBHOOK_SECRET"
  exit 1
fi

if ! grep -q 'TRADINGVIEW_WEBHOOK_SECRET=' "$REPO/.env"; then
  echo "WARN: Add TRADINGVIEW_WEBHOOK_SECRET=... to .env"
fi

echo "[1/4] Install tradingview-webhook.service"
cp "$REPO/deploy/tradingview-webhook.service" /etc/systemd/system/tradingview-webhook.service
systemctl daemon-reload
systemctl enable tradingview-webhook
systemctl restart tradingview-webhook
systemctl --no-pager --full status tradingview-webhook || true
echo

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "Installing cloudflared..."
  curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o /tmp/cloudflared.deb
  dpkg -i /tmp/cloudflared.deb
fi

mkdir -p /etc/cloudflared

if [[ ! -f /etc/cloudflared/config.yml ]]; then
  echo "[2/4] Cloudflare named tunnel NOT configured yet."
  echo
  echo "For a STABLE TradingView URL you need:"
  echo "  - Free Cloudflare account: https://dash.cloudflare.com"
  echo "  - A domain on Cloudflare (e.g. yourdomain.com)"
  echo
  echo "Then run these commands (replace YOUR-DOMAIN):"
  echo "  cloudflared tunnel login"
  echo "  cloudflared tunnel create forex-tv"
  echo "  cloudflared tunnel route dns forex-tv tv.YOUR-DOMAIN.com"
  echo
  echo "Copy deploy/cloudflared-config.yml.example -> /etc/cloudflared/config.yml"
  echo "Edit tunnel ID, credentials path, and hostname."
  echo
  echo "Then run again: bash tools/setup_tv_systemd.sh"
  echo
  echo "TEMP fallback: quick tunnel (URL CHANGES on reboot — update TradingView each time):"
  read -r -p "Install temporary quick-tunnel service now? [y/N] " ans
  if [[ "${ans,,}" != "y" ]]; then
    echo "Webhook is 24/7. Configure Cloudflare named tunnel for stable TV URL."
    exit 0
  fi
  cat >/etc/systemd/system/cloudflared-quick-tunnel.service <<EOF
[Unit]
Description=Cloudflare quick tunnel (temporary URL for TradingView)
After=network-online.target tradingview-webhook.service

[Service]
Type=simple
ExecStart=/usr/bin/cloudflared tunnel --url http://127.0.0.1:${WEBHOOK_PORT}
Restart=always
RestartSec=15
StandardOutput=append:/var/log/cloudflared-quick.log
StandardError=append:/var/log/cloudflared-quick.log

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable cloudflared-quick-tunnel
  systemctl restart cloudflared-quick-tunnel
  sleep 8
  echo "Look for trycloudflare.com URL in:"
  grep -o 'https://[^ ]*trycloudflare.com' /var/log/cloudflared-quick.log | tail -1 || tail -20 /var/log/cloudflared-quick.log
  echo
  echo "TradingView webhook URL:"
  echo "  https://YOUR-URL-ABOVE/tv/webhook?token=YOUR_SECRET"
  exit 0
fi

echo "[2/4] Install cloudflared-tunnel.service (named tunnel)"
cp "$REPO/deploy/cloudflared-tunnel.service" /etc/systemd/system/cloudflared-tunnel.service
systemctl daemon-reload
systemctl enable cloudflared-tunnel
systemctl restart cloudflared-tunnel
systemctl --no-pager --full status cloudflared-tunnel || true
echo

echo "[3/4] Health check (local webhook)"
curl -sf "http://127.0.0.1:${WEBHOOK_PORT}/health" && echo || echo "WARN: local health failed"
echo

echo "[4/4] Done"
echo "Stable TradingView URL (use your hostname from config.yml):"
echo "  https://tv.YOUR-DOMAIN.com/tv/webhook?token=YOUR_SECRET"
echo
echo "Close all SSH sessions — services keep running."
systemctl is-active forex-agent tradingview-webhook cloudflared-tunnel 2>/dev/null || \
  systemctl is-active forex-agent tradingview-webhook cloudflared-quick-tunnel 2>/dev/null || true
