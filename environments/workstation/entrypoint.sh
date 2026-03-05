#!/usr/bin/env bash
# =============================================================================
# Container entrypoint — generates random credentials for all web services
# and starts them in the background.
#
# Services configured:
#   code-server (VS Code)  — port 8080, password auth
#   JupyterLab             — port 8888, token auth
#   VNC (TurboVNC)         — port 5901, password auth (first 8 chars of service pw)
#   noVNC (browser VNC)    — port 6080, proxies to VNC :5901
#   video2robot UI         — port 8000, HTTP Basic Auth
#
# The generated password is printed to stdout so it can be retrieved with:
#   docker logs <container> 2>&1 | grep PASSWORD
# =============================================================================

# Generate random password (22 chars, URL-safe base64)
SERVICE_PASSWORD="${SERVICE_PASSWORD:-$(python3 -c "import secrets; print(secrets.token_urlsafe(16))")}"
export SERVICE_PASSWORD

# Update code-server config
if [ -f /root/.config/code-server/config.yaml ]; then
    sed -i "s/^password:.*/password: ${SERVICE_PASSWORD}/" /root/.config/code-server/config.yaml
fi

# Update JupyterLab config
if [ -f /root/.jupyter/jupyter_server_config.json ]; then
    python3 -c "
import json, pathlib
p = pathlib.Path('/root/.jupyter/jupyter_server_config.json')
cfg = json.loads(p.read_text())
cfg['ServerApp']['token'] = '${SERVICE_PASSWORD}'
p.write_text(json.dumps(cfg, indent=2))
"
fi

# Set VNC password (truncated to 8 chars — TurboVNC limit)
VNC_PASS="${SERVICE_PASSWORD:0:8}"
mkdir -p ~/.vnc
printf "${VNC_PASS}\n${VNC_PASS}\nn\n" | /opt/TurboVNC/bin/vncpasswd >/dev/null 2>&1

echo ""
echo "============================================"
echo "  SERVICE PASSWORD: ${SERVICE_PASSWORD}"
echo "============================================"
echo "  code-server (VS Code): http://<host>:8080"
echo "  JupyterLab:            http://<host>:8888"
echo "  video2robot UI:        http://<host>:8000"
echo "  VNC:                   <host>:5901 (pw: ${VNC_PASS})"
echo "  noVNC (browser VNC):   http://<host>:6080"
echo "============================================"
echo ""

# Start services in background
nohup code-server --bind-addr 0.0.0.0:8080 &>/tmp/code-server.log &
nohup /opt/noVNC/utils/novnc_proxy --vnc localhost:5901 --listen 6080 &>/tmp/novnc.log &
nohup python3 /usr/local/bin/video2robot-server.py &>/tmp/video2robot.log &

# Trust desktop icons (must run after XFCE/dbus starts — needs gvfsd-metadata)
(sleep 5; for f in ~/Desktop/*.desktop; do
    [ -f "$f" ] || continue
    chmod +x "$f"
    gio set -t string "$f" metadata::xfce-exe-checksum \
        "$(sha256sum "$f" | awk '{print $1}')" 2>/dev/null
done; xfdesktop --reload 2>/dev/null) &

# Execute the original command
exec "$@"
