#!/usr/bin/env bash
# =============================================================================
# Container entrypoint — generates random credentials for web services
# and starts the requested command.
#
# Services configured:
#   code-server (VS Code)  — port 8080, password auth
#   JupyterLab             — port 8888, token auth
#
# The generated password is printed to stdout so it can be retrieved with:
#   docker logs <container> 2>&1 | grep PASSWORD
# =============================================================================

# Generate random password (16 chars, alphanumeric)
SERVICE_PASSWORD="${SERVICE_PASSWORD:-$(python3 -c "import secrets; print(secrets.token_urlsafe(16))")}"

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

echo ""
echo "============================================"
echo "  SERVICE PASSWORD: ${SERVICE_PASSWORD}"
echo "============================================"
echo "  code-server (VS Code): http://<host>:8080"
echo "  JupyterLab:            http://<host>:8888"
echo "============================================"
echo ""

# Execute the original command
exec "$@"
