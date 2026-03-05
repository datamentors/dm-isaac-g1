# Environments TODO

All items below have been implemented. Keeping for reference.

---

## ~~VNC: Use service password instead of hardcoded password~~ DONE

Implemented: VNC password is now set at runtime by `entrypoint.sh` using the first 8 chars
of the randomly generated service password. Build-time hardcoded `datament` password removed
from both Dockerfiles.

---

## ~~noVNC: Browser-based VNC access~~ DONE

Implemented: noVNC + websockify installed in both Dockerfiles. Started by `entrypoint.sh`
on port 6080, proxying to TurboVNC on :5901. Port 6080 added to ECS security group.

---

## ~~Desktop Icons: Show shortcuts on XFCE desktop~~ DONE

Implemented: xfdesktop config pre-seeded with `style=2` (file icons mode), `gvfs` and
`librsvg2-common` installed. Desktop icons trusted at session startup via SHA256 checksum
(`gio set metadata::xfce-exe-checksum`).

Note: `ding@rastersoft.com` is a GNOME Shell extension — does not work with XFCE4.

---

## ~~video2robot UI: Password-protected web interface~~ DONE

Implemented: Auth wrapper (`video2robot-server.py`) adds HTTP Basic Auth using the service
password. Started by `entrypoint.sh` on port 8000. Port 8000 added to ECS security group.
