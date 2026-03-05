# Environments TODO

## VNC: Use service password instead of hardcoded password

**Priority**: Low (quality-of-life)
**Affects**: workstation + spark Dockerfiles, entrypoint.sh, ECS task definition

Currently VNC uses a hardcoded password (`datament`, set at build time via `vncpasswd`),
while code-server and JupyterLab use a randomly generated service password from `entrypoint.sh`.
Unify them so VNC also uses the service password.

### Implementation

1. **Remove build-time VNC password** from both Dockerfiles:
   - Delete the `RUN mkdir -p ~/.vnc && printf "datament\n..." | vncpasswd` step
   - Keep the TurboVNC install and `TVNC_WM=xfce` env var

2. **Move VNC password setup to `entrypoint.sh`** (both workstation + spark):
   - Generate service password first (already done)
   - Set VNC password: `printf "${SVC_PASSWORD:0:8}\n${SVC_PASSWORD:0:8}\nn\n" | /opt/TurboVNC/bin/vncpasswd`
     - Note: TurboVNC vncpasswd truncates to 8 chars max
   - Alternative: use TurboVNC's OTP mode (`-otp`) which generates a one-time password
     and prints it to stdout — could capture and display alongside the service password

3. **Update ECS task definition** command:
   - The `vncserver :1` startup in the task command should work as-is since the password
     file will exist by the time it runs (entrypoint runs first)

4. **Update docker-compose.yml** startup scripts if they start VNC separately

### Considerations

- TurboVNC `vncpasswd` truncates passwords to 8 characters — the full service password
  won't work as-is. Options:
  - a) Truncate to 8 chars (simple, slightly less secure)
  - b) Use TurboVNC OTP mode (`vncserver -otp`) — generates a fresh password each start
  - c) Use TurboVNC plain-text auth (`-securitytypes TLSPlain`) — supports full-length
    passwords but requires TLS setup
- Option (a) is simplest: just use first 8 chars of the service password

---

## noVNC: Browser-based VNC access

**Priority**: Medium (removes need for VNC client app)
**Affects**: workstation + spark Dockerfiles, entrypoint.sh, ECS task definition

Add [noVNC](https://novnc.com/) so users can access the XFCE desktop from a browser
(no TurboVNC Viewer or other native client needed).

### Implementation

1. **Install noVNC + websockify** in both Dockerfiles (groot stage):
   ```dockerfile
   RUN git clone --depth 1 https://github.com/novnc/noVNC.git /opt/noVNC && \
       git clone --depth 1 https://github.com/novnc/websockify.git /opt/noVNC/utils/websockify && \
       ln -s /opt/noVNC/vnc.html /opt/noVNC/index.html
   EXPOSE 6080
   ```

2. **Start noVNC in `entrypoint.sh`** (or ECS task command):
   ```bash
   /opt/noVNC/utils/novnc_proxy --vnc localhost:5901 --listen 6080 &
   ```

3. **Access**: `http://<host>:6080` — opens VNC in the browser, prompts for VNC password

4. **Port 6080** already added to ECS security group (`sg-0b92ac8b6ca93371e`)

### Considerations

- noVNC connects to the existing TurboVNC server on :5901 via websockify
- ~5MB install footprint (minimal)
- Works with the unified password from the VNC password TODO above
- Can optionally add `--web /opt/noVNC` to serve the HTML client directly
