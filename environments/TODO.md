# Environments TODO

## Upgrade Workstation Isaac Sim 5.0.0 → 5.1.0

The workstation Dockerfile uses `isaacsim==5.0.0`, while the Spark image and team's TeleOp
setup both use `5.1.0`. Update the workstation builder stage:

```dockerfile
# Line ~74 in environments/workstation/Dockerfile
RUN pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

Note: May require upgrading CUDA base image or torch version. Test IsaacLab compat after.

## Add Sim2Sim C++ build dependencies

The Sim2Sim pipeline requires these C++ dev libraries (currently installed manually via
`apt install`). Add to the groot stage `apt-get install` in **both** Dockerfiles:

```
libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

Needed for building the Unitree sim2sim bridge (CycloneDDS, robot SDK bindings).

Affects: `environments/workstation/Dockerfile` + `environments/spark/Dockerfile`

---

## Completed

- ~~VNC password~~ — runtime service password (entrypoint.sh)
- ~~noVNC~~ — browser VNC on port 6080
- ~~Desktop icons~~ — xfdesktop style=2 + gvfs trust
- ~~video2robot UI~~ — HTTP Basic Auth on port 8000
