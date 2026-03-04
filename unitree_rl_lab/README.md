# unitree_rl_lab

This is the custom `unitree_rl_lab` we are running on the G1 robot with the real ONNX.

To run the policies on the real robot, use:

```
./g1_ctrl --network enP8p1s0
```

This currently has a custom marching policy that can be executed.

Policy commands:
- Passive: `LT + B.on_pressed`
- Normal_Walking: `R1 + X.on_pressed`
- Mimic_Dance_102: `LT(2s) + down.on_pressed`
- Mimic_Gangnam_Style: `LT(2s) + left.on_pressed`
- Military_March: `LT(2s) + right.on_pressed`

If you encounter any error when running because of CycloneDDS, execute these 3 commands:

```
unset ROS_DISTRO AMENT_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/aarch64-linux-gnu
export CYCLONEDDS_URI='<CycloneDDS><Domain><SharedMemory><Enable>false</Enable></SharedMemory></Domain></CycloneDDS>'
```
