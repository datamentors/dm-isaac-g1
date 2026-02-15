# GROOT Fine-Tuning Progress Log

## Session: 2026-02-14

### Objective
Set up and run GROOT N1.6 fine-tuning with available datasets. Start with small test runs, then scale to full training.

---

## Timeline

### ✅ [COMPLETED] Full Training

**Status**: Training completed successfully (5000 steps in ~22 minutes)

#### Tasks:
- [x] SSH into workstation (192.168.1.205) ✅
- [x] Check GPU and Docker status ✅
- [x] Verify Isaac-GR00T repository ✅
- [x] Download full dataset (103 episodes + videos) ✅
- [x] Fix dependencies (FFmpeg for torchcodec) ✅
- [x] Run test fine-tuning (100 steps) ✅
- [x] Verify checkpoints save correctly ✅
- [x] Start full training ✅
- [x] Monitor and verify completion ✅
- [ ] Deploy to GROOT server

---

## Log Entries

### Entry 1: Workstation Connection Established
**Timestamp**: 2026-02-14 ~Current Time

```
✅ Connected to workstation
Host: datamentors
GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
VRAM: 97887 MiB total, 84437 MiB free (86% available)
Docker: isaac-sim container running (Up 7 days)
```

**Next Steps**:
1. Check if Isaac-GR00T is already cloned
2. Set up Python environment for fine-tuning
3. Download small test dataset

---

### Entry 2: Environment Verified
**Timestamp**: 2026-02-14 13:26

GR00T environment already set up in container:
- `grootenv` conda environment with Python 3.10.19
- Packages installed: gr00t 0.1.0, torch 2.7.1+cu128, transformers 4.51.3
- Fine-tuning scripts available at `/workspace/Isaac-GR00T/examples/`

---

### Entry 3: Dataset Download Started
**Timestamp**: 2026-02-14 13:28

```
Dataset: nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
Subset: unitree_g1.LMPnPAppleToPlateDC
Status: Downloading...
Progress: 63MB downloaded (in progress)
```

Running in tmux session `groot_download` on workstation.

---

### Entry 4: Partial Dataset Downloaded + Meta Files
**Timestamp**: 2026-02-14 14:00

```
Dataset Status:
- Episodes downloaded: 31/103 (episodes 0-30)
- Meta files: All present (info.json, episodes.jsonl, tasks.jsonl, stats.json, modality.json)
- Issue: HuggingFace rate limiting (HTTP 429) slowing downloads
```

Fine-tuning script tested and working. Need full dataset to complete training.

---

### Entry 5: All Episodes Downloaded
**Timestamp**: 2026-02-14 14:05

All 103 parquet episode files downloaded successfully.

---

### Entry 6: FFmpeg Installation Required
**Timestamp**: 2026-02-14 14:06

torchcodec failed due to missing FFmpeg. Installed FFmpeg via apt-get:
```bash
apt-get update && apt-get install -y ffmpeg
```

Both torchcodec (0.4.0) and PyAV (15.0.0) now working.

---

### Entry 7: Video Files Required
**Timestamp**: 2026-02-14 14:08

Dataset requires video files in addition to parquet data:
```
videos/chunk-000/observation.images.ego_view/episode_XXXXXX.mp4
```

Downloaded 103 video files via curl.

---

### Entry 8: Test Fine-tuning SUCCESSFUL ✅
**Timestamp**: 2026-02-14 14:13

```
Configuration:
- Steps: 100
- Batch size: 4
- Learning rate: 1e-4

Results:
- Duration: ~64 seconds
- Speed: 4.57 it/s
- GPU Memory: 49GB used
- Checkpoints saved: step 50, step 100

Checkpoint files:
- model-00001-of-00002.safetensors (4.9GB)
- model-00002-of-00002.safetensors (4.8GB)
- optimizer.pt (10.4GB)
```

Test training validated. Ready for full training.

---

### Entry 9: Initial Training Attempt (FAILED)
**Timestamp**: 2026-02-14 14:14

First training attempt failed due to missing Flash Attention 2.

---

### Entry 10: Flash Attention 2 Installation
**Timestamp**: 2026-02-14 14:17 - 15:18

```
Issue: ImportError - FlashAttention2 not installed
Solution: pip install flash-attn --no-build-isolation
Duration: ~60 minutes (compilation from source)
Result: flash-attn-2.8.3 installed successfully
```

---

### Entry 11: Training Restart (Resume Error)
**Timestamp**: 2026-02-14 15:20

Training failed trying to resume from non-existent checkpoint-1000.
Solution: Clean checkpoint directory and start fresh.

---

### Entry 12: FULL TRAINING RESTARTED 🚀
**Timestamp**: 2026-02-14 15:21

```
Configuration:
- Base model: nvidia/GR00T-N1.6-3B
- Dataset: unitree_g1.LMPnPAppleToPlateDC (103 episodes)
- Steps: 5000
- Save checkpoints: every 1000 steps
- Batch size: 8
- Learning rate: 1e-4
- Workers: 4
- Output: /workspace/checkpoints/groot_g1_full

Training Speed: ~4.3 it/s
GPU Memory: 42GB used (43% of 98GB)
Expected duration: ~19-20 minutes per 5000 steps
Expected completion: ~2026-02-14 15:41
```

Running as background process with nohup.

To monitor:
```bash
ssh datamentors@192.168.1.205
docker exec isaac-sim bash -c 'tail -50 /tmp/finetune_full.log'
```

---

### Entry 13: First Checkpoint Saved ✅
**Timestamp**: 2026-02-14 15:26

```
Checkpoint: checkpoint-1000
Files:
- model-00001-of-00002.safetensors (4.7GB)
- model-00002-of-00002.safetensors (4.5GB)
- optimizer.pt (13GB)

Training Progress:
- Current step: ~1150/5000 (23%)
- Speed: ~4.3 it/s
- GPU Memory: 42GB used
```

---

### Entry 14: TRAINING COMPLETED ✅ 🎉
**Timestamp**: 2026-02-14 15:43

```
=== TRAINING COMPLETED SUCCESSFULLY ===

Duration: ~22 minutes (15:21 - 15:43)
Final Step: 5000/5000 (100%)
Final Loss: ~0.04-0.06

Checkpoints Saved:
- checkpoint-1000 (15:26)
- checkpoint-2000 (15:30)
- checkpoint-3000 (15:34)
- checkpoint-4000 (15:39)
- checkpoint-5000 (15:43) ← FINAL

Final Checkpoint Files (checkpoint-5000):
- model-00001-of-00002.safetensors (4.7GB)
- model-00002-of-00002.safetensors (4.5GB)
- optimizer.pt (13GB)
- trainer_state.json (training history)
- Total: ~22GB

Output Directory: /workspace/checkpoints/groot_g1_full/
```

---

## Summary

| Phase | Status | Notes |
|-------|--------|-------|
| Environment Setup | ✅ Complete | grootenv with GR00T 0.1.0 |
| Dataset Download | ✅ Complete | 103 episodes + 103 videos |
| Flash Attention 2 | ✅ Complete | Compiled from source (~60 min) |
| Test Training (100 steps) | ✅ Complete | 64 seconds, checkpoints verified |
| **Full Training (5000 steps)** | ✅ Complete | 22 minutes, all checkpoints saved |

---

## Dataset Used

| Attribute | Value |
|-----------|-------|
| Repository | `nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim` |
| Subset | `unitree_g1.LMPnPAppleToPlateDC` |
| Task | Pick and Place Apple to Plate (Direct Control) |
| Episodes | 103 |
| Format | LeRobot v2 (Parquet + MP4 videos) |
| Embodiment Tag | `UNITREE_G1` |

---

## Next Steps

1. **Deploy to GROOT Server (192.168.1.237)**
   ```bash
   scp -r /workspace/checkpoints/groot_g1_full/checkpoint-5000 nvidia@192.168.1.237:/workspace/
   ```

2. **Update GROOT Server with Fine-tuned Model**
   ```bash
   # On Spark server
   python gr00t/eval/run_gr00t_server.py \
       --model-path /workspace/checkpoint-5000 \
       --embodiment-tag UNITREE_G1 \
       --port 5555
   ```

3. **Test Inference in MuJoCo**
   ```bash
   python scripts/policy_inference_mujoco.py \
       --server-host 192.168.1.237 \
       --server-port 5555
   ```

---

## Session 2: G1 Teleop Fine-tuning (Real Robot Data)

### Entry 15: G1 Teleop Dataset Downloaded
**Timestamp**: 2026-02-14 18:21

```
Dataset: nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1
Subset: g1-pick-apple (real robot data)
Episodes: 311
Format: LeRobot v2 (Parquet + MP4)
Download Method: git clone with LFS
```

### Entry 16: Custom Modality Config Created
**Timestamp**: 2026-02-14 18:24

The G1 Teleop dataset uses a different modality structure (upper body control with 43 dims) than the simulated data. Created custom config:
- State: left_leg, right_leg, waist, left_arm, left_hand, right_arm, right_hand
- Action: Same joint groups
- Video: rs_view (RealSense camera)
- Using `NEW_EMBODIMENT` tag with custom modality config

### Entry 17: G1 Teleop Training Started 🚀
**Timestamp**: 2026-02-14 18:24

```
Configuration:
- Base model: nvidia/GR00T-N1.6-3B
- Dataset: g1-pick-apple (311 episodes, REAL robot data)
- Embodiment Tag: NEW_EMBODIMENT
- Modality Config: /workspace/Isaac-GR00T/g1_teleop_config.py
- Steps: 5000
- Batch size: 8
- Learning rate: 1e-4
- Output: /workspace/checkpoints/groot_g1_teleop

Training Speed: ~5.25 it/s
Expected duration: ~16-17 minutes
Expected completion: ~2026-02-14 18:42
```

To monitor:
```bash
ssh datamentors@192.168.1.205
docker exec isaac-sim bash -c 'tail -50 /tmp/finetune_teleop.log'
```
