"""Load ONNX or JIT policies for sim2sim inference."""

from pathlib import Path

import numpy as np


def load_policy(policy_path):
    """Load ONNX or JIT policy. Returns (infer_fn, obs_dim).

    infer_fn: callable(obs: np.ndarray) -> np.ndarray
    obs_dim: int, expected observation dimension
    """
    ext = Path(policy_path).suffix.lower()

    if ext == ".onnx":
        return _load_onnx(policy_path)
    elif ext == ".pt":
        return _load_jit(policy_path)
    else:
        raise ValueError(f"Unsupported policy format: {ext} (use .onnx or .pt)")


def _load_onnx(policy_path):
    import onnxruntime as ort

    session = ort.InferenceSession(
        policy_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name
    obs_dim = input_shape[-1]

    print(f"[sim2sim] ONNX policy: input={input_name} shape={input_shape}, "
          f"output={output_name}")

    def infer(obs):
        obs_np = obs.astype(np.float32).reshape(1, -1)
        result = session.run([output_name], {input_name: obs_np})
        return result[0].flatten()

    return infer, obs_dim


def _load_jit(policy_path):
    import torch

    model = torch.jit.load(policy_path, map_location="cpu")
    model.eval()

    obs_dim = None
    for name, param in model.named_parameters():
        if "weight" in name:
            obs_dim = param.shape[1]
            break
    if obs_dim is None:
        obs_dim = 47  # fallback for G1 29dof

    print(f"[sim2sim] JIT policy: obs_dim={obs_dim}")

    def infer(obs):
        with torch.inference_mode():
            obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
            action = model(obs_t)
            return action.squeeze(0).numpy()

    return infer, obs_dim
