"""
Checkpoint utilities for saving and loading model state.
"""
from pathlib import Path
from typing import Optional, Union

import torch


def save_checkpoint(
    model,
    optimizer,
    step: int,
    cfg,
    path: Union[str, Path],
    verbose: bool = True,
):
    """Save training checkpoint.

    Args:
        model: Model to save (handles DDP wrapped models).
        optimizer: Optimizer state to save.
        step: Current training step.
        cfg: Config object (will be converted to dict if OmegaConf).
        path: Path to save checkpoint.
        verbose: Print save message.
    """
    # Handle DDP wrapped models
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    # Convert OmegaConf to dict if needed
    if hasattr(cfg, "to_container"):
        from omegaconf import OmegaConf
        config_dict = OmegaConf.to_container(cfg)
    else:
        config_dict = dict(cfg) if not isinstance(cfg, dict) else cfg

    state = {
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": config_dict,
    }

    try:
        torch.save(state, path)
        if verbose:
            print(f"Saved checkpoint to {path}")
    except Exception as e:
        print(f"Warning: Failed to save checkpoint to {path}: {e}")


def load_checkpoint(
    path: Union[str, Path],
    model,
    optimizer=None,
    device: Optional[torch.device] = None,
) -> int:
    """Load training checkpoint.

    Args:
        path: Path to checkpoint file.
        model: Model to load weights into.
        optimizer: Optional optimizer to load state into.
        device: Device to map tensors to (defaults to CPU).

    Returns:
        Training step from checkpoint (0 if not found).
    """
    map_location = device if device is not None else "cpu"
    state = torch.load(path, map_location=map_location)

    model.load_state_dict(state["model"])

    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])

    return state.get("step", 0)
