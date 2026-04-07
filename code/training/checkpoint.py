"""
Save / load training checkpoints with a single canonical schema.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn


def save_checkpoint(
    path: str,
    epoch: int,
    metrics_dict: Dict[str, Any],
    encoder: nn.Module,
    decoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    word2idx: dict,
    idx2word: dict,
    hparams_dict: Dict[str, Any],
    tracking: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write a checkpoint with separate ``metrics`` and ``hparams`` sections.

    Args:
        tracking: optional keys such as ``best_value``, ``best_epoch``, ``best_metric_key``
    """
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "metrics": dict(metrics_dict),
        "hparams": dict(hparams_dict),
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "word2idx": word2idx,
        "idx2word": idx2word,
    }
    if tracking:
        payload["tracking"] = dict(tracking)
    torch.save(payload, path)


def load_checkpoint(path: str, map_location=None) -> Dict[str, Any]:
    """Load checkpoint dict from ``path`` (``torch.load``)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_model_weights(encoder: nn.Module, decoder: nn.Module, ckpt: Dict[str, Any]) -> None:
    """Restore encoder/decoder weights (strict)."""
    encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
    decoder.load_state_dict(ckpt["decoder_state_dict"], strict=True)


def restore_training(
    ckpt: Dict[str, Any],
    encoder: nn.Module,
    decoder: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Load weights and optimizer state; return the epoch index to resume from (epoch + 1).
    """
    load_model_weights(encoder, decoder, ckpt)
    if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = int(ckpt.get("epoch", 0))
    return epoch + 1


def checkpoint_hparams(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize ``hparams`` from new or legacy (``config``) checkpoints."""
    if "hparams" in ckpt:
        return dict(ckpt["hparams"])
    return dict(ckpt.get("config", {}))


def checkpoint_metrics(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize ``metrics`` from new or legacy flat checkpoints."""
    if "metrics" in ckpt:
        return dict(ckpt["metrics"])
    legacy = {}
    for k in ("train_loss", "val_loss"):
        if k in ckpt:
            legacy[k] = ckpt[k]
    return legacy


def load_models_from_checkpoint_path(
    path: str,
    encoder: nn.Module,
    decoder: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location=None,
) -> Dict[str, Any]:
    """
    High-level checkpoint restore from path.

    Restores model weights always and optimizer state when provided.
    Returns normalized metadata useful for resume / reporting.
    """
    ckpt = load_checkpoint(path, map_location=map_location)

    if optimizer is None:
        load_model_weights(encoder, decoder, ckpt)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
    else:
        start_epoch = restore_training(ckpt, encoder, decoder, optimizer)

    return {
        "start_epoch": start_epoch,
        "tracking": dict(ckpt.get("tracking", {})),
        "metrics": checkpoint_metrics(ckpt),
        "hparams": checkpoint_hparams(ckpt),
        "checkpoint": ckpt,
    }
