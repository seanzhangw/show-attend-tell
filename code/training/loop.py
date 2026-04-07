"""
Training and validation loops for show-attend-tell.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from eval.caption_metrics import run_metrics
from eval.corpus_predictions import gather_greedy_corpus

from .checkpoint import (
    load_models_from_checkpoint_path,
    save_checkpoint,
)


def compute_caption_loss(logits, captions, criterion):
    """
    Args:
        logits:   (B, T-1, V)
        captions: (B, T)

    Returns:
        ce_loss: scalar
    """
    targets = captions[:, 1:]  # (B, T-1)
    bsz, time_steps, vocab_size = logits.shape

    logits_flat = logits.reshape(bsz * time_steps, vocab_size)
    targets_flat = targets.reshape(bsz * time_steps)

    return criterion(logits_flat, targets_flat)


def train_one_epoch(
    encoder,
    decoder,
    loader,
    criterion,
    optimizer,
    device,
    lambda_att=1.0,
    grad_clip=5.0,
    print_every=100,
    freeze_encoder=True,
):
    """One training epoch. Returns average total loss."""
    if freeze_encoder:
        encoder.eval()
    else:
        encoder.train()
    decoder.train()

    running_loss = 0.0
    t0 = time.time()

    for batch_idx, (images, captions) in enumerate(loader, start=1):
        images = images.to(device)
        captions = captions.to(device)

        optimizer.zero_grad()

        if freeze_encoder:
            with torch.no_grad():
                features = encoder(images)
        else:
            features = encoder(images)

        logits, alphas = decoder(features, captions)

        ce_loss = compute_caption_loss(logits, captions, criterion)
        attention_penalty = ((1.0 - alphas.sum(dim=1)) ** 2).mean()
        loss = ce_loss + lambda_att * attention_penalty
        loss.backward()

        if grad_clip is not None:
            clip_grad_norm_(decoder.parameters(), grad_clip)
            if not freeze_encoder:
                clip_grad_norm_(encoder.parameters(), grad_clip)

        optimizer.step()

        running_loss += loss.item()

        if batch_idx % print_every == 0:
            elapsed = time.time() - t0
            it_per_sec = batch_idx / max(elapsed, 1e-8)
            sec_per_it = 1.0 / max(it_per_sec, 1e-8)
            eta_sec = (len(loader) - batch_idx) * sec_per_it
            avg_so_far = running_loss / batch_idx
            print(
                f"[Train] {batch_idx:>5}/{len(loader)} "
                f"loss={avg_so_far:.4f} (CE={ce_loss.item():.4f}, "
                f"Att={attention_penalty.item():.4f}) | "
                f"{it_per_sec:.2f} it/s | ETA {eta_sec/60:.1f} min"
            )

    return running_loss / max(1, len(loader))


@torch.no_grad()
def validate(encoder, decoder, loader, criterion, device, lambda_att=1.0):
    """Validation epoch. Returns average total loss."""
    encoder.eval()
    decoder.eval()

    running_loss = 0.0

    for images, captions in loader:
        images = images.to(device)
        captions = captions.to(device)

        features = encoder(images)
        logits, alphas = decoder(features, captions)

        ce_loss = compute_caption_loss(logits, captions, criterion)
        attention_penalty = ((1.0 - alphas.sum(dim=1)) ** 2).mean()
        loss = ce_loss + lambda_att * attention_penalty

        running_loss += loss.item()

    return running_loss / max(1, len(loader))


def _is_better(value: float, best: float, mode: str) -> bool:
    if mode == "min":
        return value < best
    if mode == "max":
        return value > best
    raise ValueError("best_mode must be 'min' or 'max'")


def _init_best(mode: str) -> float:
    return float("inf") if mode == "min" else float("-inf")


@torch.no_grad()
def evaluate_caption_metrics(
    encoder,
    decoder,
    val_image_ids,
    full_captions_map: dict,
    image_dir: str,
    transform,
    word2idx: dict,
    idx2word: dict,
    device: torch.device,
    *,
    eval_metric_names: Sequence[str] = ("bleu1", "bleu2", "bleu3", "bleu4"),
    max_decode_len: int = 20,
    max_images: Optional[int] = 500,
) -> Dict[str, float]:
    """Run caption metrics on provided models without any training step."""
    list_of_references, hypotheses = gather_greedy_corpus(
        encoder,
        decoder,
        val_image_ids,
        full_captions_map,
        image_dir,
        transform,
        word2idx,
        idx2word,
        device,
        max_len=max_decode_len,
        max_images=max_images,
    )
    return run_metrics(eval_metric_names, list_of_references, hypotheses)


def run_training_loop(
    encoder,
    decoder,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device: torch.device,
    epochs: int,
    val_image_ids,
    full_captions_map: dict,
    transform,
    word2idx: dict,
    idx2word: dict,
    image_dir: str,
    *,
    checkpoint_dir: str = "./checkpoints",
    eval_metric_names: Sequence[str] = ("bleu1", "bleu2", "bleu3", "bleu4"),
    best_by: str = "val_loss",
    best_mode: str = "min",
    resume_path: Optional[str] = None,
    max_decode_len: int = 20,
    bleu_max_images: Optional[int] = 500,
    lambda_att: float = 1.0,
    grad_clip: float = 5.0,
    print_every: int = 100,
    freeze_encoder: bool = True,
    hparams: Optional[Dict[str, Any]] = None,
    best_ckpt_filename: Optional[str] = None,
) -> Tuple[float, str, Dict[str, Any]]:
    """
    Full training loop with validation, one greedy corpus eval per epoch, and best checkpointing.

    Args:
        best_by: metric key in the per-epoch metrics dict (e.g. ``val_loss``, ``bleu4``).
        best_mode: ``min`` for loss-like metrics, ``max`` for scores like BLEU.
        resume_path: checkpoint to load for weights + optimizer + epoch continuation.
        best_ckpt_filename: filename under ``checkpoint_dir``; default
            ``best_by_{best_by}.pt``.

    Returns:
        (best_value, best_ckpt_path, last_tracking_dict)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    if best_ckpt_filename is None:
        safe = str(best_by).replace(os.sep, "_").replace("/", "_")
        best_ckpt_filename = f"best_by_{safe}.pt"
    best_ckpt_path = os.path.join(checkpoint_dir, best_ckpt_filename)

    hp = dict(hparams or {})
    start_epoch = 1
    best_value = _init_best(best_mode)
    best_epoch: Optional[int] = None

    if resume_path and os.path.isfile(resume_path):
        resume_info = load_models_from_checkpoint_path(
            resume_path,
            encoder,
            decoder,
            optimizer=optimizer,
            map_location=device,
        )
        start_epoch = int(resume_info["start_epoch"])
        track = resume_info.get("tracking", {})
        if "best_value" in track and "best_mode" in track:
            if track["best_mode"] == best_mode and track.get("best_metric_key") == best_by:
                best_value = float(track["best_value"])
                best_epoch = track.get("best_epoch")
        else:
            m = resume_info.get("metrics", {})
            if best_by in m:
                best_value = float(m[best_by])
                best_epoch = int(resume_info["checkpoint"].get("epoch", 0))
        saved_hp = resume_info.get("hparams", {})
        if saved_hp:
            mismatches = [k for k in saved_hp if k in hp and saved_hp[k] != hp[k]]
            if mismatches:
                print(
                    "[resume] Warning: hparam mismatch vs checkpoint for keys:",
                    ", ".join(mismatches),
                )

    if start_epoch > epochs:
        print(f"[fit] start_epoch {start_epoch} > epochs {epochs}; nothing to run.")
        return best_value, best_ckpt_path, {
            "best_value": best_value,
            "best_epoch": best_epoch,
            "best_metric_key": best_by,
            "best_mode": best_mode,
        }

    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss = train_one_epoch(
            encoder=encoder,
            decoder=decoder,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            lambda_att=lambda_att,
            grad_clip=grad_clip,
            print_every=print_every,
            freeze_encoder=freeze_encoder,
        )

        val_loss = validate(
            encoder=encoder,
            decoder=decoder,
            loader=val_loader,
            criterion=criterion,
            device=device,
            lambda_att=lambda_att,
        )

        eval_scores = evaluate_caption_metrics(
            encoder,
            decoder,
            val_image_ids,
            full_captions_map,
            image_dir,
            transform,
            word2idx,
            idx2word,
            device,
            eval_metric_names=eval_metric_names,
            max_decode_len=max_decode_len,
            max_images=bleu_max_images,
        )

        metrics: Dict[str, Any] = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            **eval_scores,
        }

        print(
            f"[Epoch {epoch}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            + " | ".join(f"{k}={v:.4f}" for k, v in eval_scores.items())
        )

        if best_by not in metrics:
            raise KeyError(
                f"best_by={best_by!r} not in metrics keys {sorted(metrics.keys())!r}"
            )
        cur = float(metrics[best_by])
        if _is_better(cur, best_value, best_mode):
            best_value = cur
            best_epoch = epoch
            tracking = {
                "best_value": best_value,
                "best_epoch": best_epoch,
                "best_metric_key": best_by,
                "best_mode": best_mode,
            }
            save_checkpoint(
                best_ckpt_path,
                epoch,
                metrics,
                encoder,
                decoder,
                optimizer,
                word2idx,
                idx2word,
                hp,
                tracking=tracking,
            )
            print(
                f"  -> Saved new best checkpoint: {best_ckpt_path} "
                f"({best_by}={best_value:.4f})"
            )

    print(f"\nBest {best_by}: {best_value:.4f}" + (f" (epoch {best_epoch})" if best_epoch else ""))
    print(f"Best checkpoint path: {best_ckpt_path}")
    return best_value, best_ckpt_path, {
        "best_value": best_value,
        "best_epoch": best_epoch,
        "best_metric_key": best_by,
        "best_mode": best_mode,
    }
