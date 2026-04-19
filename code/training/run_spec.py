"""
Structured inputs for ``run_training_loop`` and caption corpus evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence


@dataclass
class CorpusEvalSpec:
    """Validation corpus, paths, and greedy-decode / metric settings."""

    val_image_ids: Any
    full_captions_map: dict
    image_dir: str
    transform: Any
    word2idx: dict
    idx2word: dict
    eval_metric_names: Sequence[str] = field(
        default_factory=lambda: ("bleu1", "bleu2", "bleu3", "bleu4")
    )
    max_decode_len: int = 20
    max_images: Optional[int] = 500


@dataclass
class TrainLoopOptions:
    """Checkpoint directory, which metric defines ``best``, resume path, and train-step knobs."""

    checkpoint_dir: str = "./checkpoints"
    best_by: str = "val_loss"
    resume_path: Optional[str] = None
    lambda_att: float = 1.0
    grad_clip: float = 5.0
    print_every: int = 100
