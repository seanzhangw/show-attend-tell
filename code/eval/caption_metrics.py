"""
Pluggable captioning metrics on precomputed (references, hypotheses) corpora.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Sequence

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

logger = logging.getLogger(__name__)

# (list_of_references, hypotheses) -> float
MetricFn = Callable[[List, List], float]


def corpus_bleu_n(list_of_references, hypotheses, n: int) -> float:
    """Corpus BLEU with cumulative n-gram weights up to order n (1 <= n <= 4)."""
    if n < 1 or n > 4:
        raise ValueError(f"n must be in [1, 4], got {n}")

    # Cumulative BLEU weights:
    # n=1 -> (1,0,0,0)
    # n=2 -> (0.5,0.5,0,0)
    # n=3 -> (1/3,1/3,1/3,0)
    # n=4 -> (0.25,0.25,0.25,0.25)
    weights = tuple((1.0 / n) if i < n else 0.0 for i in range(4))

    smooth = SmoothingFunction().method1
    return float(
        corpus_bleu(
            list_of_references,
            hypotheses,
            weights=weights,
            smoothing_function=smooth,
        )
    )

def _bleu1(refs, hyps) -> float:
    return corpus_bleu_n(refs, hyps, 1)


def _bleu2(refs, hyps) -> float:
    return corpus_bleu_n(refs, hyps, 2)


def _bleu3(refs, hyps) -> float:
    return corpus_bleu_n(refs, hyps, 3)


def _bleu4(refs, hyps) -> float:
    return corpus_bleu_n(refs, hyps, 4)


def _meteor_corpus(list_of_references, hypotheses) -> float:
    """
    Per-image: max METEOR over references; then mean over images.

    Requires ``nltk`` with WordNet data::
        import nltk
        nltk.download("wordnet")
        nltk.download("omw-1.4")
    """
    try:
        from nltk.translate.meteor_score import meteor_score
    except ImportError as e:
        raise RuntimeError("METEOR requires nltk.") from e

    scores: List[float] = []
    for refs, hyp in zip(list_of_references, hypotheses):
        if not hyp:
            scores.append(0.0)
            continue
        best = max(float(meteor_score(ref, hyp)) for ref in refs)
        scores.append(best)
    return float(sum(scores) / len(scores)) if scores else 0.0


EVAL_METRICS: Dict[str, MetricFn] = {
    "bleu1": _bleu1,
    "bleu2": _bleu2,
    "bleu3": _bleu3,
    "bleu4": _bleu4,
    "meteor": _meteor_corpus,
}


def run_metrics(
    names: Sequence[str],
    list_of_references,
    hypotheses,
) -> Dict[str, float]:
    """
    Run named metrics from :data:`EVAL_METRICS`.

    Unknown names are skipped with a warning. METEOR may raise if WordNet is missing.
    """
    out: Dict[str, float] = {}
    for name in names:
        key = name.lower()
        fn = EVAL_METRICS.get(key)
        if fn is None:
            logger.warning("Unknown eval metric %r; skipping.", name)
            continue
        try:
            out[key] = fn(list_of_references, hypotheses)
        except LookupError as e:
            if key == "meteor":
                logger.warning(
                    "METEOR skipped (missing NLTK data). "
                    "Run nltk.download('wordnet') and nltk.download('omw-1.4'). %s",
                    e,
                )
                continue
            raise
    return out
