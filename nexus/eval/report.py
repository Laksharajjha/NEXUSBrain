"""Markdown reports from result records."""

from __future__ import annotations

from typing import Any, Sequence

SPLIT_ORDER = (
    "train",
    "dev",
    "test_novel_combination",
    "test_novel_word_in_known_frame",
    "test_longer",
    "test_contradiction",
)


def fmt(stat: dict[str, Any] | None, digits: int = 3) -> str:
    """Format a ``summarize`` record as ``mean ± std``."""
    if not stat or stat.get("mean") is None:
        n_missing = stat.get("n_missing", 0) if stat else 0
        return f"n/a ({n_missing} missing)" if n_missing else "n/a"
    text = f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"
    if stat.get("n_missing"):
        text += f" ({stat['n']}/{stat['n'] + stat['n_missing']} seeds)"
    return text


def split_table(results: Sequence[dict[str, Any]]) -> str:
    """Accuracy per split, one column per learner, with split size n."""
    header = "| split | n | " + " | ".join(r["learner"] for r in results) + " |"
    lines = [header, "|" + "---|" * (len(results) + 2)]
    for split in SPLIT_ORDER:
        n_stat = results[0]["aggregate"].get(f"splits.{split}.n") if results else None
        n_str = f"{int(round(n_stat['mean']))}" if n_stat and n_stat.get("mean") is not None else "n/a"
        cells = [fmt(r["aggregate"].get(f"splits.{split}.accuracy")) for r in results]
        lines.append(f"| {split} | {n_str} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def calibration_table(result: dict[str, Any]) -> str:
    """ECE / Brier / mean confidence per split for one learner, including n per split."""
    agg = result["aggregate"]
    lines = ["| split | n | accuracy | ECE | Brier | mean confidence |", "|---|---|---|---|---|---|"]
    for split in SPLIT_ORDER:
        p = f"splits.{split}."
        n_stat = agg.get(p + "n")
        n_str = f"{int(round(n_stat['mean']))}" if n_stat and n_stat.get("mean") is not None else "n/a"
        lines.append(
            f"| {split} | {n_str} | {fmt(agg.get(p + 'accuracy'))} | {fmt(agg.get(p + 'ece'))} | "
            f"{fmt(agg.get(p + 'brier'))} | {fmt(agg.get(p + 'mean_confidence'))} |"
        )
    return "\n".join(lines)


def checkpoint_table(result: dict[str, Any], split: str) -> str:
    """Accuracy at each support checkpoint of a support split."""
    agg = result["aggregate"]
    prefix = f"splits.{split}.checkpoints."
    labels = list(result["per_seed"][0]["splits"][split]["checkpoints"])
    lines = ["| checkpoint | accuracy |", "|---|---|"]
    for lab in labels:
        lines.append(f"| {lab} | {fmt(agg.get(prefix + lab + '.accuracy'))} |")
    return "\n".join(lines)


def learning_table(result: dict[str, Any]) -> str:
    """Examples-to-criterion and retention per construction."""
    agg = result["aggregate"]
    constructions = sorted({k.split(".")[1] for k in agg if k.startswith("retention.")})
    lines = [
        "| construction | examples-to-criterion (total seen) | retention: at stage end | retention: final |",
        "|---|---|---|---|",
    ]
    for c in constructions:
        lines.append(
            f"| {c} | {fmt(agg.get(f'examples_to_criterion.{c}.n_seen'), 0)} | "
            f"{fmt(agg.get(f'retention.{c}.at_stage_end'))} | {fmt(agg.get(f'retention.{c}.final'))} |"
        )
    return "\n".join(lines)


def full_report(result: dict[str, Any]) -> str:
    """A complete markdown report for one learner."""
    agg = result["aggregate"]
    return "\n\n".join([
        f"# {result['experiment']} — {result['learner']} ({result['n_seeds']} seeds)",
        "## Accuracy and calibration (mean ± std over seeds)",
        calibration_table(result),
        "## Support-stream checkpoints: test_novel_word_in_known_frame",
        checkpoint_table(result, "test_novel_word_in_known_frame"),
        "## Support-stream checkpoints: test_contradiction",
        checkpoint_table(result, "test_contradiction"),
        "## Learning curves (dev)",
        learning_table(result),
        f"Model size: {fmt(agg.get('model_size'), 1)}",
    ]) + "\n"
