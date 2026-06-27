#!/usr/bin/env python3
"""Evaluate claim review strategies against sample_claims.csv."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings, repo_root
from src.data_loader import (
    ClaimRow,
    read_claims_csv,
    read_evidence_requirements,
    read_user_history,
)
from src.metrics import UsageMetrics
from src.pipeline import run_pipeline
from src.schema import OUTPUT_COLUMNS


COMPARE_FIELDS = [
    "evidence_standard_met",
    "issue_type",
    "object_part",
    "claim_status",
    "severity",
    "valid_image",
]


def sample_rows_to_claims(path: Path) -> tuple[list[ClaimRow], pd.DataFrame]:
    df = pd.read_csv(path, dtype=str).fillna("")
    claims: list[ClaimRow] = []
    for idx, record in df.iterrows():
        claims.append(
            ClaimRow(
                user_id=record["user_id"],
                image_paths=record["image_paths"],
                user_claim=record["user_claim"],
                claim_object=record["claim_object"],
                row_index=int(idx),
            )
        )
    return claims, df


def score_predictions(expected: pd.DataFrame, predicted: list[dict]) -> dict:
    pred_df = pd.DataFrame(predicted)
    total = len(expected)
    exact_row_matches = 0
    field_scores = {field: 0 for field in COMPARE_FIELDS}

    for idx in range(total):
        row_exact = True
        for field in COMPARE_FIELDS:
            exp = str(expected.iloc[idx][field]).strip().lower()
            got = str(pred_df.iloc[idx][field]).strip().lower()
            if exp == got:
                field_scores[field] += 1
            else:
                row_exact = False
        if row_exact:
            exact_row_matches += 1

    return {
        "rows": total,
        "exact_row_matches": exact_row_matches,
        "exact_row_accuracy": round(exact_row_matches / total, 4) if total else 0.0,
        "field_accuracy": {
            field: round(count / total, 4) if total else 0.0
            for field, count in field_scores.items()
        },
        "claim_status_accuracy": round(field_scores["claim_status"] / total, 4) if total else 0.0,
    }


def write_report(
    path: Path,
    results: dict,
    metrics_a: UsageMetrics,
    metrics_b: UsageMetrics,
) -> None:
    winner = results["winner"]
    lines = [
        "# Evaluation Report",
        "",
        "## Strategies Compared",
        "",
        "1. **single_pass** – one vision call with all images and context",
        "2. **multi_stage** – claim extraction, per-image VLM analysis, synthesis",
        "",
        "## Sample Set Metrics",
        "",
        f"- Single pass exact-row accuracy: **{results['single_pass']['exact_row_accuracy']}**",
        f"- Multi stage exact-row accuracy: **{results['multi_stage']['exact_row_accuracy']}**",
        f"- Single pass claim_status accuracy: **{results['single_pass']['claim_status_accuracy']}**",
        f"- Multi stage claim_status accuracy: **{results['multi_stage']['claim_status_accuracy']}**",
        "",
        "### Field Accuracy (multi_stage)",
        "",
    ]
    for field, value in results["multi_stage"]["field_accuracy"].items():
        lines.append(f"- `{field}`: {value}")
    lines.extend(
        [
            "",
            f"## Final Strategy",
            "",
            f"Selected strategy for test predictions: **{winner}**",
            "",
            "## Operational Analysis",
            "",
            "### Sample processing",
            f"- Single pass model calls: {metrics_a.model_calls}",
            f"- Multi stage model calls: {metrics_b.model_calls}",
            f"- Single pass images processed: {metrics_a.images_processed}",
            f"- Multi stage images processed: {metrics_b.images_processed}",
            f"- Single pass input tokens: {metrics_a.input_tokens}",
            f"- Multi stage input tokens: {metrics_b.input_tokens}",
            f"- Single pass output tokens: {metrics_a.output_tokens}",
            f"- Multi stage output tokens: {metrics_b.output_tokens}",
            f"- Single pass runtime (s): {round(metrics_a.elapsed_seconds, 2)}",
            f"- Multi stage runtime (s): {round(metrics_b.elapsed_seconds, 2)}",
            "",
            "### Test set projection (44 claims)",
            f"- Estimated test cost single pass (USD): {metrics_a.estimated_cost_usd()}",
            f"- Estimated test cost multi stage (USD): {metrics_b.estimated_cost_usd()}",
            "",
            "### Rate limits and reliability",
            "- Requests use exponential backoff retries (tenacity, max 3 attempts).",
            "- Per-image and per-claim JSON caching under `code/.cache/` avoids repeated VLM calls.",
            "- Configurable `REQUEST_DELAY_SECONDS` throttles RPM for OpenAI tier limits.",
            "- Multi-stage uses `gpt-4o-mini` for text steps and `gpt-4o` for vision to balance cost and accuracy.",
            "",
            "## Notes",
            "",
            "Exact-row accuracy is strict; claim_status and evidence fields are the primary decision metrics.",
            "Hidden test cases are evaluated separately by the HackerRank judge.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Evaluate claim review strategies.")
    parser.add_argument(
        "--sample",
        type=Path,
        default=root / "dataset" / "sample_claims.csv",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=root / "dataset",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parent / "evaluation_report.md",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=Path(__file__).resolve().parent / "results.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = Settings()
    settings.validate()

    claims, expected_df = sample_rows_to_claims(args.sample)
    history_map = read_user_history(args.dataset_root / "user_history.csv")
    requirements = read_evidence_requirements(args.dataset_root / "evidence_requirements.csv")

    print("Evaluating single_pass...")
    single_outputs, metrics_a = run_pipeline(
        claims=claims,
        dataset_root=args.dataset_root,
        history_map=history_map,
        requirements=requirements,
        strategy="single_pass",
        settings=settings,
    )
    single_scores = score_predictions(expected_df, single_outputs)

    print("Evaluating multi_stage...")
    multi_outputs, metrics_b = run_pipeline(
        claims=claims,
        dataset_root=args.dataset_root,
        history_map=history_map,
        requirements=requirements,
        strategy="multi_stage",
        settings=settings,
    )
    multi_scores = score_predictions(expected_df, multi_outputs)

    winner = "multi_stage"
    if single_scores["claim_status_accuracy"] > multi_scores["claim_status_accuracy"]:
        winner = "single_pass"
    elif (
        single_scores["claim_status_accuracy"] == multi_scores["claim_status_accuracy"]
        and single_scores["exact_row_accuracy"] > multi_scores["exact_row_accuracy"]
    ):
        winner = "single_pass"

    results = {
        "single_pass": single_scores,
        "multi_stage": multi_scores,
        "winner": winner,
    }
    args.results_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(args.report, results, metrics_a, metrics_b)

    print(json.dumps(results, indent=2))
    print(f"Report written to {args.report}")
    print(f"Recommended strategy: {winner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
