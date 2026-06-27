#!/usr/bin/env python3
"""Entry point for damage claim evidence review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import Settings, dataset_dir, repo_root
from src.data_loader import (
    read_claims_csv,
    read_evidence_requirements,
    read_user_history,
    write_output_csv,
)
from src.metrics import UsageMetrics
from src.pipeline import run_pipeline
from src.schema import OUTPUT_COLUMNS


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Run multimodal damage claim review.")
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "dataset" / "claims.csv",
        help="Input claims CSV path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "output.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=root / "dataset",
        help="Dataset root containing images/",
    )
    parser.add_argument(
        "--strategy",
        choices=["single_pass", "multi_stage"],
        default="multi_stage",
        help="Review strategy",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Optional JSON file for usage metrics",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = Settings()
    settings.validate()

    claims = read_claims_csv(args.input)
    history_map = read_user_history(args.dataset_root / "user_history.csv")
    requirements = read_evidence_requirements(args.dataset_root / "evidence_requirements.csv")

    print(f"Processing {len(claims)} claims with strategy={args.strategy}...")
    outputs, metrics = run_pipeline(
        claims=claims,
        dataset_root=args.dataset_root,
        history_map=history_map,
        requirements=requirements,
        strategy=args.strategy,
        settings=settings,
    )

    write_output_csv(args.output, outputs, OUTPUT_COLUMNS)
    print(f"Wrote {len(outputs)} rows to {args.output}")
    print(json.dumps(metrics.summary(), indent=2))
    print(f"Estimated cost (USD): {metrics.estimated_cost_usd()}")

    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(
            json.dumps(metrics.summary(), indent=2),
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
