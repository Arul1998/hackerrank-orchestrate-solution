from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ClaimRow:
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str
    row_index: int = 0

    def image_path_list(self) -> list[str]:
        return [p.strip() for p in self.image_paths.split(";") if p.strip()]

    def image_ids(self) -> list[str]:
        return [Path(p).stem for p in self.image_path_list()]


@dataclass
class UserHistory:
    user_id: str
    past_claim_count: int
    accept_claim: int
    manual_review_claim: int
    rejected_claim: int
    last_90_days_claim_count: int
    history_flags: str
    history_summary: str


@dataclass
class EvidenceRequirement:
    requirement_id: str
    claim_object: str
    applies_to: str
    minimum_image_evidence: str


def read_claims_csv(path: Path) -> list[ClaimRow]:
    df = pd.read_csv(path, dtype=str).fillna("")
    rows: list[ClaimRow] = []
    for idx, record in df.iterrows():
        rows.append(
            ClaimRow(
                user_id=record["user_id"],
                image_paths=record["image_paths"],
                user_claim=record["user_claim"],
                claim_object=record["claim_object"],
                row_index=int(idx),
            )
        )
    return rows


def read_user_history(path: Path) -> dict[str, UserHistory]:
    df = pd.read_csv(path, dtype=str).fillna("")
    history: dict[str, UserHistory] = {}
    for _, record in df.iterrows():
        history[record["user_id"]] = UserHistory(
            user_id=record["user_id"],
            past_claim_count=int(record["past_claim_count"] or 0),
            accept_claim=int(record["accept_claim"] or 0),
            manual_review_claim=int(record["manual_review_claim"] or 0),
            rejected_claim=int(record["rejected_claim"] or 0),
            last_90_days_claim_count=int(record["last_90_days_claim_count"] or 0),
            history_flags=record["history_flags"] or "none",
            history_summary=record["history_summary"],
        )
    return history


def read_evidence_requirements(path: Path) -> list[EvidenceRequirement]:
    df = pd.read_csv(path, dtype=str).fillna("")
    requirements: list[EvidenceRequirement] = []
    for _, record in df.iterrows():
        requirements.append(
            EvidenceRequirement(
                requirement_id=record["requirement_id"],
                claim_object=record["claim_object"],
                applies_to=record["applies_to"],
                minimum_image_evidence=record["minimum_image_evidence"],
            )
        )
    return requirements


def resolve_image_path(dataset_root: Path, relative_path: str) -> Path:
    return (dataset_root / relative_path).resolve()


def image_cache_key(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(str(path).encode("utf-8"))
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_output_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})
