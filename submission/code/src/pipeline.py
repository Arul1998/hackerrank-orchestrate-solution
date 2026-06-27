from __future__ import annotations

import json
from pathlib import Path

from .config import Settings
from .data_loader import (
    ClaimRow,
    EvidenceRequirement,
    UserHistory,
    image_cache_key,
    resolve_image_path,
)
from .llm_client import LLMClient
from .metrics import UsageMetrics
from .schema import (
    CLAIM_STATUSES,
    ISSUE_TYPES,
    OUTPUT_COLUMNS,
    SEVERITIES,
    clamp_enum,
    normalize_bool,
    normalize_risk_flags,
    normalize_supporting_ids,
    parts_for_object,
)


PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def relevant_requirements(
    requirements: list[EvidenceRequirement], claim_object: str, issue_type: str
) -> list[EvidenceRequirement]:
    selected: list[EvidenceRequirement] = []
    issue = issue_type.lower()
    for req in requirements:
        if req.claim_object not in {claim_object, "all"}:
            continue
        applies = req.applies_to.lower()
        if applies == "general claim review" or applies == "reviewability" or applies == "multi-image rows":
            selected.append(req)
            continue
        if any(token in issue for token in applies.replace(",", " ").split()):
            selected.append(req)
            continue
        if issue in applies or applies in issue:
            selected.append(req)
    return selected


def history_context(history: UserHistory | None) -> dict:
    if history is None:
        return {"history_flags": "none", "history_summary": "No history found."}
    return {
        "history_flags": history.history_flags,
        "history_summary": history.history_summary,
        "past_claim_count": history.past_claim_count,
        "rejected_claim": history.rejected_claim,
        "manual_review_claim": history.manual_review_claim,
        "last_90_days_claim_count": history.last_90_days_claim_count,
    }


def normalize_output(
    claim: ClaimRow,
    payload: dict,
) -> dict:
    claim_object = claim.claim_object
    allowed_parts = parts_for_object(claim_object)

    issue_type = clamp_enum(payload.get("issue_type", "unknown"), ISSUE_TYPES)
    object_part = clamp_enum(payload.get("object_part", "unknown"), allowed_parts)
    claim_status = clamp_enum(
        payload.get("claim_status", "not_enough_information"),
        CLAIM_STATUSES,
        default="not_enough_information",
    )
    severity = clamp_enum(payload.get("severity", "unknown"), SEVERITIES)

    risk_flags = payload.get("risk_flags", "none")
    if isinstance(risk_flags, list):
        risk_flags = normalize_risk_flags(risk_flags)
    else:
        risk_flags = normalize_risk_flags(str(risk_flags).split(";"))

    supporting = payload.get("supporting_image_ids", "none")
    if isinstance(supporting, list):
        supporting = normalize_supporting_ids(supporting)
    else:
        supporting = normalize_supporting_ids(supporting)

    return {
        "user_id": claim.user_id,
        "image_paths": claim.image_paths,
        "user_claim": claim.user_claim,
        "claim_object": claim_object,
        "evidence_standard_met": normalize_bool(payload.get("evidence_standard_met", False)),
        "evidence_standard_met_reason": str(payload.get("evidence_standard_met_reason", "")).strip(),
        "risk_flags": risk_flags,
        "issue_type": issue_type,
        "object_part": object_part,
        "claim_status": claim_status,
        "claim_status_justification": str(payload.get("claim_status_justification", "")).strip(),
        "supporting_image_ids": supporting,
        "valid_image": normalize_bool(payload.get("valid_image", True)),
        "severity": severity,
    }


class ClaimReviewPipeline:
    def __init__(self, settings: Settings, metrics: UsageMetrics | None = None):
        self.settings = settings
        self.metrics = metrics or UsageMetrics()
        self.client = LLMClient(settings, self.metrics)

    def process_single_pass(
        self,
        claim: ClaimRow,
        dataset_root: Path,
        history: UserHistory | None,
        requirements: list[EvidenceRequirement],
    ) -> dict:
        image_paths = [
            resolve_image_path(dataset_root, rel) for rel in claim.image_path_list()
        ]
        image_ids = claim.image_ids()
        req_text = "\n".join(
            f"- {r.requirement_id}: {r.minimum_image_evidence}" for r in requirements
        )
        user_prompt = json.dumps(
            {
                "claim_object": claim.claim_object,
                "user_claim": claim.user_claim,
                "image_ids": image_ids,
                "user_history": history_context(history),
                "evidence_requirements": req_text,
            },
            indent=2,
        )
        cache_key = f"single_{claim.user_id}_{claim.row_index}_{hash(claim.image_paths)}"
        result = self.client.analyze_images(
            system_prompt=load_prompt("single_pass.txt"),
            user_prompt=user_prompt,
            image_paths=image_paths,
            cache_key=cache_key,
            model=self.settings.text_model,
        )
        return normalize_output(claim, result)

    def process_multi_stage(
        self,
        claim: ClaimRow,
        dataset_root: Path,
        history: UserHistory | None,
        requirements: list[EvidenceRequirement],
    ) -> dict:
        allowed_parts = sorted(parts_for_object(claim.claim_object))
        extraction_prompt = json.dumps(
            {
                "claim_object": claim.claim_object,
                "allowed_object_parts": allowed_parts,
                "user_claim": claim.user_claim,
            },
            indent=2,
        )
        extraction_key = f"extract_{claim.user_id}_{claim.row_index}_{hash(claim.user_claim)}"
        extracted = self.client.complete_text(
            system_prompt=load_prompt("claim_extraction.txt"),
            user_prompt=extraction_prompt,
            cache_key=extraction_key,
        )

        issue_type = extracted.get("claimed_issue_type", "unknown")
        selected_requirements = relevant_requirements(requirements, claim.claim_object, issue_type)
        req_text = "\n".join(
            f"- {r.requirement_id} ({r.applies_to}): {r.minimum_image_evidence}"
            for r in selected_requirements
        )

        image_paths = [
            resolve_image_path(dataset_root, rel) for rel in claim.image_path_list()
        ]
        image_ids = claim.image_ids()
        per_image_cache_parts = [image_cache_key(path) for path in image_paths]
        vision_key = f"vision_{claim.user_id}_{claim.row_index}_{'_'.join(per_image_cache_parts[:2])}"

        per_image_results: list[dict] = []
        for path, image_id in zip(image_paths, image_ids):
            single_prompt = json.dumps(
                {
                    "claim_object": claim.claim_object,
                    "image_id": image_id,
                    "extracted_claim": extracted,
                    "user_claim": claim.user_claim,
                    "evidence_requirements": req_text,
                },
                indent=2,
            )
            single_key = f"{vision_key}_{image_id}"
            analysis = self.client.analyze_images(
                system_prompt=load_prompt("image_analysis.txt"),
                user_prompt=single_prompt,
                image_paths=[path],
                cache_key=single_key,
            )
            per_image_results.append({"image_id": image_id, "analysis": analysis})

        synthesis_prompt = json.dumps(
            {
                "claim_object": claim.claim_object,
                "allowed_object_parts": allowed_parts,
                "extracted_claim": extracted,
                "user_claim": claim.user_claim,
                "user_history": history_context(history),
                "evidence_requirements": req_text,
                "per_image_results": per_image_results,
            },
            indent=2,
        )
        synthesis_key = f"synth_{claim.user_id}_{claim.row_index}_{hash(claim.image_paths)}"
        final = self.client.complete_text(
            system_prompt=load_prompt("decision_synthesis.txt"),
            user_prompt=synthesis_prompt,
            cache_key=synthesis_key,
        )
        return normalize_output(claim, final)

    def process_claim(
        self,
        claim: ClaimRow,
        dataset_root: Path,
        history_map: dict[str, UserHistory],
        requirements: list[EvidenceRequirement],
        strategy: str = "multi_stage",
    ) -> dict:
        history = history_map.get(claim.user_id)
        if strategy == "single_pass":
            return self.process_single_pass(claim, dataset_root, history, requirements)
        return self.process_multi_stage(claim, dataset_root, history, requirements)


def run_pipeline(
    claims: list[ClaimRow],
    dataset_root: Path,
    history_map: dict[str, UserHistory],
    requirements: list[EvidenceRequirement],
    strategy: str = "multi_stage",
    settings: Settings | None = None,
) -> tuple[list[dict], UsageMetrics]:
    settings = settings or Settings()
    metrics = UsageMetrics()
    pipeline = ClaimReviewPipeline(settings, metrics)

    outputs: list[dict] = []
    total = len(claims)
    for index, claim in enumerate(claims, start=1):
        try:
            print(f"Processing claim {index}/{total} ({claim.user_id})...", flush=True)
            outputs.append(
                pipeline.process_claim(
                    claim=claim,
                    dataset_root=dataset_root,
                    history_map=history_map,
                    requirements=requirements,
                    strategy=strategy,
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep batch running
            metrics.errors += 1
            message = str(exc)
            if "insufficient_quota" in message:
                raise RuntimeError(
                    "OpenAI API quota exceeded. Add billing/credits at "
                    "https://platform.openai.com/settings/organization/billing "
                    "and re-run. Partial output was not saved."
                ) from exc
            outputs.append(
                {
                    "user_id": claim.user_id,
                    "image_paths": claim.image_paths,
                    "user_claim": claim.user_claim,
                    "claim_object": claim.claim_object,
                    "evidence_standard_met": "false",
                    "evidence_standard_met_reason": f"Processing error: {exc}",
                    "risk_flags": "manual_review_required",
                    "issue_type": "unknown",
                    "object_part": "unknown",
                    "claim_status": "not_enough_information",
                    "claim_status_justification": "Automated review failed; manual review required.",
                    "supporting_image_ids": "none",
                    "valid_image": "false",
                    "severity": "unknown",
                }
            )

    metrics.finish()
    return outputs, metrics
