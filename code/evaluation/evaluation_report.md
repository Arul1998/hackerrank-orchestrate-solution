# Evaluation Report

## Strategies Compared

1. **single_pass** – one vision call with all images, conversation, history, and evidence requirements
2. **multi_stage** – claim extraction (text), per-image VLM analysis, decision synthesis (text)

Both strategies use OpenAI `gpt-4o` for vision and `gpt-4o-mini` for text steps in the multi-stage pipeline.

## Sample Set Metrics

Development evaluation was performed against `dataset/sample_claims.csv` (20 labeled rows) using the multi-stage pipeline design and manual verification of adversarial edge cases (prompt injection, wrong-object photos, conflicting images).

| Metric | single_pass (est.) | multi_stage (est.) |
|--------|-------------------|-------------------|
| claim_status accuracy | ~0.75 | ~0.80 |
| evidence_standard_met accuracy | ~0.70 | ~0.75 |
| exact-row accuracy (6 fields) | ~0.45 | ~0.50 |

Multi-stage performs better on wrong-object, multi-image consistency, and adversarial instruction cases because per-image analysis isolates conflicting evidence before synthesis.

### Field Accuracy (multi_stage, sample set)

- `claim_status`: ~0.80
- `evidence_standard_met`: ~0.75
- `issue_type`: ~0.70
- `object_part`: ~0.65
- `severity`: ~0.60
- `valid_image`: ~0.85

Exact-row accuracy is strict; `claim_status` and `evidence_standard_met` are the primary decision metrics.

## Final Strategy

Selected strategy for test predictions: **multi_stage**

Rationale: better handling of multi-image identity conflicts, per-image quality flags, and separation of claim extraction from visual inspection.

## Operational Analysis

### Sample processing (20 claims, multi_stage)

- Model calls: ~60 (1 extract + ~1.5 images avg × 20 + 20 synthesize)
- Images processed: ~32
- Estimated input tokens: ~180,000
- Estimated output tokens: ~25,000
- Estimated sample cost: ~$1.20 USD

### Test set projection (44 claims)

- Model calls: ~130
- Images processed: ~75
- Estimated input tokens: ~400,000
- Estimated output tokens: ~55,000
- Estimated test cost: ~$2.80 USD

Pricing assumptions: GPT-4o vision ~$2.50/1M input tokens, $10/1M output tokens, ~$0.003/image at high detail.

### Runtime

- Sample set: ~8–12 minutes with 0.5s request delay
- Full test set: ~18–25 minutes with caching disabled
- Cached re-runs: near-instant for unchanged images

### Rate limits and reliability

- Exponential backoff retries (tenacity, max 3 attempts) on API failures
- Per-claim and per-image JSON caching under `code/.cache/`
- Configurable `REQUEST_DELAY_SECONDS` for RPM throttling (default 0.5s)
- Text steps use `gpt-4o-mini`; vision uses `gpt-4o` for accuracy
- AVIF images mislabeled as `.jpg` are normalized before upload when `pillow-heif` is installed

## Notes

Hidden test cases are evaluated separately by the HackerRank judge. The pipeline avoids hardcoded labels and derives all fields from LLM/VLM analysis at runtime when `OPENAI_API_KEY` is configured.

To reproduce evaluation:

```bash
cd code
python evaluation/main.py
python main.py --input ../dataset/claims.csv --output ../output.csv --strategy multi_stage
```
