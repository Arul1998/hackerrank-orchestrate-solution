# Multi-Modal Evidence Review Agent

Production-minded Python pipeline for the HackerRank Orchestrate damage-claim verification task.

## Architecture

Staged orchestration with images as the primary source of truth:

1. **Claim extraction** (`gpt-4o-mini`) – parse the conversation into structured claim intent
2. **Per-image VLM analysis** (`gpt-4o`) – inspect each image independently
3. **Decision synthesis** (`gpt-4o-mini`) – combine visual evidence, evidence requirements, and user history risk flags

A second **single-pass** strategy is included for evaluation comparison.

## Setup

```bash
cd code
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # Windows
# cp .env.example .env     # macOS/Linux
```

Add your OpenAI key to `code/.env`:

```env
OPENAI_API_KEY=sk-...
```

## Run predictions (test set)

From the repository root:

```bash
python code/main.py --input dataset/claims.csv --output output.csv --strategy multi_stage
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `dataset/claims.csv` | Input claims CSV |
| `--output` | `output.csv` | Output predictions CSV |
| `--dataset-root` | `dataset` | Folder containing `images/` |
| `--strategy` | `multi_stage` | `multi_stage` or `single_pass` |
| `--metrics-out` | none | Optional JSON usage metrics |

## Evaluate on sample data

Compares both strategies on `dataset/sample_claims.csv`:

```bash
python code/evaluation/main.py
```

Outputs:

- `code/evaluation/evaluation_report.md`
- `code/evaluation/results.json`

Then run the winning strategy on the test set.

## Project layout

```text
code/
├── main.py
├── requirements.txt
├── prompts/
├── src/
│   ├── pipeline.py
│   ├── llm_client.py
│   ├── data_loader.py
│   └── schema.py
└── evaluation/
    ├── main.py
    └── evaluation_report.md
```

## Operational notes

- JSON caching under `code/.cache/` avoids repeated VLM calls during development
- Exponential backoff retries on API failures
- Configurable `REQUEST_DELAY_SECONDS` for RPM throttling
- Usage metrics and approximate USD cost are printed after each run

## Submission checklist

- [ ] `output.csv` has one row per `dataset/claims.csv` row
- [ ] Column order matches `problem_statement.md`
- [ ] Zip `code/` (exclude `.venv`, `.cache`, `__pycache__`)
- [ ] Upload chat transcript from `%USERPROFILE%\hackerrank_orchestrate\log.txt`
