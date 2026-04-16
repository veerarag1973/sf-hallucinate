# CLI Reference

`sf-hallucinate` ships a standalone command-line tool with three sub-commands.

**Entry points:** `sf-hallucinate` (installed script) or `python -m sf_hallucinate`.

---

## Global options

| Flag | Default | Description |
|---|---|---|
| `--threshold FLOAT` | `0.5` | Hallucination-risk threshold. Risk above this value = failure. |
| `--format {human,json,jsonl}` | `human` | Output format. |
| `--fail-on-any` | `False` | (batch only) Exit 1 if *any* pair fails. |
| `--backend {hybrid,embedding,llm-nli}` | `hybrid` | Similarity backend. |
| `--embedding-model TEXT` | `all-MiniLM-L6-v2` | Model for embedding backend. |
| `--llm-model TEXT` | `gpt-4o-mini` | Model for LLM-NLI backend. |
| `--llm-api-key TEXT` | — | API key for LLM backends. Falls back to `OPENAI_API_KEY` env var. |
| `--language {ar,de,en,es,fr,ja,ko,pt,ru,zh}` | `en` | Language for tokenization and scoring. |
| `--no-contradiction-detection` | — | Disable contradiction detection. |

---

## Exit codes

| Code | Meaning |
|---|---|
| `0` | All outputs passed the threshold. |
| `1` | One or more outputs failed, a file was not found, or an input error occurred. |
| `130` | Interrupted by `Ctrl-C` / `SIGINT`. |

---

## `score` — Score a single pair of strings

```
sf-hallucinate score --output TEXT --reference TEXT [--threshold FLOAT] [--format FORMAT]
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--output TEXT` | Yes | The LLM-generated text to evaluate. |
| `--reference TEXT` | Yes | The ground-truth / source document. |
| `--threshold FLOAT` | No (0.5) | Risk threshold. |
| `--format FORMAT` | No (human) | Output format: `human`, `json`, or `jsonl`. |

### Examples

```bash
# Human-readable output
sf-hallucinate score \
    --output "The Eiffel Tower is in Berlin." \
    --reference "The Eiffel Tower is in Paris, France."

# JSON output (machine-readable)
sf-hallucinate score \
    --output "The Eiffel Tower is in Berlin." \
    --reference "The Eiffel Tower is in Paris, France." \
    --format json

# Strict threshold
sf-hallucinate score \
    --output "Paris is the capital of France." \
    --reference "Paris is the capital city of France." \
    --threshold 0.3

# LLM-NLI backend
sf-hallucinate score \
    --output "Paris is the capital." \
    --reference "Paris is the capital of France." \
    --backend llm-nli \
    --llm-api-key sk-... \
    --llm-model gpt-4o-mini

# Multi-language
sf-hallucinate score \
    --output "París es la capital." \
    --reference "París es la capital de Francia." \
    --language es

# Disable contradiction detection
sf-hallucinate score \
    --output "The tower is 200m tall." \
    --reference "The tower is 330m tall." \
    --no-contradiction-detection
```

### JSON output shape

```json
{
  "hallucination_risk": 0.742,
  "faithfulness_score": 0.258,
  "grounded_claim_count": 0,
  "total_claim_count": 1,
  "grounding_rate": 0.0,
  "claim_results": [
    {
      "claim": "The Eiffel Tower is in Berlin.",
      "best_match": "The Eiffel Tower is in Paris, France.",
      "similarity": 0.258,
      "grounded": false,
      "contradiction_detected": false,
      "entailment_label": "",
      "confidence": 1.0
    }
  ],
  "threshold": 0.5,
  "passed": false,
  "metadata": { "scorer": "faithfulness", ... },
  "confidence": 0.72,
  "contradiction_count": 0
}
```

---

## `score-file` — Score from files

```
sf-hallucinate score-file --output FILE --reference FILE [--threshold FLOAT] [--format FORMAT]
```

Reads the full contents of each file as UTF-8 text, then scores them.

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--output FILE` | Yes | Path to the file containing the LLM output. |
| `--reference FILE` | Yes | Path to the reference document file. |
| `--threshold FLOAT` | No (0.5) | Risk threshold. |
| `--format FORMAT` | No (human) | Output format. |

### Examples

```bash
sf-hallucinate score-file \
    --output answer.txt \
    --reference source.txt \
    --threshold 0.5 \
    --format json
```

---

## `batch` — Score a JSONL file of pairs

```
sf-hallucinate batch --input FILE [--threshold FLOAT] [--format FORMAT] [--fail-on-any]
```

Reads a `.jsonl` file where each line is a JSON object with `output` and `reference` keys. Lines that are blank, malformed JSON, or missing required fields are silently skipped.

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--input FILE` | Yes | Path to the JSONL input file. |
| `--threshold FLOAT` | No (0.5) | Applied to every pair. |
| `--format FORMAT` | No (human) | `human` prints a summary table; `jsonl` emits one JSON object per line; `json` is an alias for `jsonl` in batch mode. |
| `--fail-on-any` | No | Exit 1 if any single pair exceeds the threshold. Default: exit 1 only if *all* pairs fail. |

### Input file format

```jsonl
{"output": "Paris is the capital of France.", "reference": "Paris is the capital city of France, located in northern France."}
{"output": "The Eiffel Tower is made of concrete.", "reference": "The Eiffel Tower is made of iron."}
{"output": "The tower receives 7 million visitors per year.", "reference": "The Eiffel Tower receives about 7 million visitors annually."}
```

### Examples

```bash
# Human summary
sf-hallucinate batch --input pairs.jsonl --threshold 0.5

# JSONL output — one result per line
sf-hallucinate batch --input pairs.jsonl --format jsonl

# Fail if any pair exceeds threshold
sf-hallucinate batch --input pairs.jsonl --fail-on-any
```

### JSONL output shape

Each line is a complete `ScorerResult` JSON object (same shape as `score --format json`) with an additional `"index"` field:

```json
{"index": 1, "hallucination_risk": 0.12, "faithfulness_score": 0.88, "passed": true, ...}
{"index": 2, "hallucination_risk": 0.83, "faithfulness_score": 0.17, "passed": false, ...}
```

---

## Human-readable output

The `human` format prints a visual risk bar and per-claim breakdown:

```
Hallucination Risk  [████████████░░░░░░░░░░░░░░░░░░] 0.412  → FAIL  (threshold=0.4)
    Faithfulness score : 0.588
    Grounded claims    : 2 / 3  (67%)
    ─── Per-claim breakdown ───
    ✓ [1] sim=0.821  claim: Paris is the capital city of France.
    ✓ [2] sim=0.744  claim: The Eiffel Tower stands 330 metres tall.
    ✗ [3] sim=0.152  claim: The tower was built using reinforced concrete.
          best match: The Eiffel Tower is made of iron.
```

---

## Using as a Python module entry point

```bash
python -m sf_hallucinate score --output "..." --reference "..."
```

Identical behaviour to the `sf-hallucinate` script.
