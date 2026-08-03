# QA: two-stage qualitative error analysis

Replaces the old single-pass, closed-model error mapping (git `main`:
`scripts/run_analyses.py` + `src/qa`, which mapped ~1,600 free-text verifier
strings into seven categories by keyword matching). That approach had two
weaknesses reviewers would flag: a closed model saw MIMIC discharge notes, and the
categories were induced and counted on the same data with brittle string matching.

This version follows standard qualitative content analysis — **open coding then
closed coding** — with an open-source judge (Qwen3-32B, same family and timeframe
as the evaluated models, served locally so notes stay compliant).

## Workflow

1. **Stage 1 — induce the taxonomy** (dev split)

   ```
   python scripts/qa_analysis.py stage1
   ```

   The judge writes a free-text root-cause analysis per mispredicted case, then
   consolidates them into a candidate taxonomy of 6–9 error classes.
   Outputs: `data/qa/stage1_rca.json`, `data/qa/taxonomy_candidate.json`.

2. **Human review (required)**

   Edit `taxonomy_candidate.json` into `taxonomy.json`: rename, merge, drop, and
   sharpen the `definition`/`decision_rule` of each class, then set
   `"approved": true`. This human step is what makes the classes defensible.
   Stage 2 refuses to run on an unapproved taxonomy.

3. **Stage 2 — quantify** (test split, once test-split evals exist)

   ```
   python scripts/qa_analysis.py stage2
   ```

   One independent yes/no per class per case (guided decoding → flat boolean
   object, temperature 0). Outputs the per-case matrix
   `data/qa/qa_classifications_{model}.parquet` and the error-shift table
   `data/qa/qa_error_shift.csv` (per-class counts, rates, and base→fine-tuned
   delta — the replacement for the old Table 3).

4. **Validate the judge** (optional but recommended)

   ```
   python scripts/qa_analysis.py validate --export   # writes annotation template
   # ... a human fills the human__* columns with 0/1 ...
   python scripts/qa_analysis.py validate --score    # per-class Cohen's kappa
   ```

   Turns "the LLM said so" into a measured instrument: report the kappa table
   next to the error-shift table.

## Configuration

Everything except the stage is in `scripts/qa_config.yaml` (server, round, target
model list, sampling, thresholds). The CLI only picks the stage — consistent with
`scripts/eval_analysis.py`.

## What one command does

`stage1`/`stage2` are self-contained: deploy the Qwen3-32B judge through the
shared eval `server_template` (deployment + service `vllm-server-qa`, 32k context
— inputs reach ~11k tokens at the tail; don't shrink below 16k), auto
`kubectl port-forward`, join discharge notes (union of the ICD-10 hosp splits on
`hadm_id`, 100% coverage), run every configured result set, write `data/qa/*`,
mirror preview tables + full artifacts to wandb (project `merlin_qa`), and tear
the server down. `QA.server` controls this: `keep_alive: true` leaves the server
running between stages; `connection: localhost` skips deploy entirely (own
port-forward or local vLLM); `wandb: false` runs offline.

## Notes on the model choice

Qwen3 was the newest family when the paper started, and using it as the judge keeps
the analysis within the same model family/era as the systems under study. The main
caveat is self-preference bias (judging Qwen3 outputs with a Qwen3 judge); the
stage-2 decisions are constrained yes/no calls against fixed definitions rather than
open-ended preference judgements, and the validation step quantifies judge
reliability directly. If a reviewer pushes on this, a second-family judge (e.g.
Llama-3.3-70B) can be dropped into `server` and the two rate columns compared.
