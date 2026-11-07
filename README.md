[![ai-eval CI](https://github.com/SDETBMan/qa-automation-portfolio/actions/workflows/ai-eval.yml/badge.svg)](https://github.com/SDETBMan/qa-automation-portfolio/actions/workflows/ai-eval.yml)

# ai-eval

A production-grade **LLM evaluation framework** built with **DeepEval + Pytest**, testing a RAG (Retrieval-Augmented Generation) pipeline grounded in the Swag Labs / SauceDemo knowledge base. Demonstrates how to apply automated quality gates to AI-generated answers. Covering: relevancy, faithfulness, hallucination, safety, and JSON schema correctness.

## Key Features

* **Ten DeepEval metrics:** `AnswerRelevancyMetric`, `GEval` (faithfulness), `HallucinationMetric`, `HallucinationMetric` (aggregate benchmark), `ToxicityMetric` (safety), `BiasMetric` (bias), `JsonCorrectnessMetric`, `ContextualPrecisionMetric`, `ContextualRecallMetric`, `ContextualRelevancyMetric`: each with a configurable threshold; faithfulness, answer relevancy, and bias use Ollama Cloud Gemma 4.
* **RAG pipeline under test:** Most evaluations use ChromaDB (in-memory, ephemeral) + OpenAI `text-embedding-3-small` for semantic retrieval and GPT-4o-mini for answer generation. Answer relevancy and faithfulness can run independently with Ollama Cloud Gemma 4 and local keyword retrieval.
* **Golden dataset:** `datasets/golden_dataset.json` contains ground-truth Q&A pairs covering the SauceDemo FAQ (products, checkout, shipping, returns, accounts). Each case carries `smoke` or `regression` tags consumed directly by pytest markers.
* **Pytest markers:** `smoke` (fast, push-safe), `regression` (full suite, nightly), `safety` (toxicity and bias), `retrieval` (contextual metrics), `canary` (negative-control metric validation). Filter with `-m smoke`, `-m regression`, `-m retrieval`, or `-m canary`.
* **DataDog observability:** Suite-level pass/fail/skip/duration metrics plus per-call `llm.eval.*` scores and `llm.api.latency_ms` / token usage metrics. Faithfulness prompts, answers, scores, and outcomes are also sent as searchable Datadog logs. Skips gracefully without `DD_API_KEY`.
* **Transient-failure resilience:** `pytest-rerunfailures` retries up to 5× with 60 s delay (`--reruns 5 --reruns-delay 60`) to tolerate transient OpenAI API timeouts in CI.

## Tech Stack

| Layer | Technology |
|---|---|
| LLM evaluation | DeepEval 1.0+ |
| LLM / judge | OpenAI GPT-4o-mini; Ollama Cloud Gemma 4 for answer relevancy, faithfulness, and bias |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | ChromaDB (ephemeral, in-memory) |
| Test runner | Pytest 8+ |
| Observability | DataDog CI Visibility + custom GAUGE metrics |
| CI/CD | GitHub Actions |

## Test Coverage

| File | Metric | Threshold | What is evaluated |
|---|---|---|---|
| `test_answer_relevancy.py` | `AnswerRelevancyMetric` | 0.7 | Generated answer directly addresses the question, regardless of source |
| `test_faithfulness.py` | `GEval` (faithfulness rubric) | 0.7 | Answer is grounded in retrieved context. No hallucinated claims |
| `test_hallucination.py` | `GEval` (contradiction-focused rubric) | 0.8 | Answer does not make claims contradicted by retrieved context; unsupported-but-not-contradicted claims are not penalized |
| `test_answer_correctness.py` | `GEval` | 0.7 | Answer matches the golden expected answer |
| `test_safety.py` | `ToxicityMetric` | 0.5 | Answer does not contain harmful, toxic, or inappropriate content |
| `test_bias.py` | `BiasMetric` | 0.5 | Answer does not contain gender, age, racial, or socioeconomic bias |
| `test_json_correctness.py` | `JsonCorrectnessMetric` | 0.9 | Answer matches expected Pydantic schema when structured output is requested |
| `test_hallucination_benchmark.py` | `HallucinationMetric` (aggregate) | 80% pass rate | Aggregate benchmark — all 10 cases run, sentinel test asserts >= 80% pass rate |
| `test_retrieval_quality.py` | `ContextualPrecisionMetric`, `ContextualRecallMetric`, `ContextualRelevancyMetric` | 0.7 / 0.7 / 0.05 | Retriever chunk relevance, coverage, and alignment — relevancy threshold is lower because coarse FAQ chunks yield low sentence-level relevancy ratios |
| `test_canary.py` | `GEval` (Faithfulness) · `HallucinationMetric` · `AnswerRelevancyMetric` · `BiasMetric` · `ToxicityMetric` | inverted | Canary (negative-control) tests — hardcoded known-bad inputs with inverted assertions validate that each metric detects failures |

All nine test files are parametrized over the golden dataset or inline safety/canary cases. The seven per-case files fail individually; the hallucination benchmark only fails at the aggregate level. The canary file uses inverted assertions (score must be bad) to validate the metrics themselves.

## How to Run

**Prerequisite:** [Python 3.11+](https://python.org). Most evaluations require `OPENAI_API_KEY` in `ai-eval/.env`. The answer-relevancy test can run separately with `OLLAMA_API_KEY`.

```bash
# Install dependencies
pip install -r requirements.txt

# Ollama Cloud answer relevancy (uses OLLAMA_API_KEY)
pytest evals/test_answer_relevancy.py -v

# Ollama Cloud faithfulness (uses OLLAMA_API_KEY)
pytest evals/test_faithfulness.py -v

# Ollama Cloud hallucination detection and golden-answer correctness
pytest evals/test_hallucination.py -v
pytest evals/test_answer_correctness.py -v

# Smoke tests (fast — subset of golden dataset)
pytest -m smoke -v

# Safety tests only
pytest -m safety -v

# Canary tests — negative-control metric validation
pytest -m canary -v

# Retrieval quality tests only
pytest -m retrieval -v

# Full regression suite
pytest -m regression -v

# All tests
pytest -v
```

> Copy `.env.example` to `.env`. Add `OLLAMA_API_KEY` for the standalone answer-relevancy test, or `OPENAI_API_KEY` for the other evaluations and the full suite.

## CI/CD Pipeline

The `ai-eval.yml` workflow triggers on every push/PR to `main` that touches `ai-eval/**`, on a nightly schedule (`05:00 UTC`), and via `workflow_dispatch`.

**Manual dispatch input:**

| Input | Description |
|---|---|
| `marker` | pytest marker filter: `smoke`, `regression`, `safety`, `canary`. Leave blank to run `smoke` (CI default). |

**Pipeline steps:**

1. Checkout code
2. Set up Python 3.11 with pip cache
3. `pip install -r requirements.txt`
4. Run pytest (`--reruns 5 --reruns-delay 60`, `DEEPEVAL_DISABLE_TIMEOUTS=true`)
5. Upload JUnit XML to DataDog CI Visibility (`if: always()`, `continue-on-error: true`)
6. Upload `.deepeval/` artifact (retained 30 days)

> **Secret required:** `OPENAI_API_KEY` must be set in repository secrets. `DD_API_KEY` (optional) enables DataDog CI Visibility and custom metrics.

## Project Structure

```
ai-eval/
├── datasets/
│   └── golden_dataset.json         # Ground-truth Q&A pairs with smoke/regression tags
├── evals/
│   ├── test_answer_relevancy.py    # AnswerRelevancyMetric (threshold 0.7)
│   ├── test_faithfulness.py        # FaithfulnessMetric (threshold 0.7)
│   ├── test_hallucination.py       # GEval contradiction check (threshold 0.8)
│   ├── test_answer_correctness.py  # GEval against golden expected answers (threshold 0.7)
│   ├── test_safety.py              # ToxicityMetric (threshold 0.5)
│   ├── test_bias.py                # BiasMetric (threshold 0.5)
│   ├── test_json_correctness.py    # JsonCorrectnessMetric (threshold 0.9)
│   ├── test_hallucination_benchmark.py  # HallucinationMetric aggregate benchmark (80% pass rate)
│   ├── test_retrieval_quality.py    # ContextualPrecision/Recall/Relevancy (threshold 0.7)
│   └── test_canary.py               # Canary (negative-control) tests — 5 metrics with inverted assertions
├── rag/
│   └── document.py                 # SauceDemo FAQ chunks (knowledge base)
├── utils/
│   └── datadog_reporter.py         # GAUGE metrics: test suite + per-eval scores
├── conftest.py                     # Session fixtures: OpenAI client · ChromaDB · retriever · answer_generator
├── pytest.ini                      # markers: smoke · regression · safety · canary
└── requirements.txt
```

## Observed Evaluation Results

The following results are transcribed from the Datadog dashboard screenshots provided on **November 7, 2026**. They are a snapshot, not a guarantee of later runs; results can change with the model, prompt, FAQ, golden dataset, and dashboard time range.

| Evaluation | Result shown | Details |
|---|---:|---|
| Answer Relevancy | **7 passed, 3 failed (70%)** | Dashboard summary; threshold 0.7. The visible case table shows q4 and q9 as failed, while q5 scored 0.75 and passed. The visible rows appear to show 8 passed and 2 failed, which does not match the dashboard's 7/3 summary; verify the dashboard widgets' time range and aggregation. |
| Answer Correctness | **6 passed, 4 failed (60%); average score 0.73** | Dashboard summary; threshold 0.7. |
| Bias | **5/5 passed (100%)** | All five displayed responses scored 1.0 against a 0.5 threshold. This means the generated responses were judged unbiased; it does not assess whether the FAQ itself contains bias or whether the answers are complete. |

### Answer Relevancy Test Result

Corrected Datadog dashboard screenshot supplied on **November 7, 2026**:

![Datadog Answer Relevancy results: dashboard reports 7 passed, 3 failed, and a 70% pass rate](docs/images/answer-relevancy-results.png)

The dashboard summary reports **7 passed / 3 failed (70%)**. In the case table, q4 and q9 are visibly marked failed with scores of 0.67 against the 0.70 threshold; q5 scored 0.75 and passed. The table appears to show eight passing rows and two failing rows, so its visible case count conflicts with the summary widgets.

### Bias Test Result

Datadog dashboard screenshot supplied on **November 7, 2026**:

![Datadog Bias results: 5 cases passed, 0 failed, 100% pass rate](docs/images/bias-results.png)

The dashboard reports **5 passed / 0 failed (100%)**; each displayed case scored 1.0 against the 0.5 threshold. This measures bias in the generated responses, not whether the FAQ source itself is unbiased or whether responses are complete.

### Answer Correctness Test Result

Corrected Datadog dashboard screenshot supplied on **November 7, 2026**:

![Datadog Answer Correctness results: average score 0.73, 6 passed, 4 failed, 60% pass rate](docs/images/answer-correctness-results.png)

The dashboard reports an average score of **0.73**, with **6 passed / 4 failed (60%)** at the 0.7 threshold. The visible failures are q1, q5, q6, and q9.

### Hallucination Test Result

Datadog dashboard screenshot supplied on **November 7, 2026**:

![Datadog Hallucination results: average score 1.0, 10 passed, 0 failed, 100% pass rate](docs/images/hallucination-results.png)

The dashboard reports an average score of **1.0** and **10 passed / 0 failed (100%)**. The case table shows thresholds from 0.65 to 0.80, so this screenshot may include results from more than one threshold configuration or run; treat the summary as a dashboard snapshot rather than a single reproducible test run.

## DataDog Metrics

Faithfulness, hallucination, and answer-correctness prompts and generated answers are sent separately as Datadog logs (not metric tags). Find them in **Logs → Explorer** using `service:ai-eval @metric_name:llm.eval.faithfulness`, `service:ai-eval @metric_name:llm.eval.hallucination`, or `service:ai-eval @metric_name:llm.eval.answer_correctness`. Correctness logs also include the expected response. These evaluations are independent: hallucination checks contradiction with retrieved context; correctness compares with the golden expected answer. Use `pytest ... -s` to print the same information in the terminal, or `--junitxml=results.xml` to include it as test properties in a JUnit report.

| Metric | Description |
|---|---|
| `test.suite.passed/failed/skipped` | Suite-level counts tagged `framework:ai-eval` |
| `test.suite.duration_ms` | Total session wall-clock duration |
| `llm.eval.answer_relevancy` | Per-case AnswerRelevancyMetric score (0–1) |
| `llm.eval.answer_relevancy.passed/failed` | Per-case binary result based on the 0.7 answer-relevancy threshold |
| `llm.eval.faithfulness` | Per-case FaithfulnessMetric score (0–1) |
| `llm.eval.hallucination` | Per-case HallucinationMetric score (0–1) |
| `llm.eval.answer_correctness` | Per-case score comparing the generated answer to the golden expected answer (0–1) |
| `llm.eval.safety` | Per-case ToxicityMetric safety score (0–1) |
| `llm.eval.bias` | Per-case BiasMetric bias score (0–1) |
| `llm.eval.json_correctness` | Per-case JsonCorrectnessMetric score (0–1) |
| `llm.eval.hallucination_benchmark.pass_rate` | Aggregate hallucination benchmark pass rate (0–100) |
| `llm.eval.hallucination_benchmark.mean_score` | Aggregate hallucination benchmark mean score (0–1) |
| `llm.eval.hallucination_benchmark.sample_count` | Number of cases in the hallucination benchmark run |
| `llm.eval.contextual_precision` | Per-case ContextualPrecisionMetric score (0–1) |
| `llm.eval.contextual_recall` | Per-case ContextualRecallMetric score (0–1) |
| `llm.eval.contextual_relevancy` | Per-case ContextualRelevancyMetric score (0–1) |
| `llm.eval.canary.faithfulness` | Canary faithfulness score — inverted assertion (0–1) |
| `llm.eval.canary.hallucination` | Canary hallucination score — inverted assertion (0–1) |
| `llm.eval.canary.answer_relevancy` | Canary answer relevancy score — inverted assertion (0–1) |
| `llm.eval.canary.bias` | Canary bias score — inverted assertion (0–1) |
| `llm.eval.canary.toxicity` | Canary toxicity score — inverted assertion (0–1) |
| `llm.api.latency_ms` | Per-call answer generation latency |
| `llm.api.prompt_tokens` | Per-call prompt token count |
| `llm.api.completion_tokens` | Per-call completion token count |
| `llm.api.total_tokens` | Per-call total token count |
