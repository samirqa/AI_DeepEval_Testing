"""
test_answer_relevancy.py

Evaluates whether the RAG pipeline's generated answers actually address
the user's question — regardless of factual accuracy.

Metric: AnswerRelevancyMetric
  - Scores how directly the actual_output responds to the input question.
  - An answer that is factually correct but off-topic will score low.
  - Threshold: 0.7 (70% relevancy required to pass).

Ollama Cloud Gemma 4 generates and evaluates each answer.
"""

import json
import os
import time
from pathlib import Path

import pytest
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from rag.retrieval import retrieve_faq_chunks
from utils.datadog_reporter import send_eval_case, send_eval_score

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "golden_dataset.json"
with open(DATASET_PATH) as f:
    _raw = json.load(f)

# Convert dataset tags → pytest marks so `-m smoke` / `-m regression` work
DATASET = [
    pytest.param(c, id=c["id"], marks=[getattr(pytest.mark, t) for t in c.get("tags", [])])
    for c in _raw
]

@pytest.mark.parametrize("case", DATASET)
def test_answer_relevancy(
    case,
    ollama_cloud_client,
    ollama_cloud_judge,
    record_property,
):
    """
    RAG pipeline flow:
      1. Retrieve FAQ chunks using local keyword overlap.
      2. Generate an answer with Ollama Cloud Gemma 4.
      3. Assert the answer is relevant to the original question.
    """
    context = retrieve_faq_chunks(case["question"])
    context_text = "\n\n".join(context)

    started_at = time.time()
    response = ollama_cloud_client.chat.completions.create(
        model=os.getenv("OLLAMA_MODEL_NAME", "nemotron-3-nano:30b-cloud"),
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful customer support assistant for Swag Labs. "
                    "Answer questions using ONLY the information provided below. "
                    "If the answer is not in the provided information, say so.\n\n"
                    f"{context_text}"
                ),
            },
            {"role": "user", "content": case["question"]},
        ],
    )
    latency_ms = (time.time() - started_at) * 1000
    actual_output = response.choices[0].message.content
    if actual_output is None:
        raise RuntimeError("Ollama Cloud returned an empty answer.")

    model_name = os.getenv("OLLAMA_MODEL_NAME", "nemotron-3-nano:30b-cloud")
    tags = [f"model:{model_name}", "framework:ai-eval"]
    send_eval_score("llm.api.latency_ms", latency_ms, tags)
    if response.usage:
        send_eval_score("llm.api.prompt_tokens", response.usage.prompt_tokens, tags)
        send_eval_score("llm.api.completion_tokens", response.usage.completion_tokens, tags)
        send_eval_score("llm.api.total_tokens", response.usage.total_tokens, tags)

    test_case = LLMTestCase(
        input=case["question"],
        actual_output=actual_output,
        retrieval_context=context,
    )

    metric = AnswerRelevancyMetric(threshold=0.7, model=ollama_cloud_judge)
    score = metric.measure(test_case)
    passed = metric.is_successful() is True
    record_property("question", case["question"])
    record_property("actual_response", actual_output)
    record_property("answer_relevancy_score", score)
    record_property("passed", passed)
    record_property("failed", not passed)

    print(f"\nQuestion: {case['question']}")
    print(f"Actual response: {actual_output}")
    print(f"Answer relevancy score: {score:.2f} (threshold: {metric.threshold:.2f})")
    print(f"Answer relevancy result: {'PASSED' if passed else 'FAILED'}")

    send_eval_score("llm.eval.answer_relevancy", score, tags)
    send_eval_score("llm.eval.answer_relevancy.passed", int(passed), tags)
    send_eval_score("llm.eval.answer_relevancy.failed", int(not passed), tags)
    send_eval_case(
        case_id=case["id"],
        question=case["question"],
        actual_response=actual_output,
        metric_name="llm.eval.answer_relevancy",
        score=score,
        passed=passed,
        threshold=metric.threshold,
        reason=metric.reason,
        model=model_name,
    )
    assert passed, (
        f"Answer relevancy score {score:.2f} is below the threshold "
        f"of {metric.threshold:.2f}. {metric.reason or ''}"
    )
