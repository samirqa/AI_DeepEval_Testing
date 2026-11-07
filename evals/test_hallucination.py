"""Evaluate whether answers contradict retrieved FAQ context."""

import json
import os
import time
from pathlib import Path

import pytest
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from rag.retrieval import retrieve_faq_chunks
from utils.datadog_reporter import send_eval_case, send_eval_score

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "golden_dataset.json"
with open(DATASET_PATH) as f:
    _raw = json.load(f)

DATASET = [
    pytest.param(c, id=c["id"], marks=[getattr(pytest.mark, t) for t in c.get("tags", [])])
    for c in _raw
]


@pytest.mark.parametrize("case", DATASET)
def test_hallucination(
    case,
    ollama_cloud_client,
    ollama_cloud_judge,
    record_property,
):
    """
    Checks whether the answer contradicts the retrieved FAQ context.
    Answer correctness against the golden expected answer is tested separately
    in test_answer_correctness.py.
    """
    model_name = os.getenv("OLLAMA_MODEL_NAME", "gemma4:31b")
    context = retrieve_faq_chunks(case["question"], n_results=3)
    context_text = "\n\n".join(context)

    started_at = time.time()
    response = ollama_cloud_client.chat.completions.create(
        model=model_name,
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

    tags = [f"model:{model_name}", f"case:{case['id']}"]
    send_eval_score("llm.api.latency_ms", latency_ms, tags)
    if response.usage:
        send_eval_score("llm.api.prompt_tokens", response.usage.prompt_tokens, tags)
        send_eval_score("llm.api.completion_tokens", response.usage.completion_tokens, tags)
        send_eval_score("llm.api.total_tokens", response.usage.total_tokens, tags)

    metric = GEval(
        name="Hallucination",
        criteria=(
            "Identify factual claims in the actual output that contradict the retrieval "
            "context. Penalize claims only when the context provides incompatible facts. "
            "Do not treat information as a contradiction merely because the context does "
            "not mention it. Score highly when the answer contains no claims contradicted "
            "by the context."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        threshold=0.8,
        model=ollama_cloud_judge,
    )

    test_case = LLMTestCase(
        input=case["question"],
        actual_output=actual_output,
        retrieval_context=context,
    )

    score = metric.measure(test_case)
    passed = metric.is_successful() is True
    record_property("question", case["question"])
    record_property("actual_response", actual_output)
    record_property("hallucination_score", score)
    record_property("passed", passed)
    record_property("failed", not passed    )

    print(f"\nQuestion: {case['question']}")
    print(f"Actual response: {actual_output}")
    print(
        f"Hallucination score: {score:.2f} "
        f"(threshold: {metric.threshold:.2f})"
    )
    print(
        f"Hallucination check: "
        f"{'PASSED' if passed else 'FAILED'}"
    )
    send_eval_score("llm.eval.hallucination", score, tags)
    send_eval_score("llm.eval.hallucination.passed", int(passed), tags)
    send_eval_score("llm.eval.hallucination.failed", int(not passed), tags)
    send_eval_case(
        case_id=case["id"],
        question=case["question"],
        actual_response=actual_output,
        metric_name="llm.eval.hallucination",
        score=score,
        passed=passed,
        threshold=metric.threshold,
        reason=metric.reason,
        model=model_name,
    )
    assert passed, (
        f"Hallucination score {score:.2f} did not meet the "
        f"threshold {metric.threshold:.2f}. {metric.reason or ''}"
    )
