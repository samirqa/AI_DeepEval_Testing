"""
test_answer_correctness.py

Evaluates whether each generated answer matches the golden expected answer.
This is separate from hallucination detection, which checks contradiction with
the retrieved context.
"""

import json
import os
import time
from pathlib import Path

import pytest
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

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
def test_answer_correctness(
    case,
    ollama_cloud_client,
    ollama_cloud_judge,
    record_property,
):
    """Compare the generated answer with expected_output in the golden dataset."""
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

    test_case = LLMTestCase(
        input=case["question"],
        actual_output=actual_output,
        expected_output=case["expected_output"],
    )
    metric = GEval(
        name="Answer Correctness",
        criteria=(
            "Compare the actual output with the expected output for the input "
            "question. Score highly only when the actual output gives the same "
            "substantive answer as the expected output. Penalize incorrect facts, "
            "omissions of information required by the expected output, and refusals "
            "when the expected output answers the question. Ignore harmless wording "
            "differences. Do not treat retrieved context as a replacement for the "
            "expected output."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
        model=ollama_cloud_judge,
    )

    score = metric.measure(test_case)
    passed = metric.is_successful() is True
    record_property("question", case["question"])
    record_property("expected_response", case["expected_output"])
    record_property("actual_response", actual_output)
    record_property("answer_correctness_score", score)
    record_property("passed", passed)
    record_property("failed", not passed)

    print(f"\nQuestion: {case['question']}")
    print(f"Expected response: {case['expected_output']}")
    print(f"Actual response: {actual_output}")
    print(f"Answer correctness score: {score:.2f} (threshold: {metric.threshold:.2f})")
    print(f"Answer correctness result: {'PASSED' if passed else 'FAILED'}")

    send_eval_score("llm.eval.answer_correctness", score, tags)
    send_eval_score("llm.eval.answer_correctness.passed", int(passed), tags)
    send_eval_score("llm.eval.answer_correctness.failed", int(not passed), tags)
    send_eval_case(
        case_id=case["id"],
        question=case["question"],
        actual_response=actual_output,
        expected_response=case["expected_output"],
        metric_name="llm.eval.answer_correctness",
        score=score,
        passed=passed,
        threshold=metric.threshold,
        reason=metric.reason,
        model=model_name,
    )

    assert passed, (
        f"Answer correctness score {score:.2f} is below the threshold "
        f"of {metric.threshold:.2f}. {metric.reason or ''}"
    )
