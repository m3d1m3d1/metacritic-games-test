import json
from pathlib import Path

import pytest

from app.llm import ReviewSummaryResult
from scripts.evaluate_summaries import (
    ReviewSummaryEvalCase,
    load_eval_cases,
    score_summary,
)


def test_synthetic_evaluation_dataset_is_valid_and_unique():
    cases = load_eval_cases()
    assert len(cases) == 3
    assert len({case.id for case in cases}) == len(cases)
    assert all(case.reviews for case in cases)


def test_summary_scorer_reports_signal_coverage_and_forbidden_claims():
    case = ReviewSummaryEvalCase(
        id="unit-test",
        audience="critic",
        platform="fictional",
        reviews=["Fictional test review."],
        required_strength_signals=[["responsive controls", "precise controls"]],
        required_weakness_signals=[["repetitive levels"]],
        forbidden_claims=["online co-op"],
    )
    result = ReviewSummaryResult(
        summary="The game has responsive controls and repetitive levels.",
        likes=["Responsive controls"],
        dislikes=["Repetitive levels", "Online co-op"],
    )

    report = score_summary(case, result)

    assert report["strength_signal_coverage"] == 1.0
    assert report["weakness_signal_coverage"] == 1.0
    assert report["forbidden_claim_matches"] == ["online co-op"]
    assert report["passed"] is False


def test_summary_scorer_checks_expected_empty_feedback_lists():
    case = ReviewSummaryEvalCase(
        id="no-invented-downsides",
        audience="user",
        platform="fictional",
        reviews=["Every synthetic reviewer likes the game."],
        required_strength_signals=[["warm dialogue"]],
        required_weakness_signals=[],
        expected_empty_dislikes=True,
    )
    result = ReviewSummaryResult(
        summary="Reviewers praise the warm dialogue.",
        likes=["Warm dialogue"],
        dislikes=[],
    )

    report = score_summary(case, result)

    assert report["expected_empty_dislikes_met"] is True
    assert report["passed"] is True


def test_eval_dataset_rejects_duplicate_case_ids(tmp_path: Path):
    payload = [
        {
            "id": "duplicate",
            "audience": "user",
            "platform": "fictional",
            "reviews": ["Synthetic review."],
            "required_strength_signals": [],
            "required_weakness_signals": [],
        },
        {
            "id": "duplicate",
            "audience": "critic",
            "platform": "fictional",
            "reviews": ["Another synthetic review."],
            "required_strength_signals": [],
            "required_weakness_signals": [],
        },
    ]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="IDs must be unique"):
        load_eval_cases(path)
