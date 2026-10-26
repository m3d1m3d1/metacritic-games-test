from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from app.llm import ReviewSummaryResult, summarize_reviews
from app.scraper.parsers import Review, ReviewsPage

EVAL_PATH = Path(__file__).resolve().parent.parent / "evals" / "review_summary_cases.json"
MIN_SIGNAL_COVERAGE = 0.5


class ReviewSummaryEvalCase(BaseModel):
    id: str
    audience: Literal["critic", "user"]
    platform: str
    reviews: list[str] = Field(min_length=1)
    required_strength_signals: list[list[str]]
    required_weakness_signals: list[list[str]]
    expected_empty_likes: bool = False
    expected_empty_dislikes: bool = False
    forbidden_claims: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_signal_aliases(self):
        if any(not aliases or any(not item.strip() for item in aliases) for aliases in (
            self.required_strength_signals + self.required_weakness_signals
        )):
            raise ValueError("Each expected signal must have at least one non-empty alias")
        return self


def load_eval_cases(path: Path = EVAL_PATH) -> list[ReviewSummaryEvalCase]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to load evaluation cases from {path}: {exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError("Evaluation dataset must be a non-empty JSON array")
    try:
        cases = [ReviewSummaryEvalCase.model_validate(item) for item in payload]
    except ValidationError as exc:
        raise ValueError(f"Evaluation dataset has an invalid case: {exc}") from exc
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation case IDs must be unique")
    return cases


def _signal_coverage(signals: list[list[str]], text: str) -> float:
    if not signals:
        return 1.0
    normalized = text.casefold()
    matched = sum(
        any(alias.casefold() in normalized for alias in aliases)
        for aliases in signals
    )
    return matched / len(signals)


def score_summary(
    case: ReviewSummaryEvalCase,
    result: ReviewSummaryResult,
) -> dict[str, object]:
    text = " ".join(
        [result.summary, *result.likes, *result.dislikes]
    ).casefold()
    strengths = _signal_coverage(case.required_strength_signals, text)
    weaknesses = _signal_coverage(case.required_weakness_signals, text)
    forbidden_matches = [
        claim for claim in case.forbidden_claims if claim.casefold() in text
    ]
    empty_likes_match = not case.expected_empty_likes or not result.likes
    empty_dislikes_match = not case.expected_empty_dislikes or not result.dislikes
    passed = (
        strengths >= MIN_SIGNAL_COVERAGE
        and weaknesses >= MIN_SIGNAL_COVERAGE
        and not forbidden_matches
        and empty_likes_match
        and empty_dislikes_match
    )
    return {
        "case_id": case.id,
        "model": result.model,
        "summary": result.summary,
        "likes": result.likes,
        "dislikes": result.dislikes,
        "strength_signal_coverage": strengths,
        "weakness_signal_coverage": weaknesses,
        "forbidden_claim_matches": forbidden_matches,
        "expected_empty_likes_met": empty_likes_match,
        "expected_empty_dislikes_met": empty_dislikes_match,
        "passed": passed,
    }


async def run_evaluation(
    cases: list[ReviewSummaryEvalCase],
) -> dict[str, object]:
    results = []
    for case in cases:
        page = ReviewsPage(
            kind=case.audience,
            platform=case.platform,
            reviews=[
                Review(kind=case.audience, text=text)
                for text in case.reviews
            ],
        )
        result = await summarize_reviews(page)
        results.append(score_summary(case, result))

    return {
        "dataset": str(EVAL_PATH.name),
        "metric_note": (
            "Signal coverage uses exact case-insensitive phrase alternatives. "
            "It is a heuristic regression signal, not a proof of factuality or "
            "a substitute for human review."
        ),
        "case_count": len(results),
        "passed_count": sum(bool(item["passed"]) for item in results),
        "average_strength_signal_coverage": sum(
            float(item["strength_signal_coverage"]) for item in results
        )
        / len(results),
        "average_weakness_signal_coverage": sum(
            float(item["weakness_signal_coverage"]) for item in results
        )
        / len(results),
        "cases": results,
    }


async def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate review-summary behavior against synthetic review cases."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and list cases without making LLM requests.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the live evaluation report to this path instead of stdout.",
    )
    args = parser.parse_args()

    cases = load_eval_cases()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dataset_valid",
                    "case_count": len(cases),
                    "case_ids": [case.id for case in cases],
                    "llm_requests": 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    report = await run_evaluation(cases)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Evaluation report written to {args.output}")
    else:
        print(rendered)
    return 0 if report["passed_count"] == report["case_count"] else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
