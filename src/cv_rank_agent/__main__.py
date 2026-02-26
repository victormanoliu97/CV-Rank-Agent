"""cv_rank_agent — CV-to-job-opening ranking agent.

Entry point: uv run python -m cv_rank_agent
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import asyncio
from pathlib import Path

from cv_rank_agent.graph import build_graph
from cv_rank_agent.models import ScoreResult

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cv_rank_agent",
        description="Rank a CV against job opening URLs using LLM + cosine similarity.",
    )
    parser.add_argument(
        "cv",
        type=Path,
        help="Path to the CV file (PDF or DOCX).",
    )
    parser.add_argument(
        "jobs",
        type=Path,
        help='Path to a JSON file containing job URLs (e.g. samples/jobs.json). '
             'Expected format: {"jobs": ["url1", "url2", ...]}',
    )
    return parser.parse_args(argv)


def load_job_urls(jobs_path: Path) -> list[str]:
    """Load and validate job URLs from a JSON file."""
    try:
        data = json.loads(jobs_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Error: Invalid JSON in {jobs_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, dict) or "jobs" not in data:
        print(
            f'Error: {jobs_path} must contain a JSON object with a "jobs" key.',
            file=sys.stderr,
        )
        sys.exit(1)

    urls = data["jobs"]
    if not isinstance(urls, list) or not all(isinstance(u, str) for u in urls):
        print(
            f'Error: "jobs" in {jobs_path} must be a list of URL strings.',
            file=sys.stderr,
        )
        sys.exit(1)

    return urls


def print_results(scores: list[ScoreResult]) -> None:
    """Pretty-print the scoring results to stdout."""
    if not scores:
        print("\nNo scores to display.")
        return

    # Sort by overall_fit_score descending
    ranked = sorted(scores, key=lambda s: s.overall_fit_score, reverse=True)

    print(f"\n{'=' * 80}")
    print(f"  RANKING RESULTS — {len(ranked)} job(s) evaluated")
    print(f"{'=' * 80}")

    for i, score in enumerate(ranked, start=1):
        print(f"\n  #{i}  {score.job_reference}")
        print(f"  {'─' * 76}")
        print(f"  Overall Fit:       {score.overall_fit_score:.0%}")
        print(f"  Skill Match:       {score.skill_match_score:.0%}")
        print(f"  Experience Match:  {score.experience_match_score:.0%}")
        if score.cosine_similarity_score is not None:
            print(f"  Cosine Similarity: {score.cosine_similarity_score:.0%}")
        if score.identified_gaps:
            print(f"  Gaps:              {', '.join(score.identified_gaps)}")
        print(f"  Explanation:       {score.llm_explanation}")

    print(f"\n{'=' * 80}")


async def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)

    cv_path: Path = args.cv
    jobs_path: Path = args.jobs

    if not cv_path.exists():
        print(f"Error: CV file not found: {cv_path}", file=sys.stderr)
        sys.exit(1)

    if not jobs_path.exists():
        print(f"Error: Jobs file not found: {jobs_path}", file=sys.stderr)
        sys.exit(1)

    urls = load_job_urls(jobs_path)

    if not urls:
        print("Error: No job URLs found in the jobs file.", file=sys.stderr)
        sys.exit(1)

    if len(urls) > 50:
        print("Error: Maximum 50 job URLs allowed.", file=sys.stderr)
        sys.exit(1)

    logger.info("CV:   %s", cv_path)
    logger.info("Jobs: %d URL(s) from %s", len(urls), jobs_path)

    graph = build_graph()
    logger.info("Graph compiled — starting execution")
    result = await graph.ainvoke({"cv_path": str(cv_path), "job_urls": urls})
    logger.info("Graph execution complete")

    print_results(result["score_results"])


if __name__ == "__main__":
    asyncio.run(main())
