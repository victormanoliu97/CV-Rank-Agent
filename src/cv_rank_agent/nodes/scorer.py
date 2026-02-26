"""Node 4 — LLM deep-scores CV against job descriptions."""

import logging

from cv_rank_agent.models import JobDescription, ScoreResult
from cv_rank_agent.nodes.embedder import _cv_to_text
from cv_rank_agent.state import OverallState
from langchain_ollama import ChatOllama
from cv_rank_agent.config import settings
from cv_rank_agent.prompts.scorer import SCORER_PROMPT

logger = logging.getLogger(__name__)


def _get_job_content_text(job: JobDescription) -> str:
    """Build a natural text representation of the job description for scoring."""
    parts: list[str] = []
    if job.title:
        parts.append(f"Title: {job.title}")
    if job.company:
        parts.append(f"Company: {job.company}")
    if job.location:
        parts.append(f"Location: {job.location}")
    if job.requirements:
        parts.append("Requirements:\n- " + "\n- ".join(job.requirements))
    if job.responsibilities:
        parts.append("Responsibilities:\n- " + "\n- ".join(job.responsibilities))
    if job.job_description:
        parts.append("Full Description:\n" + job.job_description)
    return "\n".join(parts)


def scorer(state: OverallState) -> dict:
    """LangGraph node: deep-score CV against job descriptions.

    If cosine_results exist in state (Option B), score those with cosine scores.
    Otherwise (Option A), score all job_descriptions directly.
    """
    parsed_cv = state["parsed_cv"]
    results: list[ScoreResult] = []

    llm = ChatOllama(model=settings.llm_model, temperature=settings.temperature)
    llm = llm.with_structured_output(ScoreResult)

    if "cosine_results" in state:
        jobs_to_score = state["cosine_results"]
        logger.info("Scoring %d job(s) (Option B — with cosine pre-filter)", len(jobs_to_score))
        # Option B: score the top-N jobs that came from the embedder
        for i, (job, cosine_score) in enumerate(jobs_to_score, start=1):
            logger.info("[%d/%d] Scoring %s...", i, len(jobs_to_score), job.source_url)
            prompt = SCORER_PROMPT.format(
                cv_content=_cv_to_text(parsed_cv),
                job_content=_get_job_content_text(job),
            )
            score_result = llm.invoke(prompt)
            score_result.cosine_similarity_score = cosine_score
            score_result.job_reference = job.source_url
            logger.info("[%d/%d] Scored — overall fit: %.0f%%", i, len(jobs_to_score), score_result.overall_fit_score * 100)
            results.append(score_result)
    else:
        jobs_to_score = state["job_descriptions"]
        logger.info("Scoring %d job(s) (Option A — LLM only)", len(jobs_to_score))
        # Option A: score all jobs directly (no cosine pre-filter)
        for i, job in enumerate(jobs_to_score, start=1):
            logger.info("[%d/%d] Scoring %s...", i, len(jobs_to_score), job.source_url)
            prompt = SCORER_PROMPT.format(
                cv_content=_cv_to_text(parsed_cv),
                job_content=_get_job_content_text(job),
            )
            score_result = llm.invoke(prompt)
            score_result.job_reference = job.source_url
            logger.info("[%d/%d] Scored — overall fit: %.0f%%", i, len(jobs_to_score), score_result.overall_fit_score * 100)
            results.append(score_result)

    logger.info("All %d job(s) scored", len(results))
    return {"score_results": results}