"""Node 3 — Embed CV + jobs with nomic-embed-text, cosine-rank (Option B only)."""

import logging

import numpy as np

from cv_rank_agent.models import JobDescription, ParsedCV
from cv_rank_agent.config import settings
from cv_rank_agent.state import OverallState
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    vec_a = np.array(a)
    vec_b = np.array(b)
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


def _cv_to_text(parsed_cv: ParsedCV) -> str:
    """Build a natural text representation of the CV for embedding."""
    parts: list[str] = []
    if parsed_cv.summary:
        parts.append(parsed_cv.summary)
    if parsed_cv.skills:
        parts.append("Skills: " + ", ".join(parsed_cv.skills))
    for exp in parsed_cv.experience:
        entry = f"{exp.role} at {exp.company}"
        if exp.duration:
            entry += f" ({exp.duration})"
        if exp.description:
            entry += f" — {exp.description}"
        parts.append(entry)
    return "\n".join(parts)


def embedder(state: OverallState) -> dict:
    """LangGraph node: embed CV + all jobs, cosine-rank, return top N."""
    parsed_cv = state["parsed_cv"]
    job_descriptions = state["job_descriptions"]

    logger.info("Embedding CV and %d job(s) with %s", len(job_descriptions), settings.embedding_model)
    embed_model = OllamaEmbeddings(model=settings.embedding_model)

    # Embed CV as natural text (one call)
    cv_text = _cv_to_text(parsed_cv)
    cv_embedding = embed_model.embed_query(cv_text)
    logger.info("CV embedded — %d dimensions", len(cv_embedding))

    # Embed all job descriptions in one batch call
    job_embeddings = embed_model.embed_documents([job.job_description for job in job_descriptions])
    logger.info("All %d job(s) embedded", len(job_embeddings))

    # Compute cosine similarity for each job
    scored: list[tuple[JobDescription, float]] = []
    for job, job_embedding in zip(job_descriptions, job_embeddings):
        score = _cosine_similarity(cv_embedding, job_embedding)
        scored.append((job, score))

    # Sort by score descending, return top N
    scored.sort(key=lambda x: x[1], reverse=True)
    top_n = scored[: settings.llm_top_n]
    logger.info("Top %d jobs selected (scores: %s)", len(top_n), ", ".join(f"{s:.3f}" for _, s in top_n))
    return {"cosine_results": top_n}