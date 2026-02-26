"""TypedDict / Pydantic state schema shared across graph nodes."""
from __future__ import annotations

from typing import NotRequired, TypedDict

from cv_rank_agent.models import JobDescription, ParsedCV, ScoreResult


class InputState(TypedDict):
    """Inputs to the graph, set before execution."""
    cv_path: str
    job_urls: list[str]


class OverallState(TypedDict):
    """Superset of all data flowing through the graph.

    This is the main state passed to StateGraph(). It contains
    input fields, intermediate data populated by nodes, and output fields.
    """
    # Inputs
    cv_path: str
    job_urls: list[str]
    # Intermediate (populated by nodes)
    parsed_cv: ParsedCV
    job_descriptions: list[JobDescription]
    cosine_results: NotRequired[list[tuple[JobDescription, float]]]  # only in Option B
    # Output
    score_results: list[ScoreResult]


class OutputState(TypedDict):
    """Final output data from the graph."""
    score_results: list[ScoreResult]