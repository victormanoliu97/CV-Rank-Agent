"""LangGraph StateGraph definition, edges, and conditional routing."""
import logging

from cv_rank_agent.state import InputState, OverallState, OutputState
from cv_rank_agent.config import settings
from cv_rank_agent.nodes.cv_parser import cv_parser
from cv_rank_agent.nodes.job_parser import job_parser
from cv_rank_agent.nodes.embedder import embedder
from cv_rank_agent.nodes.scorer import scorer
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


def route_after_job_parser(state: OverallState) -> str:
    """Decide whether to use the embedder (Option B) or go straight to scorer (Option A)."""
    job_count = len(state["job_descriptions"])
    if job_count > settings.llm_only_threshold:
        logger.info("Routing: %d jobs > threshold %d → Option B (embedder + scorer)", job_count, settings.llm_only_threshold)
        return "many_jobs"
    logger.info("Routing: %d jobs <= threshold %d → Option A (scorer only)", job_count, settings.llm_only_threshold)
    return "few_jobs"


def build_graph() -> CompiledStateGraph:
    """Construct and compile the LangGraph graph with nodes and edges."""

    graph = StateGraph(OverallState, input=InputState, output=OutputState)
    graph.add_node("cv_parser", cv_parser)
    graph.add_node("job_parser", job_parser)
    graph.add_node("embedder", embedder)
    graph.add_node("scorer", scorer)

    graph.add_edge(START, "cv_parser")
    graph.add_edge("cv_parser", "job_parser")
    graph.add_conditional_edges(
        "job_parser",
        route_after_job_parser,
        {"many_jobs": "embedder", "few_jobs": "scorer"},
    )
    graph.add_edge("embedder", "scorer")
    graph.add_edge("scorer", END)

    return graph.compile()