"""Node 2 — Tool-calling node: crawl URLs and extract job descriptions."""

import logging

from cv_rank_agent.tools.web_crawl import web_crawl
from cv_rank_agent.prompts.job_parser import JOB_PARSER_PROMPT
from cv_rank_agent.models import JobDescription
from cv_rank_agent.config import settings
from cv_rank_agent.state import OverallState
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


async def job_parser(state: OverallState) -> dict:
    """LangGraph node: crawl all job URLs and extract structured job data."""
    llm = ChatOllama(model=settings.llm_model, temperature=settings.temperature)
    llm = llm.with_structured_output(JobDescription)

    job_descriptions: list[JobDescription] = []
    for i, url in enumerate(state["job_urls"], start=1):
        logger.info("[%d/%d] Crawling %s", i, len(state["job_urls"]), url)
        raw_content = await web_crawl(url)
        logger.info("[%d/%d] Crawled — %d characters, sending to LLM...", i, len(state["job_urls"]), len(raw_content))
        result = await llm.ainvoke(JOB_PARSER_PROMPT.format(content=raw_content))
        result.source_url = url
        logger.info("[%d/%d] Parsed — %s at %s", i, len(state["job_urls"]), result.title, result.company)
        job_descriptions.append(result)

    logger.info("All %d job(s) parsed", len(job_descriptions))
    return {"job_descriptions": job_descriptions}