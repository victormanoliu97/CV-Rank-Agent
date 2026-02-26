"""Node 1 — Parse CV (PDF/DOCX) into structured data."""
import logging

from cv_rank_agent.tools.file_load import load_cv
from cv_rank_agent.models import ParsedCV
from cv_rank_agent.prompts.cv_parser import CV_PARSER_PROMPT
from cv_rank_agent.config import settings
from cv_rank_agent.state import InputState
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


def cv_parser(state: InputState) -> dict:
    """LangGraph node: extract structured CV data from raw file."""
    logger.info("Loading CV from %s", state["cv_path"])
    raw_text = load_cv(state["cv_path"])
    logger.info("CV loaded — %d characters", len(raw_text))

    llm = ChatOllama(model=settings.llm_model, temperature=settings.temperature)
    llm = llm.with_structured_output(ParsedCV)

    logger.info("Sending CV to LLM for parsing...")
    result = llm.invoke(CV_PARSER_PROMPT.format(content=raw_text))
    logger.info("CV parsed — name: %s, skills: %d, experience: %d", result.name, len(result.skills), len(result.experience))
    return {"parsed_cv": result}