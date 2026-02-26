"""Pydantic data models: ParsedCV, JobDescription, ScoreResult."""
from __future__ import annotations

from pydantic import BaseModel

class WorkExperience(BaseModel):
    """Structured work experience entry."""
    company: str
    role: str
    duration: str | None = None        # e.g. "June 2020 - July 2023" — not always stated
    description: str | None = None

class Education(BaseModel):
    """Structured education entry."""
    institution: str
    degree: str
    year: str | None = None             # graduation year — not always stated

class LanguageSkill(BaseModel):
    """Structured language skill entry."""
    language: str
    proficiency: str | None = None      # e.g. "native", "fluent"

class ParsedCV(BaseModel):
    """Structured CV data extracted from raw PDF/DOCX content."""
    name: str
    email: str | None = None             # not every CV includes an email
    phone: str | None = None             # not every CV includes a phone number
    location: str | None = None          # some CVs omit location or just say "remote"
    summary: str | None = None           # some CVs skip the summary section
    skills: list[str] = []               # defaults to empty if no skills found
    experience: list[WorkExperience] = []  # work history entries
    education: list[Education] = []        # education entries
    certifications: list[str] = []         # defaults to empty if none listed
    languages: list[LanguageSkill] = []    # language skills entries

class JobDescription(BaseModel):
    """Structured job description data extracted from a URL."""
    title: str
    company: str | None = None           # not always clearly stated
    location: str | None = None          # some postings are "remote" or unspecified
    requirements: list[str] = []         # required skills/qualifications
    responsibilities: list[str] = []     # job duties
    job_description: str                 # full description text
    source_url: str                      # original URL
    
class ScoreResult(BaseModel):
    """LLM scoring result for a CV against a job description."""
    job_reference: str                            # Job URL
    overall_fit_score: float                      # 0.0 to 1.0
    skill_match_score: float                      # 0.0 to 1.0
    experience_match_score: float                 # 0.0 to 1.0
    identified_gaps: list[str] = []               # areas where CV falls short
    llm_explanation: str                          # LLM's reasoning behind the scores
    cosine_similarity_score: float | None = None  # only present in Option B (hybrid mode)