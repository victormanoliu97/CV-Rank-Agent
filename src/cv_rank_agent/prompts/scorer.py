"""Prompt templates for scoring and reasoning."""
SCORER_PROMPT = """\
Role: You are an expert recruiter and talent-match analyst.

Task: Evaluate how well the candidate's CV matches the given job description.

Provide the following scores (each between 0.0 and 1.0):
- overall_fit_score: How well the candidate fits the role overall
- skill_match_score: How well the candidate's skills align with the job requirements
- experience_match_score: How relevant the candidate's work experience is to the role

Also provide:
- identified_gaps: A list of specific areas where the CV falls short of the job requirements (skills missing, experience lacking, qualifications not met). Return an empty list if there are no gaps.
- llm_explanation: A concise paragraph explaining your reasoning behind the scores, highlighting key strengths and weaknesses of the candidate for this role.

Scoring guidelines:
- 0.0 = No match at all
- 0.3 = Poor match, major gaps
- 0.5 = Partial match, some relevant skills or experience
- 0.7 = Good match, most requirements met
- 0.9 = Excellent match, nearly all requirements met
- 1.0 = Perfect match

Be objective and base your assessment strictly on the information provided. Do not assume skills or experience not mentioned in the CV.

Candidate CV:
{cv_content}

Job Description:
{job_content}
"""