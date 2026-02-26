"""Prompt templates for job description extraction."""

JOB_PARSER_PROMPT = """\
Role: You are a job description extraction specialist.

Task: Extract structured job information from the following crawled Markdown content.

Fields to extract:
- title: Job title/position name
- company: Hiring company name
- location: Job location(s)
- requirements: Required qualifications, skills, and experience (as a list)
- responsibilities: Key duties and responsibilities (as a list)
- job_description: Full job description text

Guidelines:
- Ignore navigation, cookie notices, sign-in prompts, footer content, and "Similar Jobs" sections
- Focus only on the primary job listing
- If a field cannot be found, omit it

Content:
{content}
"""