# cv_rank_agent

> **⚠️ Disclaimer:** This is a **learning/personal project** built for educational purposes. It is fully open source — anyone is welcome to download it, run it locally, experiment with it, and suggest improvements via issues or pull requests. No warranties or guarantees are provided.

A Python agent that receives a CV (PDF or DOCX) and up to **50 job opening URLs**, then **ranks and scores** the CV against each job description using a hybrid approach: **cosine similarity** for fast pre-ranking + **LLM** for deep, detailed scoring — all running **100% locally** via [Ollama](https://ollama.com/). No API keys needed, no data leaves your machine.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output](#output)
- [Project Structure](#project-structure)
- [Upgrade Path](#upgrade-path)
- [Running Tests](#running-tests)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Fully local** — no API keys, no cloud services, all inference runs on your hardware
- **Adaptive scoring strategy** — automatically picks the best pipeline based on the number of jobs
- **Structured CV parsing** — extracts name, skills, experience, education, certifications, and languages from PDF/DOCX
- **Web crawling** — fetches and parses job postings directly from URLs (e.g. LinkedIn)
- **Cosine similarity pre-ranking** — fast vector-based filtering using `nomic-embed-text` embeddings (768 dimensions)
- **LLM deep scoring** — detailed skill match, experience match, gap analysis, and reasoning per job
- **Configurable** — all thresholds, models, and behavior controlled via a single `.env` file

---

## Architecture

The agent is built as a **[LangGraph](https://github.com/langchain-ai/langgraph) StateGraph** with conditional routing. It chooses between two execution paths depending on how many jobs are provided, controlled by the `LLM_ONLY_THRESHOLD` setting:

### Option A — LLM Only (small batch: `num_jobs ≤ LLM_ONLY_THRESHOLD`)

All jobs are scored directly by the LLM. Simpler, no embedding step needed.

```
[START] → cv_parser → job_parser → scorer → [END]
```

### Option B — Hybrid (large batch: `num_jobs > LLM_ONLY_THRESHOLD`)

Cosine similarity pre-ranks all jobs first, then the LLM deep-scores only the top N.

```
[START] → cv_parser → job_parser → embedder → scorer → [END]
```

### Node Descriptions

| Node | Description |
|------|-------------|
| **cv_parser** | Parses a CV file (PDF/DOCX) into structured data (name, skills, experience, education, etc.) using the LLM with structured output |
| **job_parser** | Crawls each job URL with [Crawl4AI](https://github.com/unclecode/crawl4ai) to extract markdown content, then sends it to the LLM for structured extraction |
| **embedder** | *(Option B only)* Embeds the CV and all job descriptions into 768-dim vectors using `nomic-embed-text`, computes cosine similarity, and selects the top N |
| **scorer** | LLM reads CV + job description text and produces a detailed score: overall fit, skill match, experience match, identified gaps, and full reasoning |

### Cosine Similarity (embedder node)

- The embedding model converts CV and each job description into 768-dimensional vectors
- Cosine similarity is pure math (no LLM involved): $\frac{A \cdot B}{\|A\| \times \|B\|}$
- Result ranges from 0.0 (completely unrelated) to 1.0 (identical meaning)
- Used **only** for fast pre-ranking — the LLM does the real quality scoring

### Conditional Routing Logic

```python
def route_after_job_parser(state):
    if len(state["job_descriptions"]) <= settings.llm_only_threshold:
        return "few_jobs"    # Option A: skip embedder, go straight to scorer
    return "many_jobs"       # Option B: pre-rank with embedder first
```

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.13+ |
| **Agent Framework** | [LangGraph](https://github.com/langchain-ai/langgraph) |
| **LLM Runtime** | [Ollama](https://ollama.com/) (local inference) |
| **LLM Model** | `llama3.1:8b` — 4.7 GB, 128K context, tool calling support |
| **Embedding Model** | `nomic-embed-text` — 274 MB, 768 dims, 8K context |
| **LLM Integration** | [langchain-ollama](https://python.langchain.com/docs/integrations/llms/ollama/) |
| **Web Crawling** | [Crawl4AI](https://github.com/unclecode/crawl4ai) |
| **PDF Parsing** | [PyMuPDF](https://pymupdf.readthedocs.io/) |
| **DOCX Parsing** | [python-docx](https://python-docx.readthedocs.io/) |
| **Data Validation** | [Pydantic](https://docs.pydantic.dev/) |
| **Configuration** | [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) + `.env` files |
| **Math / Vectors** | [NumPy](https://numpy.org/) |
| **Package Manager** | [uv](https://docs.astral.sh/uv/) |

---

## Prerequisites

1. **Python 3.13+** installed

2. **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
   ```bash
   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

   # macOS / Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **[Ollama](https://ollama.com/)** installed and running

4. **Required models** pulled into Ollama:
   ```bash
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```

5. **Hardware:** 32 GB RAM recommended for comfortable CPU inference with `llama3.1:8b`

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/cv-rank-agent.git
cd cv-rank-agent

# Install all dependencies
uv sync

# Copy the example environment file
cp .env.example .env    # Linux/macOS
copy .env.example .env  # Windows
```

---

## Configuration

All settings are managed via the `.env` file in the project root. Copy `.env.example` to `.env` and customize as needed:

```env
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.1:8b
EMBEDDING_MODEL=nomic-embed-text
TEMPERATURE=0.0

# Scoring strategy
LLM_ONLY_THRESHOLD=5        # <= this many jobs: LLM scores all (Option A)
LLM_TOP_N=10                # > threshold: LLM deep-scores top N after cosine ranking (Option B)

# Limits
MAX_JOBS=50
```

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `llama3.1:8b` | Model used for CV parsing, job parsing, and scoring |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Model used for cosine similarity embeddings (Option B) |
| `TEMPERATURE` | `0.0` | LLM temperature (0.0 = deterministic output) |
| `LLM_ONLY_THRESHOLD` | `5` | Jobs at or below this count skip the embedder (Option A) |
| `LLM_TOP_N` | `10` | Number of top jobs to deep-score after cosine ranking (Option B) |
| `MAX_JOBS` | `50` | Maximum number of job URLs accepted |

---

## Usage

### 1. Prepare your inputs

- **CV file** — a `.pdf` or `.docx` file containing your resume
- **Jobs file** — a JSON file listing the job URLs to evaluate against

Jobs file format:
```json
{
  "jobs": [
    "https://www.linkedin.com/jobs/view/1234567890",
    "https://www.linkedin.com/jobs/view/0987654321",
    "https://example.com/careers/senior-developer"
  ]
}
```

A sample is provided at [`samples/jobs.json`](samples/jobs.json).

### 2. Make sure Ollama is running

```bash
ollama serve
```

### 3. Run the agent

```bash
uv run python -m cv_rank_agent <path-to-cv> <path-to-jobs-json>
```

**Examples:**
```bash
# Score a PDF CV against sample jobs
uv run python -m cv_rank_agent my_cv.pdf samples/jobs.json

# Score a DOCX CV
uv run python -m cv_rank_agent my_cv.docx samples/jobs.json
```

---

## Output

The agent prints a ranked list of all evaluated jobs to stdout, sorted by overall fit score (highest first):

```
================================================================================
  RANKING RESULTS — 3 job(s) evaluated
================================================================================

  #1  https://www.linkedin.com/jobs/view/1234567890
  ────────────────────────────────────────────────────────────────────────────────
  Overall Fit:       85%
  Skill Match:       90%
  Experience Match:  78%
  Cosine Similarity: 82%          ← only shown in Option B (hybrid mode)
  Gaps:              Kubernetes, AWS certification
  Explanation:       Strong match on core Python and backend skills...

  #2  https://example.com/careers/senior-developer
  ────────────────────────────────────────────────────────────────────────────────
  Overall Fit:       72%
  Skill Match:       80%
  Experience Match:  65%
  Gaps:              React, TypeScript, 5+ years frontend
  Explanation:       Good backend alignment but lacks frontend experience...

================================================================================
```

Each scored job includes:

| Field | Description |
|-------|-------------|
| **Overall Fit** | Weighted assessment of how well the CV matches the role (0–100%) |
| **Skill Match** | How well candidate skills align with job requirements (0–100%) |
| **Experience Match** | Relevance and depth of work experience vs. what's needed (0–100%) |
| **Cosine Similarity** | Vector similarity pre-rank score — *Option B only* (0–100%) |
| **Gaps** | Specific skills, qualifications, or experience the CV is missing |
| **Explanation** | The LLM's full reasoning behind the scores |

---

## Project Structure

```
cv_rank_agent/
├── .env                             # Local config (git-ignored)
├── .env.example                     # Template committed to git
├── .gitignore
├── pyproject.toml                   # Project config & dependencies
├── README.md
├── src/
│   └── cv_rank_agent/               # Main package
│       ├── __init__.py
│       ├── __main__.py              # CLI entry point
│       ├── config.py                # pydantic-settings (reads .env)
│       ├── graph.py                 # LangGraph StateGraph, edges & routing
│       ├── state.py                 # TypedDict state schema for the graph
│       ├── models.py                # Pydantic models (ParsedCV, JobDescription, ScoreResult)
│       ├── nodes/
│       │   ├── cv_parser.py         # Node 1: CV → structured data
│       │   ├── job_parser.py        # Node 2: URLs → structured job descriptions
│       │   ├── embedder.py          # Node 3: embed + cosine rank (Option B)
│       │   └── scorer.py            # Node 4: LLM deep-scoring
│       ├── tools/
│       │   ├── file_load.py         # PDF/DOCX text extraction utilities
│       │   └── web_crawl.py         # Crawl4AI web crawling tool
│       └── prompts/
│           ├── cv_parser.py         # Prompt templates for CV extraction
│           ├── job_parser.py        # Prompt templates for job parsing
│           └── scorer.py            # Prompt templates for scoring/reasoning
├── tests/
│   ├── test_cv_parser.py
│   ├── test_job_parser.py
│   ├── test_embedder.py
│   ├── test_scorer.py
│   └── fixtures/                    # Test data
└── samples/
    └── jobs.json                    # Sample job URLs file
```

### Design Principles

- **`state.py`** — defines what flows through the LangGraph graph (TypedDict), separate from data models
- **`models.py`** — structured data shapes (`ParsedCV`, `JobDescription`, `ScoreResult`), reusable outside the graph
- **`prompts/`** — all LLM prompt text isolated from node logic for easy tuning and iteration
- **`src/` layout** — follows the [PyPA recommended](https://packaging.python.org/en/latest/tutorials/packaging-projects/) project structure

---

## Upgrade Path

If scoring quality is insufficient with `llama3.1:8b`, you can swap to a larger model by updating `LLM_MODEL` in `.env`:

| Model | Size | Trade-off |
|-------|------|-----------|
| `llama3.1:8b` *(default)* | ~4.7 GB | Fast, good quality |
| `qwen2.5:14b` | ~9 GB | Better reasoning |
| `mistral-small:22b` | ~13 GB | Best quality, slower on CPU |

```bash
# Pull the new model
ollama pull qwen2.5:14b
```

```env
# Update .env
LLM_MODEL=qwen2.5:14b
```

---

## Contributing

This is a learning project, but contributions are welcome! Feel free to:

- Open an **issue** to report bugs or suggest features
- Submit a **pull request** with improvements
- Share feedback on the architecture or scoring approach

---

## License

This project is open source. Feel free to use, modify, and distribute it.
