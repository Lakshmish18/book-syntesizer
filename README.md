# Book Synthesizer

A research-document synthesis pipeline that indexes source PDFs, builds an outline, retrieves relevant evidence, deduplicates overlapping content, and generates a structured book-style output. The project combines local PageIndex and ChatIndex modules with OpenAI-backed generation, persistent state files, and validation stages for repeatable long-form synthesis experiments.

> Note: the repository name is currently `book-syntesizer`; the documentation uses the clearer product spelling "Book Synthesizer".

## What This Project Demonstrates

- Multi-stage AI pipeline orchestration
- Retrieval and deduplication over source PDFs
- Outline-driven long-form generation
- State persistence for resume/retry workflows
- Reconstruction mode for query-driven knowledge graph style outputs
- Basic validation and test coverage for reconstruction modules

## Pipeline Overview

```text
PDF corpus
  |
  v
Index stage
  |
  +-- PageIndex document structure
  +-- document registry/state
  v
Outline stage
  |
  v
Retrieve stage
  |
  v
Deduplicate stage
  |
  v
Generate stage
  |
  v
Validate stage -> outputs/final_book.md or PDF artifacts
```

## Features

- Runs a full synthesis pipeline or individual stages with `--stage`.
- Supports resume mode to skip completed stateful stages.
- Provides reconstruction mode for focused query-based outputs.
- Persists intermediate state under `book_synthesiser/state/`.
- Writes generated artifacts under `book_synthesiser/outputs/`.
- Includes system checks for API keys, PDF availability, PageIndex imports, and ChatIndex imports.

## Tech Stack

| Area | Tools |
| --- | --- |
| Runtime | Python |
| LLM | OpenAI SDK / openai-agents |
| Retrieval helpers | Local PageIndex and ChatIndex modules |
| Data processing | NumPy, tqdm |
| Configuration | python-dotenv |
| Tests | pytest-compatible test files |

## Repository Structure

```text
.
├── README.md
└── book_synthesiser/
    ├── main.py
    ├── indexer.py
    ├── outline_builder.py
    ├── retrieval_agent.py
    ├── deduplicator.py
    ├── book_generator.py
    ├── reconstruction_agent.py
    ├── schema_validator.py
    ├── PageIndex/
    ├── ChatIndex/
    ├── pdfs/
    ├── state/
    ├── outputs/
    └── tests/
```

## Getting Started

### Prerequisites

- Python 3.10+
- Source PDFs in `book_synthesiser/pdfs/`
- API keys for the workflows you run

### Installation

```bash
git clone https://github.com/Lakshmish18/book-syntesizer.git
cd book-syntesizer
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r book_synthesiser/requirements.txt
```

### Configuration

Create `book_synthesiser/.env` or export environment variables in your shell:

```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key_if_required_by_selected_flow
```

The current system check reports both keys, but individual stages may only require the provider used by that stage.

## Usage

Run a system check:

```bash
python book_synthesiser/main.py --test
```

Run the full synthesis pipeline:

```bash
python book_synthesiser/main.py --stage all
```

Resume from saved state:

```bash
python book_synthesiser/main.py --stage all --resume
```

Run a single stage:

```bash
python book_synthesiser/main.py --stage index
python book_synthesiser/main.py --stage outline
python book_synthesiser/main.py --stage retrieve
python book_synthesiser/main.py --stage dedup
python book_synthesiser/main.py --stage generate
python book_synthesiser/main.py --stage validate
```

Run reconstruction mode:

```bash
python book_synthesiser/main.py --mode reconstruct --query "Build a complete cache-memory concept map"
```

## Generated Files

The pipeline writes generated and intermediate files to:

- `book_synthesiser/state/` for registries, outlines, retrieval results, validation results, and logs
- `book_synthesiser/outputs/` for final generated artifacts
- `book_synthesiser/logs/` for run-specific JSON logs
- `book_synthesiser/workspace/` for PageIndex workspace artifacts

These outputs are useful for debugging, but large generated artifacts should be reviewed before being committed.

## Testing

```bash
python -m pytest book_synthesiser/tests
```

Some tests or stages may require local PDFs and provider credentials.

## Production Readiness Notes

Recommended next steps:

- Rename the repository to `book-synthesizer` for discoverability.
- Move committed runtime state, logs, generated PDFs, and workspace outputs into ignored sample/output directories.
- Add a `.env.example` that documents required and optional keys.
- Convert path setup into a package-friendly import structure instead of mutating `sys.path` in entry points.
- Add tests for stage skipping, validation behavior, and error handling around malformed state files.
- Add a small deterministic fixture corpus for CI that does not require private or large PDFs.

## Security and Data Notes

- Do not commit private source PDFs, API keys, or generated content that contains restricted material.
- Treat synthesized output as AI-assisted draft content that requires human review.
- Keep provider keys in environment variables or a secret manager.

## Contact

Lakshmish M Devadiga

- GitHub: [Lakshmish18](https://github.com/Lakshmish18)
- LinkedIn: [lakshmish-m-devadiga](https://www.linkedin.com/in/lakshmish-m-devadiga)