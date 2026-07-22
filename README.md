# X-IMS maintenance-impact identification code

## Purpose

Sanitized Jupyter notebooks and Python exports for interpretable maintenance-impact identification using public C-MAPSS and NGAFID research data.

## First validation command

```powershell
python -m compileall src
```

Install optional dependencies with `python -m pip install -r requirements.txt` in an isolated environment.

## Data and path configuration

Raw C-MAPSS and NGAFID files are not included; obtain them from their authoritative public sources.

Local absolute paths in the working copies were replaced with placeholders such as `<TEP_CASE_DIR>`, `<NGAFID_DATA_DIR>`, `<DATA_ROOT>`, and `<FIGURE_SOURCE_DIR>`. Search for angle-bracket placeholders and configure them for the target machine before execution.

## Release status

- Notebook outputs and execution counters were removed.
- Caches, generated outputs, raw data, weights, backups, manuscripts, and publication PDFs were excluded.
- No credential value is intentionally included; optional Kimi access reads environment variables only.
- Public release remains subject to dataset terms, contributor approval, and selection of a software license.

## Repository status

This repository is a private, sanitized research-software pre-release. No explicit software licence has yet been approved, so the repository must not be made public or redistributed until the rights holders and contributors select a licence and confirm compatibility with the reused datasets and libraries.

The DMP OPIDoR field-by-field software questionnaire is available at `docs/DMP_OPIDoR_questionnaire.md`.
