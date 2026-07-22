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
- Public reuse remains subject to dataset terms, contributor approval, and selection of a software license.

## Repository status

This repository is a public, sanitized research-software pre-release. No explicit software licence has yet been approved. Public visibility does not by itself grant general permission to copy, modify or redistribute the software; those reuse terms will be defined after the rights holders and contributors select a licence and confirm compatibility with the reused datasets and libraries.

Current tagged pre-release: `v0.2.0` (2026-07-23).

The DMP OPIDoR field-by-field software questionnaire is available at `docs/DMP_OPIDoR_questionnaire.md`.

## Web GUI

Open the public, browser-based maintenance-impact explorer:

**https://pimlphm.github.io/xims-maintenance-impact-code/**

The GUI can generate a synthetic example or analyse a numeric signal from a local CSV file. Data remains in the browser and is not uploaded. The interface reports pre-/post-maintenance means, absolute and relative change, standardized effect size, an interpretation, and an interactive trajectory plot. It is an exploratory companion to the research code, not a substitute for the full C-MAPSS or NGAFID workflows.

## Standalone software

MII-CODE is distributed as standalone software: it does not require a host application or plug-in framework. Run the graphical interface locally with only Python's standard library:

```powershell
python launch_gui.py
```

The command starts a local-only web server and opens the same interface published through GitHub Pages. Use `python launch_gui.py --help` for host, port, browser, and directory options. The standalone GUI remains separate from the data-intensive research notebooks; it provides a bounded, auditable maintenance-event comparison without redistributing C-MAPSS or NGAFID.

## Architecture and interfaces

- [Software architecture](docs/SOFTWARE_ARCHITECTURE.md)
- [Live standalone interface](https://pimlphm.github.io/xims-maintenance-impact-code/)
- CSV interface: header row plus one or more numeric signal columns, maximum 5 MB in the public GUI
- JavaScript analysis interface: `docs/assets/analysis.mjs`
- Standalone launcher interface: `launch_gui.py`
