# MII-CODE — DMP OPIDoR popup fields v2

Updated: 2026-07-22  
Public repository: https://github.com/pimlphm/xims-maintenance-impact-code  
Versioned public release: https://github.com/pimlphm/xims-maintenance-impact-code/releases/tag/v0.1.0

This supplement gives one value for every field shown in the DMP OPIDoR popups. Add each record separately.

## 1. Documentation — Linked references

### Record 1 — README and user guide

- **Resource title:** `MII-CODE README and user guide`
- **Resource identifier:** `https://github.com/pimlphm/xims-maintenance-impact-code/blob/main/README.md`
- **Resource identifier type:** `URL`
- **Audience:** `Users and developers`

### Record 1A — Interactive web GUI

- **Resource title:** `MII-CODE interactive maintenance-impact web interface`
- **Resource identifier:** `https://pimlphm.github.io/xims-maintenance-impact-code/`
- **Resource identifier type:** `URL`
- **Audience:** `Researchers, users and developers`

### Record 2 — Code catalogue

- **Resource title:** `MII-CODE code catalogue`
- **Resource identifier:** `https://github.com/pimlphm/xims-maintenance-impact-code/blob/main/CODE_CATALOG.md`
- **Resource identifier type:** `URL`
- **Audience:** `Developers and code reviewers`

### Record 3 — Dependency declaration

- **Resource title:** `MII-CODE Python dependency declaration`
- **Resource identifier:** `https://github.com/pimlphm/xims-maintenance-impact-code/blob/main/requirements.txt`
- **Resource identifier type:** `URL`
- **Audience:** `Users and developers`

### Record 4 — Source and checksum manifest

- **Resource title:** `MII-CODE source and integrity manifest`
- **Resource identifier:** `https://github.com/pimlphm/xims-maintenance-impact-code/blob/main/SOURCE_MANIFEST.csv`
- **Resource identifier type:** `URL`
- **Audience:** `Developers, reviewers and repository curators`

### Record 5 — Security and limitations

- **Resource title:** `MII-CODE security, data-exclusion and limitation notice`
- **Resource identifier:** `https://github.com/pimlphm/xims-maintenance-impact-code/blob/main/SECURITY_AND_LIMITATIONS.md`
- **Resource identifier type:** `URL`
- **Audience:** `Users, developers and data stewards`

### Record 6 — DMP questionnaire

- **Resource title:** `MII-CODE DMP OPIDoR software questionnaire`
- **Resource identifier:** `https://github.com/pimlphm/xims-maintenance-impact-code/blob/main/docs/DMP_OPIDoR_questionnaire.md`
- **Resource identifier type:** `URL`
- **Audience:** `Project members and data stewards`

### Documentation — Additional information

> The public repository provides user, developer, integrity, security and data-management documentation. The README explains the software purpose, installation entry point, external-data requirements and path configuration. The code catalogue maps the packaged notebooks and Python exports. The requirements file records the declared dependencies, while the source manifest records provenance and integrity information. Security, data-exclusion, reuse and known limitations are documented separately. Documentation is written in English and versioned with the public source release.

## 2. Development cost popup

Do not keep `Training` unless training is genuinely the cost being recorded.

- **Cost type:** `Personnel / staff time` — select the closest available value, or type this value.
- **Amount:** `[enter the amount approved in the X-IMS/ANR project budget]`
- **Currency:** `EUR`
- **Title:** `MII-CODE research-software development, documentation and preservation`
- **Description:** `Development involves project-funded research staff time and institutional computing and storage resources. Additional effort is required for data curation, testing, documentation, repository preparation and long-term preservation. The distributed package relies on open-source Python libraries and does not currently require a paid runtime licence. The amount reported here corresponds only to the software-related budget line identified by the project and does not represent the value of third-party datasets.`

The exact amount cannot be inferred from the code. It must come from the approved project budget. Do not invent an amount. If no separate software budget line exists and the form requires a numeric direct cost, record `0 EUR` only if the title is changed to `No separately identified direct software cost` and the description explicitly states that staff time and institutional resources are provided in kind.

## 3. Programming language popup

- **Name of the programming language:** `Python`
- **Version of the programming language:** `Python 3 (minor version not pinned in v0.1.0)`

Jupyter Notebook is an execution/documentation environment, not a second programming language.

## 4. Dependencies popups

Add one record per dependency. Versions are deliberately reported as unpinned because `v0.1.0` does not contain a lock file.

| Dependency Name | Dependency version | Dependency URI |
| --- | --- | --- |
| JupyterLab or Jupyter Notebook | Not pinned in v0.1.0 | https://jupyter.org/ |
| matplotlib | Not pinned in v0.1.0 | https://pypi.org/project/matplotlib/ |
| NumPy | Not pinned in v0.1.0 | https://pypi.org/project/numpy/ |
| pandas | Not pinned in v0.1.0 | https://pypi.org/project/pandas/ |
| scikit-learn | Not pinned in v0.1.0 | https://pypi.org/project/scikit-learn/ |
| SciPy | Not pinned in v0.1.0 | https://pypi.org/project/scipy/ |
| PyTorch | Not pinned in v0.1.0 | https://pypi.org/project/torch/ |

### Runtime environment — Additional information

> MII-CODE is distributed as two Jupyter notebooks and two corresponding Python code-cell exports. It was prepared and syntax-validated on Microsoft Windows. A Python 3 environment with Jupyter Notebook or JupyterLab is required. The declared external dependencies are matplotlib, NumPy, pandas, scikit-learn, SciPy and PyTorch; exact versions are not pinned in v0.1.0. A CUDA-capable GPU may accelerate training, but CPU paths are present. Raw C-MAPSS and NGAFID data and trained weights are not distributed and must be obtained separately under the providers' terms.

## 5. Software preservation — catalogue/repository popup

- **Select or add the catalogue (or repository) in which the software is referenced:** `GitHub`
- **Indicate the persistent identifier:** `https://github.com/pimlphm/xims-maintenance-impact-code/releases/tag/v0.1.0`
- **Then specify the ID type:** `URL`

### Referencing — Additional information

> Version 0.1.0 is publicly referenced by a version-specific GitHub release URL. No software DOI or Software Heritage persistent identifier has yet been assigned. A later approved release should be deposited in a trusted research repository such as Zenodo or Recherche Data Gouv to obtain a DOI; the DMP record must then be updated without replacing the existing GitHub release reference.

### Software Heritage field

Leave the SWHID field empty until an actual `swh:1:...` identifier has been verified.

## 6. Scientific dissemination popups

### Has information about this software already been published? — Record 1

- **Resource title:** `MII-CODE public source-code release v0.1.0`
- **Resource identifier:** `https://github.com/pimlphm/xims-maintenance-impact-code/releases/tag/v0.1.0`
- **Resource identifier type:** `URL`
- **Audience:** `Researchers, users and developers` — fill this only if the popup displays an Audience field.

### Associated publication — Record 1

- **Resource title:** `Cross scenarios interpretable quantification of maintenance action effects on system health via liquid Kolmogorov–Arnold operator based framework`
- **Resource identifier:** `10.1016/j.ress.2026.112898`
- **Resource identifier type:** `DOI`

### Associated publication — Record 2

- **Resource title:** `Interpretable Maintenance Impact Quantification for Aircraft Using Boltzmann KAN with Self-Supervised Learning`
- **Resource identifier:** `10.19124/ima.2025.01.02`
- **Resource identifier type:** `DOI`

## 7. Values that must remain unclaimed

- Software DOI: not assigned.
- Software Heritage SWHID: not assigned.
- Exact dependency versions: not pinned.
- Development cost amount: must be taken from the approved budget.
- Software licence: not yet assigned; public visibility alone does not grant general reuse permission.
