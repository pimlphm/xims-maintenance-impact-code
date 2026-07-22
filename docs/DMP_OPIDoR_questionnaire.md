# MII-CODE — DMP OPIDoR “Software, source code” questionnaire

This document follows the exact five-part **Codes and Softwares** questionnaire shown in DMP OPIDoR for the research output **MII-CODE**.

It does **not** modify or replace the existing **TEP / HI CONSTRUCTION CODE / Simulation data Tennessee Eastman Process** research output.

## 使用说明

- **[选择]**：在下拉框中选择最接近的值；如果没有完全相同的值，选择本文列出的第一个近似值。
- **[逐项添加]**：点击 **Add an element**，每行内容单独添加。
- **[直接粘贴]**：复制英文段落到富文本框。
- **[暂留空]**：目前没有经过确认的 URL、DOI 或 SWHID，不应编造。
- **[需确认]**：保存前由项目负责人确认日期、作者身份或许可决定。

## Research-output identity

- **Short name:** `MII-CODE`
- **Name:** `Code for interpretable maintenance-impact identification using C-MAPSS and NGAFID`
- **Questionnaire:** `Software, source code`
- **Guidance:** `Recherche Data Gouv - Collège Codes sources et logiciels`
- **Current package:** `XIMS_Maintenance_Impact_Code_20260722.zip`

# 1. General description of the software

## 1.1 Definition and name of the software

### Full name of the software

**[直接填写]**

> Code for interpretable maintenance-impact identification using C-MAPSS and NGAFID

### Description of the software

**[直接粘贴]**

> MII-CODE is a research-software package for the interpretable identification and quantification of maintenance-action impacts on system health. It contains sanitized Jupyter notebooks and Python code-cell exports implementing maintenance-event analysis, health-indicator learning, sensor- and operator-level interpretation, and pre-/post-maintenance comparison. The package supports two case studies: controlled maintenance-effect scenarios derived from the NASA C-MAPSS turbofan degradation data and real aviation maintenance analysis using NGAFID. The software is intended for research on predictive maintenance, explainable artificial intelligence, health assessment and reliability engineering. Raw datasets, trained model weights, notebook outputs, publication files and credentials are not distributed. Users must obtain the source datasets from their authoritative providers and configure the documented data-path placeholders before execution.

### Research output type

**[选择]**

> Software

### URL of the site describing the software (or project)

**[暂留空]**

当前没有已确认的公开项目网页或代码仓库 URL。不要填写本地磁盘路径。创建公开仓库或存储库记录后，再填写其稳定 URL。

### Development start date

**[需确认；日期框格式为 mm/dd/yyyy]**

> 04/07/2026

这是当前保存的两份源 notebook 中最早可核实的本地代码日期。若项目记录证明开发开始得更早，应以项目记录为准并替换该日期。

### First version release date

**[当前建议暂留空]**

`07/22/2026` 是净化上传包的制备日期，但不是已经公开发布的证据。只有在项目组将该 ZIP 正式认定为第一版发布物时，才填写：

> 07/22/2026

否则应在首次公开存储库发布时填写实际发布日期。

### Principal areas of application

**[选择；按优先顺序选择界面中存在的最接近值]**

1. Engineering and technology
2. Computer and information sciences
3. Artificial intelligence
4. Aerospace engineering

### Others applications

**[逐项添加]**

- Predictive maintenance
- Maintenance-effect assessment
- Aircraft and turbofan health monitoring
- Reliability and prognostics engineering
- Explainable decision support for maintenance

### Keywords (include vocabulary, thesaurus or lexicon)

**[逐项添加]**

- Predictive maintenance
- Maintenance impact identification
- Explainable artificial intelligence
- Health indicator
- C-MAPSS
- NGAFID
- Kolmogorov-Arnold operator
- Self-supervised learning
- Aviation maintenance
- Reliability engineering

### Software version number

**[建议填写]**

> 0.1.0

当前包仍处于首次公开发布之前，且软件许可、公开仓库和完整复现实验尚未最终确认，因此建议使用预发布版本 `0.1.0`。首次正式公开发布后可改为 `1.0.0`。

### What is (or are) the software typology(ies)?

**[选择或逐项输入]**

- Research software
- Scientific software
- Machine-learning software
- Data-analysis software
- Computational notebooks

### Is your software linked to any other research products?

**[逐项添加]**

1. **NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set** — Dataset used for controlled degradation and maintenance-effect scenarios.
2. **NGAFID public aviation maintenance dataset** — Dataset used for real maintenance-event analysis; public research deposit: `https://doi.org/10.5281/zenodo.6624956`.
3. **Cross scenarios interpretable quantification of maintenance action effects on system health via liquid Kolmogorov-Arnold operator based framework** — Associated journal article; `https://doi.org/10.1016/j.ress.2026.112898`.
4. **Interpretable Maintenance Impact Quantification for Aircraft Using Boltzmann KAN with Self-Supervised Learning** — Associated conference publication; `https://doi.org/10.19124/ima.2025.01.02`.

### Does the development of this software involve costs?

**[建议选择]**

> Yes

如该字段允许说明，可填写：

> Development involves project-funded research staff time and institutional computing and storage resources. Additional effort is required for data curation, testing, documentation, repository preparation and long-term preservation. The distributed package relies on open-source Python libraries and does not currently require a paid runtime licence. No separate software-specific cost line has been identified in this DMP record.

### Contact person

**[保留当前界面中已有人员]**

> Nguyen Thi Phuong Khanh — Contact Person

# 2. Tools and execution environment

## 2.1 Development environment

### Do you use a software forge?

**[当前选择或输入]**

> No — not yet. A Git-based public software forge is planned after rights-holder and licence approval.

如果公开 GitHub、GitLab 或其他 forge 仓库已由项目组创建，则将此项改为相应 forge 名称。

### If so, provide the link to the code repository or the access URL

**[暂留空]**

公开仓库建立后填写仓库主页 URL；不要填写本地 ZIP 路径或临时共享链接。

### How to access the source code?

**[当前选择最接近的值]**

> Restricted or controlled access during pre-release; public access is planned after contributor and licence approval.

若下拉框只有简化选项，当前选择 **Restricted access**。公开存储库发布后改为 **Open access**。

### Do you use a Version Control System (VCS)?

**[当前建议]**

> No formal VCS was identified in the retained source snapshot; Git is planned for the public repository.

不要仅因为建议使用 Git 就把尚未建立的版本库登记为现状。当前包通过版本化 ZIP 名称、源文件清单和 SHA-256 校验值记录来源与完整性。

### What tests were used to verify the code quality?

**[选择界面中存在的最接近值]**

- Syntax or compilation tests
- Functional or smoke tests
- Manual review
- Reproducibility or archive-integrity checks
- Security or sensitive-information scanning

当前 MII-CODE 包没有独立的完整单元测试套件，因此不要选择 **Unit tests**，除非之后确实添加并执行了单元测试。

### Do you use a continuous integration framework?

**[选择]**

> No

### What, if any, is the URL of the continuous integration framework?

**[暂留空]**

### Additional information

**[直接粘贴]**

> The upload package was quality-checked by parsing the Jupyter Notebook files, compiling the exported Python source files, extracting the ZIP into a fresh directory, and scanning the package for machine-specific absolute paths, credential patterns, caches and generated outputs. Stored notebook outputs and execution counters were removed. Duplicate backup notebooks, raw data, trained model weights, generated figures, manuscripts and publication PDFs were excluded. Machine-specific paths were replaced by documented placeholders. The package does not yet contain a dedicated unit-test suite, and full scientific reproduction requires the external C-MAPSS and NGAFID datasets together with the documented path configuration.

## 2.2 Documentation

### Provide links to user and/or developer documentation, if possible

**[暂留空]**

目前文档位于上传 ZIP 内，而不是稳定公开 URL。公开仓库建立后，应逐项添加以下文件的永久链接：

- `README.md`
- `CODE_CATALOG.md`
- `requirements.txt`
- `SOURCE_MANIFEST.csv`
- `SECURITY_AND_LIMITATIONS.md`
- `LICENSE_REVIEW_REQUIRED.md`，或许可确认后替换为正式 `LICENSE`

### Additional information

**[直接粘贴]**

> The software package includes a README describing the purpose, first validation command, external-data requirements and path configuration; a code catalogue describing every packaged notebook and Python export; a requirements file listing the scientific Python dependencies; a source manifest recording provenance, file size, checksum and release status; and security, limitation and licence-review notices. Each Jupyter notebook has a corresponding Python code-cell export to support review outside the notebook interface. Documentation is written in English and distributed with the versioned ZIP. Stable documentation URLs will be added after repository deposit.

## 2.3 Runtime environment

### What programming languages are used in the software?

**[逐项添加]**

- Python

Jupyter Notebook 是执行和记录环境；notebook 内的程序语言仍为 Python，可在下方 Additional information 中说明。

### What operating system(s) allows the code to run?

**[已验证的选择]**

- Microsoft Windows

Python 科学计算栈通常可在 Linux 和 macOS 上运行，但当前净化包尚未完成这些系统上的完整数据驱动验证。因此，在验证之前不要把 Linux/macOS 标记为已验证平台。

### What dependencies are required to use the software?

**[逐项添加]**

- Python 3
- Jupyter Notebook or JupyterLab
- matplotlib
- numpy
- pandas
- scikit-learn
- scipy
- torch (PyTorch)

依赖包目前没有锁定精确版本；项目若发布正式版本，应补充经过验证的 Python 和依赖版本或环境锁文件。

### Additional information

**[直接粘贴]**

> The software is distributed as two Jupyter notebooks and two matching Python code-cell exports. It was prepared and syntax-validated on Microsoft Windows. A Python 3 environment with Jupyter Notebook or JupyterLab is required to execute the notebooks. The declared runtime dependencies are matplotlib, NumPy, pandas, scikit-learn, SciPy and PyTorch. A CUDA-capable GPU may accelerate model training, but the code also contains CPU execution paths; full runs can be computationally demanding. Raw C-MAPSS and NGAFID data and trained weights are not included. Users must obtain the datasets under their original terms and configure the documented path placeholders. Exact dependency versions are not yet pinned.

# 3. Software preservation

## 3.1 Referencing

### Indicate the catalogue(s) where the software is referenced, and its persistent identifier

**[当前暂留空]**

目前没有经过确认的公开软件目录记录或持久标识符。发布后建议逐项添加：

- 存储库或软件目录名称，例如 Zenodo、Recherche Data Gouv 或机构存储库；
- 软件记录的 DOI；
- 如有公开 forge，再添加代码仓库 URL。

### Additional information

**[直接粘贴]**

> No public catalogue record or persistent software identifier has yet been assigned. The preservation package currently consists of the versioned ZIP, README, code catalogue, dependency list, source manifest, security and limitation notes, and SHA-256 integrity information. After contributor and licence approval, a tagged release should be deposited in a trusted research repository that assigns a DOI. The DOI and repository URL will then be recorded in this DMP. Raw third-party datasets and trained model weights are not part of the preserved public software package.

## 3.2 Long-term archiving

### If the software is archived in the Software Heritage archive, indicate its SWHID identifier

**[暂留空]**

当前没有 SWHID。只有在公开源代码仓库被 Software Heritage 收录后，才填写实际 `swh:1:...` 标识符。

# 4. Legal issues

## 4.1 What are the legal aspects related to this software?

### List the authors of the software

**[需确认后添加]**

- Weikun DENG — Software author / Lead developer

这是根据保存代码与关联论文角色形成的建议。DMP 中的“software author”应反映实际代码著作权贡献，保存前需要项目组确认，不能仅凭论文署名自动认定所有论文作者都是软件作者。

### List the main contributors to the software and assign each one a role

**[需项目组确认后逐项添加]**

- Weikun DENG — Conceptualization; Methodology; Software; Investigation; Validation; Documentation
- Nguyen Thi Phuong Khanh — Supervision; Methodology; Validation; Contact person
- Phuc DO — Supervision; Methodology; Validation
- Kamal MEDJAHER — Supervision; Project administration; Validation

这些角色依据关联论文作者信息和项目上下文整理，保存前应由相关人员确认。

### Under what licence(s) is, or will, the software be distributed?

**[当前不要选择尚未批准的开源许可]**

> Licence not yet assigned; rights-holder review required before public distribution.

建议项目组在确认权属和第三方兼容性后考虑 **MIT License**，但在全体权利人批准之前，不能把 MIT 登记为当前许可。许可获批后，应在 ZIP 中加入正式 `LICENSE` 文件，并将本字段改为批准的许可名称。

### Additional information

**[直接粘贴]**

> No explicit software licence was found in the retained source folders. Public distribution therefore requires confirmation of software authorship and rights holders, approval by the contributors and relevant institutions, selection of an explicit software licence, and a compatibility review covering reused libraries, datasets and any third-party material. The package does not redistribute raw C-MAPSS or NGAFID data; users must obtain these datasets from their authoritative sources and comply with their original terms. Third-party Python libraries remain subject to their own licences. No personal data or credentials are intentionally included. Until a licence is approved, the prepared ZIP should be treated as a controlled pre-release package rather than openly reusable software.

# 5. Scientific dissemination

## 5.1 Scientific promotion

### Has information about this software already been published?

**[选择]**

> Yes — the methods and experimental use of the software have been described in associated scientific publications, although the sanitized software package itself has not yet been publicly deposited.

如该字段要求逐项添加信息，可添加：

- Journal article describing the liquid Kolmogorov-Arnold operator framework and the C-MAPSS/NGAFID case studies.
- MIMAR 2025 conference paper describing aircraft maintenance-impact quantification using Boltzmann KAN and self-supervised learning.

### Is this software associated with one or more scientific publications?

**[选择 Yes，并逐项添加]**

1. Weikun Deng, Khanh T. P. Nguyen, Phuc Do and Kamal Medjaher (2026). **Cross scenarios interpretable quantification of maintenance action effects on system health via liquid Kolmogorov-Arnold operator based framework.** *Reliability Engineering & System Safety*, 276, 112898. DOI: `https://doi.org/10.1016/j.ress.2026.112898`.
2. Weikun Deng, Khanh Nguyen and Kamal Medjaher (2025). **Interpretable Maintenance Impact Quantification for Aircraft Using Boltzmann KAN with Self-Supervised Learning.** 13th IMA International Conference on Modelling in Industrial Maintenance and Reliability (MIMAR 2025). DOI: `https://doi.org/10.19124/ima.2025.01.02`.

# 保存前必须确认的项目级信息

以下内容不能仅凭本地代码自动最终决定：

1. 实际开发开始日期是否早于 `04/07/2026`。
2. `07/22/2026` 是否被正式认定为第一版发布日期；否则 First version release date 留空。
3. 软件作者和每位贡献者的 CRediT/项目角色。
4. 软件权利人、机构审批以及最终软件许可。
5. 公开 forge、文档和 CI 的真实 URL。
6. 存储库 DOI、软件目录记录和 Software Heritage SWHID。
7. 首次公开发布时采用 `0.1.0` 还是 `1.0.0`。

# 当前代码包与表单内容的对应关系

- `notebooks/Maintenance_impact_identification_NGFAID.ipynb` — NGAFID 真实航空维护事件的可解释维护影响识别。
- `notebooks/Mimar_extension_simulated_real_maintenance_data_identification_CMPASS.ipynb` — C-MAPSS 受控维护效果情景和维护前后分析。
- `src/*.py` — 两份 notebook 的 Python code-cell 导出，便于审阅和语法验证。
- `README.md`、`CODE_CATALOG.md`、`requirements.txt`、`SOURCE_MANIFEST.csv` — 用户说明、代码目录、依赖和来源/校验信息。
- `SECURITY_AND_LIMITATIONS.md`、`LICENSE_REVIEW_REQUIRED.md` — 安全限制和发布前许可审查说明。
- 不包含原始数据、训练权重、notebook 输出、论文/PDF、缓存、备份、凭据或个人本机绝对路径。
