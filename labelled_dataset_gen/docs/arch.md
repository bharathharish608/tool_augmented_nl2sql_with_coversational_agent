# High-Level Architecture for NL2SQL Dataset Generation (TPCDS)

## Overview
This architecture outlines the modular Python scripts and components required to generate a high-quality, fully-covered NL2SQL dataset as described in `approach.md`. The design ensures full schema coverage, diversity, quality, and extensibility.

---

## 1. **Core Modules & Scripts**

### a. **Schema Loader**
- **File:** `schema_loader.py`
- **Responsibility:** Load and parse the TPCDS schema from JSON or other formats. Provide schema access to other modules.

### b. **Coverage Tracker**
- **File:** `coverage_tracker.py`
- **Responsibility:** Track which tables and columns have been used in generated queries. Expose functions to update and report coverage.

### c. **Prompt Generator**
- **File:** `prompt_generator.py`
- **Responsibility:** Construct LLM prompts with rotating schema context, business categories, and complexity. Enforce prompt template guidelines.

### d. **LLM Interface**
- **File:** `llm_interface.py`
- **Responsibility:** Handle all interactions with LLMs (OpenAI, local models, etc.), including query generation and paraphrasing.

### e. **Paraphrase Generator**
- **File:** `paraphrase_generator.py`
- **Responsibility:** Generate multiple NL paraphrases for each SQL using LLMs or paraphrase models.

### f. **Negative Pair Generator**
- **File:** `contrastive_pair_generator.py`
- **Responsibility:** Generate negative NL/SQL pairs for contrastive learning.

### g. **Validator**
- **File:** `validator.py`
- **Responsibility:** Validate SQL queries (syntax, schema adherence), NL clarity, and metadata completeness.

### h. **Metadata Annotator**
- **File:** `metadata_annotator.py`
- **Responsibility:** Annotate each example with tables/columns used, business intent, complexity, etc.

### i. **Dataset Writer**
- **File:** `dataset_writer.py`
- **Responsibility:** Store generated data in the desired format (JSONL, CSV, etc.), ensuring deduplication and reproducibility.

### j. **Reporting & QA**
- **File:** `reporting.py`
- **Responsibility:** Generate reports on schema coverage, dataset statistics, and quality metrics. Support human review workflows.

### k. **Configuration & Utilities**
- **File:** `config.py`, `utils.py`
- **Responsibility:** Centralized configuration management (e.g., with Pydantic), random seed control, and shared utility functions.

---

## 2. **Pipeline Flow**

1. **Load Schema** (`schema_loader.py`)
2. **Initialize Coverage Tracker** (`coverage_tracker.py`)
3. **For each batch:**
    - Generate prompt (`prompt_generator.py`)
    - Generate NL/SQL pairs via LLM (`llm_interface.py`)
    - Generate paraphrases (`paraphrase_generator.py`)
    - Generate negative pairs (`contrastive_pair_generator.py`)
    - Validate outputs (`validator.py`)
    - Annotate metadata (`metadata_annotator.py`)
    - Update coverage tracker
    - Write to dataset (`dataset_writer.py`)
4. **After generation:**
    - Run deduplication and QA
    - Generate coverage and quality reports (`reporting.py`)

---

## 3. **Extensibility & Integration**
- All modules should be designed for easy extension to new schemas, SQL dialects, or LLM providers.
- Support for plug-in paraphrase and validation models.
- Configurable via `config.py` (Pydantic recommended).

---

## 4. **Example Directory Structure**

```
labelled_dataset_gen/
├── main.py                # Entry point for the full pipeline
├── schema_loader.py
├── coverage_tracker.py
├── prompt_generator.py
├── llm_interface.py
├── paraphrase_generator.py
├── contrastive_pair_generator.py
├── validator.py
├── metadata_annotator.py
├── dataset_writer.py
├── reporting.py
├── config.py
├── utils.py
└── docs/
    ├── approach.md
    └── arch.md
```