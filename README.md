# Jeopardy Dataset Analysis & Generation Pipeline

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)
![uv](https://img.shields.io/badge/uv-managed-purple.svg)
![Marimo](https://img.shields.io/badge/made%20with-marimo-orange.svg)

This project contains a set of Marimo notebooks to explore, classify, sampling, and verify subdatasets from Jeopardy questions.

For more details about how I performed the process to generate subdatasets, see [Description.md](Description.md).

**Subdatasets Location:** `generated_samples/`  
**Classified Groups:** `group1_numbers.json`, `group2_non_english.json`, `group3_uncommon_proper_nouns.json`

## Prerequisite

Ensure you have the environment set up. This project uses `uv` for dependency management.

### Install uv

If you haven't installed `uv` yet, run:

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For Windows or other methods, see the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).
```bash
uv sync
```

## Workflow Execution Steps

Run the scripts in the following order to reproduce the full pipeline.

### 1. Data Exploration
**Script:** `exploration.py`

Explores the dataset structure, statistics, and distributions (categories, values, rounds, time).

- **Interactive Mode (Recommended):**
  ```bash
  uv run marimo edit exploration.py
  ```
- **Run as Script:**
  ```bash
  uv run python exploration.py
  ```

### 2. Dataset Classification
**Script:** `classify_dataset.py`

Classifies questions into three groups (Numbers, Non-English, Uncommon Proper Nouns) and exports them to JSON files (`group1_numbers.json`, etc.).

- **Run as Script (Recommended for generation):**
  ```bash
  uv run python classify_dataset.py
  ```
- **Interactive Mode:**
  ```bash
  uv run marimo edit classify_dataset.py
  ```

### 3. Subdataset Generation
**Script:** `generate_subdatasets.py`

Generates stratified samples from the classified groups. Creates a `generated_samples/` directory with the subdatasets and `metadata.json`.

- **Run as Script (Recommended for generation):**
  ```bash
  uv run python generate_subdatasets.py
  ```
- **Interactive Mode:**
  ```bash
  uv run marimo edit generate_subdatasets.py
  ```

### 4. Subdataset Verification
**Script:** `verify_subdatasets.py`

Visually compares the distributions (Round, Category) and calculates KL Divergence between the original groups and the generated subdatasets to ensure quality.

- **Interactive Mode (Recommended for visualization):**
  ```bash
  uv run marimo edit verify_subdatasets.py
  ```