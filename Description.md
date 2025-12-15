# Dataset Curation Process Description

This document outlines the end-to-end process used to curate, classify, and sample the Jeopardy dataset to create high-quality, lightweight benchmarks for Language Models.

## 1. Data Loading
The process begins by loading the full `JEOPARDY_QUESTIONS1.json` dataset.
- I use **Pandas** to ingest the JSON data into a DataFrame.
- Basic exploration is performed (`exploration.py`) to understand the structure, examining columns like 'Category', 'Value', 'Round', and 'Air Date'.

## 2. Classification & Group Treatment
I filter the dataset into three distinct, mutually non-exclusive groups to test different capabilities of Language Models. This logic is implemented in `classify_dataset.py`.

### Group 1: Numbers
*   **Objective**: Isolate questions requiring numerical reasoning or identification.
*   **Method**: Used **Regular Expressions** (`re`) to scan both Question and Answer text.
*   **Result**: Any record containing a digit (`\d`) is classified into this group.

### Group 2: Non-English Words
*   **Objective**: Identify questions involving foreign languages or loanwords.
*   **Method**:
    1.  Tokenized text and filtered out standard English words using **spaCy's** vocabulary (`en_core_web_sm`).
    2.  Filtered out **Acronyms** (defined as all-caps words ≤ 4 characters) to reduce false positives.
    3.  Excluded Proper Nouns (`PROPN` tag) to ensure we capture actual foreign terms, not just names.

### Group 3: Uncommon Proper Nouns
*   **Objective**: Find questions containing "Long Tail" entities (obscure names, places) that models might hallucinate.
*   **Method**:
    1.  **Extraction**: Used **spaCy NER** (Named Entity Recognition) to extract all proper nouns/entities from the entire corpus.
    2.  **Frequency Analysis**: Computed a global frequency count for every entity.
    3.  **Filtration**: Selected questions containing entities that appear **less than 2 times** in the entire dataset (Frequency < 2).

## 3. Subset Creation (Stratified Sampling)
To create lightweight benchmarks (~1000 samples) that statistically represent the huge original groups, I employed **Stratified Sampling** in `generate_subdatasets.py`.

*   **Strategy**: Random sampling is insufficient as it might miss rare categories or bias the difficulty distribution.
*   **Stratification Variables**: I stratified based on a composite key of:
    *   `Category` (Topic)
    *   `Value` (Difficulty/Reward)
    *   `Round` (Game stage)
*   **Implementation**: I used `sklearn.utils.resample` for efficient, replacement-enabled sampling that respects the proportional weight of every stratum in the original group.

*   **Iterative Refinement (KL Divergence)**:
    *   To ensure the generated subcategory is statistically representative, I implemented an iterative check using **Kullback-Leibler (KL) Divergence**.
    *   The script calculates the KL Divergence for 'category', 'round', and 'value' between the sample and the original group.
    *   If the divergence exceeds a threshold (0.75), the sampling is retried (up to 10 attempts) to find a subset that minimizes distribution drift.



## 4. Distribution Verification & Differences
The quality of the curation is validated in `verify_subdatasets.py`. I explicitly measure how close the subsets are to the original data.

*   **Visual Comparison**: I generate side-by-side histograms for "Round" and "Top 20 Categories".
    *   *Result*: The charts show near-identical shape, confirming that the subdatasets preserve the structural characteristics of the original data.
*   **Statistical Metric (KL Divergence)**: I calculate the **Kullback-Leibler (KL) Divergence** to quantify information loss.
    *   Low KL scores (near 0) indicate high representativeness.
    *   I generate multiple candidate subdatasets (e.g., 20) and allow selection of the one with the lowest KL score, resulting in a subset that is statistically indistinguishable from the original group but smaller in size.
