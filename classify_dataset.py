import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo  # For interactive notebook UI and markdown
    import pandas as pd  # For DataFrame manipulation and data analysis
    import numpy as np  # For high-performance numerical operations
    import spacy  # For NLP tasks like Named Entity Recognition and POS tagging
    import re  # For regex-based text pattern matching (finding numbers, etc.)
    import time  # To track the duration of processing steps
    from collections import Counter  # To count frequencies of proper nouns efficiently
    return Counter, mo, np, pd, re, spacy, time


@app.cell
def __(mo):
    mo.md("""
    # Jeopardy Dataset Classification
    
    This notebook classifies the Jeopardy dataset into three groups:
    1. **Group 1**: Questions/answers containing numbers
    2. **Group 2**: Questions/answers containing non-English words (excluding acronyms)
    3. **Group 3**: Questions/answers containing uncommon proper nouns
    """)
    return


@app.cell
def __(pd):
    # Load the dataset
    df = pd.read_json('JEOPARDY_QUESTIONS1.json')
    print(f"Loaded {len(df):,} questions")
    return df,


@app.cell
def __(spacy):
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model loaded successfully")
    return nlp,


@app.cell
def __(mo):
    mo.md("## Classification Functions")
    return


@app.cell
def __(re):
    def has_numbers(text):
        """
        Check if text contains any numbers.

        Args:
            text: The input text to check.

        Returns:
            bool: True if the text contains a digit, False otherwise.
        """
        if pd.isna(text):
            return False
        return bool(re.search(r'\d', str(text)))
    return has_numbers,


@app.cell
def __(nlp, re):
    def has_non_english_words(text):
        """
        Detect non-English words in the text, excluding acronyms.

        Acronyms are defined as all-caps words or very short (<=4 chars).

        Args:
            text: The input text to check.

        Returns:
            bool: True if non-English words are found, False otherwise.
        """
        if pd.isna(text):
            return False
        
        text = str(text)
        
        # Common English words set (using spaCy's vocabulary)
        # Split into words
        words = re.findall(r'\b[A-Za-z]+\b', text)
        
        for word in words:
            # Skip acronyms (all caps and short or no vowels)
            if word.isupper() and (len(word) <= 4):
                continue
            
            # Check if word is in English vocabulary
            # Use spaCy's vocabulary check
            if word.lower() not in nlp.vocab.strings:
                # Additional check: if it's a proper noun, it might still be valid
                doc = nlp(word)
                if not any(token.pos_ == "PROPN" for token in doc):
                    return True
        
        return False
    return has_non_english_words,


@app.cell
def __(Counter, nlp):
    def extract_proper_nouns_batch(texts, batch_size=1000):
        """
        Extract proper nouns from a list of texts using spaCy's efficient pipe() method.

        Args:
            texts (list): A list of text strings.
            batch_size (int, optional): Batch size for processing. Defaults to 1000.

        Returns:
            list: A list of lists, where each inner list contains proper nouns found in the corresponding text.
        """
        import pandas as pd
        
        results = []
        
        # Filter out NaN values and convert to strings
        processed_texts = []
        indices = []
        for idx, text in enumerate(texts):
            if pd.notna(text):
                processed_texts.append(str(text))
                indices.append(idx)
            else:
                results.append([])  # Placeholder for NaN
        
        # Use spaCy's pipe for efficient batch processing
        # disable=['parser', 'lemmatizer'] speeds up processing since we only need NER and POS
        docs = nlp.pipe(processed_texts, batch_size=batch_size, disable=['lemmatizer'])
        
        proper_nouns_list = []
        for doc in docs:
            proper_nouns = []
            
            # Get PROPN tokens
            for token in doc:
                if token.pos_ == "PROPN":
                    proper_nouns.append(token.text.lower())
            
            # Get named entities
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"]:
                    proper_nouns.append(ent.text.lower())
            
            proper_nouns_list.append(proper_nouns)
        
        return proper_nouns_list
    
    return extract_proper_nouns_batch,


@app.cell
def __(mo):
    mo.md("## Applying Classifications")
    return


@app.cell
def __(df, has_non_english_words, has_numbers, mo):
    # Apply classifications
    mo.status.spinner(title="Classifying questions...")
    
    # Combine question and answer for analysis
    df['combined_text'] = df['question'].fillna('') + ' ' + df['answer'].fillna('')
    
    # Group 1: Contains numbers
    df['has_numbers'] = df['combined_text'].apply(has_numbers)
    
    # Group 2: Contains non-English words
    df['has_non_english'] = df['combined_text'].apply(has_non_english_words)
    
    print("Classifications for Group 1 and 2 complete")
    return


@app.cell
def __(mo):
    mo.md("### Computing Proper Noun Frequencies")
    return


@app.cell
def __(Counter, df, extract_proper_nouns_batch, time):
    # First pass: Extract all proper nouns and compute frequencies using batch processing
    print("Extracting proper nouns from all questions (using fast batch processing)...")
    print("This may take a few minutes...")
    
    start_time = time.time()
    
    all_proper_nouns = []
    processing_batch_size = 5000  # Process 5000 texts at a time
    
    for i in range(0, len(df), processing_batch_size):
        batch_texts = df['combined_text'].iloc[i:i+processing_batch_size].tolist()
        batch_nouns_list = extract_proper_nouns_batch(batch_texts, batch_size=256)
        
        # Flatten the results
        for nouns in batch_nouns_list:
            all_proper_nouns.extend(nouns)
        
        if i % 10000 == 0:
            elapsed = time.time() - start_time
            print(f"Extracted from {i:,} / {len(df):,} questions ({elapsed:.1f}s elapsed)")
    
    # Count frequencies
    proper_noun_freq = Counter(all_proper_nouns)
    total_time = time.time() - start_time
    print(f"✅ Extraction complete in {total_time:.1f} seconds")
    print(f"Found {len(proper_noun_freq):,} unique proper nouns")
    print(f"Total proper noun occurrences: {sum(proper_noun_freq.values()):,}")
    return all_proper_nouns, batch_nouns_list, i, nouns, processing_batch_size, proper_noun_freq, start_time, total_time


@app.cell
def __(mo):
    mo.md("### Classifying Uncommon Proper Nouns")
    return


@app.cell
def __(df, extract_proper_nouns_batch, proper_noun_freq, time):
    # Second pass: Check for uncommon proper nouns using batch processing
    common_threshold = 2
    print(f"Classifying questions with uncommon proper nouns (threshold < {common_threshold})...")
    print("Using fast batch processing...")
    
    _start_time = time.time()
    
    _results = []
    _processing_batch_size = 5000
    
    for _i in range(0, len(df), _processing_batch_size):
        _batch_texts = df['combined_text'].iloc[_i:_i+_processing_batch_size].tolist()
        _batch_nouns_list = extract_proper_nouns_batch(_batch_texts, batch_size=256)
        
        # Check each text's proper nouns against frequency threshold
        for _nouns in _batch_nouns_list:
            if not _nouns:
                _results.append(False)
            else:
                # Check if any proper noun is uncommon
                _has_uncommon = any(proper_noun_freq.get(noun, 0) < common_threshold for noun in _nouns)
                _results.append(_has_uncommon)
        
        if _i % 10000 == 0:
            _elapsed = time.time() - _start_time
            print(f"Classified {_i:,} / {len(df):,} questions ({_elapsed:.1f}s elapsed)")
    
    df['has_proper_nouns'] = _results
    _total_time = time.time() - _start_time
    print(f"✅ Classification complete in {_total_time:.1f} seconds")
    return _batch_nouns_list, _batch_texts, common_threshold, _elapsed, _has_uncommon, _i, _nouns, _processing_batch_size, _results, _start_time, _total_time


@app.cell
def __(df):
    # Create the three groups
    group1_numbers = df[df['has_numbers']].copy()
    group2_non_english = df[df['has_non_english']].copy()
    group3_proper_nouns = df[df['has_proper_nouns']].copy()
    
    # Remove the helper columns before export
    columns_to_keep = ['category', 'air_date', 'question', 'value', 'answer', 'round', 'show_number']
    
    group1_export = group1_numbers[columns_to_keep]
    group2_export = group2_non_english[columns_to_keep]
    group3_export = group3_proper_nouns[columns_to_keep]
    return (
        columns_to_keep,
        group1_export,
        group1_numbers,
        group2_export,
        group2_non_english,
        group3_export,
        group3_proper_nouns,
    )


@app.cell
def __(group1_numbers, group2_non_english, group3_proper_nouns, mo):
    mo.md(f"""
    ## Classification Results
    
    - **Group 1 (Numbers)**: {len(group1_numbers):,} questions ({len(group1_numbers)/len(df)*100:.1f}%)
    - **Group 2 (Non-English Words)**: {len(group2_non_english):,} questions ({len(group2_non_english)/len(df)*100:.1f}%)
    - **Group 3 (Uncommon Proper Nouns)**: {len(group3_proper_nouns):,} questions ({len(group3_proper_nouns)/len(df)*100:.1f}%)
    """)
    return


@app.cell
def __(mo):
    mo.md("### Sample from Group 1 (Contains Numbers)")
    return


@app.cell
def __(group1_numbers, mo):
    mo.ui.table(group1_numbers[['category', 'question', 'answer']].head(10))
    return


@app.cell
def __(mo):
    mo.md("### Sample from Group 2 (Contains Non-English Words)")
    return


@app.cell
def __(group2_non_english, mo):
    mo.ui.table(group2_non_english[['category', 'question', 'answer']].head(10))
    return


@app.cell
def __(mo):
    mo.md("### Sample from Group 3 (Contains Uncommon Proper Nouns)")
    return


@app.cell
def __(group3_proper_nouns, mo):
    mo.ui.table(group3_proper_nouns[['category', 'question', 'answer']].head(10))
    return


@app.cell
def __(mo):
    mo.md("## Export Results")
    return


@app.cell
def __(group1_export, group2_export, group3_export):
    # Export to JSON files
    group1_export.to_json('group1_numbers.json', orient='records', indent=2)
    group2_export.to_json('group2_non_english.json', orient='records', indent=2)
    group3_export.to_json('group3_uncommon_proper_nouns.json', orient='records', indent=2)
    
    print("✅ Exported all groups to JSON files:")
    print(f"  - group1_numbers.json ({len(group1_export):,} records)")
    print(f"  - group2_non_english.json ({len(group2_export):,} records)")
    print(f"  - group3_uncommon_proper_nouns.json ({len(group3_export):,} records)")
    return


if __name__ == "__main__":
    #initilization
    app.run()
