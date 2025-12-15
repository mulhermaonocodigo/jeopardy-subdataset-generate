import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo  # For interactive notebook interface and UI elements
    import pandas as pd  # Data manipulation and reading/writing JSON files
    import numpy as np  # Numerical operations for array handling
    import json  # Reading and writing metadata JSON files
    from pathlib import Path  # Object-oriented filesystem paths
    import gc  # Garbage collection to free up memory manually
    from sklearn.utils import resample  # Efficient stratified sampling with replacement
    import scipy.stats as stats  # For KL divergence calculation
    return Path, gc, json, mo, np, pd, resample, stats


@app.cell
def __(mo):
    mo.md("""
    # Subdataset Generation for LLM Validation
    
    This notebook generates multiple representative 1000-sample subdatasets from each classified group using stratified sampling.
    
    **Sampling Strategy**: Stratified sampling based on Category, Value, and Round with replacement.
    """)
    return


@app.cell
def __(mo):
    mo.md("## Load Classified Groups")
    return


@app.cell
def __(pd):
    # Load the three group files
    group1 = pd.read_json('group1_numbers.json')
    group2 = pd.read_json('group2_non_english.json')
    group3 = pd.read_json('group3_uncommon_proper_nouns.json')
    
    print(f"Group 1 (Numbers): {len(group1):,} samples")
    print(f"Group 2 (Non-English): {len(group2):,} samples")
    print(f"Group 3 (Proper Nouns): {len(group3):,} samples")
    return group1, group2, group3


@app.cell
def __(mo):
    mo.md("## Stratified Sampling Function")
    return


@app.cell
def __(resample):
    def create_stratified_sample(df, sample_size=1000, random_seed=None, stratify=None):
        """
        Create a random sample using sklearn's resample for optimal performance.
        Uses sampling with replacement.
        """
        # sklearn's resample is highly optimized and much faster
        return resample(df, n_samples=sample_size, replace=True, random_state=random_seed, stratify=stratify)
    
    return create_stratified_sample,


@app.cell
def __(mo):
    mo.md("## Generate Subdatasets")
    return


@app.cell
def __(Path, create_stratified_sample, gc, group1, group2, group3, json):
    # Function to calculate KL Divergence
    def get_kl_divergence(series_p, series_q, top_n=None):
        """Calculate KL Divergence between two series distributions.
        If top_n is specified, calculates based on the top N most frequent items in P.
        """
        # Calculate probabilities
        p_counts = series_p.value_counts(normalize=True)
        q_counts = series_q.value_counts(normalize=True)
        
        if top_n:
            # Focus on top N categories from the original distribution
            top_indices = p_counts.head(top_n).index
            p_counts = p_counts.reindex(top_indices)
            q_counts = q_counts.reindex(top_indices, fill_value=0)
            
            # Re-normalize to sum to 1 for just these top N
            p_counts = p_counts / p_counts.sum()
            q_counts = q_counts / q_counts.sum()
        
        # Align indexes (intersection if top_n used, union otherwise)
        all_cats = p_counts.index.union(q_counts.index)
        p_probs = p_counts.reindex(all_cats, fill_value=0).values
        q_probs = q_counts.reindex(all_cats, fill_value=0).values
        
        # Add epsilon and normalize
        epsilon = 1e-9
        p_probs = (p_probs + epsilon) / (p_probs.sum() + epsilon * len(p_probs))
        q_probs = (q_probs + epsilon) / (q_probs.sum() + epsilon * len(q_probs))
        
        return stats.entropy(p_probs, q_probs)

    # Function to generate subdataset with validation
    def generate_valid_subdataset(group_df, sample_size=1000, seed_base=0, max_retries=10, kl_threshold=0.75):
        """Generate a subdataset and validate its distribution"""
        best_subdataset = None
        best_kl_details = {}
        lowest_max_kl = float('inf')
        
        for attempt in range(max_retries):
            # Generate sample
            current_seed = seed_base + (attempt * 1000)
            # Stratify by ROUND to guarantee good distribution there
            subdataset = create_stratified_sample(
                group_df, 
                sample_size=sample_size, 
                random_seed=current_seed, 
                stratify=group_df['round']  # EXPLICIT STRATIFICATION
            )
            
            # Check Distributions
            # Use Top 50 for high cardinality columns to avoid "missing tail" penalty
            kl_cat = get_kl_divergence(group_df['category'], subdataset['category'], top_n=50)
            kl_round = get_kl_divergence(group_df['round'], subdataset['round'])
            kl_value = get_kl_divergence(group_df['value'], subdataset['value'], top_n=50)
            
            max_kl = max(kl_cat, kl_round, kl_value)
            
            if max_kl < lowest_max_kl:
                lowest_max_kl = max_kl
                best_subdataset = subdataset
                best_kl_details = {'cat': kl_cat, 'round': kl_round, 'val': kl_value}
            
            if max_kl <= kl_threshold:
                # Validation passed
                return subdataset, attempt + 1, max_kl, best_kl_details
            
        # If we exhausted retries, return the best one found
        print(f" --  KL threshold {max_kl}. Best Max KL: {lowest_max_kl:.4f} (Cat: {best_kl_details['cat']:.4f}, Rnd: {best_kl_details['round']:.4f}, Val: {best_kl_details['val']:.4f})")
        return best_subdataset, max_retries, lowest_max_kl, best_kl_details

    # Determine number of subdatasets per group
    def calculate_num_subdatasets(group_size):
        """Calculate number of subdatasets based on group size"""
        if group_size < 1000:
            return 1
        elif group_size < 5000:
            return 3
        elif group_size < 10000:
            return 5
        else:
            return max(5, min(30, group_size // 2000))
    
    num_subdatasets_g1 = calculate_num_subdatasets(len(group1))
    num_subdatasets_g2 = calculate_num_subdatasets(len(group2))
    num_subdatasets_g3 = calculate_num_subdatasets(len(group3))
    
    print(f"Generating {num_subdatasets_g1} subdatasets from Group 1")
    print(f"Generating {num_subdatasets_g2} subdatasets from Group 2")
    print(f"Generating {num_subdatasets_g3} subdatasets from Group 3")
    
    # Create output directory
    output_dir = Path('generated_samples')
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Initialize metadata
    metadata = {
        'group1_numbers': {'num_subdatasets': num_subdatasets_g1, 'files': []},
        'group2_non_english': {'num_subdatasets': num_subdatasets_g2, 'files': []},
        'group3_proper_nouns': {'num_subdatasets': num_subdatasets_g3, 'files': []}
    }
    
    # Helper to process a group
    def process_group(group_df, group_key, num_subs, start_seed_base):
        print(f"\n--- Generating and Saving {group_key} Subdatasets ---")
        first_sub = None
        
        for i in range(num_subs):
            # Seed base shifts for each subdataset so they are distinct
            # But generate_valid_subdataset logic handles retries with its own seed offsets
            subdataset, attempts, final_kl, kl_details = generate_valid_subdataset(
                group_df, 
                sample_size=1000, 
                seed_base=start_seed_base + i,
                max_retries=10,
                kl_threshold=0.75
            )
            
            if i == 0:
                first_sub = subdataset.copy()
            
            # Save
            _filename = f'{group_key.replace("_numbers", "").replace("_non_english", "").replace("_proper_nouns", "")}_subdataset_{i+1}.json'
            # Fix naming to match previous pattern exactly "group1_subdataset_1.json"
            if "group1" in group_key: prefix = "group1"
            elif "group2" in group_key: prefix = "group2"
            else: prefix = "group3"
            _filename = f'{prefix}_subdataset_{i+1}.json'

            _filepath = output_dir / _filename
            subdataset.to_json(_filepath, orient='records', indent=2)
            metadata[group_key]['files'].append(_filename)
            
            print(f"  ✅ Saved {_filename} (Max KL: {final_kl:.4f}, Cat: {kl_details['cat']:.4f}, Rnd: {kl_details['round']:.4f}, Val: {kl_details['val']:.4f})")
            
            del subdataset
            gc.collect()
            
        return first_sub

    # Process Groups
    first_subdataset_g1 = process_group(group1, 'group1_numbers', num_subdatasets_g1, 42)
    process_group(group2, 'group2_non_english', num_subdatasets_g2, 100)
    process_group(group3, 'group3_proper_nouns', num_subdatasets_g3, 200)

    # Export metadata
    _metadata_file = output_dir / 'metadata.json'
    with open(_metadata_file, 'w') as _f:
        json.dump(metadata, _f, indent=2)
    
    print(f"\n✅ All subdatasets exported to '{output_dir}' directory")
    print(f"✅ Metadata saved to '{_metadata_file}'")
    return (
        calculate_num_subdatasets,
        first_subdataset_g1,
        generate_valid_subdataset,
        get_kl_divergence,
        process_group,
        metadata,
        num_subdatasets_g1,
        num_subdatasets_g2,
        num_subdatasets_g3,
        output_dir,
        _f,
        _metadata_file,
    )


@app.cell
def __(mo):
    mo.md("## Distribution Analysis")
    return


@app.cell
def __(first_subdataset_g1, group1, mo, pd):
    # Compare distributions for Group 1
    if first_subdataset_g1 is not None:
        original_category_dist = group1['category'].value_counts(normalize=True).head(10)
        sample_category_dist = first_subdataset_g1['category'].value_counts(normalize=True).head(10)
        
        comparison_df = pd.DataFrame({
            'Original': original_category_dist,
            'Sample': sample_category_dist
        }).fillna(0)
        
        mo.md(f"""
        ### Group 1 - Top Categories Distribution Comparison
        
        Original vs First Subdataset (showing top 10 categories):
        """)
        mo.ui.table(comparison_df)
    return comparison_df, original_category_dist, sample_category_dist


@app.cell
def __(first_subdataset_g1, group1, mo, pd):
    # Compare round distributions
    if first_subdataset_g1 is not None:
        original_round_dist = group1['round'].value_counts(normalize=True)
        sample_round_dist = first_subdataset_g1['round'].value_counts(normalize=True)
        
        round_comparison = pd.DataFrame({
            'Original': original_round_dist,
            'Sample': sample_round_dist
        }).fillna(0)
        
        mo.md("""
        ### Group 1 - Round Distribution Comparison
        """)
        mo.ui.table(round_comparison)
    return original_round_dist, round_comparison, sample_round_dist





@app.cell
def __(metadata, mo):
    mo.md(f"""
    ## Summary
    
    Successfully generated and exported:
    - **Group 1 (Numbers)**: {metadata['group1_numbers']['num_subdatasets']} subdatasets
    - **Group 2 (Non-English)**: {metadata['group2_non_english']['num_subdatasets']} subdatasets
    - **Group 3 (Proper Nouns)**: {metadata['group3_proper_nouns']['num_subdatasets']} subdatasets
    
    Each subdataset contains 1000 samples and maintains the distribution of its parent group.
    """)
    return


if __name__ == "__main__":
    app.run()
