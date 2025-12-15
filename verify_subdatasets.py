import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo  # For interactive notebook UI and creating widgets
    import pandas as pd  # Data manipulation/analysis and JSON loading
    import matplotlib.pyplot as plt  # For creating distribution plots and charts
    import numpy as np  # Numerical array operations
    from pathlib import Path  # Handling filesystem paths in a cross-platform way
    import json  # Parsing metadata files
    import scipy.stats as stats  # For calculating KL Divergence entropy
    return Path, json, mo, pd, plt, stats


@app.cell
def _(mo):
    mo.md("""
    # Subdataset Verification

    This notebook analyzes the distribution of 'Category' and 'Round' columns in the generated subdatasets
    comparing them against the original original groups.
    """)
    return


@app.cell
def _(Path, json, pd):
    # Load metadata and original groups
    generated_dir = Path("generated_samples")
    metadata_path = generated_dir / "metadata.json"

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Load original groups (using caching would be better but simple read is fine for now)
    g1 = pd.read_json("group1_numbers.json")
    g2 = pd.read_json("group2_non_english.json")
    g3 = pd.read_json("group3_uncommon_proper_nouns.json")

    original_groups = {
        "Group 1 (Numbers)": g1,
        "Group 2 (Non-English)": g2,
        "Group 3 (Proper Nouns)": g3
    }

    metadata_map = {
        "Group 1 (Numbers)": metadata['group1_numbers'],
        "Group 2 (Non-English)": metadata['group2_non_english'],
        "Group 3 (Proper Nouns)": metadata['group3_proper_nouns']
    }
    return generated_dir, metadata_map, original_groups


@app.cell
def _(mo, original_groups):
    # UI Controls
    group_selector = mo.ui.dropdown(
        options=list(original_groups.keys()),
        value="Group 1 (Numbers)",
        label="Select Group"
    )
    return (group_selector,)


@app.cell
def _(group_selector):
    group_selector.value
    return


@app.cell
def _(group_selector, metadata_map, mo):
    # Dynamic subdataset selector based on chosen group
    _selected_meta = metadata_map[group_selector.value]
    _num_subs = _selected_meta['num_subdatasets']

    subdataset_selector = mo.ui.slider(
        start=1, 
        stop=_num_subs, 
        step=1, 
        value=1, 
        label=f"Select Subdataset (1-{_num_subs})"
    )

    mo.md(f"""
    ### Configuration
    {group_selector}

    {subdataset_selector}
    """)
    return (subdataset_selector,)


@app.cell
def _(group_selector):
    group_selector.value
    return


@app.cell
def _(subdataset_selector):
    subdataset_selector.value
    return


@app.cell
def _(
    generated_dir,
    group_selector,
    metadata_map,
    original_groups,
    pd,
    subdataset_selector,
):
    # Get Data
    selected_group_name = group_selector.value
    selected_sub_idx = subdataset_selector.value

    # Original Data
    df_orig = original_groups[selected_group_name]

    # Subdataset Data
    # Files are named like 'group1_subdataset_1.json'
    # metadata has the file list

    _files = metadata_map[selected_group_name]['files']
    # user index is 1-based, list is 0-based
    _filename = _files[selected_sub_idx - 1] 

    df_sub = pd.read_json(generated_dir / _filename)
    return df_orig, df_sub, selected_group_name, selected_sub_idx


@app.cell
def _(mo):
    mo.md(f"""
    ### Round Distribution
    Matching strict proportions of Round types (Jeopardy!, Double Jeopardy!, Final Jeopardy!) is crucial.
    """)
    return


@app.cell
def _(df_orig, df_sub, plt, selected_sub_idx):
    # Round Distribution Analysis
    def plot_round_dist():
        """
        Plots the distribution of rounds for the original dataset and the selected subdataset.

        Returns:
            matplotlib.figure.Figure: The plot figure.
        """
        counts_orig = df_orig['round'].value_counts(normalize=True)
        counts_sub = df_sub['round'].value_counts(normalize=True)

        # Align indexes
        all_rounds = counts_orig.index.union(counts_sub.index)
        counts_orig = counts_orig.reindex(all_rounds, fill_value=0)
        counts_sub = counts_sub.reindex(all_rounds, fill_value=0)

        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(all_rounds))
        width = 0.35

        ax.bar([i - width/2 for i in x], counts_orig, width, label='Original')
        ax.bar([i + width/2 for i in x], counts_sub, width, label=f'Subdataset {selected_sub_idx}')

        ax.set_ylabel('Proportion')
        ax.set_title('Distribution of Rounds')
        ax.set_xticks(x)
        ax.set_xticklabels(all_rounds, rotation=45)
        ax.legend()

        return fig

    # Return figure directly
    return (plot_round_dist,)


@app.cell
def _(mo):
    mo.md(f"""
    ### Value Distribution (Top 20)
    Comparing the frequency of the top 20 most common values in the original dataset vs their capture rate in the sample.
    """)
    return


@app.cell
def _(df_orig, df_sub, plt, selected_sub_idx):
    # Value Distribution Analysis
    def plot_value_dist():
        """
        Plots the distribution of the top 20 values for the original dataset and the selected subdataset.

        Returns:
            matplotlib.figure.Figure: The plot figure.
        """
        # Top 20 values in ORIGINAL dataset
        top_vals_orig = df_orig['value'].value_counts(normalize=True).head(20)
        top_val_names = top_vals_orig.index

        # Calculate subdataset proportions for the SAME values
        counts_sub_all = df_sub['value'].value_counts(normalize=True)
        counts_sub = counts_sub_all.reindex(top_val_names, fill_value=0)

        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(top_val_names))
        width = 0.35

        ax.bar([i - width/2 for i in x], top_vals_orig, width, label='Original')
        ax.bar([i + width/2 for i in x], counts_sub, width, label=f'Subdataset {selected_sub_idx}')

        ax.set_ylabel('Proportion')
        ax.set_title('Top 20 Values (by Original Frequency)')
        ax.set_xticks(x)
        ax.set_xticklabels(top_val_names, rotation=90)
        ax.legend()
        plt.tight_layout()

        return fig

    # Return figure directly
    return (plot_value_dist,)


@app.cell
def _(mo):
    mo.md(f"""
    ### Category Distribution (Top 20)
    Comparing the frequency of the top 20 most common categories in the original dataset vs their capture rate in the sample.
    """)
    return


@app.cell
def _(df_orig, df_sub, plt, selected_sub_idx):
    # Category Distribution Analysis
    def plot_category_dist():
        """
        Plots the distribution of the top 20 categories for the original dataset and the selected subdataset.

        Returns:
            matplotlib.figure.Figure: The plot figure.
        """
        # Top 20 categories in ORIGINAL dataset
        top_cats_orig = df_orig['category'].value_counts(normalize=True).head(20)
        top_cat_names = top_cats_orig.index

        # Calculate subdataset proportions for the SAME categories
        counts_sub_all = df_sub['category'].value_counts(normalize=True)
        counts_sub = counts_sub_all.reindex(top_cat_names, fill_value=0)

        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(top_cat_names))
        width = 0.35

        ax.bar([i - width/2 for i in x], top_cats_orig, width, label='Original')
        ax.bar([i + width/2 for i in x], counts_sub, width, label=f'Subdataset {selected_sub_idx}')

        ax.set_ylabel('Proportion')
        ax.set_title('Top 20 Categories (by Original Frequency)')
        ax.set_xticks(x)
        ax.set_xticklabels(top_cat_names, rotation=90)
        ax.legend()
        plt.tight_layout()

        return fig

    # Return figure directly
    return (plot_category_dist,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Subset 1 - Group1
    """)
    return


@app.cell
def _(plot_round_dist):
    plot_round_dist()
    return


@app.cell
def _(plot_category_dist):
    plot_category_dist()
    return


@app.cell
def _(plot_value_dist):
    plot_value_dist()
    return


@app.cell
def _(mo):
    mo.md(f"""
    ### Kullback-Leibler Divergence Analysis
    Analyzing the distributional difference (information loss) between the original dataset and EACH generated subdataset.
    Lower values indicate better representativeness.
    """)
    return


@app.cell
def _(
    df_orig,
    generated_dir,
    metadata_map,
    pd,
    plt,
    selected_group_name,
    stats,
):
    def calculate_kl_divergence(p_series, q_series):
        """
        Calculates the Kullback-Leibler divergence between two pandas series.

        Args:
            p_series (pd.Series): The reference series (original dataset).
            q_series (pd.Series): The query series (subdataset).

        Returns:
            float: The KL divergence score.
        """
        # Calculate probabilities
        p_counts = p_series.value_counts(normalize=True)
        q_counts = q_series.value_counts(normalize=True)

        # Align indexes (union of all categories)
        all_cats = p_counts.index.union(q_counts.index)

        # Reindex and fill missing with 0
        p_probs = p_counts.reindex(all_cats, fill_value=0).values
        q_probs = q_counts.reindex(all_cats, fill_value=0).values

        # Add small epsilon to avoid div by zero or log(0)
        epsilon = 1e-9
        p_probs = p_probs + epsilon
        q_probs = q_probs + epsilon

        # Re-normalize
        p_probs = p_probs / p_probs.sum()
        q_probs = q_probs / q_probs.sum()

        return stats.entropy(p_probs, q_probs)

    # Process all subdatasets for the selected group
    kl_results = []

    _files = metadata_map[selected_group_name]['files']

    for idx, filename in enumerate(_files):
        _sub_df = pd.read_json(generated_dir / filename)

        kl_round = calculate_kl_divergence(df_orig['round'], _sub_df['round'])
        kl_category = calculate_kl_divergence(df_orig['category'], _sub_df['category'])
        kl_value = calculate_kl_divergence(df_orig['value'], _sub_df['value'])

        kl_results.append({
            'Subdataset': idx + 1,
            'KL_Round': kl_round,
            'KL_Category': kl_category,
            'KL_Value': kl_value
        })

    df_kl = pd.DataFrame(kl_results)

    # Plot KL Scores
    def plot_kl_scores():
        """
        Plots the KL divergence scores for all subdatasets.
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

        # Round KL
        ax1.bar(df_kl['Subdataset'], df_kl['KL_Round'], color='skyblue')
        ax1.set_title('KL Divergence - Round Distribution (Lower is Better)')
        ax1.set_ylabel('KL Divergence')
        ax1.set_xlabel('Subdataset Index')
        ax1.grid(axis='y', alpha=0.3)

        # Category KL
        ax2.bar(df_kl['Subdataset'], df_kl['KL_Category'], color='lightgreen')
        ax2.set_title('KL Divergence - Category Distribution (Lower is Better)')
        ax2.set_ylabel('KL Divergence')
        ax2.set_xlabel('Subdataset Index')
        ax2.grid(axis='y', alpha=0.3)

        # Value KL
        ax3.bar(df_kl['Subdataset'], df_kl['KL_Value'], color='salmon')
        ax3.set_title('KL Divergence - Value Distribution (Lower is Better)')
        ax3.set_ylabel('KL Divergence')
        ax3.set_xlabel('Subdataset Index')
        ax3.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        return fig
    return (plot_kl_scores,)


@app.cell
def _(plot_kl_scores):
    plot_kl_scores()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
