import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo  # For interactive notebook UI and markdown
    import pandas as pd  # For DataFrame operations and handling dataset
    import numpy as np  # For numerical arrays and efficient computation
    import spacy  # For Natural Language Processing (Lemmatization, POS tagging)
    import re  # For regular expression matching (detecting numbers/words)
    import time  # For measuring performance execution time
    from collections import Counter  # For efficient counting of unique elements
    import matplotlib.pyplot as plt  # For creating static and interactive visualizations
    return mo, pd, np, spacy, re, time, Counter, plt


@app.cell
def _(mo):
    mo.md("""
    # Jeopardy Dataset Exploration

    This notebook explores the Jeopardy questions dataset with basic statistical analyses and visualizations.
    """)
    return


@app.cell
def _(pd):
    # Load the JSON data
    df = pd.read_json('JEOPARDY_QUESTIONS1.json')
    return (df,)


@app.cell
def _(df, mo):
    mo.md(f"""
    ## Dataset Overview

    The dataset contains **{len(df):,}** Jeopardy questions with the following structure:
    """)
    return


@app.cell
def _(df, mo):
    mo.ui.table(df.head(10))
    return


@app.cell
def _(df, mo):
    mo.md(f"""
    ## Data Information

    **Columns:** {', '.join(df.columns.tolist())}

    **Data Types:**
    """)
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df, mo):
    mo.md(f"""
    ## Basic Statistics

    - **Total Questions:** {len(df):,}
    - **Unique Categories:** {df['category'].nunique():,}
    - **Unique Shows:** {df['show_number'].nunique():,}
    - **Date Range:** {df['air_date'].min()} to {df['air_date'].max()}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Top 10 Categories
    """)
    return


@app.cell
def _(df, mo, plt):
    def plot_top_categories(df):
        """
        Plots the top 10 most common categories as a horizontal bar chart.

        Args:
            df (pd.DataFrame): The dataframe containing the Jeopardy questions.

        Returns:
            matplotlib.figure.Figure: The bar chart figure.
        """
        top_categories = df['category'].value_counts().head(10)

        fig, ax = plt.subplots(figsize=(10, 6))
        top_categories.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel('Number of Questions')
        ax.set_title('Top 10 Most Common Categories')
        ax.invert_yaxis()
        plt.tight_layout()
        return fig

    mo.mpl.interactive(plot_top_categories(df))
    return plot_top_categories,


@app.cell
def _(mo):
    mo.md("""
    ## Distribution of Question Values
    """)
    return


@app.cell
def _(df, mo, pd, plt):
    def plot_value_distribution(df):
        """
        Plots the distribution of question values as a histogram.
        
        Cleans the 'value' column by removing symbols and converting to numeric
        before plotting.

        Args:
            df (pd.DataFrame): The dataframe containing the Jeopardy questions.

        Returns:
            matplotlib.figure.Figure: The histogram figure.
        """
        # Clean and convert value column
        df_with_value = df[df['value'].notna()].copy()
        df_with_value['value_clean'] = df_with_value['value'].str.replace('$', '').str.replace(',', '')
        df_with_value['value_numeric'] = pd.to_numeric(df_with_value['value_clean'], errors='coerce')
        df_with_value = df_with_value[df_with_value['value_numeric'].notna()]

        fig, ax = plt.subplots(figsize=(10, 6))
        df_with_value['value_numeric'].hist(bins=30, ax=ax, color='coral', edgecolor='black')
        ax.set_xlabel('Question Value ($)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Question Values')
        plt.tight_layout()
        return fig

    mo.mpl.interactive(plot_value_distribution(df))
    return plot_value_distribution,


@app.cell
def _(mo):
    mo.md("""
    ## Round Distribution
    """)
    return


@app.cell
def _(df, mo, plt):
    def plot_round_distribution(df):
        """
        Plots the distribution of questions by round using a pie chart.

        Args:
            df (pd.DataFrame): The dataframe containing the Jeopardy questions.

        Returns:
            matplotlib.figure.Figure: The pie chart figure.
        """
        round_counts = df['round'].value_counts()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(round_counts, labels=round_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
        ax.set_title('Distribution of Questions by Round')
        return fig

    mo.mpl.interactive(plot_round_distribution(df))
    return plot_round_distribution,


@app.cell
def _(mo):
    mo.md("""
    ## Questions Over Time
    """)
    return


@app.cell
def _(df, mo, pd, plt):
    def plot_questions_over_time(df):
        """
        Plots the number of questions aired per year as a line chart.

        Args:
            df (pd.DataFrame): The dataframe containing the Jeopardy questions.

        Returns:
            matplotlib.figure.Figure: The line chart figure.
        """
        # Convert air_date to datetime
        df_time = df.copy()
        df_time['air_date_dt'] = pd.to_datetime(df_time['air_date'])
        df_time['year'] = df_time['air_date_dt'].dt.year

        questions_by_year = df_time.groupby('year').size()

        fig, ax = plt.subplots(figsize=(12, 6))
        questions_by_year.plot(kind='line', ax=ax, marker='o', color='purple', linewidth=2)
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Questions')
        ax.set_title('Jeopardy Questions Over Time')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    mo.mpl.interactive(plot_questions_over_time(df))
    return plot_questions_over_time,


@app.cell
def _(mo):
    mo.md("""
    ## Summary Statistics
    """)
    return


@app.cell
def _(df, mo, pd):
    summary_stats = {
        "Total Questions": len(df),
        "Unique Categories": df['category'].nunique(),
        "Unique Shows": df['show_number'].nunique(),
        "Questions with Values": df['value'].notna().sum(),
        "Questions without Values": df['value'].isna().sum(),
    }

    mo.ui.table(pd.DataFrame([summary_stats]).T.rename(columns={0: 'Count'}))
    return


if __name__ == "__main__":
    app.run()
