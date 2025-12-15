import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Libraries imported successfully.")
    print(f"Pandas version: {pd.__version__}")
    print(f"Numpy version: {np.__version__}")
    
    try:
        df = pd.read_json('JEOPARDY_QUESTIONS1.json')
        print("\nJSON file loaded successfully.")
        print(df.info())
        print("\nFirst 5 rows:")
        print(df.head())
    except Exception as e:
        print(f"\nError loading JSON file: {e}")

if __name__ == "__main__":
    main()
