# data_cleaning.py
import pandas as pd

def fill_missing_values(df: pd.DataFrame, fill_value=0):
    """Fill missing values in a DataFrame with a specified value."""
    return df.fillna(fill_value)
