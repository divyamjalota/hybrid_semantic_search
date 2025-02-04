import pandas as pd

def load_jobs_from_excel(file_path):
    """
    Load job postings from an Excel file.

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        DataFrame: A pandas DataFrame containing job postings.
    """
    return pd.read_csv(file_path)
