from scripts.load_data import load_jobs_from_excel
from scripts.process_data import process_job_descriptions
from scripts.create_db import create_chroma_db

EXCEL_PATH = "./data/jobs.csv"
CHROMA_PATH = "./chroma_db"

def main():
    """
    Main workflow:
      1. Load job postings from the CSV.
      2. Process job descriptions into document chunks.
      3. Create and populate the Chroma vector database.
    """
    jobs_df = load_jobs_from_excel(EXCEL_PATH)
    documents = process_job_descriptions(jobs_df)
    create_chroma_db(documents, CHROMA_PATH)

if __name__ == "__main__":
    main()
