import sys
import os
import streamlit as st

# Add the project root to the Python path so that "scripts" can be imported.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)

from scripts.hybrid_search import hybrid_search_jobs

st.title("Semantic Job search")

query = st.text_input("Enter your job search query:")
experience = st.text_input("Enter your experience (e.g., '4+ years'):")

if st.button("Search"):
    if query and experience:
        with st.spinner("Searching..."):
            result = hybrid_search_jobs(query, experience, os.path.join(project_root, "chroma_db"))
        if result:
            st.subheader("Search Results (JSON):")
            st.json(result)
        else:
            st.warning("No valid results found. Check the logs for details.")
    else:
        st.warning("Please enter both query and experience.")
