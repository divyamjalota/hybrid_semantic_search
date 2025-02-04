import unittest
from scripts.hybrid_search import extract_experience_range, aggregate_results

# A simple fake document class to simulate documents returned from the vector store.
class FakeDocument:
    def __init__(self, metadata, page_content):
        self.metadata = metadata
        self.page_content = page_content

class TestSearchJobs(unittest.TestCase):
    def test_extract_experience_range(self):
        self.assertEqual(extract_experience_range("3-5 years"), (3, 5))
        self.assertEqual(extract_experience_range("3 - 5 years"), (3, 5))
        self.assertEqual(extract_experience_range("5+ years experience"), (5, None))
        self.assertEqual(extract_experience_range("5 years"), (5, None))
        self.assertEqual(extract_experience_range("Minimum 7 years required"), (7, None))
        self.assertEqual(extract_experience_range("No experience required"), (None, None))
        
    def test_aggregate_results_candidate_experience_4(self):
        docs = [
            FakeDocument(
                metadata={
                    "job_id": "1",
                    "title": "Frontend Developer",
                    "experience_required": "2-4 years"
                },
                page_content="Looking for a creative frontend developer."
            ),
            FakeDocument(
                metadata={
                    "job_id": "2",
                    "title": "Frontend Developer",
                    "experience_required": "4-6 years"
                },
                page_content="Join our frontend team."
            ),
            FakeDocument(
                metadata={
                    "job_id": "3",
                    "title": "Senior Frontend Developer",
                    "experience_required": "5+ years"
                },
                page_content="Experienced frontend developer needed."
            ),
        ]
        # For candidate_experience = 4:
        # - Job "1": "2-4 years" qualifies because 4 is within [2, 4].
        # - Job "2": "4-6 years" qualifies because 4 is within [4, 6].
        # - Job "3": "5+ years" is rejected because 4 < 5.
        aggregated = aggregate_results(docs, 4)
        job_ids = set(job["job_id"] for job in aggregated)
        self.assertIn("1", job_ids)
        self.assertIn("2", job_ids)
        self.assertNotIn("3", job_ids)

    def test_aggregate_results_candidate_experience_5(self):
        docs = [
            FakeDocument(
                metadata={
                    "job_id": "1",
                    "title": "Frontend Developer",
                    "experience_required": "2-4 years"
                },
                page_content="Looking for a creative frontend developer."
            ),
            FakeDocument(
                metadata={
                    "job_id": "2",
                    "title": "Frontend Developer",
                    "experience_required": "4-6 years"
                },
                page_content="Join our frontend team."
            ),
            FakeDocument(
                metadata={
                    "job_id": "3",
                    "title": "Senior Frontend Developer",
                    "experience_required": "5+ years"
                },
                page_content="Experienced frontend developer needed."
            ),
        ]
        # For candidate_experience = 5:
        # - Job "1": "2-4 years" is rejected because 5 > 4.
        # - Job "2": "4-6 years" qualifies because 5 is within [4, 6].
        # - Job "3": "5+ years" qualifies because 5 >= 5.
        aggregated = aggregate_results(docs, 5)
        job_ids = set(job["job_id"] for job in aggregated)
        self.assertNotIn("1", job_ids)
        self.assertIn("2", job_ids)
        self.assertIn("3", job_ids)

if __name__ == "__main__":
    unittest.main()
