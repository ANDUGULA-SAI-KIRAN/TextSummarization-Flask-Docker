import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.summarizer import summarizer_instance

class TestSummarizer(unittest.TestCase):
    def test_validate_input(self):
        with self.assertRaises(ValueError):
            summarizer_instance.validate_input("")
        with self.assertRaises(ValueError):
            summarizer_instance.validate_input("   ")
        with self.assertRaises(ValueError):
            summarizer_instance.validate_input(None)

    def test_summarization_shape(self):
        # We assume the model loads correctly.
        # This test might be slow if it actually runs inference.
        # Ideally we mock the model, but for integration, we'll test a short string.
        text = "This is a test sentence related to artificial intelligence. AI is good."
        summary = summarizer_instance.summarize(text)
        self.assertIsInstance(summary, str)
        self.assertTrue(len(summary) > 0)

if __name__ == '__main__':
    unittest.main()
