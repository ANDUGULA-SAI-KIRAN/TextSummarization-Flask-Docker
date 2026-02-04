import pytest
from unittest.mock import MagicMock, patch
import sys

# We need to mock src.summarizer.pipeline and AutoTokenizer BEFORE importing src.app
# because src.app initializes Summarizer at module level.
with patch('src.summarizer.pipeline') as mock_pipeline, \
     patch('src.summarizer.AutoTokenizer') as mock_tokenizer:
    from src.summarizer import Summarizer
    from src.app import app

@pytest.fixture
def summarizer_instance():
    # Return a fresh instance with mocked dependencies
    with patch('src.summarizer.pipeline') as mock_pipeline, \
         patch('src.summarizer.AutoTokenizer') as mock_tokenizer:
        
        # Setup mock tokenizer behavior for basic calls
        mock_tokenizer_obj = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_obj
        
        summ = Summarizer()
        summ.tokenizer = mock_tokenizer_obj
        summ.summarizer = MagicMock()
        return summ

def test_ingest_data(summarizer_instance):
    docs = [{"text": "Hello world."}, {"text": "   Foo bar.  "}]
    result = summarizer_instance.ingest_data(docs)
    assert result == "Hello world. Foo bar."
    
    # Test empty or malformed
    assert summarizer_instance.ingest_data([]) == ""
    assert summarizer_instance.ingest_data([{"foo": "bar"}]) == ""

def test_preprocess(summarizer_instance):
    raw = "Hello   \n world.  "
    assert summarizer_instance.preprocess(raw) == "Hello world."

def test_chunk_text(summarizer_instance):
    # Mock tokenizer to return specific token counts
    # Text: "A B C D E F" -> 6 tokens.
    # We want to test chunking. Implementation uses 900 chunk size.
    # We should set a small chunk size on the instance for testing.
    summarizer_instance.chunk_size = 900 # Default in code is hardcoded in method?
    # Wait, the method had `chunk_size = 900` hardcoded? I should check. 
    # Yes, it was hardcoded. I should probably monkeypatch or modify behavior.
    
    # Let's mock the tokenizer output to simulate a VERY long sequence.
    # Simulate 2000 tokens.
    long_ids = list(range(2000))
    summarizer_instance.tokenizer.return_value = {"input_ids": [long_ids]}
    summarizer_instance.tokenizer.decode.return_value = "chunked_text"
    
    chunks = summarizer_instance.chunk_text("some long text")
    
    # 2000 tokens with chunk size 900 and overlap 50.
    # Stride = 850.
    # 0-900 (chunk 1)
    # 850-1750 (chunk 2)
    # 1700-2600 (chunk 3) -> 1700-2000
    # Should get 3 chunks.
    assert len(chunks) == 3

def test_summarize_flow(summarizer_instance):
    # Mock ingest and chunking to keep it simple, or test full flow
    summarizer_instance.summarizer.return_value = [{'summary_text': 'summary'}]
    summarizer_instance.tokenizer.return_value = {"input_ids": [[1, 2, 3]]} # Small input
    summarizer_instance.tokenizer.decode.return_value = "decoded"
    
    result = summarizer_instance.summarize("some text")
    assert result == "summary"
    
    # Test recursive flow?
    # If combined summary is too long.
    # First call returns 'summary'.
    # We need 'summary' to be SHORT enough to stop recursion.
    # The mock returns [1, 2, 3] which is < 1024, so it stops.

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_api_endpoint(client):
    # We need to patch the global summarizer_service in app
    with patch('src.app.summarizer_service') as mock_service:
        mock_service.ingest_data.return_value = "valid text"
        mock_service.summarize.return_value = "mocked summary"
        
        payload = {"documents": [{"text": "doc1"}]}
        response = client.post('/summarize', json=payload)
        
        assert response.status_code == 200
        assert response.json == {"summary": "mocked summary"}

def test_api_invalid_payload(client):
    response = client.post('/summarize', json={})
    assert response.status_code == 400
