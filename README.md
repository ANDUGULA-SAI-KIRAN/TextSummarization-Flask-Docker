# Text Summarization Flask API

A Flask-based API that ingests multiple text documents, concatenates them, and returns a coherent summary using the Hugging Face T5-Small Transformer model.

## 🚀 Key Features
- **Text Ingestion**: Accepts multiple text documents via JSON.
- **Sliding Window Logic**: Automatically handles "Long Context" by chunking text into 1000-token windows with a 50-token stride to preserve semantic meaning.
- **Recursive Summarization**: If a combined summary exceeds the model's capacity, the system performs a second summarization pass.
- **Parallel Inference**: Utilizes `ThreadPoolExecutor` to process text chunks simultaneously, optimizing GPU/CPU utilization.
- **Logging**: Logging monitors chunking decisions, thread performance, and model loading states.
- **Custom HTML Reporting**: Includes a custom test runner that generates visual HTML reports of the test suite.
- **Dockerized**: Easy to deploy using Docker.
- **REST API**: Simple `/summarize` endpoint.

---

## Project Structure
```text

.
├── src/
│   ├── app.py            # API Gateway with error handlers & logging
│   ├── preprocessing.py  # Text cleaning & concatenation logic
│   ├── summarizer.py     # T5 Model interface & Sliding Window logic
│   └── utils.py          # Logger configuration
├── tests/
│   └── test_api.py       # Unit tests with custom HTML reporting
├── model_cache/          # Local storage for model weights (persistent)
├── reports/              # Generated test reports (HTML)
├── Dockerfile            # Container configuration
├── requirements.txt      # Dependency list
└── README.md             # Project Documentation
```

## Technical Details
### Sliding Window Summarization
For documents exceeding the model's token limit (1024 tokens), the system employs a sliding window approach:
1. **Tokenization**: Input text is tokenized using the T5 tokenizer.
2. **Chunking**: Tokens are split into chunks of 1000 tokens with a 50-token overlap (stride) to preserve context boundaries.
3. **Parallel Processing**: Chunks are summarized in parallel using a thread pool to improve performance.
4. **Aggregation**: Individual chunk summaries are concatenated. If the result is still too long, the process repeats recursively.

### Status Code Logic
The API follows standard REST principles for error reporting. You can observe these codes in Postman: | Code | Meaning | Trigger Scenario | | :--- | :--- | :--- | | 200 | OK | Successful summarization. | | 400 | Bad Request | Malformed JSON (e.g., unescaped quotes) or empty text. | | 415 | Unsupported Media | Content-Type is not application/json. | | 422 | Unprocessable Entity | Missing documents key or invalid data type (e.g., string instead of list). | | 500 | Server Error | Internal model inference or hardware failure. |

## Requirements
- Python 3.12.7
- Docker

## Setup & Installation

### Local Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python -m src.app
   ```

### Docker Setup
1. Build the image:
   ```bash
   docker build -t text-summarizer-api .
   ```
2. Run the container:
   ```bash
   docker run -p 5000:5000 -v $(pwd)/model_cache:/app/model_cache text-summarizer-api
   ```

## API Usage

### Endpoint: `/summarize`
- **Method**: POST
- **Content-Type**: application/json
- **Body**:
  ```json
  {
    "documents": [
      "Text document 1 content...",
      "Text document 2 content..."
    ]
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "summary": "Concatenated and summarized text..."
  }
  ```

## Testing & Reporting
The project includes a robust testing suite that mocks model inference for speed while verifying pathing and logic.
Run Tests and Generate Report:
```bash
python -m tests.test_api
```
After execution, open reports/report.html in your browser to view the visual test results as requested in the deliverables.