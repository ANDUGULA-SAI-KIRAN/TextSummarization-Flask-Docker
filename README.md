# TextSummarization-Flask-Docker
Building a Flask-based API that ingests multiple text documents in JSON format, concatenates them, and returns a coherent summary using a transformer-based model.

## Setup and Running

### Prerequisites
- Python 3.12+
- Docker (optional)

### Running Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   python -m src.app
   ```
   The API will be available at `http://localhost:5000`.

### Running with Docker
1. Build the image:
   ```bash
   docker build -t text-summarizer .
   ```
2. Run the container:
   ```bash
   docker run -p 5000:5000 text-summarizer
   ```

## Features
- **Transformer-based Summarization**: Uses `disilbart-cnn-12-6` model.
- **Robustness**: Implements `tenacity` for retries on transient failures.
- **Validation**: Strict input validation using `Pydantic`.
- **Observability**: Structured logging for easier debugging.

## API Documentation

### POST /summarize
Summarizes a list of text documents.

**Endpoint:** `http://localhost:5000/summarize`
**Method:** `POST`
**Content-Type:** `application/json`

**Request Body Schema:**
```json
{
  "documents": [
    { "text": "string (min_length=1)" }
  ]
}
```

**Success Response (200 OK):**
```json
{
  "summary": "string (generated summary)"
}
```

**Error Responses:**
- **400 Bad Request**: Validation failure (e.g., empty text, invalid format).
  ```json
  {
    "error": "Validation failed",
    "details": [ ... ]
  }
  ```
- **500 Internal Server Error**: Unexpected server-side processing error.

## Sample Requests

### cURL
```bash
curl -X POST http://localhost:5000/summarize \
-H "Content-Type: application/json" \
-d '{
    "documents": [
        {"text": "The quick brown fox jumps over the lazy dog."},
        {"text": "This is a second document to be concatenated."}
    ]
}'
```

### Postman
1. Import the provided `postman_collection.json` file into Postman.
2. The collection includes requests for both **Valid Summarization** and **Validation Error Testing**. 

## Testing
To run the automated unit tests:
```bash
pytest
```
To generate a simple test report:
```bash
pytest > test_report.txt
```
