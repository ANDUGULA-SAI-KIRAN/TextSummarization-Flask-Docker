import logging
from flask import Flask, request, jsonify
from src.summarizer import Summarizer
from src.schemas import SummarizeRequest
from pydantic import ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize summarizer
# We initialize it here so the model loads when the app starts, not per request
summarizer_service = Summarizer()

@app.route('/summarize', methods=['POST'])
def summarize_route():
    try:
        data = request.get_json()
        if not data:
             return jsonify({"error": "Invalid payload. JSON body required."}), 400
        
        # Validate input using Pydantic
        try:
            request_model = SummarizeRequest(**data)
        except ValidationError as e:
            logger.warning(f"Validation error: {e.errors()}")
            return jsonify({"error": "Validation failed", "details": e.errors()}), 400
            
        documents = [doc.model_dump() for doc in request_model.documents]
        
        logger.info(f"Received request to summarize {len(documents)} documents")

        # Ingest
        full_text = summarizer_service.ingest_data(documents)
        
        if not full_text:
            logger.warning("No text found in provided documents")
            return jsonify({"error": "No text found in documents."}), 400
            
        # Summarize
        summary = summarizer_service.summarize(full_text)
        
        logger.info("Request processed successfully")
        return jsonify({"summary": summary})
        
    except Exception as e:
        logger.error(f"Unexpected error in summarize_route: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
