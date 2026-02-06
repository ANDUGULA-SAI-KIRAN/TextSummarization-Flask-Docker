import os
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException
from src.preprocessing import concatenate_texts
from src.summarizer import summarize_text
from src.utils import logger
app = Flask(__name__)

# --- Error Handlers ---

@app.errorhandler(HTTPException)
def handle_http_exception(e):
    """Handles logic errors like 404, 405, and malformed JSON (400)"""
    logger.warning(f"HTTP {e.code} Error: {e.description}")
    return jsonify({
        "status_code": e.code,
        "error": e.name,
        "details": e.description
    }), e.code

@app.errorhandler(Exception)
def handle_unhandled_exception(e):
    """Catch-all for actual code crashes (500)"""
    logger.error(f"CRITICAL SYSTEM ERROR: {str(e)}", exc_info=True)
    return jsonify({
        "status_code": 500,
        "error": "Internal Server Error",
        "message": "The server encountered a problem it couldn't handle."
    }), 500

# --- Routes ---
# @app.route('/summarize', methods=['POST'])
# def summarize_endpoint():
#     try:
#         data = request.get_json()
#         if not data:
#             return jsonify({"error": "Invalid JSON"}), 400
        
#         documents = data.get("documents", [])
#         if not documents or not isinstance(documents, list):
#             return jsonify({"error": "'documents' must be a non-empty list of strings"}), 400
            
#         # Preprocess: Concatenate all docs
#         full_text = concatenate_texts(documents)
        
#         if not full_text:
#              return jsonify({"error": "No valid text content found in documents"}), 400

#         # Summarize
#         summary = summarize_text(full_text)
        
#         return jsonify({"summary": summary}), 200

#     except ValueError as ve:
#         return jsonify({"error": str(ve)}), 400
#     except Exception as e:
#         logger.error(f"Error during summarization: {e}")
#         return jsonify({"error": "An error occurred during processing"}), 500

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)
@app.route('/summarize', methods=['POST'])
def summarize_endpoint():
    # 1. Audit Incoming Request
    logger.info("--- New Summarization Request Received ---")
    
    # Check if header is application/json
    if not request.is_json:
        logger.warning("Request rejected: Content-Type is not application/json")
        return jsonify({"error": "Content-Type must be application/json"}), 415

    # 2. Parse JSON safely
    try:
        data = request.get_json()
    except Exception as e:
        logger.error(f"JSON Parsing Failed: {str(e)}")
        return jsonify({"error": "Malformed JSON input", "details": str(e)}), 400

    # 3. Validate Data Structure
    documents = data.get("documents")
    
    if documents is None:
        logger.warning("Validation Failed: Missing 'documents' key.")
        return jsonify({"error": "Missing 'documents' key in request body"}), 422

    if not isinstance(documents, list) or not documents:
        logger.warning(f"Validation Failed: 'documents' is {type(documents)} (expected non-empty list).")
        return jsonify({"error": "'documents' must be a non-empty list of strings"}), 422

    # 4. Processing Phase
    try:
        logger.info(f"Preprocessing {len(documents)} documents...")
        full_text = concatenate_texts(documents)
        
        if not full_text or not full_text.strip():
            logger.warning("Preprocessing resulted in empty text.")
            return jsonify({"error": "Documents contain no valid text to summarize"}), 400

        logger.info(f"Starting model inference (Text length: {len(full_text)} chars)...")
        summary = summarize_text(full_text)
        
        logger.info("Summarization successful. Sending response.")
        return jsonify({
            "status": "success",
            "summary": summary
        }), 200

    except ValueError as ve:
        logger.warning(f"Application Logic Error: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        # exc_info=True prints the file name and line number of the crash
        logger.error(f"Error during summarization pipeline: {str(e)}", exc_info=True)
        return jsonify({"error": "Inference failed", "details": str(e)}), 500

if __name__ == "__main__":
    # Check if model_cache exists locally
    if not os.path.exists("model_cache"):
        logger.info("Warning: 'model_cache' not found. Model will be downloaded on first request.")
    
    app.run(host="0.0.0.0", port=5000, debug=True)