import logging
import re
from transformers import pipeline, AutoTokenizer
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Summarizer:
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        self.summarizer = pipeline("summarization", model=model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Model max position embeddings usually 1024 for BART
        self.max_input_tokens = 1024 
        self.chunk_overlap = 50

    def ingest_data(self, documents: list) -> str:
        """
        Concatenates text from a list of document dictionaries.
        Expected format: [{"text": "doc1 content"}, {"text": "doc2 content"}]
        """
        full_text = ""
        for doc in documents:
            if "text" in doc and isinstance(doc["text"], str):
                full_text += doc["text"].strip() + " "
        return full_text.strip()

    def preprocess(self, text: str) -> str:
        """
        Basic cleaning: remove excessive whitespace.
        """
        # Remove multiple spaces/newlines
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def chunk_text(self, text: str) -> list:
        """
        Splits text into chunks explicitly using the tokenizer to ensure 
        we fit within the model's context window.
        """
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"][0]
        
        # Limit for generation + prompt. 
        # Reserve some tokens for safety and generation logic if needed, 
        # though pipeline handles generation limits. 
        # We need to chunk input so it fits into the encoder.
        # Max input for this model is 1024. Let's suggest 900 to be safe.
        chunk_size = 900 
        stride = chunk_size - self.chunk_overlap
        
        chunks = []
        if len(input_ids) <= chunk_size:
            chunks.append(text)
            return chunks

        for i in range(0, len(input_ids), stride):
            chunk_ids = input_ids[i : i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text)
            
        return chunks

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10),
           retry=retry_if_exception_type(Exception),
           reraise=True)
    def summarize(self, text: str) -> str:
        """
        Orchestrates the summarization process with chunking.
        """
        logger.info(f"Starting summarization for text of length {len(text)}")
        try:
            clean_text = self.preprocess(text)
            chunks = self.chunk_text(clean_text)
            
            logger.info(f"Text split into {len(chunks)} chunks")
            
            summaries = []
            for i, chunk in enumerate(chunks):
                # Generate summary for each chunk
                # min_length and max_length can be tuned.
                # For chunks, we want concise summaries.
                logger.debug(f"Summarizing chunk {i+1}/{len(chunks)}")
                summary_output = self.summarizer(chunk, max_length=150, min_length=40, do_sample=False)
                summaries.append(summary_output[0]['summary_text'])
                
            combined_summary = " ".join(summaries)
            logger.info("Chunks summarized and concatenated")
            
            # If we had multiple chunks, the combined summary might still be long.
            # Check if we need a recursive pass.
            # Check token length of combined summary
            combined_inputs = self.tokenizer(combined_summary, return_tensors="pt", add_special_tokens=False)
            token_count = len(combined_inputs["input_ids"][0])
            
            if token_count > self.max_input_tokens:
                 logger.info(f"Combined summary (tokens={token_count}) exceeds max input. Recursively summarizing.")
                 # Recursive call
                 return self.summarize(combined_summary)
                 
            logger.info("Summarization complete")
            return combined_summary

        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}", exc_info=True)
            raise
