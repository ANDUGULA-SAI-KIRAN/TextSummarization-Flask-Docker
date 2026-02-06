import os
import torch
import concurrent.futures
from transformers import T5Tokenizer, T5ForConditionalGeneration
from src.utils import logger

# Define model path relative to this file
# MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "t5-small")
# If model doesn't exist locally, we can download it to this path or let transformers cache it.
# To ensure it works offline after build, we'll try to load from local, else download.
# Use a folder that is in your .gitignore
CACHE_DIR = os.path.join(os.getcwd(), "model_cache")
MODEL_NAME = "t5-small"

class Summarizer:
    def __init__(self):
        # self.tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, cache_dir=os.path.dirname(MODEL_DIR))
        # self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, cache_dir=os.path.dirname(MODEL_DIR))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.model.to(self.device)
        self.tokenizer = None
        self.model = None
        logger.info(f"Summarizer initialized using device: {self.device}")

    def _load_model(self):
        try:
            logger.info(f"Attempting to load model '{MODEL_NAME}' from {CACHE_DIR}...")
            # If the folder is empty, this downloads it. If not, it loads locally.
            self.tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
            self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
            self.model.to(self.device)
            logger.info("Model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            self.model = None

    def validate_input(self, text: str):
        if not text or not isinstance(text, str) or not text.strip():
            logger.warning("Validation failed: Input text is empty or not a string.")
            raise ValueError("Input text cannot be empty.")

    def _summarize_chunk(self, text: str, max_length: int = 150, min_length: int = 40) -> str:
        # Internal method to summarize a single piece of text.
        try:
            input_text = "summarize: " + text
            inputs = self.tokenizer.encode(
                input_text,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            ).to(self.device)

            summary_ids = self.model.generate(
                inputs,
                max_length=max_length,
                min_length=min_length,
                length_penalty=2.0,
                num_beams=4,
                repetition_penalty=2.5,
                early_stopping=True
            )
            return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error in _summarize_chunk: {str(e)}")
            raise

    def summarize(self, text: str, max_length: int = 150, min_length: int = 40) -> str:
        # Main entry point for summarization with sliding window support.
        if self.model is None or self.tokenizer is None:
            self._load_model()
            if self.model is None:
                raise RuntimeError("Summarization model is not loaded.")
        
        self.validate_input(text)
        
        # Calculate tokens to determine if chunking is needed
        tokens = self.tokenizer.encode(text)
        token_count = len(tokens)
        logger.info(f"Processing text with {token_count} tokens.")

        # Simple Case: No chunking needed
        if token_count <= 1024:
            logger.info("Text fits within single window (<=1024 tokens). Summarizing directly.")
            return self._summarize_chunk(text, max_length, min_length)
        
        # Sliding window approach
        chunk_size = 1000 # Leave some space for special tokens
        stride = 50
        chunks = []
        
        for i in range(0, token_count, chunk_size - stride):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            if i + chunk_size >= token_count:
                break

        num_chunks = len(chunks)
        logger.info(f"Text exceeds 1024 tokens. Split into {num_chunks} chunks for parallel processing.")
        
        # Parallel processing of chunks
        summaries = [None] * num_chunks # Pre-allocate to maintain order if needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_chunks, 4)) as executor:
            future_to_index = {
                executor.submit(self._summarize_chunk, chunk, max_length, min_length): idx 
                for idx, chunk in enumerate(chunks)
            }
            
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    summaries[idx] = future.result()
                    logger.debug(f"Chunk {idx+1}/{num_chunks} summarized successfully.")
                except Exception as exc:
                    logger.error(f"Chunk {idx+1} generated an exception: {exc}")
                    summaries[idx] = "" # Fallback to empty string for failed chunk

        combined_summary = " ".join(filter(None, summaries))
        combined_token_count = len(self.tokenizer.encode(combined_summary))
        logger.info(f"Combined summary length: {combined_token_count} tokens.")

        # Recursive Check
        if combined_token_count > 1024:
            logger.warning("Combined summary still exceeds 1024 tokens. Starting recursive summarization pass.")
            return self.summarize(combined_summary, max_length, min_length)
             
        logger.info("Summarization process complete.")
        return combined_summary
    
        # # Parallel processing of chunks
        # summaries = []
        # with concurrent.futures.ThreadPoolExecutor() as executor:
            
        #     future_to_chunk = {executor.submit(self._summarize_chunk, chunk, max_length, min_length): chunk for chunk in chunks}
        #     for future in concurrent.futures.as_completed(future_to_chunk):
        #         try:
        #             summaries.append(future.result())
        #         except Exception as exc:
        #             print(f"Chunk summarization generated an exception: {exc}")
        
        # # Join input summaries and second pass summarization if needed
        # combined_summary = " ".join(summaries)
        
        # # If combined summary is still too long, we might want to summarize again (recursive).
        # # For this logic, we will inspect the length.
        # combined_tokens = self.tokenizer.encode(combined_summary)
        # if len(combined_tokens) > 1024:
        #      # Recursive call
        #      return self.summarize(combined_summary, max_length, min_length)
             
        # return combined_summary

# Singleton instance
summarizer_instance = Summarizer()

def summarize_text(text, max_length=150, min_length=40):
    return summarizer_instance.summarize(text, max_length, min_length)
