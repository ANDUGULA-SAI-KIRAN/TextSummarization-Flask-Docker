from pydantic import BaseModel, Field
from typing import List

class DocumentInput(BaseModel):
    text: str = Field(..., min_length=1, description="The text content of the document to be summarized.")

class SummarizeRequest(BaseModel):
    documents: List[DocumentInput] = Field(..., min_length=1, description="A non-empty list of documents.")
