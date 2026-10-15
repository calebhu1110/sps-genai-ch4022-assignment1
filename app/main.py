from fastapi import FastAPI
from pydantic import BaseModel
import spacy

from app.cnn.api import router as cnn_router


# Load spaCy model
nlp = spacy.load("en_core_web_lg")


# Create FastAPI app
app = FastAPI(
    title="SPS GenAI Assignment 1",
    description="FastAPI project for spaCy word embeddings and CNN image classification"
)

# CNN image classifier (Assignment 2): adds POST /predict_cnn
app.include_router(cnn_router)


# Request format
class WordRequest(BaseModel):
    word: str


# Root endpoint
@app.get("/")
def root():
    return {"message": "SPS GenAI API is running"}


# Word embedding function from Module 2
def calculate_embedding(input_word):
    word = nlp(input_word)
    return word.vector


# Embedding endpoint
@app.post("/embedding")
def get_embedding(request: WordRequest):
    vector = calculate_embedding(request.word)

    return {
        "word": request.word,
        "embedding": vector.tolist()
    }
