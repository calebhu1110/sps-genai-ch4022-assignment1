# SPS GenAI Assignment 1

This project implements a FastAPI application with spaCy word embeddings.

## Features

- FastAPI REST API
- spaCy `en_core_web_lg` word embeddings
- Docker deployment
- Interactive API documentation with Swagger UI

## API Endpoints

### GET /

Checks whether the API is running.

### POST /embedding

Returns the spaCy word embedding vector for an input word.

Example request:

{
  "word": "king"
}

## Run Locally

Install dependencies:

uv sync

Download the spaCy model:

uv run python -m spacy download en_core_web_lg

Start FastAPI:

uv run fastapi dev app/main.py

Open the API documentation at:

http://127.0.0.1:8000/docs

## Run with Docker

Build the Docker image:

docker build -t sps-genai .

Run the container:

docker run -p 8000:80 sps-genai

Then open:

http://127.0.0.1:8000/docs
