# SPS GenAI Assignment 1

This project implements a FastAPI application with spaCy word embeddings (Assignment 1) and a CNN image classifier trained on CIFAR10 (Assignment 2).

## Features

- FastAPI REST API
- spaCy `en_core_web_lg` word embeddings
- PyTorch CNN image classifier (CIFAR10, 10 classes)
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

### POST /predict_cnn

Classifies an uploaded image into one of the 10 CIFAR10 classes
(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
The image is resized to 64x64 and passed through the CNN.

Example request:

curl -X POST -F "file=@my_image.jpg" http://127.0.0.1:8000/predict_cnn

Example response:

{
  "predicted_class": "ship",
  "class_index": 8,
  "confidence": 0.93
}

### GET /generate_digit

Generates a handwritten-digit image with a GAN trained on MNIST (Assignment 3)
and returns it as a PNG. Optional query parameters: `seed` (same seed gives the
same image) and `scale` (upscale factor, default 8).

curl "http://127.0.0.1:8000/generate_digit?seed=42" -o digit.png

## GAN Architecture (Assignment 3)

Generator: noise (100) -> Linear(7*7*128) -> reshape -> ConvTranspose2d(128->64, k4, s2, p1)
-> BatchNorm -> ReLU (14x14) -> ConvTranspose2d(64->1, k4, s2, p1) -> Tanh (28x28)

Discriminator: image (1x28x28) -> Conv2d(1->64, k4, s2, p1) -> LeakyReLU(0.2) (14x14)
-> Conv2d(64->128, k4, s2, p1) -> BatchNorm -> LeakyReLU(0.2) (7x7) -> Flatten -> Linear(1)

Code: `app/gan/model.py`. Trained Generator weights: `app/gan/generator.pth`.

Re-train (downloads MNIST into `data/`, overwrites `app/gan/generator.pth`):

uv run python -m training.train_gan

## CNN Architecture

Input 3x64x64 -> Conv(16, 3x3, s1, p1) -> ReLU -> MaxPool(2x2, s2)
-> Conv(32, 3x3, s1, p1) -> ReLU -> MaxPool(2x2, s2) -> Flatten (8192)
-> Linear(100) -> ReLU -> Linear(10)

Code: `app/cnn/model.py`. Trained weights: `app/cnn/model.pth`.

## Re-train the model (optional)

uv run --group train python -m training.train

This downloads CIFAR10 into `data/` (git-ignored) and overwrites `app/cnn/model.pth`.

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
