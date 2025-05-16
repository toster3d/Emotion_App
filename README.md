# Audio Emotion Detection App

A web application that records audio from the user's browser and analyzes emotions using a PyTorch **ResNet18 model trained on mel spectrograms**. It features a backend API built with FastAPI for processing and a frontend built with React Bootstrap for the user interface.

## Features

- Real-time audio recording in the browser
- Audio emotion detection using a PyTorch **ResNet18 model**
- Handles various audio formats (mp3, wav, ogg, flac, m4a, webm) through the API
- Responsive UI built with React Bootstrap
- Fast and efficient API using FastAPI with full OpenAPI documentation
- Module for recording sound directly from the browser
- Implemented ResNet model adapted for audio analysis
- Docker support for easy deployment

## Tech Stack

- **Backend**: FastAPI, PyTorch
- **Frontend**: React, Bootstrap
- **Machine Learning**: PyTorch **ResNet18 model** for audio feature extraction
- **Containerization**: Docker

## Getting Started

### Prerequisites

- Python 3.13+
- Node.js 18+
- [uv](https://github.com/astral-sh/uv) for Python package management

### Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd [repository-name]
   ```

2. Set up the Python environment with uv:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e .
   ```

3. Set up the frontend:
   ```bash
   cd app/frontend
   npm install
   ```

4. Prepare the model:
   Place the trained model file in the `model_outputs` directory, typically named `best_model.pt` or the appropriate filename (e.g., `ensemble_model.pt`). If the model is tracked by Git LFS, ensure LFS is installed and run `git lfs pull` after cloning.

### Running the Application Locally (without Docker)

You have two options to run the application locally:

1.  **Using separate commands:**
    Start the backend:
    Make sure your Python virtual environment is active.
    ```bash
    uvicorn app.main:app --reload
    ```

    Start the frontend:
    Navigate to the frontend directory.
    ```bash
    cd app/frontend
    npm start
    ```

    Open your browser and navigate to:
    ```
    http://localhost:3000
    ```

2.  **Using the `run.py` helper script:**
    This script simplifies starting different parts of the application.

    Start the backend:
    ```bash
    python run.py backend
    ```

    Start the frontend:
    ```bash
    python run.py frontend
    ```

    Open your browser and navigate to:
    ```
    http://localhost:3000
    ```

## Using Docker

To build and run the application using Docker Compose:

```bash
docker-compose up --build
```

## API Documentation

Once the application (backend) is running, you can access the API documentation (Swagger UI) at:

```
http://localhost:8000/docs
```

## API Endpoints

- `GET /api/v1/health` - Checks the status of the server and model.
- `POST /api/v1/predict` - Analyzes an uploaded audio file.
- `POST /api/v1/record` - Analyzes audio recorded directly in the browser.


## Recognized Emotions

- anger
- fear
- happiness
- neutral
- sadness
- surprised

## License

MIT 