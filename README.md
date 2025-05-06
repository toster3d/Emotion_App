# Audio Emotion Detection App

A web application that records audio from the user's browser and analyzes emotions using a PyTorch ensemble model.

## Features

- Real-time audio recording in the browser
- Audio emotion detection using a PyTorch ensemble model
- Responsive UI built with React Bootstrap
- Fast and efficient API using FastAPI
- Docker support for easy deployment

## Tech Stack

- **Backend**: FastAPI, PyTorch
- **Frontend**: React, Bootstrap
- **Machine Learning**: PyTorch ensemble model using ResNet18 for audio feature extraction
- **Containerization**: Docker

## Getting Started

### Prerequisites

- Python 3.10+
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

### Running the Application

1. Start the backend:
   ```bash
   uvicorn app.main:app --reload
   ```

2. Start the frontend:
   ```bash
   cd app/frontend
   npm start
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:3000
   ```

## Using Docker

Build and run the Docker container:

```bash
docker-compose up --build
```

## API Documentation

Once the application is running, you can access the API documentation at:

```
http://localhost:8000/docs
```

## License

MIT 