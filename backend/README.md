# Anya Backend API

FastAPI-based backend for the Anya AI News Journalist application.

## Setup

1. Create a virtual environment and install dependencies:

```bash
# Create and activate virtual environment
./setup.sh
```

2. Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Start the development server:

```bash
source venv/bin/activate
uvicorn main:app --reload
```

4. Access the API documentation at:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## Project Structure

```
backend/
├── api/                # API endpoints
│   └── v1/             # API version 1
│       └── endpoints/  # API route handlers
├── core/               # Core application components
│   ├── config.py       # Application settings
│   └── database.py     # Database connection
├── models/             # Data models
├── services/           # Business logic
├── main.py             # Application entry point
└── requirements.txt    # Dependencies
```