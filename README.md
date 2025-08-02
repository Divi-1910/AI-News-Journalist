# AI News Journalist (Infiya)

An intelligent news aggregation and personalization platform that delivers curated news content through conversational AI with customizable personalities and preferences.

## 🚀 Features

- **Conversational AI Interface** - Chat with AI news anchors with different personalities
- **Multi-Agent Processing Pipeline** - Intelligent news classification, extraction, and summarization
- **Personalized Content** - Customizable news personalities and topic preferences
- **Real-time Updates** - Live agent status updates during news processing
- **Google OAuth Integration** - Secure authentication with Google accounts
- **Responsive Design** - Modern React frontend with Tailwind CSS
- **Scalable Architecture** - Microservices with FastAPI backend and Go pipeline

## 🏗️ Architecture
High Level Design : https://github.com/Divi-1910/AI-News-Journalist/blob/main/High-Level-Design.png

Agent Pipeline Architecture : <img width="1195" height="719" alt="Image" src="https://github.com/user-attachments/assets/0e6a1f02-707f-43e6-b4ce-acc94dde79e8" />


### Components

- **Frontend**: React + Vite + Tailwind CSS
- **Backend**: FastAPI + MongoDB + Redis
- **Pipeline**: Go microservice with multi-agent workflow
- **Authentication**: Google OAuth 2.0
- **Database**: MongoDB for user data, Redis for caching and real-time updates

## 🛠️ Tech Stack

### Backend

- **FastAPI** - Modern Python web framework
- **MongoDB** - Document database for user data and chat history
- **Redis** - Caching and real-time agent updates
- **Pydantic** - Data validation and serialization
- **Google OAuth** - Authentication

### Frontend

- **React 18** - UI framework
- **Jotai** - State Management
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **Zustand** - State management
- **Axios** - HTTP client

### Pipeline

- **Go** - High-performance processing pipeline
- **Gin** - HTTP web framework
- **ChromaDB** - Vector database for embeddings
- **Ollama** - Local LLM integration
- **News APIs** - External news sources
- **Youtube Data v3 APIs** - External Video Description and Title

## 📋 Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.9+
- **Go** 1.21+
- **MongoDB** 4.4+
- **Redis** 6.0+
- **Docker**

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone <repository-url>
cd Infiya
```

### 2. Environment Setup

#### Backend

```bash
cd backend
cp .env.example .env
# Edit .env with your configuration
pip install -r requirements.txt
```

#### Frontend

```bash
cd frontend
npm install
cp .env.example .env
# Edit .env with your configuration
```

#### Pipeline

```bash
cd pipeline
cp .env.example .env
# Edit .env with your configuration
go mod download
```

### 3. Architecture Setup

```bash
# Start MongoDB and Redis
docker-compose up -d

```

### 4. Run Ollama with a embedding Model (I Recommend nomic-embed-text)

### 4. Run Services

#### Start Backend

```bash
cd backend
python main.py
# Runs on http://localhost:8000
```

#### Start Frontend

```bash
cd frontend
npm run dev
# Runs on http://localhost:5173
```

#### Start Pipeline

```bash
cd pipeline
go run cmd/server/main.go
# Runs on http://localhost:8080
```

### 5. Access Application

Open http://localhost:5173 in your browser

## 🔧 Configuration

### Environment Variables

#### Backend (.env)

```env
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=anya_db
REDIS_URL=redis://localhost:6379
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
JWT_SECRET_KEY=your_jwt_secret
PIPELINE_URL=http://localhost:8080
```

#### Frontend (.env)

```env
VITE_API_URL=http://localhost:8000
VITE_GOOGLE_CLIENT_ID=your_google_client_id
```

#### Pipeline (.env)

```env
REDIS_URL=
OLLAMA_URL=
CHROMADB_URL=
NEWS_API_KEY=your_news_api_key
YOUTUBE_API_KEY=your_youtube_api_key
GEMINI_API_KEY=
```

## 🤖 AI Agents

The pipeline includes multiple specialized agents:

- **Classifier** - Determines query intent and routing
- **Keyword Extractor** - Extracts relevant keywords from queries
- **Query Enhancement** - Enchanes queries for search
- **News API** - Fetches news from external sources
- **Embedding** - Creates vector embeddings for content
- **Relevancy** - Scores content relevance
- **Scraper** - Extracts full article content
- **Vector Storage** - Stores and searches for semantically similar article titles
- **Follow-up** - for Contextual Follow up queries
- **Youtube** - Fetches latest video details
- **Summarizer** - Generates concise summaries
- **Memory** - Maintains conversation context
- **Persona** - Applies personality to responses
- **Chitchat** - Handles casual conversation

## 👤 News Personalities

Choose from different AI anchor personalities:

- **Calm Anchor** - Professional and composed delivery
- **Friendly Explainer** - Approachable and educational
- **Investigative Reporter** - In-depth and analytical
- **Youthful Trendspotter** - Energetic and trend-focused
- **Global Correspondent** - International perspective
- **AI Analyst** - Technical and data-driven

## 📊 API Endpoints

### Authentication

- `POST /api/v1/auth/google` - Google OAuth login
- `POST /api/v1/auth/refresh` - Refresh JWT token
- `POST /api/v1/auth/logout` - User logout

### Chat

- `POST /api/v1/chat/message` - Send chat message
- `GET /api/v1/chat/history` - Get chat history
- `DELETE /api/v1/chat/clear` - Clear chat history

### Users

- `GET /api/v1/users/me` - Get current user
- `PUT /api/v1/users/preferences` - Update preferences
- `GET /api/v1/users/stats` - Get user statistics

### Pipeline

- `POST /api/v1/workflow/process` - Process user query
- `GET /api/v1/workflow/status/{id}` - Get workflow status
- `GET /api/v1/health` - Health check

## 🐳 Docker Deployment

```bash
# Build and run all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🧪 Testing

### Backend Tests

```bash
cd backend
pytest tests/
```

### Frontend Tests

```bash
cd frontend
npm test
```

### Pipeline Tests

```bash
cd pipeline
go test ./...
```

## 📈 Monitoring

- **Health Checks** - Available at `/health` endpoints
- **Metrics** - Prometheus metrics at `/metrics`
- **Logs** - Structured logging with configurable levels
- **Agent Status** - Real-time workflow monitoring

## 🔒 Security

- **JWT Authentication** - Secure token-based auth
- **Google OAuth** - Trusted third-party authentication
- **Input Validation** - Pydantic models for data validation
- **CORS Configuration** - Proper cross-origin setup
- **Rate Limiting** - API rate limiting (configurable)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request


**Built with ❤️ ( Inspired by a special person in my life. )**
