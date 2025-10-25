# AI-Powered Amazon Product Assistant

An intelligent conversational assistant that helps users discover and explore Amazon Electronics products through natural language queries. Built with advanced RAG (Retrieval-Augmented Generation) architecture, this system combines semantic search with large language models to provide accurate, context-aware product recommendations.

## Overview

This project transforms traditional e-commerce search into an interactive, AI-driven experience. By leveraging vector embeddings and semantic search, the assistant understands user intent beyond keyword matching and provides personalized product recommendations with detailed explanations.

### Key Features

- **Semantic Product Search**: Vector-based similarity search using OpenAI embeddings for understanding natural language queries
- **RAG Pipeline**: Retrieval-augmented generation combining Qdrant vector database with GPT-4 for contextually relevant responses
- **Multi-Provider LLM Support**: Flexible architecture supporting OpenAI, Groq, and Google Gemini models
- **Interactive Chat Interface**: Clean Streamlit UI with real-time conversations and provider selection
- **Production-Ready API**: FastAPI backend with RESTful endpoints and automatic documentation
- **Observability & Instrumentation**: Built-in logging and tracing for monitoring system performance
- **Evaluation Framework**: Comprehensive metrics and LangSmith integration for continuous quality assessment
- **Hybrid Retrieval**: Combines semantic and keyword search with re-ranking for improved relevance
- **Structured Outputs**: Type-safe LLM responses using Pydantic models
- **Advanced Chunking**: Contextual embeddings and optimized chunk strategies for better retrieval
- **Prompt Management**: Systematic context engineering and automated prompt tuning

## Architecture

The system follows a microservices architecture with three main components:

```
┌─────────────────┐      HTTP/REST      ┌──────────────────┐
│   Streamlit UI  │ ──────────────────> │  FastAPI Backend │
│   (Port 8501)   │                     │   (Port 8000)    │
└─────────────────┘                     └──────────────────┘
                                                 │
                                                 │ Vector Search
                                                 ▼
                                        ┌──────────────────┐
                                        │ Qdrant Vector DB │
                                        │  (Port 6333)     │
                                        └──────────────────┘
```

### Component Details

**Frontend (Streamlit)**
- Interactive chat interface with conversation history
- Provider and model selection sidebar
- Session state management for multi-turn conversations

**Backend (FastAPI)**
- Multi-provider LLM abstraction layer (`run_llm()` function)
- RAG pipeline orchestration (retrieval + generation)
- RESTful `/chat` endpoint with automatic OpenAPI docs
- Environment-based configuration with Pydantic Settings

**Vector Database (Qdrant)**
- Stores embeddings for Amazon Electronics product catalog
- Semantic similarity search with configurable top-k retrieval
- Persistent storage with Docker volume mounting

## Tech Stack

- **LLM Providers**: OpenAI GPT-4, Groq, Google Gemini
- **Embeddings**: OpenAI `text-embedding-3-small` (1536 dimensions)
- **Vector Database**: Qdrant (self-hosted via Docker)
- **Backend Framework**: FastAPI with async support
- **Frontend**: Streamlit for rapid UI development
- **Orchestration**: Docker Compose for multi-container setup
- **Package Management**: UV for fast, reliable dependency resolution
- **Observability**: LangSmith for experiment tracking and evaluation
- **Data Processing**: Jupyter notebooks for EDA and pipeline development

## Getting Started

### Prerequisites

- Python 3.12+
- Docker and Docker Compose
- OpenAI API key (required)
- Groq API key (optional)
- Google API key (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd 01_ai_engineering
   ```

2. **Set up environment variables**

   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=sk-...
   GROQ_API_KEY=gsk_...        # Optional
   GOOGLE_API_KEY=...          # Optional
   ```

3. **Install dependencies**
   ```bash
   uv sync
   ```

4. **Start all services**
   ```bash
   make run-docker-compose
   # or
   docker compose up --build
   ```

### Service Endpoints

Once running, access the following services:

- **Chat Interface**: http://localhost:8501
- **API Backend**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## Usage

### Chat Interface

1. Navigate to http://localhost:8501
2. Select your preferred LLM provider and model from the sidebar
3. Ask questions about electronics products, for example:
   - "What wireless earbuds do you have with noise cancellation?"
   - "Show me gaming laptops under $1000"
   - "I need a tablet for drawing, what are my options?"

### API Integration

The FastAPI backend exposes a `/chat` endpoint for programmatic access:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What headphones are available?",
    "provider": "openai",
    "model": "gpt-4"
  }'
```

## Data Pipeline

The RAG system processes Amazon Electronics data through the following stages:

1. **Data Collection**: JSONL files with product metadata (title, description, ratings, reviews)
2. **Preprocessing**: Filtering, cleaning, and formatting in `notebooks/phase_2/01_rag_preprocessing.ipynb`
3. **Embedding Generation**: OpenAI embeddings for semantic representation
4. **Vector Storage**: Upload to Qdrant collection (`Amazon-items-collection-00`)
5. **Retrieval**: Hybrid search combining semantic similarity and keyword matching with re-ranking
6. **Generation**: Context-aware responses using retrieved product information with structured outputs

### RAG Pipeline Flow

```
User Query
    │
    ▼
[Embedding] → Query Vector
    │
    ▼
[Hybrid Search] → Semantic + Keyword Search → Re-ranking
    │
    ▼
[Top-K Products] → Contextual Chunking
    │
    ▼
[Context Engineering] → Optimized prompt with product details
    │
    ▼
[LLM Generation] → Structured output (Pydantic)
    │
    ▼
Natural Language Response
```

## Development

### Project Structure

```
01_ai_engineering/
├── src/
│   ├── api/                    # FastAPI backend
│   │   ├── api/
│   │   │   ├── endpoint.py     # REST endpoints
│   │   │   ├── model.py        # Pydantic models
│   │   │   └── rag/            # RAG pipeline implementation
│   │   ├── core/
│   │   │   └── config.py       # Configuration management
│   │   └── app.py              # FastAPI application
│   └── chatbot_ui/             # Streamlit frontend
│       ├── core/
│       │   └── config.py       # UI configuration
│       └── app.py              # Streamlit app
├── notebooks/
│   ├── phase_1/                # Initial exploration
│   └── phase_2/                # RAG development
│       ├── 01_rag_preprocessing.ipynb
│       └── 02_rag_pipeline.ipynb
├── data/                       # Amazon product datasets (JSONL)
├── docker-compose.yaml         # Service orchestration
├── pyproject.toml             # Project dependencies
└── Makefile                   # Development shortcuts
```

### Development Workflow

**Working with Notebooks**

Notebooks are organized by development phase for exploratory analysis and prototyping:

```bash
# Clean notebook outputs before committing
make clean-notebook-outputs
```

**Hot Reload**

Docker Compose is configured with volume mounts for automatic code reloading:
- Edit files in `src/api/` or `src/chatbot_ui/`
- Changes are reflected immediately without rebuilding containers

**Adding Dependencies**

```bash
# Add a new package
uv add package-name

# Rebuild containers to apply changes
docker compose up --build
```

### Configuration

All services use Pydantic Settings for type-safe configuration:

- **API Config**: `src/api/core/config.py`
- **UI Config**: `src/chatbot_ui/core/config.py`

Configuration loads from `.env` file with validation at startup.

## Evaluation & Observability

### Instrumentation

The RAG pipeline includes comprehensive instrumentation:
- Request/response logging for all LLM calls
- Performance tracking for embedding generation
- Vector search latency monitoring
- End-to-end pipeline execution tracing

### Evaluation Metrics

Custom evaluation suite tracks:
- **Retrieval Quality**: Precision, recall, and relevance of retrieved products (with re-ranking metrics)
- **Generation Quality**: Answer accuracy, completeness, and coherence
- **Latency**: Response times across pipeline stages
- **Cost**: Token usage and API call optimization

All metrics are logged to LangSmith for analysis and experiment tracking.

## Roadmap

This project follows an incremental development approach. Current implementation (Sprint 2 - Complete):

- [x] Multi-provider LLM integration
- [x] Vector database setup with Qdrant
- [x] Basic RAG pipeline with semantic search
- [x] Streamlit chat interface
- [x] FastAPI backend with REST endpoints
- [x] Instrumentation and observability
- [x] Evaluation framework with LangSmith
- [x] **Pydantic structured outputs for type-safe responses**
- [x] **Hybrid retrieval with semantic + keyword search**
- [x] **Re-ranking for improved relevance**
- [x] **Advanced chunking strategies with contextual embeddings**
- [x] **Context engineering and prompt management**
- [x] **Automated prompt tuning**

### Upcoming Features

- [ ] **Agentic Workflows**: Tool-using agents with planning and reflection
- [ ] **Multi-Agent Systems**: Coordinated agents for complex product queries
- [ ] **Production Deployment**: CI/CD, semantic caching, and performance optimization

## Troubleshooting

### Qdrant Connection Issues

If notebooks or services can't connect to Qdrant:
```bash
# Verify Qdrant is running
docker compose ps

# Check Qdrant logs
docker compose logs qdrant-1

# Access dashboard to verify data
open http://localhost:6333/dashboard
```

### API Provider Errors

If LLM calls fail:
1. Verify API keys in `.env` file
2. Check model names match provider specifications
3. Review API usage limits and quotas

### Container Issues

```bash
# Rebuild all containers
docker compose down
docker compose up --build

# View service logs
docker compose logs -f streamlit-1
docker compose logs -f api-1
```

## Contributing

This is a personal portfolio project, but feedback and suggestions are welcome! Feel free to open issues or reach out with ideas.

## License

MIT License - feel free to use this project as a reference for your own AI engineering work.

## Acknowledgments

- Amazon Product Dataset from publicly available sources
- OpenAI for embeddings and language models
- Qdrant for high-performance vector search
- FastAPI and Streamlit communities for excellent frameworks
