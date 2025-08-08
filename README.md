# HackRx Intelligent Query-Retrieval System

A comprehensive LLM-powered document processing and query-retrieval system designed for insurance, legal, HR, and compliance domains. The system processes documents (PDFs, DOCX, emails) and provides intelligent answers to natural language queries using semantic search and GPT-4.

## 🚀 Features

- **Multi-format Document Processing**: Supports PDF, DOCX, DOC, TXT, and email files
- **Semantic Search**: Uses OpenAI embeddings with Pinecone vector database for accurate retrieval
- **Intelligent Answering**: GPT-4 powered answer generation with explainable reasoning
- **RESTful API**: FastAPI-based backend with comprehensive documentation
- **Authentication**: Bearer token-based security
- **Scalable Architecture**: Modular design with separate services for different concerns
- **Comprehensive Testing**: Unit tests and integration tests included
- **Production Ready**: Docker support, logging, error handling, and monitoring

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Query          │    │   Document      │
│   (API Layer)   │◄──►│   Processor      │◄──►│   Processor     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   OpenAI        │    │   Embedding      │    │   Pinecone      │
│   (LLM)         │◄──►│   Service        │◄──►│   (Vector DB)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📋 Requirements

- Python 3.10+
- OpenAI API Key
- Pinecone API Key
- PostgreSQL Database (optional, for metadata storage)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd hackrx-intelligent-query-system
```

### 2. Create Virtual Environment

```bash
python -m venv hackrx-env
source hackrx-env/bin/activate  # On Windows: hackrx-env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Copy the example environment file and configure your settings:

```bash
cp .env.example .env
```

Edit `.env` file with your configuration:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=hackrx-documents

# Database Configuration (optional)
DATABASE_URL=postgresql://username:password@localhost:5432/hackrx_db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Authentication
BEARER_TOKEN=4097f4513d2abce6e34765287189ca01fc5d42f566fdebecaa1aa41c52d83cca
```

### 5. Initialize Pinecone Index

The application will automatically create the Pinecone index on first run if it doesn't exist.

## 🚀 Running the Application

### Development Mode

```bash
python -m app.main
```

Or using uvicorn directly:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Mode

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Docker

```bash
docker-compose up -d
```

## 📚 API Documentation

Once the application is running, you can access:

- **Swagger UI**: http://localhost:8000/api/v1/docs
- **ReDoc**: http://localhost:8000/api/v1/redoc
- **OpenAPI JSON**: http://localhost:8000/api/v1/openapi.json

## 🔌 API Endpoints

### Main HackRx Endpoint

**POST** `/api/v1/hackrx/run`

Process a document and answer multiple questions.

```bash
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Authorization: Bearer 4097f4513d2abce6e34765287189ca01fc5d42f566fdebecaa1aa41c52d83cca" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
      "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
      "What is the waiting period for pre-existing diseases (PED) to be covered?",
      "Does this policy cover maternity expenses, and what are the conditions?"
    ]
  }'
```

### Other Endpoints

- **GET** `/api/v1/hackrx/health` - Health check
- **POST** `/api/v1/hackrx/query` - Process single query
- **GET** `/api/v1/hackrx/metrics` - System metrics
- **GET** `/api/v1/hackrx/document/{id}/summary` - Document summary
- **DELETE** `/api/v1/hackrx/document/{id}` - Delete document

## 🧪 Testing

### Run All Tests

```bash
pytest
```

### Run Specific Test Files

```bash
pytest tests/test_api.py
pytest tests/test_services.py
```

### Run Tests with Coverage

```bash
pytest --cov=app tests/
```

## 📊 System Components

### Document Processor
- **File Support**: PDF, DOCX, DOC, TXT, EML
- **Text Extraction**: Clean text extraction with metadata
- **Chunking**: Intelligent text chunking with overlap
- **Preprocessing**: Text cleaning and normalization

### Embedding Service
- **Model**: OpenAI text-embedding-ada-002
- **Batch Processing**: Efficient batch embedding generation
- **Similarity**: Cosine similarity calculations
- **Validation**: Embedding quality validation

### Pinecone Service
- **Vector Storage**: Scalable vector database operations
- **Semantic Search**: Fast similarity-based retrieval
- **Metadata**: Rich metadata storage with vectors
- **Management**: Index management and cleanup

### LLM Service
- **Model**: GPT-4 for answer generation
- **Structured Output**: JSON-formatted responses
- **Context Management**: Efficient context window usage
- **Quality Assessment**: Answer quality validation

## 🔧 Configuration

### Key Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `CHUNK_SIZE` | Maximum characters per chunk | 1000 |
| `CHUNK_OVERLAP` | Character overlap between chunks | 200 |
| `EMBEDDING_DIMENSION` | OpenAI embedding dimension | 1536 |
| `MAX_TOKENS` | Maximum LLM response tokens | 2000 |
| `TEMPERATURE` | LLM creativity (0.0-1.0) | 0.1 |

### Performance Tuning

- **Batch Size**: Adjust embedding batch size for API limits
- **Chunk Parameters**: Optimize chunk size and overlap for your documents
- **Similarity Threshold**: Fine-tune retrieval quality vs. recall
- **Top-K**: Balance between context quality and token usage

## 📈 Monitoring & Logging

### Health Checks

The system provides comprehensive health checks:

```bash
curl http://localhost:8000/api/v1/hackrx/health
```

### Metrics

Get system metrics and statistics:

```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/hackrx/metrics
```

### Logging

Logs are written to:
- Console output (structured logging)
- `hackrx_system.log` file
- Configurable log levels via `LOG_LEVEL` environment variable

## 🚨 Error Handling

The system includes comprehensive error handling:

- **Authentication Errors**: Invalid or missing bearer tokens
- **Validation Errors**: Invalid request parameters
- **Processing Errors**: Document processing failures
- **Service Errors**: External API failures (OpenAI, Pinecone)
- **Rate Limiting**: Graceful handling of API rate limits

## 🔒 Security

- **Authentication**: Bearer token-based API authentication
- **Input Validation**: Comprehensive request validation
- **Error Sanitization**: Secure error messages in production
- **CORS**: Configurable cross-origin resource sharing
- **Rate Limiting**: Built-in protection against abuse

## 🎯 Evaluation Criteria

The system is optimized for the HackRx evaluation parameters:

1. **Accuracy**: Semantic search with GPT-4 ensures high precision
2. **Token Efficiency**: Optimized context management and chunking
3. **Latency**: Async processing and efficient vector operations
4. **Reusability**: Modular architecture with clear interfaces
5. **Explainability**: Detailed reasoning and source attribution

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:

1. Check the [API documentation](http://localhost:8000/api/v1/docs)
2. Review the test cases for usage examples
3. Check system logs for debugging information
4. Use the health check endpoint to verify system status

## 🔄 Version History

- **v1.0.0**: Initial release with core functionality
  - Document processing for PDF, DOCX, email
  - Semantic search with Pinecone
  - GPT-4 powered answer generation
  - RESTful API with authentication
  - Comprehensive testing suite

---

Built with ❤️ for HackRx Challenge