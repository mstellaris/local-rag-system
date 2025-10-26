# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Local RAG (Retrieval-Augmented Generation) System that provides privacy-focused document management and querying capabilities using OpenSearch, Sentence Transformers, and Ollama for local LLM inference. The system features a Streamlit web interface with document upload, OCR processing, and an AI chatbot.

## Architecture

### Core Components

1. **Streamlit Multi-page App** (`Welcome.py` + `pages/`)
   - Main entry point: `Welcome.py`
   - Page routing: `pages/1_🤖_Chatbot.py`, `pages/2_📄_Upload_Documents.py`
   - Session state management for chat history and settings

2. **OpenSearch Integration** (`src/opensearch.py`, `src/ingestion.py`)
   - Hybrid search combining BM25 text matching and vector similarity
   - Document indexing with embeddings
   - Search pipeline with score normalization (0.3 text, 0.7 vector weights)

3. **Embedding System** (`src/embeddings.py`)
   - Uses Sentence Transformers (default: `all-mpnet-base-v2`)
   - 768-dimensional embeddings
   - Supports asymmetric embeddings for query/document optimization

4. **LLM Integration** (`src/chat.py`)
   - Ollama for local LLM inference (default: `llama3.2:1b`)
   - Streaming response generation
   - RAG-enhanced responses with document context

5. **Document Processing** (`src/ocr.py`, `src/ingestion.py`)
   - PDF processing with PyPDF2
   - OCR support via Tesseract for image-based PDFs
   - Text chunking (300 character chunks by default)

## Common Development Tasks

### Running the Application

```bash
# Start the Streamlit app
streamlit run Welcome.py

# Alternative: Run with specific port
streamlit run Welcome.py --server.port 8501
```

### Setting up OpenSearch

```bash
# Pull and run OpenSearch (single-node, security disabled for local dev)
docker run -d --name opensearch \
  -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "DISABLE_SECURITY_PLUGIN=true" \
  opensearchproject/opensearch:2.19.2

# Run OpenSearch Dashboards
docker run -d --name opensearch-dashboards \
  -p 5601:5601 \
  --link opensearch:opensearch \
  -e "OPENSEARCH_HOSTS=http://opensearch:9200" \
  -e "DISABLE_SECURITY_DASHBOARDS_PLUGIN=true" \
  opensearchproject/opensearch-dashboards:2.19.2
```

### Setting up Ollama

```bash
# Install Ollama from https://ollama.com/download
# Pull the default model
ollama pull llama3.2:1b

# Or pull alternative models
ollama pull qwen3:8b
```

### Code Formatting

```bash
# Format with Black (configured in pyproject.toml)
black src/ pages/ --line-length 88

# Sort imports with isort
isort src/ pages/ --profile black
```

### Testing Individual Components

```python
# Test OpenSearch connection
from src.opensearch import get_opensearch_client
client = get_opensearch_client()
print(client.info())

# Test embedding generation
from src.embeddings import get_embedding_model
model = get_embedding_model()
embedding = model.encode("test text")

# Test Ollama model
import ollama
ollama.chat(model='llama3.2:1b', messages=[{'role': 'user', 'content': 'Hello'}])
```

## Configuration

Key settings are in `src/constants.py`:

- `EMBEDDING_MODEL_PATH`: Sentence Transformer model or local path
- `EMBEDDING_DIMENSION`: Must match the model output (default: 768)
- `TEXT_CHUNK_SIZE`: Characters per text chunk (default: 300)
- `OLLAMA_MODEL_NAME`: LLM model for chat (default: "llama3.2:1b")
- `OPENSEARCH_HOST/PORT`: OpenSearch connection (default: localhost:9200)
- `OPENSEARCH_INDEX`: Index name (default: "documents")

## Project Structure

- `src/` - Core application logic
  - `chat.py` - LLM interaction and RAG implementation
  - `embeddings.py` - Embedding model management
  - `opensearch.py` - Search functionality
  - `ingestion.py` - Document indexing
  - `ocr.py` - OCR processing for PDFs
  - `constants.py` - Configuration settings
- `pages/` - Streamlit page components
- `notebooks/` - Jupyter notebooks for setup and testing
- `uploaded_files/` - Temporary storage for uploaded documents
- `logs/` - Application logs

## Important Implementation Details

1. **Hybrid Search Pipeline**: Must be created in OpenSearch before use. The pipeline normalizes and combines BM25 and vector scores using arithmetic mean with configurable weights.

2. **Model Caching**: The embedding model is cached using `@st.cache_resource` to avoid reloading on page refreshes in `src/chat.py:get_embedding_model()`.

3. **Streaming Responses**: Chat responses use Ollama's streaming API for real-time display in `src/chat.py:generate_response_streaming()`.

4. **Document Chunking**: Text is split into overlapping chunks for better context preservation during retrieval in `src/ingestion.py:chunk_text()`.

5. **Session State**: Chat history and settings are maintained in Streamlit session state, initialized in `pages/1_🤖_Chatbot.py:50-60`.

## Dependencies

Main dependencies (from requirements.txt):
- streamlit==1.39.0 - Web interface
- sentence-transformers==3.1.1 - Embeddings
- opensearch-py==2.7.1 - Search backend
- ollama==0.3.3 - Local LLM interface
- pytesseract==0.3.13 - OCR support
- pypdf2==3.0.1 - PDF processing