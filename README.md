---
title: Copyadi Finder
emoji: 🔍
colorFrom: indigo
colorTo: green
sdk: docker
pinned: false
---

# 🔍 Copyadi Finder

A powerful code similarity detection tool that uses multiple analysis methods to find copied or similar code across submissions.

## ✨ Features

- **Multi-Source Input**: Analyze code from GitHub repos, PDFs, ZIP files, or individual files
- **5 Similarity Metrics**: Raw text, Processed text, AST structure, Token sequence, and TF-IDF cosine similarity
- **Smart Preprocessing**: Configurable options to normalize code before comparison
- **Plagiarism Detection**: Side-by-side function matching with AST-based analysis (resilient to renamed variables)
- **Workspace Management**: Save and organize analysis sessions with persistent URL storage

## 📊 Similarity Metrics Explained

| Metric | Description |
|--------|-------------|
| **Raw Text** | Direct comparison of original code (no preprocessing) |
| **Processed** | Comparison after removing comments, normalizing whitespace, and optionally anonymizing literals |
| **AST (Structure)** | Compares logical code structure using Abstract Syntax Trees |
| **Token Sequence** | Detects copy-paste by finding matching token sequences |
| **Cosine (TF-IDF)** | Measures semantic similarity using term frequency analysis |

## ⚙️ Preprocessing Options

| Option | Effect |
|--------|--------|
| **Remove Comments** | Strips all comments from code |
| **Normalize Whitespace** | Standardizes indentation and spacing |
| **Preserve Variable Names** | Includes function/variable names in AST comparison (if checked) |
| **Anonymize Literals** | Replaces strings with `"STR"` and numbers with `0` (if checked) |

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker (optional, for containerized deployment)

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/Copyadi-finder.git
cd Copyadi-finder

# Install dependencies
pip install -r requirements-docker.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Start the backend API
cd backend
uvicorn api_server:app --reload --port 8000

# In a new terminal, start the Streamlit frontend
streamlit run a_streamlit_app.py
```

### Docker Deployment (Backend)

```bash
# Build and run
docker build -t copyadi-backend .
docker run -d -p 8000:7860 --name copyadi-checker copyadi-backend

# Run tests inside container
docker exec copyadi-checker python test/test_similarity.py
```

### Frontend Deployment (Streamlit)

The frontend can be deployed to [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy `a_streamlit_app.py`
4. Set the `API_BASE_URL` environment variable to your backend URL

The frontend automatically detects if a local backend is running (`localhost:8000`) and falls back to the remote API if not.

## 🌐 Live Demo

| Component | URL |
|-----------|-----|
| **Frontend** | [Streamlit App](https://your-app.streamlit.app) |
| **Backend API** | [Hugging Face Space](https://arnavam-copyadi-finder.hf.space) |

## 🏗️ Architecture

```
Copyadi-finder/
├── a_streamlit_app.py      # Frontend (Streamlit)
├── backend/
│   ├── api_server.py       # FastAPI REST API
│   ├── code_similarity_finder.py  # Core similarity algorithms
│   ├── text_extractor_llamaindex.py  # File/URL text extraction
│   └── auth.py             # Authentication & workspace management
├── test/
│   └── test_similarity.py  # Unit tests
└── Dockerfile              # Container configuration
```

## 🔐 Authentication

The app requires login to use. Register with an invite code (contact admin) or use existing credentials.

## 📝 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/github` | POST | Process GitHub URLs / PDFs |
| `/upload` | POST | Upload files for extraction |
| `/analyze` | POST | Run similarity analysis |
| `/similar-regions` | POST | Get detailed function/token matches |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

MIT License
