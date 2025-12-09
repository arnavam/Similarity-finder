# 🔍 Copyadi - Code Similarity Checker

A powerful code plagiarism detection tool that compares submissions using multiple similarity metrics including TF-IDF, AST analysis, Levenshtein distance, and token-based comparison.

## 🌐 Live Demo

- **Frontend**: [Streamlit Cloud](https://copyadi-finder.streamlit.app) *(update with your URL)*
- **API**: [Render](https://copyadi-finder.onrender.com)

## ✨ Features

- 📦 **ZIP Upload**: Upload ZIP files containing multiple submissions
- 🔗 **GitHub Integration**: Clone and analyze GitHub repositories
- 📄 **Multiple File Formats**: Supports `.py`, `.java`, `.cpp`, `.js`, `.txt`, `.pdf`, `.docx`, and more
- 📊 **5 Similarity Metrics**:
  - Raw text comparison (Levenshtein)
  - Preprocessed text similarity
  - TF-IDF cosine similarity
  - AST structure analysis
  - Token-based comparison

## 🏗️ Architecture

```
┌─────────────────┐     HTTP      ┌─────────────────┐
│   Streamlit     │ ────────────► │   FastAPI       │
│   (Frontend)    │               │   (Backend)     │
│   Cloud         │               │   Render        │
└─────────────────┘               └─────────────────┘
```

## 📁 Project Structure

```
copyadi-finder/
├── a_streamlit_app.py      # Streamlit frontend
├── backend/
│   ├── api_server.py       # FastAPI REST API
│   ├── code_similarity_finder.py  # Similarity algorithms
│   └── text_extractor.py   # File extraction utilities
├── requirements.txt        # Streamlit Cloud dependencies
├── requirements-docker.txt # Docker/API dependencies
├── Dockerfile              # Docker configuration
├── start.sh                # API startup script
└── .streamlit/
    └── config.toml         # Streamlit configuration
```

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/arnavam/Similarity-finder.git
   cd Similarity-finder
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements-docker.txt
   python -m spacy download en_core_web_sm
   ```

4. **Run the API**
   ```bash
   uvicorn backend.api_server:app --reload --port 8000
   ```

5. **Run the Streamlit app** (in a new terminal)
   ```bash
   streamlit run a_streamlit_app.py
   ```

6. Open http://localhost:8501

### Docker

```bash
docker build -t copyadi .
docker run -p 8000:8000 copyadi
```

### Docker Compose (Development)

```bash
docker-compose up --build
```

## ☁️ Deployment

### Streamlit Community Cloud (Frontend)

1. Connect your GitHub repo to [share.streamlit.io](https://share.streamlit.io)
2. Set the main file to `a_streamlit_app.py`
3. Add secret: `API_URL = "https://your-api.onrender.com"`

### Render (Backend API)

1. Connect your GitHub repo to [Render](https://render.com)
2. Create a new **Web Service** with Docker
3. The API will be available at your Render URL

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | API server port | `8000` |
| `API_URL` | Backend API URL (for Streamlit) | `http://localhost:8000` |

### Streamlit Secrets

Create `.streamlit/secrets.toml` for local development:
```toml
API_URL = "http://localhost:8000"
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/upload` | POST | Upload files for extraction |
| `/upload-individual` | POST | Upload individual submissions |
| `/github` | POST | Process GitHub URLs |
| `/analyze` | POST | Calculate similarity scores |

## 🧪 Similarity Metrics

1. **Raw Score**: Direct Levenshtein distance on original code
2. **Processed Score**: Similarity after removing comments/whitespace
3. **Cosine Score**: TF-IDF based cosine similarity
4. **AST Score**: Abstract Syntax Tree structural comparison
5. **Token Score**: Token sequence similarity using spaCy

## 📄 License

MIT License - feel free to use this for your projects!

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.
