"""
API Server for Code Similarity Checker
REST API wrapper for text extraction and similarity analysis.

Run with: uvicorn api_server:app --reload --port 8000
"""

import base64
import tempfile
import os
from typing import Dict, List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from text_extractor import (
    extract_text,
    extract_from_zip,
    extract_from_github,
    extract_from_folder,
    extract_zip_as_single,
)
from code_similarity_finder import FlexibleCodeSimilarityChecker


app = FastAPI(
    title="Code Similarity API",
    description="Extract text from files and calculate code similarity",
    version="1.0.0"
)

# Allow CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== Request/Response Models =====

class GitHubRequest(BaseModel):
    urls: List[str]
    ignore_patterns: Optional[List[str]] = None


class AnalyzeRequest(BaseModel):
    submissions: Dict[str, str]  # filename -> content
    target_file: str  # which file to compare against others
    preprocessing_options: Optional[Dict[str, bool]] = None


class ExtractResponse(BaseModel):
    submissions: Dict[str, str]
    count: int


class AnalyzeResponse(BaseModel):
    scores: Dict[str, List[float]]
    submission_names: List[str]
    target_file: str


# ===== Endpoints =====

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "service": "Code Similarity API"}


@app.post("/upload", response_model=ExtractResponse)
async def upload_files(
    files: List[UploadFile] = File(...),
    ignore_patterns: Optional[str] = None
):
    """
    Upload files (ZIP, PDF, code files, etc.) and extract text.
    
    Returns extracted text for each file.
    """
    patterns = ignore_patterns.split(",") if ignore_patterns else None
    all_submissions = {}
    
    for file in files:
        content = await file.read()
        
        if file.filename.endswith(".zip"):
            # Handle ZIP file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                submissions = extract_from_zip(tmp_path, patterns)
                all_submissions.update(submissions)
            finally:
                os.unlink(tmp_path)
        else:
            # Handle single file
            text = extract_text(content, file.filename)
            if text.strip():
                all_submissions[file.filename] = text
    
    return ExtractResponse(
        submissions=all_submissions,
        count=len(all_submissions)
    )


@app.post("/github", response_model=ExtractResponse)
async def process_github(request: GitHubRequest):
    """
    Clone GitHub repositories and extract text from all files.
    
    Accepts a list of GitHub URLs.
    """
    all_submissions = {}
    
    for url in request.urls:
        try:
            submissions = extract_from_github(url, request.ignore_patterns)
            all_submissions.update(submissions)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error cloning {url}: {str(e)}")
    
    return ExtractResponse(
        submissions=all_submissions,
        count=len(all_submissions)
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_similarity(request: AnalyzeRequest):
    """
    Calculate similarity scores between a target file and other submissions.
    
    Accepts:
    - submissions: Dict of filename -> code content
    - target_file: Which file to compare against others
    - preprocessing_options: Optional preprocessing settings
    """
    if request.target_file not in request.submissions:
        raise HTTPException(
            status_code=400,
            detail=f"Target file '{request.target_file}' not found in submissions"
        )
    
    # Initialize checker
    options = request.preprocessing_options or {
        "remove_comments": True,
        "normalize_whitespace": True,
        "preserve_variable_names": True,
        "preserve_literals": False,
    }
    checker = FlexibleCodeSimilarityChecker(preprocessing_options=options)
    
    # Get target code and add other submissions
    target_code = request.submissions[request.target_file]
    
    for name, code in request.submissions.items():
        if name != request.target_file:
            checker.add_submission(name, code)
    
    # Calculate similarities
    scores = checker.calculate_similarities(target_code)
    
    return AnalyzeResponse(
        scores=scores,
        submission_names=checker.submission_names,
        target_file=request.target_file
    )


@app.post("/extract-text")
async def extract_single_file(
    file: UploadFile = File(...)
):
    """
    Extract text from a single file.
    """
    content = await file.read()
    text = extract_text(content, file.filename)
    
    return {
        "filename": file.filename,
        "text": text,
        "length": len(text)
    }

@app.post("/upload-individual", response_model=ExtractResponse)
async def upload_individual_files(
    files: List[UploadFile] = File(...),
    ignore_patterns: Optional[str] = None
):
    """
    Upload files where each file/ZIP = ONE submission.
    
    Use this for tab3 where user submits items one by one.
    Each uploaded file becomes one submission.
    If a ZIP is uploaded, all its content is combined into one submission.
    """
    patterns = ignore_patterns.split(",") if ignore_patterns else None
    all_submissions = {}
    
    for file in files:
        content = await file.read()
        
        if file.filename.endswith(".zip"):
            # ZIP = one submission with all content combined
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                combined_text = extract_zip_as_single(tmp_path, file.filename, patterns)
                if combined_text.strip():
                    # Use ZIP filename (without .zip) as submission name
                    submission_name = file.filename.rsplit('.', 1)[0]
                    all_submissions[submission_name] = combined_text
            finally:
                os.unlink(tmp_path)
        else:
            # Single file = one submission
            text = extract_text(content, file.filename)
            if text.strip():
                all_submissions[file.filename] = text
    
    return ExtractResponse(
        submissions=all_submissions,
        count=len(all_submissions)
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
