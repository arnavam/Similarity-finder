"""
API Server for Code Similarity Checker
REST API wrapper for text extraction and similarity analysis.

Run with: uvicorn api_server:app --reload --port 8000
"""

import os
import tempfile
from typing import Dict, List, Optional

from auth import (
    require_auth,
    save_github_urls,
    get_user_github_history,
    create_instance,
    get_user_instances,
    get_instance,
    update_instance_urls,
    delete_instance,
)
from auth import router as auth_router
from code_similarity_finder import FlexibleCodeSimilarityChecker, get_similar_regions
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from text_extractor_llamaindex import (
    extract_from_github,
    extract_from_zip,
    extract_text,
    extract_zip_as_single,
    extract_from_url,
)
from submission_buffer import get_buffer

app = FastAPI(
    title="Code Similarity API",
    description="Extract text from files and calculate code similarity",
    version="2.0.0",
)

app.include_router(auth_router)

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
    submissions: Optional[Dict[str, str]] = None  # Full content (deprecated)
    buffer_id: Optional[str] = None               # Recommended
    target_file: str
    preprocessing_options: Optional[Dict[str, bool]] = None
    mode: str = "direct"  # "direct" or "preprocess"


class PreprocessRequest(BaseModel):
    submissions: Optional[Dict[str, str]] = None
    buffer_id: Optional[str] = None
    preprocessing_options: Optional[Dict[str, bool]] = None


class CompareRequest(BaseModel):
    embeddings: Dict[str, dict]  # Preprocessed embeddings
    target_file: str
    preprocessing_options: Optional[Dict[str, bool]] = None


class ExtractResponse(BaseModel):
    buffer_id: str
    filenames: List[str]
    count: int
    submissions: Optional[Dict[str, str]] = None  # Deprecated


class AnalyzeResponse(BaseModel):
    scores: Dict[str, List[float]]
    submission_names: List[str]
    target_file: str


class PreprocessResponse(BaseModel):
    embeddings: Dict[str, dict]
    count: int
    names: List[str]


class SimilarRegionsRequest(BaseModel):
    """Request to get detailed similar regions between two files."""
    buffer_id: Optional[str] = None
    file1_name: str
    file1_content: Optional[str] = None
    file2_name: str
    file2_content: Optional[str] = None
    block_threshold: float = 0.6  # Minimum similarity for function/class matches
    min_tokens: int = 10  # Minimum tokens for sequence matches


# ===== Endpoints =====


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "service": "Code Similarity API", "version": "2.0.0"}


@app.post("/extract/upload", response_model=ExtractResponse)
async def extract_text_from_upload(
    files: List[UploadFile] = File(...),
    ignore_patterns: Optional[str] = None,
    user: dict = Depends(require_auth),
):
    """Upload files (ZIP, PDF, code files, etc.) and extract text."""
    patterns = ignore_patterns.split(",") if ignore_patterns else None
    all_submissions = {}

    for file in files:
        content = await file.read()

        if file.filename.endswith(".zip"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                submissions = extract_from_zip(tmp_path, patterns)
                all_submissions.update(submissions)
            finally:
                os.unlink(tmp_path)
        else:
            text = extract_text(content, file.filename)
            if text.strip():
                all_submissions[file.filename] = text

    buffer = get_buffer()
    buffer_id = buffer.save_submissions(all_submissions)

    return ExtractResponse(
        buffer_id=buffer_id,
        filenames=list(all_submissions.keys()),
        count=len(all_submissions),
        submissions={}
    )



@app.post("/extract/urls", response_model=ExtractResponse)
async def extract_text_from_urls(request: GitHubRequest, user: dict = Depends(require_auth)):
    """Process URLs (GitHub repos or direct files) and extract text."""
    all_submissions = {}

    for url in request.urls:
        try:
            # Supports both GitHub repos and direct file URLs now
            submissions = extract_from_url(url, request.ignore_patterns)
            all_submissions.update(submissions)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Error processing {url}: {str(e)}"
            )

    # Save URLs to user's history
    if request.urls:
        save_github_urls(user["username"], request.urls)

    buffer = get_buffer()
    buffer_id = buffer.save_submissions(all_submissions)

    return ExtractResponse(
        buffer_id=buffer_id,
        filenames=list(all_submissions.keys()),
        count=len(all_submissions),
        submissions={}
    )


@app.get("/github-history")
async def get_github_history(
    instance_id: Optional[str] = None,
    user: dict = Depends(require_auth)
):
    """Get user's GitHub URL history (optionally filtered by instance)."""
    history = get_user_github_history(user["username"], instance_id)
    return {"history": history, "count": len(history)}


# ===== Instance Management Endpoints =====


@app.get("/instances")
async def list_instances(user: dict = Depends(require_auth)):
    """List all instances for the current user."""
    instances = get_user_instances(user["username"])
    return {"instances": instances, "count": len(instances)}


@app.post("/instances")
async def create_new_instance(
    name: str,
    description: str = "",
    user: dict = Depends(require_auth)
):
    """Create a new instance."""
    instance = create_instance(user["username"], name, description)
    return {"instance": instance, "message": "Instance created successfully"}


@app.get("/instances/{instance_id}")
async def get_instance_details(instance_id: str, user: dict = Depends(require_auth)):
    """Get details of a specific instance."""
    instance = get_instance(instance_id, user["username"])
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")
    return {"instance": instance}


@app.put("/instances/{instance_id}/urls")
async def update_urls(
    instance_id: str,
    urls: List[str],
    user: dict = Depends(require_auth)
):
    """Update GitHub URLs for an instance."""
    instance = get_instance(instance_id, user["username"])
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")
    
    update_instance_urls(instance_id, user["username"], urls)
    return {"message": "URLs updated successfully", "urls": urls}


@app.delete("/instances/{instance_id}")
async def delete_instance_endpoint(instance_id: str, user: dict = Depends(require_auth)):
    """Delete an instance."""
    instance = get_instance(instance_id, user["username"])
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")
    
    delete_instance(instance_id, user["username"])
    return {"message": "Instance deleted successfully"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_similarity(
    request: AnalyzeRequest, user: dict = Depends(require_auth)
):
    """
    Calculate similarity scores between a target file and other submissions.

    Modes:
    - "direct": One-pass parallel processing (memory efficient)
    - "preprocess": Uses pre-stored embeddings (for after /preprocess call)
    """
    if request.target_file not in request.submissions:
        raise HTTPException(
            status_code=400,
            detail=f"Target file '{request.target_file}' not found in submissions",
        )

    options = request.preprocessing_options or {
        "remove_comments": True,
        "normalize_whitespace": True,
        "preserve_variable_names": True,
        "preserve_literals": False,
    }

    checker = FlexibleCodeSimilarityChecker(preprocessing_options=options)

    # Resolve submissions: either from request or from buffer
    submissions = request.submissions
    if request.buffer_id:
        buffer = get_buffer()
        submissions = buffer.get_submissions(request.buffer_id)
    
    if not submissions:
        raise HTTPException(status_code=400, detail="No submissions provided or buffer expired")
    
    if request.target_file not in submissions:
        raise HTTPException(
            status_code=400,
            detail=f"Target file '{request.target_file}' not found",
        )

    # Use direct mode - parallel process + compare + discard
    result = checker.analyze_direct(request.target_file, submissions)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return AnalyzeResponse(
        scores=result["scores"],
        submission_names=result["submission_names"],
        target_file=result["target_file"],
    )


@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_submissions(
    request: PreprocessRequest, user: dict = Depends(require_auth)
):
    """
    Parallel preprocess all submissions and return embeddings.
    Raw text is discarded, only embeddings are returned.
    Use this when you want to select a target AFTER preprocessing.
    """
    options = request.preprocessing_options or {
        "remove_comments": True,
        "normalize_whitespace": True,
        "preserve_variable_names": True,
        "preserve_literals": False,
    }

    checker = FlexibleCodeSimilarityChecker(preprocessing_options=options)
    
    submissions = request.submissions
    if request.buffer_id:
        buffer = get_buffer()
        submissions = buffer.get_submissions(request.buffer_id)
        
    if not submissions:
        raise HTTPException(status_code=400, detail="No submissions provided or buffer expired")

    embeddings = checker.preprocess_all(submissions)
    
    # Save back to buffer if available for fast /compare later
    if request.buffer_id:
        buffer = get_buffer()
        for name, emb_data in embeddings.items():
            buffer.save_preprocessed(request.buffer_id, name, emb_data)

    return PreprocessResponse(
        embeddings=embeddings, count=len(embeddings), names=list(embeddings.keys())
    )


@app.post("/compare", response_model=AnalyzeResponse)
async def compare_preprocessed(
    request: CompareRequest, user: dict = Depends(require_auth)
):
    """
    Compare target against preprocessed embeddings.
    Use this after calling /preprocess endpoint.
    """
    if request.target_file not in request.embeddings:
        raise HTTPException(
            status_code=400,
            detail=f"Target file '{request.target_file}' not found in embeddings",
        )

    options = request.preprocessing_options or {}
    checker = FlexibleCodeSimilarityChecker(preprocessing_options=options)

    result = checker.compare_preprocessed(request.target_file, request.embeddings)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return AnalyzeResponse(
        scores=result["scores"],
        submission_names=result["submission_names"],
        target_file=result["target_file"],
    )


@app.post("/extract-text")
async def extract_single_file(file: UploadFile = File(...)):
    """Extract text from a single file."""
    content = await file.read()
    text = extract_text(content, file.filename)
    return {"filename": file.filename, "text": text, "length": len(text)}


@app.post("/upload-individual", response_model=ExtractResponse)
async def upload_individual_files(
    files: List[UploadFile] = File(...),
    ignore_patterns: Optional[str] = None,
    user: dict = Depends(require_auth),
):
    """Upload files where each file/ZIP = ONE submission."""
    patterns = ignore_patterns.split(",") if ignore_patterns else None
    all_submissions = {}

    for file in files:
        content = await file.read()

        if file.filename.endswith(".zip"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                combined_text = extract_zip_as_single(tmp_path, file.filename, patterns)
                if combined_text.strip():
                    submission_name = file.filename.rsplit(".", 1)[0]
                    all_submissions[submission_name] = combined_text
            finally:
                os.unlink(tmp_path)
        else:
            text = extract_text(content, file.filename)
            if text.strip():
                all_submissions[file.filename] = text

    buffer = get_buffer()
    buffer_id = buffer.save_submissions(all_submissions)

    return ExtractResponse(
        buffer_id=buffer_id,
        filenames=list(all_submissions.keys()),
        count=len(all_submissions),
        submissions={}
    )



@app.post("/similar-regions")
async def get_similar_regions(
    request: SimilarRegionsRequest,
    user: dict = Depends(require_auth)
):
    """
    Get detailed similar regions between two files.
    
    Returns:
    - function_matches: Similar functions/classes found via AST comparison
    - token_matches: Matching token sequences
    - overall_similarity: Overall Levenshtein similarity
    - stats: Analysis statistics
    """
    
    code1 = request.file1_content
    code2 = request.file2_content

    if request.buffer_id:
        buffer = get_buffer()
        submissions = buffer.get_submissions(request.buffer_id)
        code1 = submissions.get(request.file1_name, code1)
        code2 = submissions.get(request.file2_name, code2)

    if not code1 or not code2:
        raise HTTPException(status_code=400, detail="Cannot find file content for comparison")

    result = get_similar_regions(
        code1=code1,
        code2=code2,
        block_threshold=request.block_threshold,
        min_tokens=request.min_tokens,
    )
    
    return {
        "file1": request.file1_name,
        "file2": request.file2_name,
        **result,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
