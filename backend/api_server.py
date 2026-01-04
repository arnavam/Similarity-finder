"""
API Server for Code Similarity Checker
REST API wrapper for text extraction and similarity analysis.

Run with: uvicorn api_server:app --reload --port 8000
"""

import os
import tempfile
from typing import Dict, List, Optional

from auth import (
    create_instance,
    delete_instance,
    get_instance,
    get_user_github_history,
    get_user_instances,
    require_auth,
    save_github_urls,
    update_instance_urls,
)
from auth import router as auth_router
from code_extractor import (
    extract_from_url,
    extract_from_zip,
    extract_text,
    extract_zip_as_single,
)
from code_similarity_finder import (
    analyze_direct,
    analyze_full,
    analyze_hybrid,
    compare_preprocessed as compare_preprocessed_fn,
    preprocess_all,
)
from code_region_similarity_finder import get_similar_regions

from fastapi import Body, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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


# ===== request models =====


class AnalyzeRequest(BaseModel):  # preprocess + compare
    target_file: str
    mode: str = "direct"  # "direct", "preprocess", "hybrid"
    buffer_id: Optional[str] = None
    instance_id: Optional[str] = None
    preprocessing_options: Optional[Dict[str, bool]] = None


class CompareRequest(BaseModel):
    embeddings: Dict[str, dict]  # Preprocessed embeddings
    target_file: str
    preprocessing_options: Optional[Dict[str, bool]] = None


class SimilarRegionsRequest(BaseModel):
    """Request to get detailed similar regions between two files."""

    buffer_id: Optional[str] = None
    file1_name: str
    file1_content: Optional[str] = None
    file2_name: str
    file2_content: Optional[str] = None
    block_threshold: float = 0.6  # Minimum similarity for function/class matches
    min_tokens: int = 2  # Minimum tokens for sequence matches


# ===== response models =====


class ExtractResponse(BaseModel): 
    buffer_id: str 
    filenames: List[str]
    count: int


class AnalyzeResponse(BaseModel):
    scores: Dict[str, List[float]]
    submission_names: List[str]
    target_file: str


class PreprocessResponse(BaseModel):
    embeddings: Dict[str, dict]
    count: int
    names: List[str]


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
# note : add except: block here and pass the response
    buffer = get_buffer()
    buffer_id = buffer.save_submissions(all_submissions)

    return ExtractResponse(
        buffer_id=buffer_id,
        filenames=list(all_submissions.keys()),
        count=len(all_submissions),
    ) #note: why count is passed is there a need ?


@app.post("/extract/urls", response_model=ExtractResponse)
async def extract_text_from_urls(
    urls: List[str] = Body(...),
    ignore_patterns: Optional[List[str]] = Body(None),
    user: dict = Depends(require_auth),
):
    """Process URLs (GitHub repos or direct files) and extract text."""
    all_submissions = {}

    for url in urls:
        try:
            submissions = extract_from_url(url, ignore_patterns)
            all_submissions.update(submissions)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Error processing {url}: {str(e)}"
            )
    # Save URLs to user's history
    if urls:
        save_github_urls(user["username"], urls)
# note: save the urls only if there is change
    buffer = get_buffer()
    buffer_id = buffer.save_submissions(all_submissions)

    return ExtractResponse(
        buffer_id=buffer_id,
        filenames=list(all_submissions.keys()),
        count=len(all_submissions),
    )


@app.get("/github-history")
async def get_github_history(
    instance_id: Optional[str] = None, user: dict = Depends(require_auth)
):
    """Get user's GitHub URL history (optionally filtered by instance)"""
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
    discord_channel_id: str = None,
    user: dict = Depends(require_auth),
):
    """Create a new instance. Optionally link to a Discord channel for auto-loading URLs."""
    instance = create_instance(user["username"], name, description, discord_channel_id)
    return {"instance": instance, "message": "Instance created successfully"}


@app.get("/instances/{instance_id}")
async def get_instance_details(instance_id: str, user: dict = Depends(require_auth)):
    """Get details of a specific instance. Auto-loads Discord URLs if configured."""
    instance = get_instance(instance_id, user["username"])
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")

    # Auto-load Discord URLs if configured
    if instance.get("discord_channel_id"):
        try:
            from discord_scraper import get_or_scrape_urls, is_discord_configured

            if is_discord_configured():
                tag = instance.get("name", "")
                urls = get_or_scrape_urls(instance["discord_channel_id"], tag)
                if urls:
                    # Merge with existing URLs (Discord URLs first, then manual ones)
                    existing = instance.get("github_urls", [])
                    merged = urls + [u for u in existing if u not in urls]
                    instance["github_urls"] = merged
                    # SAVE the merged URLs to database
                    update_instance_urls(instance_id, user["username"], merged)
                    print(
                        f"💾 Saved {len(merged)} URLs to instance {instance.get('name', instance_id)}"
                    )
        except Exception as e:
            print(f"⚠️ Discord auto-load failed: {e}")

    return {"instance": instance}


@app.put("/instances/{instance_id}/urls")
async def update_urls(
    instance_id: str, urls: List[str], user: dict = Depends(require_auth)
):
    """Update GitHub URLs for an instance."""
    instance = get_instance(instance_id, user["username"])
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")

    update_instance_urls(instance_id, user["username"], urls)
    return {"message": "URLs updated successfully", "urls": urls}


@app.delete("/instances/{instance_id}")
async def delete_instance_endpoint(
    instance_id: str, user: dict = Depends(require_auth)
):
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

    options = request.preprocessing_options or {
        "remove_comments": True,
        "normalize_whitespace": True,
        "preserve_variable_names": True,
        "preserve_literals": False,
    }

    # Resolve submissions from buffer
    submissions = None
    if request.buffer_id:
        buffer = get_buffer()
        submissions = buffer.get_submissions(request.buffer_id)

    if not submissions:
        raise HTTPException(
            status_code=400, detail="No submissions provided or buffer expired"
        )

    if request.target_file not in submissions:
        raise HTTPException(
            status_code=400,
            detail=f"Target file '{request.target_file}' not found",
        )

    # Modes:
    # 1. fast/direct: Fast AST comparison, skips TF-IDF cosine (memory efficient)
    # 2. full: Full analysis with TF-IDF cosine similarity (all scoring methods)
    # 3. hybrid: Vector Search -> Prune -> Direct Analysis (fastest for large sets)

    result = {}

    if request.mode == "hybrid":
        if not request.instance_id:
            # Fallback if no instance_id provided
            result = analyze_direct(request.target_file, submissions, options)
        else:
            result = analyze_hybrid(
                request.target_file, submissions, namespace=request.instance_id, options=options
            )
    elif request.mode == "full":
        # Full analysis with TF-IDF cosine similarity
        result = analyze_full(request.target_file, submissions, options)
    else:
        # Default: fast/direct mode
        result = analyze_direct(request.target_file, submissions, options)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return AnalyzeResponse(
        scores=result["scores"],
        submission_names=result["submission_names"],
        target_file=result["target_file"],
    )


@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_submissions(
    buffer_id: str = Body(...),
    preprocessing_options: Optional[Dict[str, bool]] = Body(None),
    user: dict = Depends(require_auth),
):
    """
    Parallel preprocess all submissions and return embeddings.
    Raw text is discarded, only embeddings are returned.
    Use this when you want to select a target AFTER preprocessing.
    """
    options = preprocessing_options or {
        "remove_comments": True,
        "normalize_whitespace": True,
        "preserve_variable_names": True,
        "preserve_literals": False,
    }

    buffer = get_buffer()
    submissions = buffer.get_submissions(buffer_id)

    if not submissions:
        raise HTTPException(
            status_code=400, detail="No submissions provided"
        )

    embeddings = preprocess_all(submissions, options)

    # Save back to buffer for fast /compare later
    for name, emb_data in embeddings.items():
        buffer.save_preprocessed(buffer_id, name, emb_data)

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

    result = compare_preprocessed_fn(request.target_file, request.embeddings, options)

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
    )


@app.post("/similar-regions")
async def compute_similar_regions(
    request: SimilarRegionsRequest, user: dict = Depends(require_auth)
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
#note: here we only checking for if the id exists is that enough? 
    if not code1 or not code2:
        raise HTTPException(
            status_code=400, detail="Cannot find file content for comparison"
        )

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
