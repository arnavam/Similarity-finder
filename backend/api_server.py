"""
API Server for Code Similarity Checker
REST API wrapper for text extraction and similarity analysis.

Run with: uvicorn api_server:app --reload --port 8000

Modes:
- fast: Quick AST comparison, no TF-IDF, no storage
- sparse: TF-IDF analysis
    - with instance_id: Uses Pinecone INDEX_NAME2 (persistent)
    - without instance_id: Uses MongoDB buffer (temporary)
- vector: Dense vector search via Pinecone INDEX_NAME (requires instance_id)

"""

import asyncio
import os
import tempfile
from typing import Dict, List, Literal, Optional

from auth import require_auth
from auth import router as auth_router
from code_extractor import (
    extract_from_url,
    extract_from_zip,
    extract_text,
    extract_zip_as_single,
)
from code_region_similarity_finder import get_similar_regions
from code_similarity_finder import (
    analyze_fast,
    analyze_vector,
    compare_cached,
    preprocess_and_cache,
)
from fastapi import Body, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import vector_engine

import db

app = FastAPI(
    title="Copyadi Finder API",
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


# Startup event - initialize database indexes and vector engine
@app.on_event("startup")
async def startup_event():
    # Create MongoDB indexes
    await db.ensure_indexes()



# ===== Shared Models =====


DEFAULT_PREPROCESSING_OPTIONS = {
    "remove_comments": True,
    "normalize_whitespace": True,
    "preserve_variable_names": True,
    "preserve_literals": False,
}


class ProcessingContext(BaseModel):
    """Common context for processing endpoints."""
    buffer_id: Optional[str] = None
    instance_id: Optional[str] = None
    preprocessing_options: Optional[Dict[str, bool]] = None


class ExtractionOptions(BaseModel):
    """Common options for extraction endpoints."""
    ignore_patterns: Optional[List[str]] = None
    ignore: bool = True  # If True, skip matching files. If False, only include matching.


# ===== Response Models =====


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


# ===== Extraction Endpoints =====

@app.post("/extract/urls", response_model=ExtractResponse)
async def extract_text_from_urls(
    urls: List[str] = Body(...),
    instance_id: Optional[str] = Body(None),
    options: ExtractionOptions = Body(default_factory=ExtractionOptions),
    user: dict = Depends(require_auth),
):
    """Process URLs (GitHub repos, Medium articles, or direct files) and extract text.
    
    Args:
        urls: List of URLs to process
        instance_id: Optional instance to save URLs to
        options.ignore_patterns: Patterns to match
        options.ignore: If True (default), skip files matching patterns. If False, only include matching.
    """
    all_submissions = {}

    # Process all URLs in parallel (extract_from_url is now async)
    try:
        results = await asyncio.gather(*[
            extract_from_url(url, options.ignore_patterns, options.ignore) for url in urls
        ])
        for submissions in results:
            all_submissions.update(submissions)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error processing URLs: {str(e)}"
        )
    # Save URLs to user's history
    if urls:
        await db.save_github_urls(user["username"], urls, instance_id)

    buffer_id = await db.save_submissions(all_submissions)

    return ExtractResponse(
        buffer_id=buffer_id,
        filenames=list(all_submissions.keys()),
        count=len(all_submissions),
    )


@app.post("/extract/files", response_model=ExtractResponse)
async def extract_text_from_upload(
    files: List[UploadFile] = File(...),
    options: ExtractionOptions = Body(default_factory=ExtractionOptions),
    user: dict = Depends(require_auth),
):
    """Extract text from files (ZIP, PDF, code files, etc.) and extract text.
    
    Args:
        options.ignore_patterns: Patterns to match
        options.ignore: If True (default), skip files matching patterns. If False, only include matching.
    """
    all_submissions = {}

    for file in files:
        try:
            content = await file.read()

            if file.filename.endswith(".zip"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                try:
                    # CPU-bound: Zip extraction
                    submissions = await run_in_threadpool(
                        extract_from_zip, tmp_path, options.ignore_patterns, None, options.ignore
                    )
                    all_submissions.update(submissions)
                finally:
                    os.unlink(tmp_path)
            else:
                # CPU-bound: Text extraction (PDF/Docx/etc)
                text = await run_in_threadpool(extract_text, content, file.filename)
                if text.strip():
                    all_submissions[file.filename] = text
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Error processing '{file.filename}': {e}"
            )
    buffer_id = await db.save_submissions(all_submissions)

    return ExtractResponse(
        buffer_id=buffer_id,
        filenames=list(all_submissions.keys()),
        count=len(all_submissions),
    )





@app.post("/extract/zip", response_model=ExtractResponse)
async def upload_individual_files(
    files: List[UploadFile] = File(...),
    options: ExtractionOptions = Body(default_factory=ExtractionOptions),
    user: dict = Depends(require_auth),
):
    """Extract text from files where each file/ZIP = ONE submission.
    
    Args:
        options.ignore_patterns: Patterns to match
        options.ignore: If True (default), skip files matching patterns. If False, only include matching.
    """
    all_submissions = {}

    for file in files:
        content = await file.read()

        if file.filename.endswith(".zip"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                # CPU-bound: Zip extraction
                combined_text = await run_in_threadpool(
                    extract_zip_as_single, tmp_path, file.filename, options.ignore_patterns, options.ignore
                )
                if combined_text.strip():
                    submission_name = file.filename.rsplit(".", 1)[0]
                    all_submissions[submission_name] = combined_text
            finally:
                os.unlink(tmp_path)
        else:
            # CPU-bound: Text extraction
            text = await run_in_threadpool(extract_text, content, file.filename)
            if text.strip():
                all_submissions[file.filename] = text

    buffer_id = await db.save_submissions(all_submissions)

    return ExtractResponse(
        buffer_id=buffer_id,
        filenames=list(all_submissions.keys()),
        count=len(all_submissions),
    )



# ===== Instance Management Endpoints =====


@app.get("/instances")
async def list_instances(user: dict = Depends(require_auth)):
    """List all instances for the current user."""
    instances = await db.get_user_instances(user["username"])
    return {"instances": instances, "count": len(instances)}


@app.post("/instances")
async def create_new_instance(
    name: str,
    description: str = "",
    discord_channel_id: str = None,
    user: dict = Depends(require_auth),
):
    """Create a new instance. Optionally link to a Discord channel for auto-loading URLs."""
    instance = await db.create_instance(user["username"], name, description, discord_channel_id)
    return {"instance": instance, "message": "Instance created successfully"}


@app.get("/instances/{instance_id}")
async def get_instance_details(
    instance_id: str,
    sync_discord: bool = False,
    force_refresh: bool = False,
    user: dict = Depends(require_auth),
):
    """
    Get details of a specific instance.

    Args:
        sync_discord: If True, scrape Discord for URLs. If False, only read from DB.
        force_refresh: If True, ignore cache and scrape all messages.
    """
    instance = await db.get_instance(instance_id, user["username"])
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")

    if sync_discord:
        try:
            from discord_scraper import get_or_scrape_urls

            channel_id = instance.get("discord_channel_id")
            if not channel_id:
                raise Exception("This instance has no Discord channel ID linked")

            tag = instance.get("name")
            print(
                f'''🔄 Syncing Discord URLs for tag: #{tag}, channel: {
                    channel_id
                }, force={force_refresh}'''
            )

            urls = get_or_scrape_urls(channel_id, tag, force_refresh=force_refresh)
            if not urls:
                raise Exception(f"No URLs found in Discord for #{tag}")

            print(f"📥 Discord returned {len(urls)} URLs")

            # Merge logic
            existing = instance.get("github_urls", [])
            merged = urls + [u for u in existing if u not in urls]
            instance["github_urls"] = merged

            await db.update_instance_urls(instance_id, user["username"], merged)
            print(
                f'''💾 Saved {len(merged)} URLs to instance {
                    instance.get('name', instance_id)
                }'''
            )

        except Exception as e:
            print(f"⚠️ Discord sync skipped: {e}")

    return {"instance": instance}


@app.put("/instances/{instance_id}/urls")
async def update_urls(
    instance_id: str, urls: List[str], user: dict = Depends(require_auth)
):
    """Update GitHub URLs for an instance."""
    instance = await db.get_instance(instance_id, user["username"])
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")

    await db.update_instance_urls(instance_id, user["username"], urls)
    return {"message": "URLs updated successfully", "urls": urls}


@app.delete("/instances/{instance_id}")
async def delete_instance_endpoint(
    instance_id: str, user: dict = Depends(require_auth)
):
    """Delete an instance."""
    instance = await db.get_instance(instance_id, user["username"])
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")

    await db.delete_instance(instance_id, user["username"])
    return {"message": "Instance deleted successfully"}



# ===== Analysis Endpoints =====

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_similarity(
    target_file: str = Body(...),
    mode: Literal["fast", "sparse", "vector"] = Body("fast"),
    context: ProcessingContext = Body(default_factory=ProcessingContext),
    user: dict = Depends(require_auth),
):
    """
    Calculate similarity scores between a target file and other submissions.

    Modes:
    - "fast": Quick AST comparison, no TF-IDF, no storage
    - "sparse": Full analysis with TF-IDF, caches to MongoDB
    - "vector": Vector search via Pinecone (requires instance_id)
    """

    # Resolve buffer contents (single query)
    docs = await db.get_buffer_contents(context.buffer_id)
    if not docs:
        raise HTTPException(status_code=404, detail="Buffer not found or expired")

    submissions = {doc.name: doc.raw_text for doc in docs}

    if mode == "fast":
        # Fast mode: no caching, no TF-IDF
        result = await run_in_threadpool(
            analyze_fast, target_file, submissions, context.preprocessing_options
        )

    elif mode == "sparse":
        if context.instance_id:
            # Persistent mode: use Pinecone INDEX_NAME2 for sparse/TF-IDF vectors
            result = await run_in_threadpool(
                analyze_vector,
                target_file,
                submissions,
                namespace=context.instance_id,
                options=context.preprocessing_options,
                index_name=vector_engine.INDEX_NAME2,  # Sparse index
            )
        else:
            # Temporary mode: use MongoDB buffer
            embeddings = {doc.name: doc.preprocessed for doc in docs if doc.preprocessed}
            
            # Only preprocess if cache is incomplete
            if len(embeddings) < len(submissions) or target_file not in embeddings:
                embeddings = await preprocess_and_cache(
                    context.buffer_id, submissions, context.preprocessing_options
                )
                
            result = await run_in_threadpool(
                compare_cached, target_file, embeddings, context.preprocessing_options
            )

    elif mode == "vector":
        # Dense vector mode: use Pinecone INDEX_NAME (default)
        result = await run_in_threadpool(
            analyze_vector,
            target_file,
            submissions,
            namespace=context.instance_id,
            options=context.preprocessing_options,
            index_name=vector_engine.INDEX_NAME,  # Dense index
        )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return AnalyzeResponse(
        scores=result["scores"],
        submission_names=result["submission_names"],
        target_file=result["target_file"],
    )


@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_submissions(
    context: ProcessingContext = Body(...),
    user: dict = Depends(require_auth),
):
    """Preprocess submissions and cache embeddings to MongoDB. Use /compare to run comparison."""
    submissions = await db.get_submissions(context.buffer_id)
    embeddings = await preprocess_and_cache(context.buffer_id, submissions, context.preprocessing_options, context.instance_id)

    if "error" in embeddings:
        raise HTTPException(status_code=400, detail=embeddings["error"])

    return PreprocessResponse(embeddings=embeddings, count=len(embeddings), names=list(embeddings.keys()))


@app.post("/compare", response_model=AnalyzeResponse)
async def compare_preprocessed(
    target_file: str = Body(...),
    buffer_id: Optional[str] = Body(None),
    preprocessing_options: Dict[str, bool] = Body(...),
    embeddings: Optional[Dict[str, dict]] = Body(None),
    user: dict = Depends(require_auth),
):
    """Compare target file against cached embeddings. Requires /preprocess or direct embeddings."""
    resolved = (await db.get_preprocessed_batch(buffer_id) if buffer_id else None) or embeddings
    result = await run_in_threadpool(compare_cached, target_file, resolved, preprocessing_options)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return AnalyzeResponse(
        scores=result["scores"],
        submission_names=result["submission_names"],
        target_file=result["target_file"],
    )

@app.post("/similar-regions")
async def compute_similar_regions(
    file1_name: str = Body(...),
    file2_name: str = Body(...),
    buffer_id: Optional[str] = Body(None),
    block_threshold: float = Body(0.6),
    min_tokens: int = Body(2),
    user: dict = Depends(require_auth),
):
    """
    Get detailed similar regions between two files.

    Returns:
    - function_matches: Similar functions/classes found via AST comparison
    - token_matches: Matching token sequences
    - overall_similarity: Overall Levenshtein similarity
    - stats: Analysis statistics
    """
    code1 = None
    code2 = None
    
    if buffer_id:
        submissions = await db.get_submissions(buffer_id)
        code1 = submissions.get(file1_name)
        code2 = submissions.get(file2_name)

    if not code1 or not code2:
        raise HTTPException(
            status_code=400, detail="Cannot find file content for comparison"
        )

    result = get_similar_regions(
        code1=code1,
        code2=code2,
        block_threshold=block_threshold,
        min_tokens=min_tokens,
    )

    return {
        "file1": file1_name,
        "file2": file2_name,
        **result,
    }


@app.get("/history")
async def get_history(
    instance_id: Optional[str] = None, user: dict = Depends(require_auth)
):
    """Get user's submission history (optionally filtered by instance)"""
    history = await db.get_user_github_history(user["username"], instance_id)
    return {"history": history, "count": len(history)}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
