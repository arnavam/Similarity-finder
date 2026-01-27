
"""
Optimized Code Similarity Checker with parallel processing.

Pure function-based API:
1. analyze_fast: Quick AST comparison, no TF-IDF
2. analyze_vector: Vector search → prune → detailed analysis
3. preprocess_all: Parallel preprocess all → return embeddings
4. compare_cached: Compare target against preprocessed embeddings
5. preprocess_and_cache: Async function for caching with MongoDB/Pinecone
"""

import ast
import asyncio
import copy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from functools import partial
from typing import Dict, List, Optional, Tuple

import Levenshtein
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import vector_engine

import db


_nlp = None


def get_nlp():
    """Loads spaCy when first needed."""
    global _nlp
    if _nlp is None:
        import spacy

        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ===== Default Options =====

DEFAULT_OPTIONS = {
    "remove_comments": True,
    "normalize_whitespace": True,
    "preserve_variable_names": False,
    "preserve_literals": False,
}


# ===== Pure Functions =====


def sequence_similarity(seq1: List, seq2: List) -> float:
    """Calculate similarity between two sequences using SequenceMatcher."""
    if not seq1 or not seq2:
        return 0.0
    return SequenceMatcher(None, seq1, seq2).ratio()


def tokenize_code(code: str) -> List[str]:
    """Tokenize code using spaCy."""
    nlp = get_nlp()
    doc = nlp(code)
    return [token.text for token in doc]


# ===== Preprocessing Functions =====


def _preprocess_code(code: str, options: Dict) -> str:
    """Preprocess code based on options."""
    processed = code

    if options.get("remove_comments", True):
        lines = []
        for line in processed.split("\n"):
            if "#" in line:
                line = line.split("#")[0]
            lines.append(line)
        processed = "\n".join(lines)

        try:
            tree = ast.parse(processed)
            for node in ast.walk(tree):
                if isinstance(node, ast.Expr) and isinstance(
                    node.value, ast.Constant
                ):
                    node.value.s = ""
            processed = ast.unparse(tree)
        except:
            pass

    if options.get("normalize_whitespace", True):
        lines = []
        for line in processed.split("\n"):
            line = " ".join(line.split())
            if line:
                lines.append(line)
        processed = "\n".join(lines)

    # Handle literal preservation/anonymization
    if not options.get("preserve_literals", False):
        try:
            tree = ast.parse(processed)
            modified = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant):
                    if isinstance(node.value, str):
                        if node.value != "":
                            node.value = "STR"
                            modified = True
                    elif isinstance(node.value, (int, float, complex)):
                        node.value = 0
                        modified = True

            if modified:
                processed = ast.unparse(tree)
        except:
            pass

    return processed


def _extract_ast_features(code: str, options: Dict) -> dict:
    """Extract AST features while preserving information based on options."""
    try:
        tree = ast.parse(code)
        features = {
            "node_types": [],
            "function_calls": [],
            "control_flow": [],
            "operators": [],
        }

        for node in ast.walk(tree):
            features["node_types"].append(type(node).__name__)

            if isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Name)
                    and options.get("preserve_variable_names", False)
                ):
                    features["function_calls"].append(node.func.id)
                else:
                    features["function_calls"].append("FUNC_CALL")

            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                features["control_flow"].append(type(node).__name__)

            if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare)):
                features["operators"].append(type(node.op).__name__)

        return features
    except:
        return {
            "node_types": [],
            "function_calls": [],
            "control_flow": [],
            "operators": [],
        }


# MongoDB 16MB limit - set conservative threshold accounting for JSON overhead
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB per file


def process_to_embeddings(name: str, code: str, options: Optional[Dict] = None, include_dense: bool = False) -> dict:
    """Process code into embeddings dict with all features.
    
    Args:
        name: Filename/identifier
        code: Raw code content
        options: Preprocessing options
        include_dense: If True, compute dense embedding using vector_engine model
        
    Returns:
        Embeddings dict, or {"error": "..."} if file is too large
    """
    options = options or DEFAULT_OPTIONS
    start_time = time.time()
    
    # Check file size before processing
    file_size = len(code.encode("utf-8"))
    if file_size > MAX_FILE_SIZE_BYTES:
        return {
            "error": f"File '{name}' is too large ({file_size / 1024 / 1024:.2f}MB). "
                     f"Maximum size is {MAX_FILE_SIZE_BYTES / 1024 / 1024:.0f}MB."
        }

    processed = _preprocess_code(code, options)
    ast_features = _extract_ast_features(code, options)
    tokens = tokenize_code(processed)

    result = {
        "name": name,
        "processed_text": processed,
        "ast": ast_features,
        "tokens": tokens,
        "raw_code": code,
    }
    
    # Optionally compute dense embedding
    if include_dense and vector_engine.enabled and vector_engine.embed_model:
        result["dense_embedding"] = vector_engine.embed_model.get_text_embedding(processed)

    elapsed = time.time() - start_time

    print(
        f"📥 Processed '{name}': {elapsed:.4f}s | File: {file_size / 1024:.2f}KB | Tokens: {len(tokens)} | Dense: {include_dense}"
    )

    return result


def compare_embeddings(target: dict, other: dict) -> dict:
    """
    Compare two embedding dicts, return all similarity scores.

    Args:
        target: Target submission embeddings
        other: Other submission embeddings to compare

    Returns:
        dict with score types as keys
    """
    scores = {}

    # 1. Raw text similarity (Levenshtein)
    scores["raw"] = Levenshtein.ratio(target["raw_code"], other["raw_code"])

    # 2. Processed text similarity
    scores["processed"] = Levenshtein.ratio(
        target["processed_text"], other["processed_text"]
    )

    # 3. AST structure similarity
    ast_seq1 = list(target["ast"]["node_types"])
    ast_seq2 = list(other["ast"]["node_types"])

    if target["ast"]["function_calls"] and other["ast"]["function_calls"]:
        ast_seq1.extend(target["ast"]["function_calls"])
        ast_seq2.extend(other["ast"]["function_calls"])

    scores["ast"] = sequence_similarity(ast_seq1, ast_seq2)

    # 4. Token-based similarity
    scores["token"] = sequence_similarity(target["tokens"], other["tokens"])

    return scores


def _process_and_compare(
    name: str, code: str, target_emb: dict, options: Dict
) -> Tuple[str, dict]:
    """
    Process + Compare in one step for direct mode.
    Embeddings are discarded after scoring (memory freed).
    """
    emb = process_to_embeddings(name, code, options)
    scores = compare_embeddings(target_emb, emb)
    return name, scores


def _compute_cosine_similarity(texts: List[str]) -> List[float]:
    """Compute TF-IDF cosine similarity of texts[0] against texts[1:]."""
    if len(texts) <= 1:
        return []
    tfidf_matrix = TfidfVectorizer().fit_transform(texts)
    cosine_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    return [float(s) for s in cosine_scores[0]]


def _compute_dense_embeddings(texts: List[str]) -> List[List[float]]:
    """Compute dense embeddings using vector_engine's embedding model."""
    if not vector_engine.enabled or not vector_engine.embed_model:
        return []
    
    embeddings = []
    for text in texts:
        emb = vector_engine.embed_model.get_text_embedding(text)
        embeddings.append(emb)
    return embeddings


def _compute_dense_similarity(target_embedding: List[float], other_embeddings: List[List[float]]) -> List[float]:
    """Compute cosine similarity between target and other dense embeddings."""
    if not target_embedding or not other_embeddings:
        return []
    
    import numpy as np
    target = np.array(target_embedding).reshape(1, -1)
    others = np.array(other_embeddings)
    scores = cosine_similarity(target, others)
    return [float(s) for s in scores[0]]


# ===== Main API Functions =====


def preprocess_all(
    submissions: Dict[str, str],
    options: Optional[Dict] = None,
    max_workers: int = 4,
    include_dense: bool = False,
) -> Dict[str, dict]:
    """
    Parallel preprocess all submissions, return embeddings.

    Args:
        submissions: Dict of {name: raw_code}
        options: Preprocessing options
        max_workers: Number of parallel workers
        include_dense: If True, compute dense embeddings for each submission

    Returns:
        Dict of {name: embeddings}
    """
    options = {**DEFAULT_OPTIONS, **(options or {})}
    start_time = time.time()
    process = psutil.Process()
    start_ram = process.memory_info().rss / (1024 * 1024)

    embeddings = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_to_embeddings, name, code, options, include_dense): name
            for name, code in submissions.items()
        }

        for future in as_completed(futures):
            name = futures[future]
            embeddings[name] = future.result()

    elapsed = time.time() - start_time
    end_ram = process.memory_info().rss / (1024 * 1024)

    print(
        f"📊 Preprocessed {len(embeddings)} submissions: {elapsed:.4f}s | RAM: {end_ram:.1f}MB (delta: {end_ram - start_ram:+.2f}MB)"
    )

    return embeddings


def preprocess_with_cache(
    submissions: Dict[str, str],
    existing_embeddings: Dict[str, dict],
    options: Optional[Dict] = None,
    max_workers: int = 4,
    include_dense: bool = False,
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """
    Smart preprocessing with caching support.
    
    - Processes ONLY new files (not in existing_embeddings)
    - Filters out stale embeddings (files no longer in submissions)
    - Returns both merged embeddings and newly created embeddings
    
    Args:
        submissions: Dict of {name: raw_code} - current files
        existing_embeddings: Dict of {name: embeddings} - previously cached
        options: Preprocessing options
        max_workers: Number of parallel workers
        include_dense: If True, compute dense embeddings
    
    Returns:
        Tuple of (all_embeddings, new_embeddings)
        - all_embeddings: Merged dict of valid existing + new embeddings
        - new_embeddings: Only the newly processed embeddings (for saving)
    """
    options = {**DEFAULT_OPTIONS, **(options or {})}
    
    # Find NEW files (in submissions but not in existing embeddings)
    new_files = {
        name: code for name, code in submissions.items() 
        if name not in existing_embeddings
    }
    
    # Find VALID existing embeddings (filter out stale/removed files)
    valid_existing = {
        name: emb for name, emb in existing_embeddings.items() 
        if name in submissions
    }
    
    # Process only new files
    if new_files:
        print(f"🔄 Processing {len(new_files)} new files (skipping {len(valid_existing)} cached)")
        new_embeddings = preprocess_all(new_files, options, max_workers, include_dense)
    else:
        print(f"✅ All {len(valid_existing)} files already preprocessed, using cache")
        new_embeddings = {}
    
    # Merge: valid existing + newly processed
    all_embeddings = {**valid_existing, **new_embeddings}
    
    print(f"💾 Total embeddings: {len(all_embeddings)} ({len(new_embeddings)} new, {len(valid_existing)} cached)")
    
    return all_embeddings, new_embeddings


async def preprocess_and_cache(
    buffer_id: str,
    submissions: Dict[str, str],
    options: Optional[Dict] = None,
    instance_id: Optional[str] = None,
    max_workers: int = 4,
    include_dense: bool = False,
) -> Dict[str, dict]:
    """
    Async preprocessing with caching.
    
    Storage strategy:
    - If instance_id provided: Use Pinecone only (permanent storage)
    - If no instance_id: Use MongoDB buffer only (temporary storage)
    
    Args:
        buffer_id: MongoDB buffer ID for temporary caching
        submissions: Dict of {name: raw_code}
        options: Preprocessing options
        instance_id: If provided, store to Pinecone instead of MongoDB
        max_workers: Number of parallel workers
        include_dense: If True, compute dense embeddings (for temporary vector mode)
        
    Returns:
        Dict of {name: embeddings} - all embeddings (cached + new)
    """
    if not buffer_id:
        return {"error": "buffer_id is required"}
    
    if not submissions:
        return {"error": "No submissions provided"}

    options = {**DEFAULT_OPTIONS, **(options or {})}
    loop = asyncio.get_event_loop()
    
    if instance_id:
        # Pinecone path: Store to Pinecone, no MongoDB caching
        print(f"📦 Using Pinecone storage (instance: {instance_id})")
        
        # Just preprocess everything (Pinecone handles its own caching via namespace check)
        embeddings = await loop.run_in_executor(
            None,
            partial(preprocess_all, submissions, options, max_workers, include_dense)
        )
        
        # Store to Pinecone
        await loop.run_in_executor(
            None,
            partial(vector_engine.ingest_code, submissions, instance_id)
        )
        print(f"✅ Stored to Pinecone namespace: {instance_id}")
        
    else:
        # MongoDB path: Use buffer for temporary caching
        print(f"📦 Using MongoDB buffer (buffer_id: {buffer_id})")
        
        # Get existing embeddings from MongoDB
        existing_embeddings = await db.get_preprocessed_batch(buffer_id)
        
        # Smart caching - only process new files
        embeddings, new_embeddings = await loop.run_in_executor(
            None,
            partial(preprocess_with_cache, submissions, existing_embeddings, options, max_workers, include_dense)
        )
        
        # Save NEW embeddings to MongoDB
        for name, emb_data in new_embeddings.items():
            await db.save_preprocessed(buffer_id, name, emb_data)
        
        if new_embeddings:
            print(f"✅ Saved {len(new_embeddings)} new embeddings to MongoDB buffer")
    
    return embeddings


def compare_cached(
    target_name: str,
    embeddings: Dict[str, dict],
    options: Optional[Dict] = None,
) -> dict:
    """
    Compare target against all other preprocessed embeddings.

    Args:
        target_name: Which submission to compare against
        embeddings: Dict of {name: embeddings}
        options: Preprocessing options (for TF-IDF)

    Returns:
        dict with scores and submission_names
    """
    options = options or DEFAULT_OPTIONS

    if not embeddings:
        return {"error": "No embeddings provided"}

    if target_name not in embeddings:
        return {"error": f"Target '{target_name}' not found in preprocessed data"}

    start_time = time.time()
    target_emb = embeddings[target_name]

    all_scores = {
        "raw_scores": [],
        "processed_scores": [],
        "ast_scores": [],
        "token_scores": [],
        "cosine_scores": [],
    }
    submission_names = []

    for name, other_emb in embeddings.items():
        if name != target_name:
            scores = compare_embeddings(target_emb, other_emb)
            submission_names.append(name)
            for key in ["raw", "processed", "ast", "token"]:
                all_scores[f"{key}_scores"].append(scores[key])

    # TF-IDF cosine similarity
    all_texts = [target_emb["processed_text"]]
    for name in submission_names:
        all_texts.append(embeddings[name]["processed_text"])
    all_scores["cosine_scores"] = _compute_cosine_similarity(all_texts)
    
    # Dense vector similarity (if embeddings have dense_embedding)
    if "dense_embedding" in target_emb:
        target_dense = target_emb["dense_embedding"]
        other_dense = [embeddings[name].get("dense_embedding") for name in submission_names]
        # Filter out None values - only compute if all have dense embeddings
        if all(e is not None for e in other_dense):
            all_scores["vector_scores"] = _compute_dense_similarity(target_dense, other_dense)

    elapsed = time.time() - start_time
    print(
        f"🔍 Comparison: {elapsed:.4f}s | Compared against {len(submission_names)} submissions"
    )

    return {
        "scores": all_scores,
        "submission_names": submission_names,
        "target_file": target_name,
    }


def analyze_fast(
    target_name: str,
    submissions: Dict[str, str],
    options: Optional[Dict] = None,
    max_workers: int = 4,
) -> dict:
    """
    Fast, memory-efficient analysis WITHOUT TF-IDF cosine similarity.
    Process target first, then parallel process+compare others,
    discarding embeddings after each comparison.

    Args:
        target_name: Which submission to compare against
        submissions: Dict of {name: raw_code}
        options: Preprocessing options
        max_workers: Number of parallel workers

    Returns:
        dict with scores (raw, processed, ast, token) and submission_names
    """
    options = {**DEFAULT_OPTIONS, **(options or {})}
    start_time = time.time()
    process = psutil.Process()
    start_ram = process.memory_info().rss / (1024 * 1024)

    if not submissions:
        return {"error": "No submissions provided"}

    if target_name not in submissions:
        return {"error": f"Target '{target_name}' not found"}

    target_emb = process_to_embeddings(target_name, submissions[target_name], options)

    all_scores = {
        "raw_scores": [],
        "processed_scores": [],
        "ast_scores": [],
        "token_scores": [],
    }
    submission_names = []

    other_submissions = [(n, c) for n, c in submissions.items() if n != target_name]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_and_compare, name, code, target_emb, options): name
            for name, code in other_submissions
        }

        for future in as_completed(futures):
            name, scores = future.result()
            submission_names.append(name)
            for key in ["raw", "processed", "ast", "token"]:
                all_scores[f"{key}_scores"].append(scores[key])

    elapsed = time.time() - start_time
    end_ram = process.memory_info().rss / (1024 * 1024)

    print(
        f"⚡ Fast analysis: {elapsed:.4f}s | RAM: {end_ram:.1f}MB (delta: {end_ram - start_ram:+.2f}MB)"
    )

    return {
        "scores": all_scores,
        "submission_names": submission_names,
        "target_file": target_name,
    }


# Alias for API compatibility
analyze_direct = analyze_fast



def analyze_vector(
    target_name: str,
    submissions: Dict[str, str],
    namespace: str,
    options: Optional[Dict] = None,
    max_workers: int = 4,
    index_name: str = None,
) -> dict:
    """
    Vector-based analysis:
    1. Ingest all submissions into Vector DB (dense embeddings)
    2. Query Vector DB for top-k candidates similar to target
    3. Run fast analysis (AST, token, Levenshtein) on candidates
    4. Use dense vector scores instead of TF-IDF cosine

    Args:
        index_name: Which Pinecone index to use (default: vector_engine.INDEX_NAME)
    """
    options = {**DEFAULT_OPTIONS, **(options or {})}
    index_name = index_name or vector_engine.INDEX_NAME
    start_time = time.time()

    if not submissions:
        return {"error": "No submissions provided"}

    if not namespace:
        return {"error": "namespace (instance_id) is required for vector mode"}

    if not vector_engine.enabled:
        return {"error": "Vector engine is not available. Check PINECONE_API_KEY and server logs."}

    # 1. Ingest (Pinecone upsert is idempotent)
    print(f"🌲 Ingesting {len(submissions)} files into namespace '{namespace}' (index: {index_name})...")
    success = vector_engine.ingest_code(submissions, namespace, index_name=index_name)
    
    if not success:
        return {"error": "Vector ingestion failed. Please try again."}

    # 2. Retrieve Candidates with vector scores
    target_code = submissions.get(target_name)
    if not target_code:
        return {"error": f"Target '{target_name}' not found"}

    candidates = vector_engine.query_similar(target_code, namespace, index_name=index_name, top_k=10)
    
    # Capture vector scores (dense embedding similarity from Pinecone)
    vector_score_map = {c["filename"]: c["score"] for c in candidates}
    candidate_names = [c["filename"] for c in candidates if c["filename"] != target_name]

    # Build subset for detailed analysis
    subset_submissions = {target_name: target_code}
    for name in candidate_names:
        if name in submissions:
            subset_submissions[name] = submissions[name]

    print(f"🎯 Vector Pruning: {len(submissions)} → {len(subset_submissions)} candidates")

    # 3. Fast Analysis on Subset (AST, token, Levenshtein - NO TF-IDF)
    result = analyze_fast(target_name, subset_submissions, options, max_workers)
    
    if "error" in result:
        return result

    # 4. Add vector_scores (dense embeddings) instead of cosine_scores (TF-IDF)
    # Match order with submission_names in result
    vector_scores = [
        vector_score_map.get(name, 0.0) for name in result["submission_names"]
    ]
    result["scores"]["vector_scores"] = vector_scores
    result["mode"] = "vector"

    elapsed = time.time() - start_time
    print(f"🚀 Vector Analysis Complete: {elapsed:.4f}s")

    return result