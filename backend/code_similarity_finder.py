"""
Optimized Code Similarity Checker with parallel processing.

Pure function-based API:
1. analyze_direct: Process target → parallel process+compare others → discard
2. analyze_hybrid: Vector search → prune → direct analysis
3. preprocess_all: Parallel preprocess all → return embeddings
4. compare_preprocessed: Compare target against preprocessed embeddings
"""

import ast
import copy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import Levenshtein
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vector_engine import get_vector_engine


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


# ===== Core Processing Functions =====


def process_to_embeddings(name: str, code: str, options: Optional[Dict] = None) -> dict:
    """Process code into embeddings dict with all features."""
    options = options or DEFAULT_OPTIONS
    start_time = time.time()

    processed = _preprocess_code(code, options)
    ast_features = _extract_ast_features(code, options)
    tokens = tokenize_code(processed)

    elapsed = time.time() - start_time
    file_size = len(code.encode("utf-8"))

    print(
        f"📥 Processed '{name}': {elapsed:.4f}s | File: {file_size / 1024:.2f}KB | Tokens: {len(tokens)}"
    )

    return {
        "name": name,
        "processed_text": processed,
        "ast": ast_features,
        "tokens": tokens,
        "raw_code": code,
    }


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


# ===== Main API Functions =====


def preprocess_all(
    submissions: Dict[str, str],
    options: Optional[Dict] = None,
    max_workers: int = 4,
) -> Dict[str, dict]:
    """
    Parallel preprocess all submissions, return embeddings.

    Args:
        submissions: Dict of {name: raw_code}
        options: Preprocessing options
        max_workers: Number of parallel workers

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
            executor.submit(process_to_embeddings, name, code, options): name
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


def compare_preprocessed(
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


def analyze_full(
    target_name: str,
    submissions: Dict[str, str],
    options: Optional[Dict] = None,
    max_workers: int = 4,
) -> dict:
    """
    Full analysis WITH TF-IDF cosine similarity.
    Preprocesses all submissions first, then compares using compare_preprocessed.
    Higher memory usage but includes all scoring methods.

    Args:
        target_name: Which submission to compare against
        submissions: Dict of {name: raw_code}
        options: Preprocessing options
        max_workers: Number of parallel workers

    Returns:
        dict with scores (raw, processed, ast, token, cosine) and submission_names
    """
    options = {**DEFAULT_OPTIONS, **(options or {})}
    start_time = time.time()
    process = psutil.Process()
    start_ram = process.memory_info().rss / (1024 * 1024)

    if target_name not in submissions:
        return {"error": f"Target '{target_name}' not found"}

    # Preprocess all submissions (keeps embeddings in memory)
    embeddings = preprocess_all(submissions, options, max_workers)

    # Compare using existing function
    result = compare_preprocessed(target_name, embeddings, options)

    elapsed = time.time() - start_time
    end_ram = process.memory_info().rss / (1024 * 1024)

    print(
        f" Full analysis: {elapsed:.4f}s | RAM: {end_ram:.1f}MB (delta: {end_ram - start_ram:+.2f}MB)"
    )

    return result


def analyze_hybrid(
    target_name: str,
    submissions: Dict[str, str],
    namespace: str,
    options: Optional[Dict] = None,
    max_workers: int = 4,
) -> dict:
    """
    Refined analysis:
    1. Ingest all submissions into Vector DB (Namespace isolated)
    2. Query Vector DB for top-k candidates similar to target
    3. Run detailed 'Direct' analysis ONLY on those candidates
    """
    options = {**DEFAULT_OPTIONS, **(options or {})}
    start_time = time.time()
    vector_engine = get_vector_engine()

    if not vector_engine.enabled:
        print("⚠️ Vector engine disabled. Falling back to Direct Mode.")
        return analyze_full(target_name, submissions, options, max_workers)

    # 1. Ingest (if not already there - naive check, ideally we track state)
    # For this implementation, we re-ingest to ensure freshness.
    # Pinecone upsert is idempotent.
    print(f"🌲 Ingesting {len(submissions)} files into namespace '{namespace}'...")
    vector_engine.ingest_code(submissions, namespace)

    # 2. Retrieve Candidates
    target_code = submissions.get(target_name)
    if not target_code:
        return {"error": f"Target '{target_name}' not found"}

    candidates = vector_engine.query_similar(target_code, namespace, top_k=10)
    candidate_names = [c["filename"] for c in candidates if c["filename"] != target_name]

    # Add target back if missing (shouldn't happen if top_k includes it)
    # We need target + candidates for direct comparison
    subset_submissions = {target_name: target_code}
    for name in candidate_names:
        if name in submissions:
            subset_submissions[name] = submissions[name]

    print(f"🎯 Hybrid Pruning: Reduced {len(submissions)} -> {len(subset_submissions)} candidates")

    # 3. Direct Analysis on Subset
    result = analyze_full(target_name, subset_submissions, options, max_workers)
    result["mode"] = "hybrid"

    elapsed = time.time() - start_time
    print(f"🚀 Hybrid Analysis Complete: {elapsed:.4f}s")

    return result