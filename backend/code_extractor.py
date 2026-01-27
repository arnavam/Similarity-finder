"""
Text Extractor Module (LlamaIndex Version)
Drop-in replacement for text_extractor.py using LlamaIndex connectors.
Unified text extraction from various sources: files, ZIPs, GitHub repos, folders.
"""

import fnmatch
import json
import os
import re
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import git
import httpx
from newspaper import Article

# LlamaIndex imports
from llama_index.core import SimpleDirectoryReader


def extract_text(file_path_or_bytes, filename: Optional[str] = None) -> str:
    """
    Extract text from any file using LlamaIndex.

    Args:
        file_path_or_bytes: Either a file path (str) or file bytes
        filename: Required when passing bytes, used to detect file type

    Returns:
        Extracted text as string
    """
    try:
        if isinstance(file_path_or_bytes, bytes):
            # For in-memory files, write to temp file for LlamaIndex
            suffix = os.path.splitext(filename)[1] if filename else ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_path_or_bytes)
                tmp_path = tmp.name
            try:
                # Special handling for Jupyter notebooks - extract code only, no outputs
                if suffix.lower() == ".ipynb":
                    text = _extract_notebook_code(tmp_path)
                else:
                    text = _extract_from_path(tmp_path)
            finally:
                os.unlink(tmp_path)
            return text
        else:
            # For file paths
            # Special handling for Jupyter notebooks - extract code only, no outputs
            if Path(file_path_or_bytes).suffix.lower() == ".ipynb":
                return _extract_notebook_code(file_path_or_bytes)
            return _extract_from_path(file_path_or_bytes)
    except Exception:
        # Fallback: try reading as plain text
        try:
            if isinstance(file_path_or_bytes, bytes):
                return file_path_or_bytes.decode("utf-8", errors="ignore")
            else:
                with open(
                    file_path_or_bytes, "r", encoding="utf-8", errors="ignore"
                ) as f:
                    return f.read()
        except Exception:
            return ""


def _extract_from_path(file_path: str) -> str:
    """
    Extract text from a file path using LlamaIndex SimpleDirectoryReader.
    Auto-detects file type and handles PDFs, DOCX, code files, etc.
    """
    try:
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        if documents:
            return "\n\n".join(doc.text for doc in documents if doc.text)
        return ""
    except Exception:
        # Fallback: read as plain text (works for code files, txt, etc.)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def extract_from_zip(
    zip_path_or_bytes,
    ignore_patterns: Optional[List[str]] = None,
    max_workers: Optional[int] = None,
    ignore: bool = True,
) -> Dict[str, str]:
    """
    each top-level folder/file is ONE submission.

    Handles both:
        submissions.zip -> (student1/, student2/, ..)  ; student1, student2 are submissions
        submissions.zip/container -> (student1/, student2/, ..)  ; student1, student2 are submissions

    Args:
        zip_path_or_bytes: Either a file path or file-like object/bytes
        ignore_patterns: Patterns to match
        max_workers: Max threads for parallel extraction (default: CPU count)
        ignore: If True (default), skip files matching patterns. If False, only include files matching patterns.

    Returns:
        Dict mapping submission name to combined content
    """
    if ignore_patterns is None:
        ignore_patterns = []
    # Always ignore macOS junk folders
    # ignore_patterns = ["__MACOSX", "__MACOSX/*"] + ignore_patterns

    if max_workers is None:
        max_workers = os.cpu_count() or 4

    # Phase 1: Collect file info and bytes (sequential - zipfile not thread-safe)
    files_to_process = []  # list of (submission_name, filename, file_bytes)

    try:
        with zipfile.ZipFile(zip_path_or_bytes, "r") as zip_ref:
            # First, detect if there's a single container folder
            top_level_items = set()
            print("\n=== ZIP FILE STRUCTURE ===")
            for file_info in zip_ref.filelist:
                if _should_ignore(file_info.filename, ignore_patterns) == ignore:
                    continue
                print(f"  {file_info.filename}")
                if file_info.filename.endswith("/"):
                    continue
                parts = file_info.filename.split("/")
                if parts[0]:
                    top_level_items.add(parts[0])

            print(f"\n=== TOP LEVEL ITEMS (after filtering): {top_level_items} ===")

            # If only one top-level item and it's likely a folder, use depth 2
            use_depth_2 = len(top_level_items) == 1
            print(f"=== USE DEPTH 2: {use_depth_2} ===\n")

            # Collect all files with their bytes
            for file_info in zip_ref.filelist:
                if file_info.filename.endswith("/"):
                    continue
                if _should_ignore(file_info.filename, ignore_patterns) == ignore:
                    continue

                # Determine submission name based on depth
                path_parts = file_info.filename.split("/")

                if use_depth_2 and len(path_parts) >= 2:
                    submission_name = path_parts[1] if path_parts[1] else path_parts[0]
                elif len(path_parts) == 1:
                    submission_name = path_parts[0]
                else:
                    submission_name = path_parts[0]

                print(f"  File: {file_info.filename} -> Submission: {submission_name}")

                # Read bytes (sequential - zipfile not thread-safe for reads)
                with zip_ref.open(file_info.filename) as f:
                    file_bytes = f.read()
                    files_to_process.append(
                        (submission_name, file_info.filename, file_bytes)
                    )

    except Exception as e:
        print(f"Error reading ZIP: {e}")
        return {}

    # Phase 2: Extract text in parallel (thread-safe)
    submission_files = {}  # submission_name -> list of (filename, content)

    def _extract_single(item):
        """Extract text from a single file."""
        submission_name, filename, file_bytes = item
        try:
            content = extract_text(file_bytes, filename)
            if content.strip():
                return (submission_name, filename, content)
        except Exception:
            pass
        return None

    print(f"\n=== EXTRACTING {len(files_to_process)} FILES (max_workers={max_workers}) ===")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_extract_single, item): item for item in files_to_process}
        for future in as_completed(futures):
            result = future.result()
            if result:
                submission_name, filename, content = result
                if submission_name not in submission_files:
                    submission_files[submission_name] = []
                submission_files[submission_name].append((filename, content))

    print(f"\n=== FINAL SUBMISSIONS: {list(submission_files.keys())} ===\n")

    # Phase 3: Combine files for each submission
    submissions = {}
    for submission_name, files in submission_files.items():
        if len(files) == 1 and files[0][0] == submission_name:
            submissions[submission_name] = files[0][1]
        else:
            combined = "\n\n".join(
                [f"# === {filename} ===\n{content}" for filename, content in files]
            )
            submissions[submission_name] = combined

    return submissions


def extract_zip_as_single(
    zip_path_or_bytes,
    zip_name: str = "submission",
    ignore_patterns: Optional[List[str]] = None,
    ignore: bool = True,
) -> str:
    """
    Extract ALL files from a ZIP and combine into ONE submission string.

    Use this when the entire ZIP = one submission (tab3 use case).

    Args:
        zip_path_or_bytes: Either a file path or file-like object/bytes
        zip_name: Name to use for this submission
        ignore_patterns: Patterns to match
        ignore: If True (default), skip files matching patterns. If False, only include files matching patterns.

    Returns:
        Combined content of all files as single string
    """
    if ignore_patterns is None:
        ignore_patterns = []

    all_content = []

    try:
        with zipfile.ZipFile(zip_path_or_bytes, "r") as zip_ref:
            for file_info in zip_ref.filelist:
                if file_info.filename.endswith("/"):
                    continue

                if _should_ignore(file_info.filename, ignore_patterns) == ignore:
                    continue

                with zip_ref.open(file_info.filename) as f:
                    try:
                        file_bytes = f.read()
                        content = extract_text(file_bytes, file_info.filename)
                        if content.strip():
                            all_content.append(
                                f"# === {file_info.filename} ===\n{content}"
                            )
                    except Exception:
                        pass
    except Exception:
        pass

    return "\n\n".join(all_content)


def _is_medium_url(url: str) -> bool:
    """
    Check if a URL is a Medium article.
    Handles:
    - medium.com/@username/article
    - username.medium.com/article
    - Custom domains using Medium (checks response headers)
    """
    medium_patterns = [
        "medium.com",
        ".medium.com",
        "towardsdatascience.com",
        "betterprogramming.pub",
        "levelup.gitconnected.com",
        "blog.devgenius.io",
        "javascript.plainenglish.io",
        "python.plainenglish.io",
        "aws.plainenglish.io",
    ]
    return any(pattern in url.lower() for pattern in medium_patterns)


def extract_from_medium(url: str) -> Dict[str, str]:
    """
    Extract article content from a Medium URL using newspaper3k.
    
    Args:
        url: Medium article URL
        
    Returns:
        Dict with article title as key and content as value
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        # Build content with metadata
        content_parts = []
        
        if article.title:
            content_parts.append(f"# {article.title}")
        
        if article.authors:
            content_parts.append(f"**Authors:** {', '.join(article.authors)}")
        
        if article.publish_date:
            content_parts.append(f"**Published:** {article.publish_date}")
        
        if article.text:
            content_parts.append(f"\n{article.text}")
        
        combined_content = "\n\n".join(content_parts)
        
        # Use title as submission name, fallback to URL slug
        submission_name = article.title if article.title else url.split("/")[-1].split("?")[0]
        # Clean up submission name for use as key
        submission_name = re.sub(r'[^\w\s-]', '', submission_name).strip()[:100]
        
        if combined_content.strip():
            return {submission_name: combined_content}
        
        return {}
        
    except Exception as e:
        print(f"Error extracting Medium article: {e}")
        return {}


async def extract_from_url(
    url: str, ignore_patterns: Optional[List[str]] = None, ignore: bool = True
) -> Dict[str, str]:
    """
    Extract text from a URL.
    - If Medium article URL: Extracts article text using newspaper3k.
    - If GitHub repo URL: Clones and extracts all files (one submission).
    - If GitHub blob URL: Converts to raw URL and downloads the file.
    - If direct file URL (PDF, TXT, etc.): Downloads and extracts (one submission).

    Args:
        url: URL to process
        ignore_patterns: Patterns to match
        ignore: If True (default), skip files matching patterns. If False, only include files matching patterns.

    Returns:
        Dict mapping submission name to content
    """
    # Check if it's a Medium article
    if _is_medium_url(url):
        return extract_from_medium(url)
    
    # Convert GitHub blob URLs to raw URLs
    # e.g., https://github.com/user/repo/blob/main/file.py
    #    -> https://raw.githubusercontent.com/user/repo/main/file.py
    if "github.com" in url and "/blob/" in url:
        url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    
    # Check if it's a GitHub repo URL (not a file URL)
    # Exclude raw file URLs which are direct file downloads
    is_github_repo = (
        "github.com" in url
        and "raw.githubusercontent.com" not in url
        and "/blob/" not in url
        and "/tree/" not in url  # Also exclude tree views
    )

    if is_github_repo:
        return extract_from_github(url, ignore_patterns, ignore)

    # Otherwise treat as a single file download
    submissions = {}
    
    # Add User-Agent to avoid GitHub blocking requests
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()

        # Determine filename
        filename = url.split("/")[-1].split("?")[0]  # Remove query params

        # Try to get filename from Content-Disposition header
        if "Content-Disposition" in response.headers:
            fname_match = re.findall(
                r'filename="?([^"]+)"?', response.headers["Content-Disposition"]
            )
            if fname_match:
                filename = fname_match[0]

        # Read content
        content_bytes = response.content

        text = extract_text(content_bytes, filename)
        if text.strip():
            submissions[filename] = text

    return submissions


def extract_from_github(
    url: str, ignore_patterns: Optional[List[str]] = None, ignore: bool = True
) -> Dict[str, str]:
    """
    Clone a GitHub repository and extract text from all files.
    
    Args:
        url: GitHub repository URL
        ignore_patterns: Patterns to match
        ignore: If True (default), skip files matching patterns. If False, only include files matching patterns.

    Returns:
        Dict with repo name as key and combined content as value (one submission per repo)
    """
    if ignore_patterns is None:
        ignore_patterns = []

    submissions = {}

    repo_name = url.split("/")[-1].replace(".git", "")
    with tempfile.TemporaryDirectory() as temp_dir:
        git.Repo.clone_from(url, temp_dir)
        folder_submissions = extract_from_folder(temp_dir, ignore_patterns, ignore=ignore)

        # Combine all files into one submission per repo
        if folder_submissions:
            combined_content = "\n\n".join(
                [
                    f"# === {filename} ===\n{content}"
                    for filename, content in folder_submissions.items()
                ]
            )
            submissions[repo_name] = combined_content

    return submissions


def extract_from_folder(
    folder_path: str,
    ignore_patterns: Optional[List[str]] = None,
    max_workers: Optional[int] = None,
    ignore: bool = True,
) -> Dict[str, str]:
    """
    Recursively extract text from all files in a folder.

    Args:
        folder_path: Path to folder
        ignore_patterns: Patterns to match
        max_workers: Max threads for parallel extraction (default: CPU count)
        ignore: If True (default), skip files matching patterns. If False, only include files matching patterns.

    Returns:
        Dict mapping relative filename to extracted text
    """
    if ignore_patterns is None:
        ignore_patterns = []

    if max_workers is None:
        max_workers = os.cpu_count() or 4

    if not os.path.exists(folder_path):
        return {}

    # Phase 1: Collect all file paths (sequential)
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        # Filter out directories only in exclude mode (ignore=True)
        # In include mode, we need to traverse all dirs to find matching files
        if ignore:
            dirs[:] = [d for d in dirs if not _should_ignore(d, ignore_patterns)]

        for filename in files:
            if _should_ignore(filename, ignore_patterns) == ignore:
                continue

            file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(file_path, folder_path)
            file_paths.append((file_path, rel_path))

    # Phase 2: Extract text in parallel
    def _extract_single(item):
        """Extract text from a single file."""
        file_path, rel_path = item
        try:
            content = extract_text(file_path)
            if content.strip():
                return (rel_path, content)
        except Exception:
            pass
        return None

    submissions = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_extract_single, item): item for item in file_paths}
        for future in as_completed(futures):
            result = future.result()
            if result:
                rel_path, content = result
                submissions[rel_path] = content

    return submissions


def _should_ignore(name: str, patterns: List[str]) -> bool:
    """Check if a file/folder name matches any ignore pattern."""
    for pattern in patterns:
        if fnmatch.fnmatch(name, pattern):
            return True
        # Also check if the pattern is in the path
        if pattern.lstrip("*") in name:
            return True
    return False


def _extract_notebook_code(file_path: str) -> str:
    """
    Extract only code cells from a Jupyter notebook.
    Ignores markdown cells and outputs.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            notebook = json.load(f)

        code_cells = []
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                source = cell.get("source", [])
                # Handle both list of strings (standard) and single string
                if isinstance(source, list):
                    code_content = "".join(source)
                else:
                    code_content = str(source)

                if code_content.strip():
                    code_cells.append(code_content)

        return "\n\n# === CELL SEPARATOR ===\n\n".join(code_cells)
    except Exception:
        # If json load fails or structure is invalid, return empty
        return ""
