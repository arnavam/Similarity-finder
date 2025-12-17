"""
Text Extractor Module
Unified text extraction from various sources: files, ZIPs, GitHub repos, folders.
"""

import fnmatch
import json
import os
import tempfile
import zipfile
from typing import Dict, Optional, List

import git
import textract


def _extract_notebook_code(file_path_or_bytes, include_markdown: bool = True) -> str:
    """
    Extract code cells from a Jupyter notebook (.ipynb).
    Filters out all outputs, metadata, and execution data.
    
    Args:
        file_path_or_bytes: File path or bytes content
        include_markdown: If True, include markdown cells as comments
        
    Returns:
        Clean code extracted from notebook cells
    """
    try:
        if isinstance(file_path_or_bytes, bytes):
            notebook = json.loads(file_path_or_bytes.decode('utf-8'))
        else:
            with open(file_path_or_bytes, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
        
        extracted_parts = []
        
        for cell in notebook.get('cells', []):
            cell_type = cell.get('cell_type', '')
            source = cell.get('source', [])
            
            # source can be a list of lines or a single string
            if isinstance(source, list):
                content = ''.join(source)
            else:
                content = source
            
            if not content.strip():
                continue
            
            if cell_type == 'code':
                extracted_parts.append(content)
            elif cell_type == 'markdown' and include_markdown:
                # Include markdown as comments for context
                commented = '\n'.join(f'# {line}' for line in content.split('\n'))
                extracted_parts.append(f'# [Markdown Cell]\n{commented}')
        
        return '\n\n'.join(extracted_parts)
    
    except (json.JSONDecodeError, KeyError, TypeError):
        return ""


def extract_text(file_path_or_bytes, filename: Optional[str] = None) -> str:
    """
    Extract text from any file using textract.
    
    Args:
        file_path_or_bytes: Either a file path (str) or file bytes
        filename: Required when passing bytes, used to detect file type
        
    Returns:
        Extracted text as string
    """
    try:
        if isinstance(file_path_or_bytes, bytes):
            # For in-memory files, write to temp file for textract
            suffix = os.path.splitext(filename)[1] if filename else '.txt'
            
            # Special handling for Jupyter notebooks - extract code only, no outputs
            if suffix.lower() == '.ipynb':
                return _extract_notebook_code(file_path_or_bytes)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_path_or_bytes)
                tmp_path = tmp.name
            try:
                text = textract.process(tmp_path).decode('utf-8', errors='ignore')
            finally:
                os.unlink(tmp_path)
            return text
        else:
            # For file paths
            # Special handling for Jupyter notebooks - extract code only, no outputs
            if file_path_or_bytes.lower().endswith('.ipynb'):
                return _extract_notebook_code(file_path_or_bytes)
            return textract.process(file_path_or_bytes).decode('utf-8', errors='ignore')
    except Exception:
        # Fallback: try reading as plain text
        try:
            if isinstance(file_path_or_bytes, bytes):
                return file_path_or_bytes.decode('utf-8', errors='ignore')
            else:
                with open(file_path_or_bytes, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
        except Exception:
            return ""


def extract_from_zip(
    zip_path_or_bytes,
    ignore_patterns: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Extract text from a ZIP archive where each top-level folder/file is ONE submission.
    
    Handles both:
        submissions.zip/student1/, student2/   → student1, student2 are submissions
        submissions.zip/container/student1/, student2/  → student1, student2 are submissions
    
    Args:
        zip_path_or_bytes: Either a file path or file-like object/bytes
        ignore_patterns: Patterns to skip
        
    Returns:
        Dict mapping submission name to combined content
    """
    if ignore_patterns is None:
        ignore_patterns = []
    
    submission_files = {}  # submission_name -> list of (filename, content)
    

    
    try:
        with zipfile.ZipFile(zip_path_or_bytes, 'r') as zip_ref:
            # First, detect if there's a single container folder
            top_level_items = set()
            print("\n=== ZIP FILE STRUCTURE ===")
            for file_info in zip_ref.filelist:
                # Skip junk files (handled by ignore patterns)
                if _should_ignore(file_info.filename, ignore_patterns):
                    continue
                    
                print(f"  {file_info.filename}")
                if file_info.filename.endswith('/'):
                    continue
                parts = file_info.filename.split('/')
                if parts[0] and parts[0] != '__MACOSX':
                    top_level_items.add(parts[0])
            
            print(f"\n=== TOP LEVEL ITEMS (after filtering): {top_level_items} ===")
            
            # If only one top-level item and it's likely a folder, use depth 2
            use_depth_2 = len(top_level_items) == 1
            print(f"=== USE DEPTH 2: {use_depth_2} ===\n")
            
            for file_info in zip_ref.filelist:
                # Skip directories
                if file_info.filename.endswith('/'):
                    continue
                
                # Skip junk files
                if _should_ignore(file_info.filename, ignore_patterns):
                    continue
                
                # Determine submission name based on depth
                path_parts = file_info.filename.split('/')
                
                if use_depth_2 and len(path_parts) >= 2:
                    # Container folder exists - use 2nd level as submission
                    submission_name = path_parts[1] if path_parts[1] else path_parts[0]
                elif len(path_parts) == 1:
                    # File at root level = its own submission
                    submission_name = path_parts[0]
                else:
                    # Normal case - first folder is submission
                    submission_name = path_parts[0]
                
                print(f"  File: {file_info.filename} -> Submission: {submission_name}")
                
                # Extract text from file
                with zip_ref.open(file_info.filename) as f:
                    try:
                        file_bytes = f.read()
                        content = extract_text(file_bytes, file_info.filename)
                        if content.strip():
                            if submission_name not in submission_files:
                                submission_files[submission_name] = []
                            submission_files[submission_name].append(
                                (file_info.filename, content)
                            )
                    except Exception:
                        pass
        
        print(f"\n=== FINAL SUBMISSIONS: {list(submission_files.keys())} ===\n")
        
        # Combine files for each submission
        submissions = {}
        for submission_name, files in submission_files.items():
            if len(files) == 1 and files[0][0] == submission_name:
                # Single file submission - use content directly
                submissions[submission_name] = files[0][1]
            else:
                # Multiple files - combine with headers
                combined = "\n\n".join([
                    f"# === {filename} ===\n{content}"
                    for filename, content in files
                ])
                submissions[submission_name] = combined
    except Exception:
        pass
    
    return submissions


def extract_zip_as_single(
    zip_path_or_bytes,
    zip_name: str = "submission",
    ignore_patterns: Optional[List[str]] = None
) -> str:
    """
    Extract ALL files from a ZIP and combine into ONE submission string.
    
    Use this when the entire ZIP = one submission (tab3 use case).
    
    Args:
        zip_path_or_bytes: Either a file path or file-like object/bytes
        zip_name: Name to use for this submission
        ignore_patterns: Patterns to skip
        
    Returns:
        Combined content of all files as single string
    """
    if ignore_patterns is None:
        ignore_patterns = []
    
    all_content = []
    
    try:
        with zipfile.ZipFile(zip_path_or_bytes, 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                if file_info.filename.endswith('/'):
                    continue
                
                if _should_ignore(file_info.filename, ignore_patterns):
                    continue
                
                with zip_ref.open(file_info.filename) as f:
                    try:
                        file_bytes = f.read()
                        content = extract_text(file_bytes, file_info.filename)
                        if content.strip():
                            all_content.append(f"# === {file_info.filename} ===\n{content}")
                    except Exception:
                        pass
    except Exception:
        pass
    
    return "\n\n".join(all_content)

def extract_from_github(
    url: str,
    ignore_patterns: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Clone a GitHub repository and extract text from all files.
    
    Args:
        url: GitHub repository URL
        ignore_patterns: Patterns to skip
        
    Returns:
        Dict with repo name as key and combined content as value (one submission per repo)
    """
    if ignore_patterns is None:
        ignore_patterns = []
    
    submissions = {}
    
    try:
        repo_name = url.split("/")[-1].replace(".git", "")
        with tempfile.TemporaryDirectory() as temp_dir:
            git.Repo.clone_from(url, temp_dir)
            folder_submissions = extract_from_folder(temp_dir, ignore_patterns)
            
            # Combine all files into one submission per repo
            if folder_submissions:
                combined_content = "\n\n".join([
                    f"# === {filename} ===\n{content}" 
                    for filename, content in folder_submissions.items()
                ])
                submissions[repo_name] = combined_content
    except Exception:
        pass
    
    return submissions


def extract_from_folder(
    folder_path: str,
    ignore_patterns: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Recursively extract text from all files in a folder.

    Args:
        folder_path: Path to folder
        ignore_patterns: Patterns to skip

    Returns:
        Dict mapping relative filename to extracted text
    """
    if ignore_patterns is None:
        ignore_patterns = []
    
    submissions = {}
    
    if not os.path.exists(folder_path):
        return submissions
    
    for root, dirs, files in os.walk(folder_path):
        # Filter out ignored directories
        dirs[:] = [d for d in dirs if not _should_ignore(d, ignore_patterns)]
        
        for filename in files:
            if _should_ignore(filename, ignore_patterns):
                continue
            
            file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(file_path, folder_path)
            
            content = extract_text(file_path)
            if content.strip():
                submissions[rel_path] = content
    
    return submissions


def _should_ignore(name: str, patterns: List[str]) -> bool:
    """Check if a file/folder name matches any ignore pattern."""
    for pattern in patterns:
        if fnmatch.fnmatch(name, pattern):
            return True
        # Also check if the pattern is in the path
        if pattern.lstrip('*') in name:
            return True
    return False
