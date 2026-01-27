"""
Code Region Similarity Finder.

Functions for detailed code comparison:
1. extract_code_blocks: Extract functions/classes as AST nodes
2. find_similar_blocks: AST-based function/class matching
3. find_matching_token_sequences: Token sequence matching
4. get_similar_regions: Combined detailed similarity analysis
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


from code_similarity_finder import DEFAULT_OPTIONS, tokenize_code


def normalize_ast_node(node: ast.AST, options: Optional[Dict] = None) -> ast.AST:
    """
    Normalize an AST node by replacing variable/function names with placeholders.
    Optionally normalizes literals based on options.
    This makes comparison resilient to renaming and literal changes.
    """
    options = options or DEFAULT_OPTIONS
    node = copy.deepcopy(node)

    class NameNormalizer(ast.NodeTransformer):
        def __init__(self, preserve_literals: bool):
            self.name_map = {}
            self.counter = 0
            self.preserve_literals = preserve_literals

        def visit_Name(self, node):
            if node.id not in self.name_map:
                self.name_map[node.id] = f"VAR_{self.counter}"
                self.counter += 1
            node.id = self.name_map[node.id]
            return node

        def visit_FunctionDef(self, node):
            node.name = "FUNC"
            self.generic_visit(node)
            return node

        def visit_arg(self, node):
            node.arg = "ARG"
            return node

        def visit_Constant(self, node):
            if not self.preserve_literals:
                if isinstance(node.value, str):
                    node.value = "STR"
                elif isinstance(node.value, (int, float, complex)):
                    node.value = 0
            return node

    try:
        preserve_literals = options.get("preserve_literals", False)
        normalizer = NameNormalizer(preserve_literals)
        return normalizer.visit(node)
    except:
        return node


def extract_code_blocks(code: str, options: Optional[Dict] = None) -> Dict[str, dict]:
    """
    Extract functions and classes as separate AST nodes with metadata.

    Args:
        code: Source code to extract blocks from
        options: Preprocessing options (uses preserve_literals for normalization)

    Returns:
        Dict mapping block name to block info (ast_dump, lines, source)
    """
    options = options or DEFAULT_OPTIONS
    blocks = {}
    try:
        tree = ast.parse(code)
        lines = code.split("\n")

        def extract_function(node, prefix=""):
            """Helper to extract function/async function blocks."""
            if isinstance(node, ast.AsyncFunctionDef):
                block_name = f"{prefix}async:{node.name}" if prefix else f"async:{node.name}"
                block_type = "async_function"
            else:
                block_name = f"{prefix}func:{node.name}" if prefix else f"func:{node.name}"
                block_type = "function"

            try:
                source = ast.unparse(node)
            except:
                source = "\n".join(lines[node.lineno - 1 : node.end_lineno])

            normalized = normalize_ast_node(node, options)

            blocks[block_name] = {
                "type": block_type,
                "name": node.name,
                "ast_dump": ast.dump(normalized),
                "lineno": node.lineno,
                "end_lineno": node.end_lineno,
                "source": source,
            }

            # Extract nested functions
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    extract_function(child, prefix=f"{block_name}.")

        for node in ast.iter_child_nodes(tree):
            # Handle regular and async functions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                extract_function(node)

            elif isinstance(node, ast.ClassDef):
                block_name = f"class:{node.name}"
                try:
                    source = ast.unparse(node)
                except:
                    source = "\n".join(lines[node.lineno - 1 : node.end_lineno])

                normalized = normalize_ast_node(node, options)

                blocks[block_name] = {
                    "type": "class",
                    "name": node.name,
                    "ast_dump": ast.dump(normalized),
                    "lineno": node.lineno,
                    "end_lineno": node.end_lineno,
                    "source": source,
                }

                # Extract methods within the class (regular and async)
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_type = "async_method" if isinstance(item, ast.AsyncFunctionDef) else "method"
                        method_name = f"class:{node.name}.{item.name}"
                        try:
                            method_source = ast.unparse(item)
                        except:
                            method_source = "\n".join(
                                lines[item.lineno - 1 : item.end_lineno]
                            )

                        normalized_method = normalize_ast_node(item, options)

                        blocks[method_name] = {
                            "type": method_type,
                            "name": f"{node.name}.{item.name}",
                            "ast_dump": ast.dump(normalized_method),
                            "lineno": item.lineno,
                            "end_lineno": item.end_lineno,
                            "source": method_source,
                        }

                        # Extract nested functions within methods
                        for child in ast.iter_child_nodes(item):
                            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                extract_function(child, prefix=f"{method_name}.")

    except SyntaxError:
        pass
    except Exception:
        pass

    return blocks


def find_similar_blocks(
    code1: str, code2: str, threshold: float = 0.6, options: Optional[Dict] = None
) -> List[dict]:
    """
    Find similar functions/classes between two code files using AST comparison.

    Args:
        code1: First code file content
        code2: Second code file content
        threshold: Minimum similarity to report (0.0 to 1.0)
        options: Preprocessing options (uses preserve_literals for normalization)

    Returns:
        List of matches with similarity scores and source code
    """
    options = options or DEFAULT_OPTIONS
    blocks1 = extract_code_blocks(code1, options)
    blocks2 = extract_code_blocks(code2, options)

    matches = []

    for name1, block1 in blocks1.items():
        for name2, block2 in blocks2.items():
            similarity = SequenceMatcher(
                None, block1["ast_dump"], block2["ast_dump"]
            ).ratio()

            if similarity >= threshold:
                matches.append(
                    {
                        "file1_block": name1,
                        "file1_lines": [block1["lineno"], block1["end_lineno"]],
                        "file1_source": block1["source"],
                        "file2_block": name2,
                        "file2_lines": [block2["lineno"], block2["end_lineno"]],
                        "file2_source": block2["source"],
                        "similarity": round(similarity, 4),
                        "type": block1["type"],
                    }
                )

    return sorted(matches, key=lambda x: -x["similarity"])


def find_matching_token_sequences(
    code1: str, code2: str, min_tokens: int = 5
) -> List[dict]:
    """
    Find matching token sequences between two code files.
    Uses spaCy tokenization for consistency with similarity scoring.

    Args:
        code1: First code file content
        code2: Second code file content
        min_tokens: Minimum number of consecutive matching tokens to report

    Returns:
        List of matching sequences with positions and text
    """
    tokens1 = tokenize_code(code1)
    tokens2 = tokenize_code(code2)

    matcher = SequenceMatcher(None, tokens1, tokens2)
    matches = []

    for block in matcher.get_matching_blocks():
        if block.size >= min_tokens:
            matched_tokens = tokens1[block.a : block.a + block.size]
            matched_text = "".join(matched_tokens)

            # Line estimation by counting newlines
            tokens_before_1 = tokens1[: block.a]
            tokens_before_2 = tokens2[: block.b]
            text_before_1 = "".join(tokens_before_1)
            text_before_2 = "".join(tokens_before_2)
            line1 = text_before_1.count("\n") + 1
            line2 = text_before_2.count("\n") + 1

            matches.append(
                {
                    "file1_token_pos": block.a,
                    "file1_approx_line": line1,
                    "file2_token_pos": block.b,
                    "file2_approx_line": line2,
                    "token_count": block.size,
                    "matched_text": matched_text[:500],  # Truncate for display
                }
            )

    return sorted(matches, key=lambda x: -x["token_count"])


def get_similar_regions(
    code1: str, code2: str, block_threshold: float = 0.6, min_tokens: int = 10
) -> dict:
    """
    Get detailed similarity regions between two code files.
    Combines AST-based function matching with token sequence matching.

    Args:
        code1: First code file content
        code2: Second code file content
        block_threshold: Minimum similarity for function/class matches
        min_tokens: Minimum tokens for sequence matches

    Returns:
        Dict with function_matches, token_matches, and overall_similarity
    """
    start_time = time.time()

    function_matches = find_similar_blocks(code1, code2, block_threshold)
    token_matches = find_matching_token_sequences(code1, code2, min_tokens)
    overall_similarity = Levenshtein.ratio(code1, code2)

    elapsed = time.time() - start_time

    return {
        "function_matches": function_matches[:20],
        "token_matches": token_matches[:10],
        "overall_similarity": round(overall_similarity, 4),
        "stats": {
            "file1_blocks": len(extract_code_blocks(code1)),
            "file2_blocks": len(extract_code_blocks(code2)),
            "total_function_matches": len(function_matches),
            "total_token_matches": len(token_matches),
            "analysis_time": round(elapsed, 4),
        },
    }
