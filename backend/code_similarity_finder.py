"""
Optimized Code Similarity Checker with parallel processing and memory optimization.

Two modes:
1. Direct Mode: Process target → parallel process+compare others → discard
2. Preprocess Mode: Parallel preprocess all → store embeddings → select target → compare
"""

import ast
import time
import sys
import psutil
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any, Union

import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_nlp = None


def get_nlp():
    """Loads when first needed."""
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


class FlexibleCodeSimilarityChecker:
    """Memory-optimized similarity checker with parallel processing."""

    # Class-level cache for preprocessed code (shared across instances)
    _preprocess_cache = {}

    def __init__(self, preprocessing_options: Optional[Dict] = None, max_workers: int = 1):
        """
        Initialize with customizable preprocessing options.

        Args:
            preprocessing_options: Options for preprocessing
            max_workers: Number of parallel workers for processing
        """
        self.max_workers = max_workers
        
        # Default preprocessing options
        self.options = {
            "remove_comments": True,
            "normalize_whitespace": True,
            "preserve_variable_names": False,
            "preserve_literals": False,
        }

        if preprocessing_options:
            self.options.update(preprocessing_options)

        self._options_key = tuple(sorted(self.options.items()))
        
        # Storage for preprocess mode (embeddings only, no raw text)
        self._embeddings: Dict[str, dict] = {}

    # ===== Core Processing Methods =====

    def process_to_embeddings(self, name: str, code: str) -> dict:
        """
        Process code and return embeddings only. Does NOT store raw text.
        
        Returns:
            dict with: name, processed_text, ast, tokens
        """
        start_time = time.time()
        
        processed = self._get_preprocessed(code)
        ast_features = self.extract_ast_features(code)
        tokens = self.tokenize_code(processed)
        
        elapsed = time.time() - start_time
        file_size = len(code.encode('utf-8'))
        
        print(f"📥 Processed '{name}': {elapsed:.4f}s | File: {file_size/1024:.2f}KB | Tokens: {len(tokens)}")
        
        return {
            "name": name,
            "processed_text": processed,
            "ast": ast_features,
            "tokens": tokens,
            "raw_code": code,  # Kept for Levenshtein raw comparison
        }

    def compare_embeddings(self, target: dict, other: dict) -> dict:
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
        scores["processed"] = Levenshtein.ratio(target["processed_text"], other["processed_text"])
        
        # 3. AST structure similarity
        # Create new lists to avoid modifying cached embeddings
        ast_seq1 = list(target["ast"]["node_types"])
        ast_seq2 = list(other["ast"]["node_types"])
        
        # Include extracted function/variable names in comparison if available
        # (These are populated based on preserve_variable_names option during extraction)
        if target["ast"]["function_calls"] and other["ast"]["function_calls"]:
            ast_seq1.extend(target["ast"]["function_calls"])
            ast_seq2.extend(other["ast"]["function_calls"])

        if ast_seq1 and ast_seq2:
            scores["ast"] = self.sequence_similarity(ast_seq1, ast_seq2)
        else:
            scores["ast"] = 0.0
        
        # 4. Token-based similarity
        if target["tokens"] and other["tokens"]:
            scores["token"] = self.sequence_similarity(target["tokens"], other["tokens"])
        else:
            scores["token"] = 0.0
        
        return scores

    def _process_and_compare(self, name: str, code: str, target_emb: dict) -> Tuple[str, dict]:
        """
        Process + Compare in one step for direct mode.
        Embeddings are discarded after scoring (memory freed).
        """
        emb = self.process_to_embeddings(name, code)
        scores = self.compare_embeddings(target_emb, emb)
        # emb is garbage collected after return
        return name, scores

    # ===== Direct Mode =====

    def analyze_direct(self, target_name: str, submissions: Dict[str, str]) -> dict:
        """
        One-pass parallel analysis. Process target first, then parallel
        process+compare others, discarding embeddings after each comparison.
        
        Args:
            target_name: Which submission to compare against
            submissions: Dict of {name: raw_code}
            
        Returns:
            dict with scores and submission_names
        """
        start_time = time.time()
        process = psutil.Process()
        start_ram = process.memory_info().rss / (1024 * 1024)
        
        if target_name not in submissions:
            return {"error": f"Target '{target_name}' not found"}
        
        # Process target first
        target_emb = self.process_to_embeddings(target_name, submissions[target_name])
        
        # Prepare results structure
        all_scores = {
            "raw_scores": [],
            "processed_scores": [],
            "ast_scores": [],
            "token_scores": [],
            "cosine_scores": [],  # Computed separately for TF-IDF
        }
        submission_names = []
        processed_texts = [target_emb["processed_text"]]  # For TF-IDF
        
        # Parallel process + compare others
        other_submissions = [(n, c) for n, c in submissions.items() if n != target_name]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_and_compare, name, code, target_emb): name
                for name, code in other_submissions
            }
            
            for future in as_completed(futures):
                name, scores = future.result()
                submission_names.append(name)
                all_scores["raw_scores"].append(scores["raw"])
                all_scores["processed_scores"].append(scores["processed"])
                all_scores["ast_scores"].append(scores["ast"])
                all_scores["token_scores"].append(scores["token"])
                # Store processed text for TF-IDF (computed after all processing)
                # Note: we need to recompute this or pass it differently
        
        # Compute TF-IDF cosine similarity (needs all texts together)
        # Re-process just for TF-IDF - could optimize by storing processed_text temporarily
        all_texts = [target_emb["processed_text"]]
        for name, code in other_submissions:
            all_texts.append(self._get_preprocessed(code))
        
        if len(all_texts) > 1:
            tfidf_matrix = TfidfVectorizer().fit_transform(all_texts)
            cosine_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
            all_scores["cosine_scores"] = [float(s) for s in cosine_scores[0]]
        
        elapsed = time.time() - start_time
        end_ram = process.memory_info().rss / (1024 * 1024)
        
        print(f"🔍 Direct analysis: {elapsed:.4f}s | RAM: {end_ram:.1f}MB (delta: {end_ram - start_ram:+.2f}MB)")
        
        return {
            "scores": all_scores,
            "submission_names": submission_names,
            "target_file": target_name,
        }

    # ===== Preprocess Mode =====

    def preprocess_all(self, submissions: Dict[str, str]) -> Dict[str, dict]:
        """
        Parallel preprocess all submissions, store embeddings (no raw text kept).
        
        Args:
            submissions: Dict of {name: raw_code}
            
        Returns:
            Dict of {name: embeddings}
        """
        start_time = time.time()
        process = psutil.Process()
        start_ram = process.memory_info().rss / (1024 * 1024)
        
        self._embeddings = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.process_to_embeddings, name, code): name
                for name, code in submissions.items()
            }
            
            for future in as_completed(futures):
                name = futures[future]
                self._embeddings[name] = future.result()
        
        elapsed = time.time() - start_time
        end_ram = process.memory_info().rss / (1024 * 1024)
        
        print(f"📊 Preprocessed {len(self._embeddings)} submissions: {elapsed:.4f}s | RAM: {end_ram:.1f}MB (delta: {end_ram - start_ram:+.2f}MB)")
        
        return self._embeddings

    def compare_preprocessed(self, target_name: str, embeddings: Optional[Dict[str, dict]] = None) -> dict:
        """
        Compare target against all other preprocessed embeddings.
        
        Args:
            target_name: Which submission to compare against
            embeddings: Optional external embeddings dict, uses self._embeddings if None
            
        Returns:
            dict with scores and submission_names
        """
        emb = embeddings or self._embeddings
        
        if target_name not in emb:
            return {"error": f"Target '{target_name}' not found in preprocessed data"}
        
        start_time = time.time()
        target_emb = emb[target_name]
        
        all_scores = {
            "raw_scores": [],
            "processed_scores": [],
            "ast_scores": [],
            "token_scores": [],
            "cosine_scores": [],
        }
        submission_names = []
        
        # Compare against all others
        for name, other_emb in emb.items():
            if name != target_name:
                scores = self.compare_embeddings(target_emb, other_emb)
                submission_names.append(name)
                all_scores["raw_scores"].append(scores["raw"])
                all_scores["processed_scores"].append(scores["processed"])
                all_scores["ast_scores"].append(scores["ast"])
                all_scores["token_scores"].append(scores["token"])
        
        # TF-IDF cosine similarity
        all_texts = [target_emb["processed_text"]]
        for name in submission_names:
            all_texts.append(emb[name]["processed_text"])
        
        if len(all_texts) > 1:
            tfidf_matrix = TfidfVectorizer().fit_transform(all_texts)
            cosine_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
            all_scores["cosine_scores"] = [float(s) for s in cosine_scores[0]]
        
        elapsed = time.time() - start_time
        print(f"🔍 Comparison: {elapsed:.4f}s | Compared against {len(submission_names)} submissions")
        
        return {
            "scores": all_scores,
            "submission_names": submission_names,
            "target_file": target_name,
        }

    # ===== Helper Methods =====

    def _get_preprocessed(self, code: str) -> str:
        """Get preprocessed code with caching."""
        cache_key = (hash(code), self._options_key)
        if cache_key not in self._preprocess_cache:
            self._preprocess_cache[cache_key] = self.preprocess_code(code)
        return self._preprocess_cache[cache_key]

    def preprocess_code(self, code: str) -> str:
        """Preprocess code based on options."""
        processed = code

        if self.options["remove_comments"]:
            lines = []
            for line in processed.split("\n"):
                if "#" in line:
                    line = line.split("#")[0]
                lines.append(line)
            processed = "\n".join(lines)

            try:
                tree = ast.parse(processed)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                        node.value.s = ""
                processed = ast.unparse(tree)
            except:
                pass

        if self.options["normalize_whitespace"]:
            lines = []
            for line in processed.split("\n"):
                line = " ".join(line.split())
                if line:
                    lines.append(line)
            processed = "\n".join(lines)

        # Handle literal preservation (or anonymization)
        # If preserve_literals is False (default), we anonymize string/number literals
        if not self.options.get("preserve_literals", False):
            try:
                tree = ast.parse(processed)
                modified = False
                for node in ast.walk(tree):
                    if isinstance(node, ast.Constant):
                        # Anonymize strings (except docstrings which are handled by remove_comments)
                        if isinstance(node.value, str):
                            # Don't touch docstrings if they weren't removed? 
                            # Actually, just anonymize all string literals to "STR"
                            # If remove_comments ran, docstrings are already empty or gone from lines loop
                            if node.value != "": 
                                node.value = "STR"
                                modified = True
                        # Anonymize numbers
                        elif isinstance(node.value, (int, float, complex)):
                            node.value = 0
                            modified = True
                            
                if modified:
                    processed = ast.unparse(tree)
            except:
                pass

        return processed

    def extract_ast_features(self, code: str) -> dict:
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
                    if isinstance(node.func, ast.Name) and self.options["preserve_variable_names"]:
                        features["function_calls"].append(node.func.id)
                    else:
                        features["function_calls"].append("FUNC_CALL")

                if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                    features["control_flow"].append(type(node).__name__)

                if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare)):
                    features["operators"].append(type(node.op).__name__)

            return features
        except:
            return {"node_types": [], "function_calls": [], "control_flow": [], "operators": []}

    def tokenize_code(self, code: str) -> List[str]:
        """Simple tokenization of code using lazy-loaded spaCy."""
        nlp = get_nlp()
        doc = nlp(code)
        return [token.text for token in doc]

    def sequence_similarity(self, seq1: List, seq2: List) -> float:
        """Calculate similarity between two sequences."""
        if not seq1 or not seq2:
            return 0.0
        return SequenceMatcher(None, seq1, seq2).ratio()

    # ===== Similar Regions Detection =====

    def extract_code_blocks(self, code: str) -> Dict[str, dict]:
        """
        Extract functions and classes as separate AST nodes with metadata.
        
        Returns:
            Dict mapping block name to block info (ast_dump, lines, source)
        """
        blocks = {}
        try:
            tree = ast.parse(code)
            lines = code.split('\n')
            
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.FunctionDef):
                    block_name = f"func:{node.name}"
                    try:
                        source = ast.unparse(node)
                    except:
                        # Fallback: extract from source lines
                        source = '\n'.join(lines[node.lineno - 1:node.end_lineno])
                    
                    # Normalize AST for comparison (remove names)
                    normalized = self._normalize_ast_node(node)
                    
                    blocks[block_name] = {
                        "type": "function",
                        "name": node.name,
                        "ast_dump": ast.dump(normalized),
                        "lineno": node.lineno,
                        "end_lineno": node.end_lineno,
                        "source": source,
                    }
                    
                elif isinstance(node, ast.ClassDef):
                    block_name = f"class:{node.name}"
                    try:
                        source = ast.unparse(node)
                    except:
                        source = '\n'.join(lines[node.lineno - 1:node.end_lineno])
                    
                    normalized = self._normalize_ast_node(node)
                    
                    blocks[block_name] = {
                        "type": "class",
                        "name": node.name,
                        "ast_dump": ast.dump(normalized),
                        "lineno": node.lineno,
                        "end_lineno": node.end_lineno,
                        "source": source,
                    }
                    
                    # Also extract methods within the class
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_name = f"class:{node.name}.{item.name}"
                            try:
                                method_source = ast.unparse(item)
                            except:
                                method_source = '\n'.join(lines[item.lineno - 1:item.end_lineno])
                            
                            normalized_method = self._normalize_ast_node(item)
                            
                            blocks[method_name] = {
                                "type": "method",
                                "name": f"{node.name}.{item.name}",
                                "ast_dump": ast.dump(normalized_method),
                                "lineno": item.lineno,
                                "end_lineno": item.end_lineno,
                                "source": method_source,
                            }
        except SyntaxError:
            # Not valid Python, return empty
            pass
        except Exception:
            pass
            
        return blocks

    def _normalize_ast_node(self, node: ast.AST) -> ast.AST:
        """
        Normalize an AST node by replacing variable/function names with placeholders.
        This makes comparison resilient to renaming.
        """
        import copy
        node = copy.deepcopy(node)
        
        class NameNormalizer(ast.NodeTransformer):
            def __init__(self):
                self.name_map = {}
                self.counter = 0
                
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
        
        try:
            normalizer = NameNormalizer()
            return normalizer.visit(node)
        except:
            return node

    def find_similar_blocks(
        self, 
        code1: str, 
        code2: str, 
        threshold: float = 0.6
    ) -> List[dict]:
        """
        Find similar functions/classes between two code files using AST comparison.
        
        Args:
            code1: First code file content
            code2: Second code file content
            threshold: Minimum similarity to report (0.0 to 1.0)
            
        Returns:
            List of matches with similarity scores and source code
        """
        blocks1 = self.extract_code_blocks(code1)
        blocks2 = self.extract_code_blocks(code2)
        
        matches = []
        
        for name1, block1 in blocks1.items():
            for name2, block2 in blocks2.items():
                # Compare normalized AST structure
                similarity = SequenceMatcher(
                    None, block1["ast_dump"], block2["ast_dump"]
                ).ratio()
                
                if similarity >= threshold:
                    matches.append({
                        "file1_block": name1,
                        "file1_lines": [block1["lineno"], block1["end_lineno"]],
                        "file1_source": block1["source"],
                        "file2_block": name2,
                        "file2_lines": [block2["lineno"], block2["end_lineno"]],
                        "file2_source": block2["source"],
                        "similarity": round(similarity, 4),
                        "type": block1["type"],
                    })
        
        # Sort by similarity (highest first)
        return sorted(matches, key=lambda x: -x["similarity"])

    def find_matching_token_sequences(
        self, 
        code1: str, 
        code2: str, 
        min_tokens: int = 5
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
        # Use spaCy tokenization (same as similarity scoring)
        tokens1 = self.tokenize_code(code1)
        tokens2 = self.tokenize_code(code2)
        
        matcher = SequenceMatcher(None, tokens1, tokens2)
        matches = []
        
        for block in matcher.get_matching_blocks():
            if block.size >= min_tokens:
                matched_tokens = tokens1[block.a : block.a + block.size]
                matched_text = "".join(matched_tokens)  # spaCy preserves spacing info
                
                # Estimate line numbers by counting newlines in matched tokens
                tokens_before_1 = tokens1[:block.a]
                tokens_before_2 = tokens2[:block.b]
                line1 = sum(1 for t in tokens_before_1 if '\n' in t) + 1
                line2 = sum(1 for t in tokens_before_2 if '\n' in t) + 1
                
                # Better line estimation: reconstruct text and count newlines
                text_before_1 = "".join(tokens_before_1)
                text_before_2 = "".join(tokens_before_2)
                line1 = text_before_1.count('\n') + 1
                line2 = text_before_2.count('\n') + 1
                
                matches.append({
                    "file1_token_pos": block.a,
                    "file1_approx_line": line1,
                    "file2_token_pos": block.b,
                    "file2_approx_line": line2,
                    "token_count": block.size,
                    "matched_text": matched_text[:500],  # Truncate for display
                })
        
        # Sort by length (longest first)
        return sorted(matches, key=lambda x: -x["token_count"])

    def get_similar_regions(
        self, 
        code1: str, 
        code2: str,
        block_threshold: float = 0.6,
        min_tokens: int = 10
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
        
        # Get AST-based function/class matches
        function_matches = self.find_similar_blocks(code1, code2, block_threshold)
        
        # Get token sequence matches
        token_matches = self.find_matching_token_sequences(code1, code2, min_tokens)
        
        # Calculate overall similarity
        overall_similarity = Levenshtein.ratio(code1, code2)
        
        elapsed = time.time() - start_time
        
        return {
            "function_matches": function_matches[:20],  # Top 20 function matches
            "token_matches": token_matches[:10],  # Top 10 token sequences
            "overall_similarity": round(overall_similarity, 4),
            "stats": {
                "file1_blocks": len(self.extract_code_blocks(code1)),
                "file2_blocks": len(self.extract_code_blocks(code2)),
                "total_function_matches": len(function_matches),
                "total_token_matches": len(token_matches),
                "analysis_time": round(elapsed, 4),
            }
        }

    # ===== Legacy API (backwards compatible) =====

    def add_submission(self, name: str, code: str):
        """Legacy: Add submission to internal storage."""
        self._embeddings[name] = self.process_to_embeddings(name, code)

    def calculate_similarities(self, new_code: str) -> dict:
        """Legacy: Calculate similarities against stored submissions."""
        if not self._embeddings:
            return {"error": "No previous submissions available"}
        
        # Create temp embedding for new code
        new_emb = self.process_to_embeddings("__target__", new_code)
        
        result = self.compare_preprocessed("__target__", {**self._embeddings, "__target__": new_emb})
        return result.get("scores", result)

    @property
    def submission_names(self) -> List[str]:
        """Legacy: Get list of submission names."""
        return list(self._embeddings.keys())
