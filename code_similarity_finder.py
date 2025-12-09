import ast
import Levenshtein
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the English tokenizer model
nlp = spacy.load("en_core_web_sm")


class FlexibleCodeSimilarityChecker:
    def __init__(self, preprocessing_options=None):
        """
        Initialize with customizable preprocessing options

        Args:
            preprocessing_options (dict): Options for preprocessing
                - remove_comments (bool): Whether to remove comments
                - normalize_whitespace (bool): Whether to normalize whitespace
                - preserve_variable_names (bool): Whether to keep variable names in AST
                - preserve_literals (bool): Whether to keep literals in AST
        """
        self.submission_names = []  # Store submission identifiers
        self.previous_submissions = []  # Store original code
        self.previous_asts = []
        self.previous_tokens = []

        # Default preprocessing options
        self.options = {
            "remove_comments": True,
            "normalize_whitespace": True,
            "preserve_variable_names": False,
            "preserve_literals": False,
        }

        # Update with user-provided options
        if preprocessing_options:
            self.options.update(preprocessing_options)

    def add_submission(self, name, code):
        """Add a new submission to the history with an identifier"""
        self.submission_names.append(name)
        self.previous_submissions.append(code)

        # Process based on options
        processed_code = self.preprocess_code(code)
        ast_representation = self.extract_ast_features(code)
        tokens = self.tokenize_code(processed_code)

        self.previous_asts.append(ast_representation)
        self.previous_tokens.append(tokens)

    def preprocess_code(self, code):
        """Preprocess code based on options"""
        processed = code

        if self.options["remove_comments"]:
            # Remove single-line comments
            lines = []
            for line in processed.split("\n"):
                if "#" in line:
                    line = line.split("#")[0]
                lines.append(line)
            processed = "\n".join(lines)

            # Remove multi-line comments (docstrings)
            try:
                tree = ast.parse(processed)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Expr) and isinstance(
                        node.value, ast.Constant
                    ):
                        # This is a docstring, remove it
                        node.value.s = ""
                processed = ast.unparse(tree)
            except:
                pass  # If AST parsing fails, continue without removing docstrings

        if self.options["normalize_whitespace"]:
            # Normalize whitespace but preserve structure
            lines = []
            for line in processed.split("\n"):
                line = " ".join(line.split())  # Collapse multiple spaces
                if line:  # Skip empty lines
                    lines.append(line)
            processed = "\n".join(lines)

        return processed

    def extract_ast_features(self, code):
        """
        Extract AST features while preserving information based on options

        Returns a dictionary with structural information
        """
        try:
            tree = ast.parse(code)
            features = {
                "node_types": [],
                "function_calls": [],
                "control_flow": [],
                "operators": [],
            }

            for node in ast.walk(tree):
                # Always capture node types
                features["node_types"].append(type(node).__name__)

                # Capture function calls with optional name preservation
                if isinstance(node, ast.Call):
                    if (
                        isinstance(node.func, ast.Name)
                        and self.options["preserve_variable_names"]
                    ):
                        features["function_calls"].append(node.func.id)
                    else:
                        features["function_calls"].append("FUNC_CALL")

                # Capture control flow structures
                if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                    features["control_flow"].append(type(node).__name__)

                # Capture operators
                if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare)):
                    features["operators"].append(type(node.op).__name__)

            return features
        except:
            # Return empty features if parsing fails
            return {
                "node_types": [],
                "function_calls": [],
                "control_flow": [],
                "operators": [],
            }

    def tokenize_code(self, code):
        """Simple tokenization of code"""
        tokens = []
        doc = nlp(code)

        for token in doc:
            tokens.append(token.text)
        return tokens

    def calculate_similarities(self, new_code):
        """Calculate multiple similarity scores with detailed results"""
        if not self.previous_submissions:
            return {"error": "No previous submissions available"}

        # Process the new code
        processed_new_code = self.preprocess_code(new_code)
        new_ast = self.extract_ast_features(new_code)
        new_tokens = self.tokenize_code(processed_new_code)

        results = {}
        all_scores = {}

        # 1. Raw text similarity (Levenshtein)
        raw_scores = [
            Levenshtein.ratio(new_code, prev) for prev in self.previous_submissions
        ]
        all_scores["raw_scores"] = raw_scores

        # 2. Processed text similarity
        processed_scores = [
            Levenshtein.ratio(processed_new_code, self.preprocess_code(prev))
            for prev in self.previous_submissions
        ]
        all_scores["processed_scores"] = processed_scores

        # 3. TF-IDF similarity
        all_texts = [
            self.preprocess_code(prev) for prev in self.previous_submissions
        ] + [processed_new_code]
        tfidf_matrix = TfidfVectorizer().fit_transform(all_texts)
        cosine_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

        all_scores["cosine_scores"] = [float(i) for i in cosine_scores[0]]

        # 4. AST structure similarity
        ast_scores = []
        for prev_ast in self.previous_asts:
            if prev_ast["node_types"] and new_ast["node_types"]:
                # Compare node type sequences
                score = self.sequence_similarity(
                    new_ast["node_types"], prev_ast["node_types"]
                )
                ast_scores.append(score)
            else:
                ast_scores.append(0.0)

        if any(score > 0 for score in ast_scores):
            all_scores["ast_scores"] = ast_scores

        # 5. Token-based similarity
        token_scores = []
        for prev_tokens in self.previous_tokens:
            if prev_tokens and new_tokens:
                score = self.sequence_similarity(new_tokens, prev_tokens)
                token_scores.append(score)
            else:
                token_scores.append(0.0)

        all_scores["token_scores"] = token_scores

        return all_scores

    def sequence_similarity(self, seq1, seq2):
        """Calculate similarity between two sequences"""
        if not seq1 or not seq2:
            return 0

        # Find the longest common subsequence
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_length = dp[m][n]
        return lcs_length / max(m, n)
