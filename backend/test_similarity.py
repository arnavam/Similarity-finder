"""
Test file for the optimized code similarity checker.
Run with: python test_similarity.py
"""

import time
import sys

# Add backend to path
sys.path.insert(0, '/Users/arnav/Code/Copyadi-finder/backend')

from code_similarity_finder import FlexibleCodeSimilarityChecker


def test_process_to_embeddings():
    """Test that embeddings are generated correctly."""
    print("\n=== Test: process_to_embeddings ===")
    
    checker = FlexibleCodeSimilarityChecker()
    code = '''
def hello():
    print("Hello, World!")

def add(a, b):
    return a + b
'''
    
    emb = checker.process_to_embeddings('test_file.py', code)
    
    assert 'name' in emb, "Missing 'name' in embeddings"
    assert 'processed_text' in emb, "Missing 'processed_text' in embeddings"
    assert 'ast' in emb, "Missing 'ast' in embeddings"
    assert 'tokens' in emb, "Missing 'tokens' in embeddings"
    assert 'raw_code' in emb, "Missing 'raw_code' in embeddings"
    
    print(f"  ✅ Embeddings keys: {list(emb.keys())}")
    print(f"  ✅ Token count: {len(emb['tokens'])}")
    print(f"  ✅ AST node types: {len(emb['ast']['node_types'])}")


def test_compare_embeddings():
    """Test pairwise comparison of embeddings."""
    print("\n=== Test: compare_embeddings ===")
    
    checker = FlexibleCodeSimilarityChecker()
    
    code1 = 'def hello():\n    print("Hello")'
    code2 = 'def hello():\n    print("Hi")'  # Similar
    code3 = 'class Foo:\n    def bar(self): pass'  # Different
    
    emb1 = checker.process_to_embeddings('file1', code1)
    emb2 = checker.process_to_embeddings('file2', code2)
    emb3 = checker.process_to_embeddings('file3', code3)
    
    scores_similar = checker.compare_embeddings(emb1, emb2)
    scores_different = checker.compare_embeddings(emb1, emb3)
    
    print(f"  Similar (file1 vs file2): {scores_similar}")
    print(f"  Different (file1 vs file3): {scores_different}")
    
    # Similar code should have higher scores
    assert scores_similar['raw'] > scores_different['raw'], "Similar code should have higher raw score"
    print("  ✅ Similar code has higher similarity scores")


def test_analyze_direct():
    """Test direct mode: parallel process + compare."""
    print("\n=== Test: analyze_direct (parallel processing) ===")
    
    checker = FlexibleCodeSimilarityChecker(max_workers=2)
    
    submissions = {
        'target.py': 'def main():\n    print("Hello")\n    x = 1 + 2',
        'similar.py': 'def main():\n    print("Hi")\n    y = 2 + 3',
        'different.py': 'class Database:\n    def connect(self): pass',
        'another.py': 'import os\nfor i in range(10): print(i)',
    }
    
    start = time.time()
    result = checker.analyze_direct('target.py', submissions)
    elapsed = time.time() - start
    
    assert 'scores' in result, "Missing 'scores' in result"
    assert 'submission_names' in result, "Missing 'submission_names' in result"
    assert len(result['submission_names']) == 3, "Should compare against 3 other files"
    
    print(f"  ✅ Compared against {len(result['submission_names'])} submissions")
    print(f"  ✅ Score types: {list(result['scores'].keys())}")
    print(f"  ✅ Time: {elapsed:.4f}s")
    
    # Print detailed scores
    for score_type, values in result['scores'].items():
        if values:
            print(f"     {score_type}: {[f'{v:.2f}' for v in values]}")


def test_preprocess_all():
    """Test preprocess mode: parallel preprocessing."""
    print("\n=== Test: preprocess_all (parallel preprocessing) ===")
    
    checker = FlexibleCodeSimilarityChecker(max_workers=2)
    
    submissions = {
        'file1.py': 'def func1(): pass',
        'file2.py': 'def func2(): return 1',
        'file3.py': 'class MyClass: pass',
    }
    
    start = time.time()
    embeddings = checker.preprocess_all(submissions)
    elapsed = time.time() - start
    
    assert len(embeddings) == 3, "Should have 3 embeddings"
    assert all('tokens' in e for e in embeddings.values()), "All should have tokens"
    
    print(f"  ✅ Preprocessed {len(embeddings)} submissions")
    print(f"  ✅ Time: {elapsed:.4f}s")


def test_compare_preprocessed():
    """Test comparison using preprocessed embeddings."""
    print("\n=== Test: compare_preprocessed ===")
    
    checker = FlexibleCodeSimilarityChecker()
    
    submissions = {
        'target.py': 'def main(): print("Hello")',
        'other1.py': 'def main(): print("Hi")',
        'other2.py': 'class Foo: pass',
    }
    
    # First preprocess
    embeddings = checker.preprocess_all(submissions)
    
    # Then compare
    result = checker.compare_preprocessed('target.py', embeddings)
    
    assert 'scores' in result, "Missing scores"
    assert len(result['submission_names']) == 2, "Should compare against 2 others"
    
    print(f"  ✅ Compared target against {len(result['submission_names'])} submissions")
    print(f"  ✅ Scores: {result['scores']}")


def test_legacy_api():
    """Test that legacy API still works for backwards compatibility."""
    print("\n=== Test: Legacy API (backwards compatibility) ===")
    
    checker = FlexibleCodeSimilarityChecker()
    
    # Old way: add_submission + calculate_similarities
    checker.add_submission('old1.py', 'def foo(): pass')
    checker.add_submission('old2.py', 'def bar(): return 1')
    
    scores = checker.calculate_similarities('def baz(): pass')
    
    assert 'raw_scores' in scores or 'error' not in scores, "Legacy API should work"
    print(f"  ✅ Legacy API works")
    print(f"  ✅ Submission names: {checker.submission_names}")


def main():
    print("=" * 60)
    print("Testing Optimized Code Similarity Checker")
    print("=" * 60)
    
    try:
        test_process_to_embeddings()
        test_compare_embeddings()
        test_analyze_direct()
        test_preprocess_all()
        test_compare_preprocessed()
        test_legacy_api()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
