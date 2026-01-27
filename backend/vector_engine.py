"""
Vector Engine Module - handles interaction with Pinecone and LlamaIndex.
Uses module-level state (Python modules are natural singletons).
Fails fast on initialization errors.
"""

import os
import time
from typing import Dict, List

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# ===== Configuration =====
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV", "us-east-1")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "copyadi-code-index")
INDEX_NAME2 = os.environ.get("PINECONE_INDEX_NAME2", "copyadi-code-index2")
MODEL_NAME = "BAAI/bge-m3"

# ===== Initialize on module import (fail-fast) =====
print("🚀 Initializing Vector Engine...")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found.")

# 1. Initialize Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)

# 2. Check/Create Indexes (both dense and sparse)
existing_indexes = [i.name for i in pc.list_indexes()]
for idx_name in [INDEX_NAME, INDEX_NAME2]:
    if idx_name not in existing_indexes:
        print(f"📦 Creating Pinecone index: {idx_name}")
        pc.create_index(
            name=idx_name,
            dimension=1024,  # BGE-M3 dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),
        )

# 3. Load Embedding Model
print(f"🧠 Loading Embedding Model: {MODEL_NAME}...")
embed_model = HuggingFaceEmbedding(model_name=MODEL_NAME)

# 4. Simple Text Splitter
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

enabled = True
print("✅ Vector Engine initialized successfully!")


# ===== Public Functions =====


def _compute_hash(content: str) -> str:
    """Compute a short hash of content for change detection."""
    import hashlib
    return hashlib.md5(content.encode()).hexdigest()[:16]


def _get_existing_files(index_name: str, namespace: str) -> Dict[str, str]:
    """Query Pinecone to get existing files and their content hashes.
    
    Returns:
        Dict of {filename: content_hash}
    """
    index = pc.Index(index_name)
    
    # Get stats to check if namespace has any vectors
    stats = index.describe_index_stats()
    ns_stats = stats.get("namespaces", {}).get(namespace, {})
    if ns_stats.get("vector_count", 0) == 0:
        return {}
    
    # Query with a dummy vector to get all vectors with metadata
    # We use list() to get vectors by prefix filter
    try:
        # Fetch vectors using list + fetch pattern
        # Note: Pinecone list() returns vector IDs, then we fetch metadata
        results = index.query(
            vector=[0.0] * 1024,  # Dummy vector (BGE-M3 dimension)
            top_k=10000,
            namespace=namespace,
            include_metadata=True,
        )
        
        existing = {}
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            filename = metadata.get("filename")
            content_hash = metadata.get("content_hash")
            if filename and content_hash:
                existing[filename] = content_hash
        
        return existing
    except Exception as e:
        print(f"⚠️ Could not query existing files: {e}")
        return {}


def ingest_code(
    submissions: Dict[str, str],
    namespace: str,
    index_name: str = INDEX_NAME,
    force: bool = False,
) -> bool:
    """
    Ingest code into Pinecone under a specific namespace.
    Uses smart caching - only ingests new or changed files.

    Args:
        submissions: Dict[filename, code_content]
        namespace: Unique namespace (e.g. instance_id)
        index_name: Pinecone index to use (default: INDEX_NAME for dense)
        force: If True, re-ingest all files even if unchanged
    """
    start_time = time.time()
    
    # Get existing files and their hashes
    existing_files = {} if force else _get_existing_files(index_name, namespace)
    
    # Determine which files need ingestion
    new_files = {}
    for filename, code in submissions.items():
        content_hash = _compute_hash(code)
        existing_hash = existing_files.get(filename)
        
        if existing_hash != content_hash:
            new_files[filename] = (code, content_hash)
    
    if not new_files:
        print(f"✅ All {len(submissions)} files already ingested in namespace '{namespace}'. Skipping.")
        return True
    
    print(f"🔄 Ingesting {len(new_files)} new/changed files (skipping {len(submissions) - len(new_files)} cached)")
    
    # Create Documents with content hash in metadata
    documents = []
    for filename, (code, content_hash) in new_files.items():
        doc = Document(
            text=code, 
            metadata={
                "filename": filename,
                "content_hash": content_hash,
            }
        )
        documents.append(doc)

    # Connect to Namespace
    vector_store = PineconeVectorStore(
        pinecone_index=pc.Index(index_name), namespace=namespace
    )

    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build Index (This runs the pipeline: Chunk -> Embed -> Upsert)
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
        show_progress=True,
    )

    elapsed = time.time() - start_time
    print(f"✅ Ingested {len(documents)} files into namespace '{namespace}' in {elapsed:.2f}s")
    return True


def query_similar(
    query_code: str,
    namespace: str,
    index_name: str = INDEX_NAME,
    top_k: int = 10,
) -> List[dict]:
    """
    Find similar code chunks in the given namespace.

    Args:
        query_code: Code to find similar matches for
        namespace: Namespace to search in
        index_name: Pinecone index to use (default: INDEX_NAME for dense)
        top_k: Number of results to return
    """
    # Connect to Namespace
    vector_store = PineconeVectorStore(
        pinecone_index=pc.Index(index_name), namespace=namespace
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )

    # Create Retriever
    retriever = index.as_retriever(similarity_top_k=top_k)

    # Retrieve
    nodes = retriever.retrieve(query_code)

    results = []
    for node in nodes:
        results.append(
            {
                "filename": node.metadata.get("filename", "unknown"),
                "score": node.score,
                "text": node.text,
                "node_id": node.node_id,
            }
        )

    return results


def delete_namespace(namespace: str, index_name: str = INDEX_NAME):
    """Delete all vectors in a namespace (cleanup).

    Args:
        namespace: Namespace to delete
        index_name: Pinecone index to use (default: INDEX_NAME for dense)
    """
    index = pc.Index(index_name)
    index.delete(delete_all=True, namespace=namespace)
    print(f"🗑️ Deleted namespace '{namespace}' from index '{index_name}'")
