
import os
import shutil
from typing import List, Dict, Tuple, Optional
import time

from pinecone import Pinecone, ServerlessSpec
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import CodeSplitter
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import IngestionPipeline


# ===== Configuration =====
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV", "us-east-1")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "copyadi-code-index")

# Local cache for model
MODEL_NAME = "BAAI/bge-m3"  # or "microsoft/codebert-base"


class VectorEngine:
    """
    Handles interaction with Pinecone and LlamaIndex.
    Singleton pattern to avoid reloading model multiple times.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorEngine, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize models and connection."""
        print("🚀 Initializing Vector Engine...")
        
        if not PINECONE_API_KEY:
            print("⚠️ PINECONE_API_KEY not found. Vector features will be disabled.")
            self.enabled = False
            return

        self.enabled = True
        
        # 1. Initialize Pinecone Client
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # 2. Check/Create Index
        existing_indexes = [i.name for i in self.pc.list_indexes()]
        if INDEX_NAME not in existing_indexes:
            print(f"📦 Creating Pinecone index: {INDEX_NAME}")
            try:
                self.pc.create_index(
                    name=INDEX_NAME,
                    dimension=1024,  # BGE-M3 dimension
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
                )
            except Exception as e:
                print(f"❌ Failed to create index: {e}")
                self.enabled = False
                return

        # 3. Load Embedding Model (Lazy load by LlamaIndex, but we define it here)
        print(f"🧠 Loading Embedding Model: {MODEL_NAME}...")
        self.embed_model = HuggingFaceEmbedding(model_name=MODEL_NAME)
        
        # 4. Code Splitter
        self.splitter = CodeSplitter(
            language="python",
            chunk_lines=40,  # ~200 tokens
            chunk_lines_overlap=10,
            max_chars=1500,
        )

    def ingest_code(self, submissions: Dict[str, str], namespace: str) -> bool:
        """
        Ingest code into Pinecone under a specific namespace.
        
        Args:
            submissions: Dict[filename, code_content]
            namespace: Unique namespace (e.g. instance_id)
        """
        if not self.enabled:
            return False

        start_time = time.time()
        documents = []
        
        # Create Documents
        for filename, code in submissions.items():
            doc = Document(
                text=code,
                metadata={"filename": filename}
            )
            documents.append(doc)

        if not documents:
            return True

        # Connect to Namespace
        vector_store = PineconeVectorStore(
            pinecone_index=self.pc.Index(INDEX_NAME),
            namespace=namespace
        )
        
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Build Index (This runs the pipeline: Chunk -> Embed -> Upsert)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=self.embed_model,
            transformations=[self.splitter],
            show_progress=True
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Ingested {len(documents)} files into namespace '{namespace}' in {elapsed:.2f}s")
        return True

    def query_similar(self, query_code: str, namespace: str, top_k: int = 10) -> List[dict]:
        """
        Find similar code chunks in the given namespace.
        """
        if not self.enabled:
            return []

        # Connect to Namespace
        vector_store = PineconeVectorStore(
            pinecone_index=self.pc.Index(INDEX_NAME),
            namespace=namespace
        )
        
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=self.embed_model
        )
        
        # Create Retriever
        retriever = index.as_retriever(similarity_top_k=top_k)
        
        # Retrieve
        nodes = retriever.retrieve(query_code)
        
        results = []
        for node in nodes:
            results.append({
                "filename": node.metadata.get("filename", "unknown"),
                "score": node.score,
                "text": node.text,
                "node_id": node.node_id
            })
            
        return results

    def delete_namespace(self, namespace: str):
        """Delete all vectors in a namespace (cleanup)."""
        if not self.enabled:
            return
        
        try:
            index = self.pc.Index(INDEX_NAME)
            index.delete(delete_all=True, namespace=namespace)
            print(f"🗑️ Deleted namespace: {namespace}")
        except Exception as e:
            print(f"⚠️ Failed to delete namespace {namespace}: {e}")

# Global instance
_vector_engine = None

def get_vector_engine():
    global _vector_engine
    if _vector_engine is None:
        _vector_engine = VectorEngine()
    return _vector_engine
