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

        # Set defaults first to prevent AttributeError
        self.enabled = False
        self.embed_model = None
        self.splitter = None
        self.pc = None

        if not PINECONE_API_KEY:
            print("⚠️ PINECONE_API_KEY not found. Vector features will be disabled.")
            return

        try:
            # 1. Initialize Pinecone Client
            self.pc = Pinecone(api_key=PINECONE_API_KEY)

            # 2. Check/Create Index
            existing_indexes = [i.name for i in self.pc.list_indexes()]
            if INDEX_NAME not in existing_indexes:
                print(f"📦 Creating Pinecone index: {INDEX_NAME}")
                self.pc.create_index(
                    name=INDEX_NAME,
                    dimension=1024,  # BGE-M3 dimension
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),
                )

            # 3. Load Embedding Model (Lazy load by LlamaIndex, but we define it here)
            print(f"🧠 Loading Embedding Model: {MODEL_NAME}...")
            self.embed_model = HuggingFaceEmbedding(model_name=MODEL_NAME)

            # 4. Simple Text Splitter (no tree-sitter dependencies)
            self.splitter = SentenceSplitter(
                chunk_size=512,
                chunk_overlap=50,
            )

            # Only enable if everything succeeded
            self.enabled = True
            print("✅ Vector Engine initialized successfully!")

        except Exception as e:
            import traceback

            print(f"❌ Vector Engine initialization failed: {e}")
            traceback.print_exc()
            self.enabled = False

    def ingest_code(
        self, submissions: Dict[str, str], namespace: str, force: bool = False
    ) -> bool:
        """
        Ingest code into Pinecone under a specific namespace.

        Args:
            submissions: Dict[filename, code_content]
            namespace: Unique namespace (e.g. instance_id)
            force: If True, re-ingest even if namespace exists
        """
        if not self.enabled:
            return False

        # Check if namespace already has vectors (skip re-ingestion)
        if not force:
            try:
                index = self.pc.Index(INDEX_NAME)
                stats = index.describe_index_stats()
                ns_stats = stats.get("namespaces", {}).get(namespace, {})
                vector_count = ns_stats.get("vector_count", 0)
                if vector_count > 0:
                    print(
                        f"✅ Namespace '{namespace}' already has {
                            vector_count
                        } vectors. Skipping ingestion."
                    )
                    return True
            except Exception as e:
                print(f"⚠️ Could not check namespace stats: {e}")

        start_time = time.time()
        documents = []

        # Create Documents
        for filename, code in submissions.items():
            doc = Document(text=code, metadata={"filename": filename})
            documents.append(doc)

        if not documents:
            return True

        # Connect to Namespace
        vector_store = PineconeVectorStore(
            pinecone_index=self.pc.Index(INDEX_NAME), namespace=namespace
        )

        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        try:
            # Build Index (This runs the pipeline: Chunk -> Embed -> Upsert)
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=self.embed_model,
                transformations=[self.splitter],
                show_progress=True,
            )

            elapsed = time.time() - start_time
            print(
                f"✅ Ingested {len(documents)} files into namespace '{namespace}' in {
                    elapsed:.2f}s"
            )
            return True
        except Exception as e:
            import traceback

            print(f"❌ Failed to ingest code into vector store: {e}")
            traceback.print_exc()
            return False

    def query_similar(
        self, query_code: str, namespace: str, top_k: int = 10
    ) -> List[dict]:
        """
        Find similar code chunks in the given namespace.
        """
        if not self.enabled:
            return []

        try:
            # Connect to Namespace
            vector_store = PineconeVectorStore(
                pinecone_index=self.pc.Index(INDEX_NAME), namespace=namespace
            )

            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store, embed_model=self.embed_model
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
        except Exception as e:
            print(f"⚠️ Vector query failed: {e}")
            return []

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
