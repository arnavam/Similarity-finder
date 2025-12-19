"""
Submission Buffer Module for Code Similarity Checker.
Provides a TTL-based storage for extracted text and preprocessed features.
Uses an abstraction layer to allow switching databases in the future.
"""

import os
import uuid
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from pymongo import MongoClient, ASCENDING

# ===== Configuration =====
MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.environ.get("MONGO_DB", "copyadi_checker")
BUFFER_TTL_SECONDS = 3600  # 1 hour

# ===== Interface =====

class SubmissionBufferInterface(ABC):
    """Abstract base class for submission buffering."""
    
    @abstractmethod
    def save_submissions(self, submissions: Dict[str, str]) -> str:
        """Save a batch of submissions and return a unique buffer_id."""
        pass
        
    @abstractmethod
    def get_submissions(self, buffer_id: str) -> Dict[str, str]:
        """Retrieve all raw text submissions for a buffer_id."""
        pass
        
    @abstractmethod
    def save_preprocessed(self, buffer_id: str, name: str, data: Dict[str, Any]) -> bool:
        """Cache preprocessed features for a specific submission."""
        pass
        
    @abstractmethod
    def get_preprocessed_batch(self, buffer_id: str) -> Dict[str, Dict[str, Any]]:
        """Retrieve all preprocessed data for a buffer_id."""
        pass

# ===== MongoDB Implementation =====

class MongoSubmissionBuffer(SubmissionBufferInterface):
    """MongoDB implementation of the submission buffer."""
    
    def __init__(self):
        self.client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        self.db = self.client[DB_NAME]
        self.collection = self.db["submission_buffer"]
        
        # Create TTL index for automatic cleanup
        # We index 'expires_at' and MongoDB will delete docs when current time > expires_at
        self.collection.create_index("expires_at", expireAfterSeconds=0)
        # Index buffer_id for fast lookups
        self.collection.create_index([("buffer_id", ASCENDING)])
        
    def save_submissions(self, submissions: Dict[str, str]) -> str:
        """
        Save each file as a separate document to handle large batches.
        Returns a unique buffer_id.
        """
        buffer_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(seconds=BUFFER_TTL_SECONDS)
        
        docs = []
        for name, text in submissions.items():
            docs.append({
                "buffer_id": buffer_id,
                "name": name,
                "raw_text": text,
                "preprocessed": None,
                "created_at": datetime.utcnow(),
                "expires_at": expires_at
            })
            
        if docs:
            self.collection.insert_many(docs)
            print(f"📦 Stored {len(docs)} files in buffer {buffer_id}")
            
        return buffer_id
        
    def get_submissions(self, buffer_id: str) -> Dict[str, str]:
        """Retrieve all raw text for a specific session."""
        cursor = self.collection.find({"buffer_id": buffer_id})
        return {doc["name"]: doc["raw_text"] for doc in cursor}
        
    def save_preprocessed(self, buffer_id: str, name: str, data: Dict[str, Any]) -> bool:
        """Update a specific doc with its preprocessed features (tokens, ast, etc)."""
        result = self.collection.update_one(
            {"buffer_id": buffer_id, "name": name},
            {"$set": {"preprocessed": data}}
        )
        return result.modified_count > 0
        
    def get_preprocessed_batch(self, buffer_id: str) -> Dict[str, Dict[str, Any]]:
        """Retrieve all preprocessed data for a batch."""
        cursor = self.collection.find({"buffer_id": buffer_id})
        batch = {}
        for doc in cursor:
            if doc.get("preprocessed"):
                batch[doc["name"]] = doc["preprocessed"]
        return batch

# ===== Factory / Singleton =====

_buffer_instance = None

def get_buffer() -> SubmissionBufferInterface:
    """Get the global buffer instance."""
    global _buffer_instance
    if _buffer_instance is None:
        # Default to MongoDB implementation
        _buffer_instance = MongoSubmissionBuffer()
    return _buffer_instance
