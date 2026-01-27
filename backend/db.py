"""
Database Module for Code Similarity Checker.
Uses Beanie ODM for MongoDB with async Motor driver.
All models and collections are defined here with schema enforcement.
"""

import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from beanie import Document, Indexed, init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import Field
from pymongo import IndexModel

# ===== Configuration =====
MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.environ.get("MONGO_DB", "copyadi_checker")
BUFFER_TTL_SECONDS = 3600  # 1 hour for submission buffer

# ===== MongoDB Connection =====

print("🔌 Connecting to MongoDB...")
client = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=5000)
db = client[DB_NAME]
print(f"✅ MongoDB client ready: {DB_NAME}")

# ===== Document Models (Schema Definitions) =====


class User(Document):
    """User account for authentication."""
    username: Indexed(str, unique=True)
    password_hash: str
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    class Settings:
        name = "users"


class Instance(Document):
    """Workspace/project instance containing GitHub URLs."""
    instance_id: Indexed(str, unique=True) = Field(default_factory=lambda: str(uuid.uuid4()))
    username: Indexed(str)
    name: str
    description: str = ""
    github_urls: List[str] = []
    discord_channel_id: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    class Settings:
        name = "instances"


class GitHubHistory(Document):
    """History of GitHub URLs processed by a user."""
    username: Indexed(str)
    urls: List[str]
    instance_id: Optional[str] = None
    created_at: Indexed(str) = Field(default_factory=lambda: datetime.utcnow().isoformat())

    class Settings:
        name = "github_history"


class DiscordCache(Document):
    """Cache for Discord channel URL scraping."""
    channel_id: Indexed(str)
    tag: Indexed(str)
    urls: List[str] = []
    last_message_id: Optional[str] = None
    last_scraped_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "discord_cache"


class SubmissionBuffer(Document):
    """Temporary storage for uploaded code submissions (auto-expires)."""
    buffer_id: Indexed(str)
    name: str
    raw_text: str
    preprocessed: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime  # TTL field - MongoDB auto-deletes when this time is reached

    class Settings:
        name = "submission_buffer"
        indexes = [
            IndexModel(["expires_at"], expireAfterSeconds=0),  # TTL: expire at exact time
        ]


class RevokedToken(Document):
    """Revoked JWT tokens (for logout/security). Auto-expires with the token."""
    jti: Indexed(str, unique=True)  # JWT ID
    username: Indexed(str)
    revoked_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime  # TTL field - MongoDB auto-deletes when token would expire

    class Settings:
        name = "revoked_tokens"
        indexes = [
            IndexModel(["expires_at"], expireAfterSeconds=0),  # TTL: expire at exact time
        ]





# ===== Initialization (call at app startup) =====

_initialized = False


async def ensure_indexes():
    """Initialize Beanie with all document models. Call once at startup."""
    global _initialized
    if _initialized:
        return

    await init_beanie(
        database=db,
        document_models=[User, Instance, GitHubHistory, DiscordCache, SubmissionBuffer, RevokedToken]
    )
    _initialized = True
    print("✅ Beanie ODM initialized with all models")


# ===== Submission Buffer Operations =====


async def save_submissions(submissions: Dict[str, str]) -> str:
    """Save a batch of submissions. Returns a unique buffer_id."""
    await ensure_indexes()

    buffer_id = str(uuid.uuid4())
    expires_at = datetime.utcnow() + timedelta(seconds=BUFFER_TTL_SECONDS)

    docs = [
        SubmissionBuffer(
            buffer_id=buffer_id,
            name=name,
            raw_text=text,
            expires_at=expires_at,
        )
        for name, text in submissions.items()
    ]

    if docs:
        await SubmissionBuffer.insert_many(docs)
        print(f"📦 Stored {len(docs)} files in buffer {buffer_id}")

    return buffer_id


async def get_submissions(buffer_id: str) -> Dict[str, str]:
    """Retrieve all raw text for a buffer."""
    docs = await SubmissionBuffer.find(SubmissionBuffer.buffer_id == buffer_id).to_list()
    return {doc.name: doc.raw_text for doc in docs}


async def save_preprocessed(buffer_id: str, name: str, data: Dict[str, Any]) -> bool:
    """Save preprocessed features for a submission."""
    doc = await SubmissionBuffer.find_one(
        SubmissionBuffer.buffer_id == buffer_id,
        SubmissionBuffer.name == name
    )
    if doc:
        doc.preprocessed = data
        await doc.save()
        return True
    return False


async def get_preprocessed_batch(buffer_id: str) -> Dict[str, Dict[str, Any]]:
    """Retrieve all preprocessed data for a buffer."""
    docs = await SubmissionBuffer.find(SubmissionBuffer.buffer_id == buffer_id).to_list()
    return {doc.name: doc.preprocessed for doc in docs if doc.preprocessed}


async def get_buffer_contents(buffer_id: str) -> List[SubmissionBuffer]:
    """Retrieve all documents for a buffer to avoid multiple queries."""
    return await SubmissionBuffer.find(SubmissionBuffer.buffer_id == buffer_id).to_list()


# ===== User Operations =====


async def find_user(username: str) -> Optional[dict]:
    """Find a user by username."""
    user = await User.find_one(User.username == username)
    return user.model_dump() if user else None


async def create_user(username: str, password_hash: str) -> dict:
    """Create a new user."""
    user = User(username=username, password_hash=password_hash)
    await user.insert()
    return user.model_dump()


# ===== Instance Operations =====


async def create_instance(
    username: str, name: str, description: str = "", discord_channel_id: str = None
) -> dict:
    """Create a new instance for a user."""
    instance = Instance(
        username=username,
        name=name,
        description=description,
        discord_channel_id=discord_channel_id,
    )
    await instance.insert()
    return {
        "instance_id": instance.instance_id,
        "name": instance.name,
        "description": instance.description,
        "discord_channel_id": instance.discord_channel_id,
        "created_at": instance.created_at,
    }


async def get_user_instances(username: str) -> List[dict]:
    """Get all instances for a user (newest first)."""
    instances = await Instance.find(
        Instance.username == username
    ).sort(-Instance.created_at).to_list()

    return [
        {
            "instance_id": inst.instance_id,
            "name": inst.name,
            "description": inst.description,
            "github_urls": inst.github_urls,
            "discord_channel_id": inst.discord_channel_id,
            "created_at": inst.created_at,
        }
        for inst in instances
    ]


async def get_instance(instance_id: str, username: str) -> Optional[dict]:
    """Get a specific instance by ID (with ownership check)."""
    instance = await Instance.find_one(
        Instance.instance_id == instance_id,
        Instance.username == username
    )
    if not instance:
        return None
    return {
        "instance_id": instance.instance_id,
        "name": instance.name,
        "description": instance.description,
        "github_urls": instance.github_urls,
        "discord_channel_id": instance.discord_channel_id,
        "created_at": instance.created_at,
    }


async def update_instance_urls(instance_id: str, username: str, urls: List[str]) -> bool:
    """Update GitHub URLs for an instance."""
    instance = await Instance.find_one(
        Instance.instance_id == instance_id,
        Instance.username == username
    )
    if instance:
        instance.github_urls = urls
        await instance.save()
        return True
    return False


async def delete_instance(instance_id: str, username: str) -> bool:
    """Delete an instance (with ownership check)."""
    instance = await Instance.find_one(
        Instance.instance_id == instance_id,
        Instance.username == username
    )
    if instance:
        await instance.delete()
        return True
    return False


# ===== GitHub History Operations =====


async def save_github_urls(username: str, urls: List[str], instance_id: str = None):
    """Save GitHub URLs to user's history."""
    history = GitHubHistory(
        username=username,
        urls=urls,
        instance_id=instance_id,
    )
    await history.insert()


async def get_user_github_history(username: str, instance_id: str = None) -> List[dict]:
    """Get all GitHub URL history for a user."""
    query = {"username": username}
    if instance_id:
        query["instance_id"] = instance_id

    histories = await GitHubHistory.find(query).sort(-GitHubHistory.created_at).to_list()
    return [
        {"urls": h.urls, "created_at": h.created_at, "instance_id": h.instance_id}
        for h in histories
    ]


# ===== Discord Cache Operations =====


async def get_cached_discord_urls(channel_id: str, tag: str) -> Optional[dict]:
    """Get cached Discord URLs for a channel+tag combination."""
    tag = tag.lstrip("#")
    cache = await DiscordCache.find_one(
        DiscordCache.channel_id == channel_id,
        DiscordCache.tag == tag
    )
    if not cache:
        return None
    return {
        "urls": cache.urls,
        "last_message_id": cache.last_message_id,
        "last_scraped_at": cache.last_scraped_at,
    }


async def update_discord_cache(
    channel_id: str, tag: str, urls: List[str], last_message_id: str
):
    """Upsert Discord cache with new/merged URLs."""
    tag = tag.lstrip("#")
    cache = await DiscordCache.find_one(
        DiscordCache.channel_id == channel_id,
        DiscordCache.tag == tag
    )

    if cache:
        cache.urls = urls
        cache.last_message_id = last_message_id
        cache.last_scraped_at = datetime.utcnow()
        await cache.save()
    else:
        cache = DiscordCache(
            channel_id=channel_id,
            tag=tag,
            urls=urls,
            last_message_id=last_message_id,
        )
        await cache.insert()

    print(f"📦 Cached {len(urls)} URLs for #{tag} in channel {channel_id}")


# ===== Token Revocation Operations =====


async def revoke_token(jti: str, username: str, expires_at: datetime):
    """Revoke a token by its JTI."""
    await ensure_indexes()
    token = RevokedToken(jti=jti, username=username, expires_at=expires_at)
    await token.insert()
    print(f"🚫 Token revoked for user: {username}")


async def is_token_revoked(jti: str) -> bool:
    """Check if a token has been revoked."""
    await ensure_indexes()
    token = await RevokedToken.find_one(RevokedToken.jti == jti)
    return token is not None


async def revoke_all_user_tokens(username: str):
    """Revoke all tokens for a user (logout everywhere)."""
    # This doesn't actually revoke existing tokens, but you could
    # store a 'tokens_revoked_before' timestamp on the user document
    # and check against it. For now, this is a placeholder.
    pass
