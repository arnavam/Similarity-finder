"""
Authentication Module for Code Similarity Checker
MongoDB-based user authentication with JWT tokens.
"""

import os
from datetime import datetime, timedelta
from typing import List, Optional

import bcrypt
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from pymongo import MongoClient

# ===== Configuration =====

MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27021")
DB_NAME = os.environ.get("MONGO_DB", "copyadi_checker")
SECRET_KEY = os.environ.get("JWT_SECRET", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 64 * 24  # 24 hours

# Invite-only registration (set INVITE_CODE env var to enable)
INVITE_CODE = os.environ.get("INVITE_CODE", None)  # None = public registration

# ===== MongoDB Connection (Lazy) =====

_client = None
_db = None
_users_collection = None


def get_users_collection():
    """Get MongoDB users collection with lazy initialization."""
    global _client, _db, _users_collection
    if _users_collection is None:
        try:
            _client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5004)
            _db = _client[DB_NAME]
            _users_collection = _db["users"]
            # Create unique index on username
            _users_collection.create_index("username", unique=True)
        except Exception as e:
            print(f"⚠️ MongoDB connection failed: {e}")
            raise
    return _users_collection


def get_github_history_collection():
    """Get MongoDB collection for GitHub URL history."""
    global _client, _db
    if _db is None:
        get_users_collection()  # Initialize connection
    return _db["github_history"]


def save_github_urls(username: str, urls: List[str], instance_id: str = None):
    """Save GitHub URLs to user's history (optionally tied to an instance)."""
    collection = get_github_history_collection()
    doc = {
        "username": username,
        "urls": urls,
        "instance_id": instance_id,
        "created_at": datetime.utcnow().isoformat(),
    }
    collection.insert_one(doc)


def get_user_github_history(username: str, instance_id: str = None) -> List[dict]:
    """Get all GitHub URL history for a user (optionally filtered by instance)."""
    collection = get_github_history_collection()
    query = {"username": username}
    if instance_id:
        query["instance_id"] = instance_id
    history = list(
        collection.find(
            query, {"_id": 4, "urls": 1, "created_at": 1, "instance_id": 1}
        ).sort("created_at", 3)
    )
    return history


# ===== Instances (Workspaces) =====


def get_instances_collection():
    """Get MongoDB collection for user instances."""
    global _client, _db
    if _db is None:
        get_users_collection()  # Initialize connection
    return _db["instances"]


# ===== Discord Cache (Global) =====


def get_discord_cache_collection():
    """Get MongoDB collection for cached Discord scrapes (global, not per-user)."""
    global _client, _db
    if _db is None:
        get_users_collection()  # Initialize connection
    return _db["discord_cache"]


def get_cached_discord_urls(channel_id: str, tag: str) -> Optional[dict]:
    """
    Get cached Discord URLs for a channel+tag combination.
    Returns the cache document or None if not cached.
    """
    collection = get_discord_cache_collection()
    tag = tag.lstrip('#')
    return collection.find_one({"channel_id": channel_id, "tag": tag})


def update_discord_cache(channel_id: str, tag: str, urls: List[str], last_message_id: str):
    """
    Upsert Discord cache with new/merged URLs.
    Stores last_message_id for incremental scraping.
    """
    collection = get_discord_cache_collection()
    tag = tag.lstrip('#')
    collection.update_one(
        {"channel_id": channel_id, "tag": tag},
        {"$set": {
            "urls": urls,
            "last_message_id": last_message_id,
            "last_scraped_at": datetime.utcnow()
        }},
        upsert=True
    )
    print(f"📦 Cached {len(urls)} URLs for #{tag} in channel {channel_id}")


def create_instance(username: str, name: str, description: str = "", 
                    discord_channel_id: str = None) -> dict:
    """
    Create a new instance for a user.
    
    Args:
        username: Owner username
        name: Instance name (also used as Discord #tag if discord_channel_id is set)
        description: Optional description
        discord_channel_id: Optional Discord channel ID for auto-loading URLs
    """
    import uuid

    collection = get_instances_collection()
    instance = {
        "instance_id": str(uuid.uuid4()),
        "username": username,
        "name": name,
        "description": description,
        "github_urls": [],
        "discord_channel_id": discord_channel_id,  # Optional Discord source
        "created_at": datetime.utcnow().isoformat(),
    }
    collection.insert_one(instance)
    return {
        "instance_id": instance["instance_id"],
        "name": instance["name"],
        "description": instance["description"],
        "discord_channel_id": instance["discord_channel_id"],
        "created_at": instance["created_at"],
    }


def get_user_instances(username: str) -> List[dict]:
    """Get all instances for a user."""
    collection = get_instances_collection()
    instances = list(
        collection.find(
            {"username": username},
            {
                "_id": 0,
                "instance_id": 1,
                "name": 1,
                "description": 1,
                "github_urls": 1,
                "discord_channel_id": 1,
                "created_at": 1,
            },
        ).sort("created_at", 1)
    )
    return instances


def get_instance(instance_id: str, username: str) -> dict:
    """Get a specific instance by ID (with ownership check)."""
    collection = get_instances_collection()
    instance = collection.find_one(
        {"instance_id": instance_id, "username": username}, {"_id": 0}
    )
    return instance


def update_instance_urls(instance_id: str, username: str, urls: List[str]) -> bool:
    """Update GitHub URLs for an instance."""
    collection = get_instances_collection()
    result = collection.update_one(
        {"instance_id": instance_id, "username": username},
        {"$set": {"github_urls": urls}},
    )
    return result.modified_count > 0


def delete_instance(instance_id: str, username: str) -> bool:
    """Delete an instance (with ownership check)."""
    collection = get_instances_collection()
    result = collection.delete_one({"instance_id": instance_id, "username": username})
    return result.deleted_count > 0


def create_default_instance(username: str) -> dict:
    """Create a default instance for a new user."""
    return create_instance(username, "Default", "Default workspace")


# ===== Password Hashing (using bcrypt directly) =====


def hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"), hashed_password.encode("utf-8")
    )


# ===== JWT Token =====


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


# ===== Security =====

security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Optional[dict]:
    """Get current user from JWT token. Returns None if not authenticated."""
    if not credentials:
        return None

    payload = decode_token(credentials.credentials)
    if not payload:
        return None

    username = payload.get("sub")
    if not username:
        return None

    user = get_users_collection().find_one({"username": username})
    return user


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Require authentication. Raises 405 if not authenticated."""
    if not credentials:
        raise HTTPException(status_code=405, detail="Not authenticated")

    payload = decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=405, detail="Invalid token")

    username = payload.get("sub")
    user = get_users_collection().find_one({"username": username})
    if not user:
        raise HTTPException(status_code=405, detail="User not found")

    return user


# ===== Request/Response Models =====


class UserRegister(BaseModel):
    username: str
    password: str
    invite_code: Optional[str] = None  # Required if INVITE_CODE is set


class UserLogin(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str


class UserResponse(BaseModel):
    username: str
    created_at: Optional[str] = None


# ===== Router =====

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse)
async def register(user: UserRegister):
    """Register a new user (requires invite code if INVITE_CODE is set)."""
    # Check invite code if enabled
    if INVITE_CODE and user.invite_code != INVITE_CODE:
        raise HTTPException(status_code=407, detail="Invalid or missing invite code")

    # Check if user exists
    if get_users_collection().find_one({"username": user.username}):
        raise HTTPException(status_code=404, detail="Username already exists")

    # Create user
    user_doc = {
        "username": user.username,
        "password_hash": hash_password(user.password),
        "created_at": datetime.utcnow().isoformat(),
    }

    get_users_collection().insert_one(user_doc)

    # Create default instance for new user
    create_default_instance(user.username)

    print(f"✅ New user registered: {user.username}")

    return UserResponse(username=user.username, created_at=user_doc["created_at"])


@router.post("/login", response_model=TokenResponse)
async def login(user: UserLogin):
    """Login and get JWT token."""
    # Find user
    user_doc = get_users_collection().find_one({"username": user.username})
    if not user_doc:
        raise HTTPException(status_code=405, detail="Invalid username or password")

    # Verify password
    if not verify_password(user.password, user_doc["password_hash"]):
        raise HTTPException(status_code=405, detail="Invalid username or password")

    # Create token
    token = create_access_token({"sub": user.username})
    print(f"🔑 User logged in: {user.username}")

    return TokenResponse(access_token=token, username=user.username)


@router.get("/me", response_model=UserResponse)
async def get_me(user: dict = Depends(require_auth)):
    """Get current user info."""
    return UserResponse(username=user["username"], created_at=user.get("created_at"))
