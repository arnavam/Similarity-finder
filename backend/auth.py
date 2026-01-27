"""
Authentication Module for Code Similarity Checker
JWT-based authentication with async database operations.
"""

import os
import uuid
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

import db

# ===== Configuration =====

SECRET_KEY = os.environ.get("JWT_SECRET", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15  # Short-lived access token
REFRESH_TOKEN_EXPIRE_DAYS = 7     # Long-lived refresh token

# Invite-only registration (set INVITE_CODE env var to enable)
INVITE_CODE = os.environ.get("INVITE_CODE", None)  # None = public registration


# ===== JWT Token =====


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None, refresh: bool = False) -> str:
    """Create a JWT token. Set refresh=True for refresh tokens."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    elif refresh:
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode["exp"] = expire
    to_encode["jti"] = str(uuid.uuid4())  # Unique token ID for revocation
    to_encode["refresh"] = refresh  # Token type flag
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

    # Check if token is revoked
    jti = payload.get("jti")
    if jti and await db.is_token_revoked(jti):
        return None

    username = payload.get("sub")
    if not username:
        return None

    user = await db.find_user(username)
    return user


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Require authentication. Raises 401 if not authenticated."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")

    payload = decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")

    # Check if token is revoked
    jti = payload.get("jti")
    if jti and await db.is_token_revoked(jti):
        raise HTTPException(status_code=401, detail="Token has been revoked")

    username = payload.get("sub")
    user = await db.find_user(username)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    # Attach token info to user for potential logout use
    user["_token_jti"] = jti
    user["_token_exp"] = payload.get("exp")
    
    # Reject refresh tokens for regular auth (they can only be used at /refresh)
    if payload.get("refresh"):
        raise HTTPException(status_code=401, detail="Cannot use refresh token for API access")

    return user


# ===== Request/Response Models =====


class UserRegister(BaseModel):
    username: str
    password: str
    invite_code: Optional[str] = None


class UserLogin(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60  # seconds
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
    if await db.find_user(user.username):
        raise HTTPException(status_code=400, detail="Username already exists")

    # Create user
    password_hash = bcrypt.hashpw(
        user.password.encode("utf-8"), bcrypt.gensalt()
    ).decode("utf-8")
    user_doc = await db.create_user(user.username, password_hash)

    # Create default instance
    await db.create_instance(user.username, "Default", "Default workspace")

    print(f"✅ New user registered: {user.username}")
    return UserResponse(username=user.username, created_at=user_doc["created_at"])


@router.post("/login", response_model=TokenResponse)
async def login(user: UserLogin):
    """Login and get JWT token."""
    user_doc = await db.find_user(user.username)
    if not user_doc:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Verify password
    if not bcrypt.checkpw(
        user.password.encode("utf-8"), user_doc["password_hash"].encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    access_token = create_access_token({"sub": user.username})
    refresh_token = create_access_token({"sub": user.username}, refresh=True)
    
    print(f"🔑 User logged in: {user.username}")
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        username=user.username
    )


@router.get("/me", response_model=UserResponse)
async def get_me(user: dict = Depends(require_auth)):
    """Get current user info."""
    return UserResponse(username=user["username"], created_at=user.get("created_at"))


@router.post("/logout")
async def logout(user: dict = Depends(require_auth)):
    """Logout and revoke the current token."""
    jti = user.get("_token_jti")
    exp = user.get("_token_exp")
    
    if jti and exp:
        # Convert exp timestamp to datetime
        expires_at = datetime.utcfromtimestamp(exp)
        await db.revoke_token(jti, user["username"], expires_at)
        return {"message": "Successfully logged out"}
    
    return {"message": "Logged out (token had no jti)"}


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/refresh", response_model=TokenResponse)
async def refresh_tokens(request: RefreshRequest):
    """Get new access & refresh tokens using a valid refresh token."""
    payload = decode_token(request.refresh_token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    
    # Must be a refresh token
    if not payload.get("refresh"):
        raise HTTPException(status_code=401, detail="Not a refresh token")
    
    # Check if revoked
    jti = payload.get("jti")
    if jti and await db.is_token_revoked(jti):
        raise HTTPException(status_code=401, detail="Refresh token has been revoked")
    
    username = payload.get("sub")
    user = await db.find_user(username)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    # Revoke old refresh token (rotation for security)
    if jti:
        expires_at = datetime.utcfromtimestamp(payload.get("exp", 0))
        await db.revoke_token(jti, username, expires_at)
    
    # Issue new tokens
    access_token = create_access_token({"sub": username})
    refresh_token = create_access_token({"sub": username}, refresh=True)
    
    print(f"🔄 Tokens refreshed for: {username}")
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        username=username
    )
