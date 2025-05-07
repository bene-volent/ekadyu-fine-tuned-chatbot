import time
import hmac
import hashlib
import jwt
from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Optional, Callable, Any
from pydantic_settings import BaseSettings
import logging
from collections import defaultdict

# Initialize logging
logger = logging.getLogger(__name__)

class AuthSettings(BaseSettings):
    """
    Authentication settings loaded from .env
    """
    jwt_secret_key: str
    jwt_algorithm: str = 'HS256'
    request_timestamp_max_age:int = 30
    request_headers_secret: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    

# In-memory rate limiting storage - in production use Redis or similar
ip_request_timestamps = defaultdict(float)
user_request_timestamps = defaultdict(float)

# Initialize HTTP Bearer authentication
security = HTTPBearer()

def verify_timestamp_signature(timestamp: int, signature: str, secret_key: str) -> bool:
    """
    Verify the HMAC signature of a timestamp.
    
    Args:
        timestamp: Unix timestamp in seconds
        signature: HMAC signature to verify
        secret_key: Secret key for HMAC
        
    Returns:
        bool: True if signature is valid
    """
    message = str(timestamp).encode()
    expected_signature = hmac.new(
        secret_key.encode(),
        message,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(signature, expected_signature)

async def verify_jwt_and_timestamp(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    settings: AuthSettings = Depends(lambda: AuthSettings())
) -> Dict[str, Any]:
    """
    Middleware that verifies:
    1. Valid JWT token in Authorization header
    2. Valid and recent timestamp with signature in X-Request-Timestamp and X-Request-Signature headers
    3. Rate limiting based on user ID and IP
    
    Returns the JWT payload if verification passes
    """
    logger.info(f"Verifying request from {request.client.host}")
    
    # Load settings
    try:
        settings = AuthSettings()
    except Exception as e:
        logger.error(f"Failed to load auth settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error"
        )   
    
    # Get client IP for rate limiting
    client_ip = request.client.host
    
    if not credentials:
        logger.warning("Missing Authorization header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # 1. Verify JWT token
    try:
        print(credentials)
        token = credentials.credentials
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        

        logger.info(f"JWT token verified successfully")
        
        # Extract user_id for rate limiting
        user_id = payload.get('user_id')
        if not user_id:
            logger.warning("JWT token missing user_id claim")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing user_id claim"
            )
            
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # 2. Verify timestamp and signature
    timestamp_str = request.headers.get("X-Request-Timestamp")
    signature = request.headers.get("X-Request-Signature")
    
    if not timestamp_str or not signature:
        logger.warning("Missing timestamp headers")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing X-Request-Timestamp or X-Request-Signature headers"
        )
    
    try:
        timestamp = int(timestamp_str)
    except ValueError:
        logger.warning("Invalid timestamp format")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid timestamp format"
        )
    
    # Check if timestamp is too old
    current_time = int(time.time())
    if current_time - timestamp > settings.request_timestamp_max_age:
        logger.warning(f"Timestamp expired: {timestamp} (current: {current_time})")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Request timestamp expired (max age: {settings.request_timestamp_max_age}s)"
        )
    
    # Verify the signature
    if not verify_timestamp_signature(timestamp, signature, settings.request_headers_secret):
        logger.warning("Invalid timestamp signature")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid timestamp signature"
        )
    
    # 3. Rate limiting based on user ID and IP
    # Check if this user or IP has made a request in the last 30 seconds
    if current_time - user_request_timestamps.get(user_id, 0) < settings.request_timestamp_max_age:
        remaining = settings.request_timestamp_max_age - (current_time - user_request_timestamps[user_id])
        logger.warning(f"Rate limit exceeded for user {user_id}: {remaining}s remaining")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for user. Try again in {remaining} seconds"
        )
    
    if current_time - ip_request_timestamps.get(client_ip, 0) < settings.request_timestamp_max_age:
        remaining = settings.request_timestamp_max_age - (current_time - ip_request_timestamps[client_ip])
        logger.warning(f"Rate limit exceeded for IP {client_ip}: {remaining}s remaining")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for IP. Try again in {remaining} seconds"
        )
    
    # Update rate limit timestamps
    user_request_timestamps[user_id] = current_time
    ip_request_timestamps[client_ip] = current_time
    
    logger.info(f"Request authenticated for user {user_id}")
    # Return the JWT payload for use in the route handlers
    return payload