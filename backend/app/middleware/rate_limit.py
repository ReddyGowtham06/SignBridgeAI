"""
SignBridge AI - Rate Limiting Middleware
In-memory sliding window rate limiter for security-sensitive endpoints (login, register).
"""
import time
from collections import defaultdict
from typing import Dict, List
from fastapi import HTTPException, Request, status

from app.core.config import settings


class InMemoryRateLimiter:
    """
    Thread-safe in-memory rate limiter tracking timestamps per client IP.
    """
    def __init__(self, requests_per_minute: int = 20):
        self.rate = requests_per_minute
        self.window = 60.0  # seconds
        self._history: Dict[str, List[float]] = defaultdict(list)

    def check_rate_limit(self, client_key: str):
        now = time.time()
        cutoff = now - self.window
        
        # Prune old timestamps
        timestamps = [t for t in self._history[client_key] if t > cutoff]
        
        if len(timestamps) >= self.rate:
            retry_after = int(self.window - (now - timestamps[0]))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Maximum {self.rate} requests per minute. Try again in {retry_after}s.",
                headers={"Retry-After": str(max(1, retry_after))}
            )
        
        timestamps.append(now)
        self._history[client_key] = timestamps


auth_rate_limiter = InMemoryRateLimiter(
    requests_per_minute=settings.RATE_LIMIT_AUTH_PER_MINUTE
)


def rate_limit_auth(request: Request):
    """Dependency to rate limit sensitive authentication requests by client IP."""
    client_ip = request.client.host if request.client else "unknown"
    auth_rate_limiter.check_rate_limit(client_ip)
