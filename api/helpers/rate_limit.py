"""Rate limiting utilities."""

import time as _rate_time
from collections import defaultdict


class RateLimiter:
    def __init__(self, max_requests: int = 60, window_sec: int = 60):
        self.max_requests = max_requests
        self.window_sec = window_sec
        self._requests = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        now = _rate_time.time()
        window_start = now - self.window_sec
        self._requests[key] = [t for t in self._requests[key] if t > window_start]
        if len(self._requests[key]) >= self.max_requests:
            return False
        self._requests[key].append(now)
        return True


_rate_limiter = RateLimiter(max_requests=60, window_sec=60)
_chat_rate_limiter = RateLimiter(max_requests=10, window_sec=60)  # Stricter for chat
