"""Polite rate limiting for web scraping with random jitter."""

import random
import time


class RateLimiter:
    """Token-bucket style rate limiter with random jitter.

    Ensures a random delay between consecutive requests so the scraper
    behaves like a human browser rather than a bot.

    Args:
        min_delay: Minimum seconds to wait between requests.
        max_delay: Maximum seconds to wait between requests.
    """

    def __init__(self, min_delay: float = 3.0, max_delay: float = 7.0) -> None:
        self.min_delay = min_delay
        self.max_delay = max_delay
        self._last_request: float = 0.0

    def wait(self) -> None:
        """Block until enough time has passed since the last request."""
        elapsed = time.monotonic() - self._last_request
        delay = random.uniform(self.min_delay, self.max_delay)
        remaining = delay - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request = time.monotonic()
