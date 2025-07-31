"""
Rate limiter utility for OpenAI API calls.
"""

import time
import random
import asyncio
import logging

logger = logging.getLogger(__name__)


class OpenAIRateLimiter:
    """Rate limiter for OpenAI API calls to avoid hitting rate limits."""
    
    def __init__(self, requests_per_minute: int = 200, add_jitter: bool = True):
        """Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute (default 200, conservative limit)
            add_jitter: Whether to add random jitter to delays
        """
        self.requests_per_minute = requests_per_minute
        self.min_delay = 60.0 / requests_per_minute  # Minimum delay between requests
        self.add_jitter = add_jitter
        self.last_request_time = 0.0
        self.consecutive_429s = 0  # Track consecutive 429 errors
        
        logger.info(f"OpenAI rate limiter initialized: {requests_per_minute} RPM, min delay: {self.min_delay:.3f}s")
    
    async def wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_delay:
            wait_time = self.min_delay - time_since_last
            
            # Add jitter to avoid thundering herd
            if self.add_jitter:
                jitter = random.uniform(0, min(0.1, wait_time * 0.1))  # Up to 10% jitter
                wait_time += jitter
            
            logger.debug(f"Rate limiting: waiting {wait_time:.3f}s before API call")
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def wait_if_needed_sync(self):
        """Synchronous version of wait_if_needed."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_delay:
            wait_time = self.min_delay - time_since_last
            
            # Add jitter to avoid thundering herd
            if self.add_jitter:
                jitter = random.uniform(0, min(0.1, wait_time * 0.1))  # Up to 10% jitter
                wait_time += jitter
            
            logger.debug(f"Rate limiting: waiting {wait_time:.3f}s before API call")
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def handle_429_error(self):
        """Handle 429 rate limit error with exponential backoff."""
        self.consecutive_429s += 1
        backoff_delay = min(60.0, 2 ** self.consecutive_429s)  # Exponential backoff, max 60s
        logger.warning(f"Rate limit hit (429 error #{self.consecutive_429s}), backing off for {backoff_delay:.1f}s")
        time.sleep(backoff_delay)
        self.last_request_time = time.time()
    
    def reset_429_counter(self):
        """Reset 429 error counter after successful request."""
        if self.consecutive_429s > 0:
            logger.info(f"Rate limit recovered after {self.consecutive_429s} consecutive 429 errors")
            self.consecutive_429s = 0