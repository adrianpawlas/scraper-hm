"""Environment-based configuration for H&M scraper."""

import os

# Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# Scraper settings
SOURCE = "scraper-hm"
BRAND = "H&M"
SECOND_HAND = False
EMBEDDING_VERSION = 2
BATCH_SIZE = 50
REQUEST_DELAY = 1.0  # seconds between store requests
MAX_RETRIES = 3
PAGE_SIZE = 72  # max supported by H&M API
STALE_MISS_THRESHOLD = 2  # consecutive misses before deletion
