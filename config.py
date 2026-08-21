"""Environment-based configuration for H&M scraper."""

import os

# Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# HuggingFace
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
HF_API_URL = "https://api-inference.huggingface.co/models/google/siglip-base-patch16-384"

# Gemini (for text embeddings)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Scraper settings
SOURCE = "scraper-hm"
BRAND = "H&M"
SECOND_HAND = False
EMBEDDING_VERSION = 2
BATCH_SIZE = 50
HF_RATE_LIMIT_DELAY = 0.5  # seconds between HF API calls
REQUEST_DELAY = 1.0  # seconds between store requests
MAX_RETRIES = 3
PAGE_SIZE = 72  # max supported by H&M API
STALE_MISS_THRESHOLD = 2  # consecutive misses before deletion
