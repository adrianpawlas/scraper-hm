"""Embedding pipeline for image and text embeddings.

Uses:
- HuggingFace Inference API with google/siglip-base-patch16-384 for image embeddings (768-d)
- Gemini API for text/metadata embeddings (768-d)

All vectors are L2-normalized before storage.
"""

import asyncio
import base64
import io
import json
import logging
import time
from typing import Any

import httpx
from PIL import Image

from config import GEMINI_API_KEY, HF_API_TOKEN, HF_RATE_LIMIT_DELAY

logger = logging.getLogger(__name__)

HF_API_URL = "https://api-inference.huggingface.co/models/google/siglip-base-patch16-384"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"

_last_hf_call = 0.0


def _l2_normalize(vector: list[float]) -> list[float]:
    """L2-normalize a vector."""
    import math

    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def _prepare_image(image_bytes: bytes) -> bytes:
    """Resize and encode image for embedding.

    - Decode to RGB
    - Resize longest side to max 1280px (preserve aspect ratio)
    - Encode as JPEG quality ~85%
    - Return raw JPEG bytes
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    max_size = 1280
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


async def _rate_limit_hf():
    """Enforce rate limiting between HF API calls."""
    global _last_hf_call
    now = time.time()
    elapsed = now - _last_hf_call
    if elapsed < HF_RATE_LIMIT_DELAY:
        await asyncio.sleep(HF_RATE_LIMIT_DELAY - elapsed)
    _last_hf_call = time.time()


async def generate_image_embedding(
    client: httpx.AsyncClient,
    image_url: str,
) -> list[float] | None:
    """Generate a 768-dim L2-normalized image embedding using SigLIP.

    Downloads image, processes it, and sends to HuggingFace Inference API.
    Returns None on failure.
    """
    if not image_url:
        return None

    try:
        # Download image
        img_resp = await client.get(image_url, timeout=30)
        img_resp.raise_for_status()
        image_bytes = img_resp.content

        if len(image_bytes) < 100:
            logger.warning(f"Image too small ({len(image_bytes)} bytes): {image_url}")
            return None

        # Process image
        processed = _prepare_image(image_bytes)
        b64_data = base64.b64encode(processed).decode("utf-8")

        # Rate limit
        await _rate_limit_hf()

        # Call HuggingFace API
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        payload = {
            "inputs": b64_data,
            "options": {"wait_for_model": True},
        }

        resp = await client.post(
            HF_API_URL,
            json=payload,
            headers=headers,
            timeout=120,
        )
        resp.raise_for_status()
        result = resp.json()

        # Handle different response formats
        embedding = None
        if isinstance(result, list):
            if len(result) > 0:
                if isinstance(result[0], list):
                    # Batch format: [[...768 floats...]] -> take first
                    embedding = result[0]
                elif isinstance(result[0], (int, float)):
                    # Already a flat vector
                    embedding = result

        if embedding is None or len(embedding) != 768:
            logger.warning(f"Invalid embedding shape for {image_url}: {type(result)} len={len(result) if isinstance(result, list) else 'N/A'}")
            return None

        # Verify all values are finite
        import math
        if not all(math.isfinite(v) for v in embedding):
            logger.warning(f"Non-finite values in embedding for {image_url}")
            return None

        # L2-normalize
        embedding = _l2_normalize(embedding)

        return embedding

    except httpx.HTTPStatusError as e:
        logger.warning(f"HTTP error generating image embedding for {image_url}: {e.response.status_code}")
        return None
    except Exception as e:
        logger.warning(f"Error generating image embedding for {image_url}: {e}")
        return None


async def generate_text_embedding(
    client: httpx.AsyncClient,
    text: str,
) -> list[float] | None:
    """Generate a 768-dim L2-normalized text embedding using Gemini.

    Returns None on failure.
    """
    if not text or not GEMINI_API_KEY:
        return None

    try:
        # Rate limit
        await _rate_limit_hf()

        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "models/gemini-embedding-001",
            "content": {"parts": [{"text": text}]},
        }

        url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
        resp = await client.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        result = resp.json()

        embedding = result.get("embedding", {}).get("values", [])
        if not embedding or len(embedding) != 768:
            logger.warning(f"Invalid text embedding shape: {len(embedding)}")
            return None

        import math
        if not all(math.isfinite(v) for v in embedding):
            logger.warning("Non-finite values in text embedding")
            return None

        embedding = _l2_normalize(embedding)
        return embedding

    except Exception as e:
        logger.warning(f"Error generating text embedding: {e}")
        return None


def _build_info_text(product: dict[str, Any]) -> str:
    """Build text representation of product for info embedding.

    Combines title, description, category, gender, price, and metadata
    into a single text block for embedding.
    """
    parts = []

    parts.append(f"Brand: {product.get('brand', 'H&M')}")
    parts.append(f"Product: {product.get('title', '')}")

    if product.get("category"):
        parts.append(f"Category: {product['category']}")
    if product.get("gender"):
        parts.append(f"Gender: {product['gender']}")
    if product.get("price"):
        parts.append(f"Price: {product['price']}")
    if product.get("sale"):
        parts.append(f"Sale price: {product['sale']}")

    if product.get("description"):
        parts.append(f"Description: {product['description']}")

    # Parse metadata JSON
    try:
        metadata = json.loads(product.get("metadata") or "{}")
        if metadata.get("color_name"):
            parts.append(f"Color: {metadata['color_name']}")
        if metadata.get("source_category"):
            parts.append(f"Style: {metadata['source_category']}")
    except (json.JSONDecodeError, TypeError):
        pass

    return " | ".join(parts)


async def embed_products(
    products: list[dict[str, Any]],
    existing_embeddings: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Generate embeddings for products.

    Args:
        products: List of product dicts to embed.
        existing_embeddings: Map of product_url -> existing row data
            for smart re-embedding decisions.

    Returns:
        Tuple of (updated products, stats dict)
    """
    if existing_embeddings is None:
        existing_embeddings = {}

    stats = {
        "front_embeddings": 0,
        "back_embeddings": 0,
        "text_embeddings": 0,
        "skipped": 0,
    }

    async with httpx.AsyncClient() as client:
        for i, product in enumerate(products):
            product_url = product.get("product_url", "")
            existing = existing_embeddings.get(product_url, {})

            image_url = product.get("image_url", "")
            existing_image_url = existing.get("image_url", "")

            # Front embedding: generate if new or image_url changed
            if image_url and (not existing or image_url != existing_image_url):
                embedding = await generate_image_embedding(client, image_url)
                if embedding:
                    product["image_embedding"] = embedding
                    product["embedding_version"] = 2
                    stats["front_embeddings"] += 1
                    logger.debug(f"  [{i+1}/{len(products)}] Front embed: {product.get('title', '')[:40]}")
                else:
                    stats["skipped"] += 1
            else:
                # Keep existing embedding if no change
                if existing and existing.get("image_embedding"):
                    product["image_embedding"] = existing["image_embedding"]

            # Back embedding: generate if back_image_url changed
            back_url = product.get("back_image_url")
            existing_back_url = existing.get("back_image_url")
            if back_url and (not existing or back_url != existing_back_url):
                back_embedding = await generate_image_embedding(client, back_url)
                if back_embedding:
                    product["back_image_embedding"] = back_embedding
                    stats["back_embeddings"] += 1
                    logger.debug(f"  [{i+1}/{len(products)}] Back embed: {product.get('title', '')[:40]}")
            elif not back_url:
                product["back_image_embedding"] = None

            # Info embedding: generate if new or text fields changed
            info_text = _build_info_text(product)
            existing_info_text = ""
            if existing:
                existing_info_text = _build_info_text(existing)

            if not existing or info_text != existing_info_text:
                text_embedding = await generate_text_embedding(client, info_text)
                if text_embedding:
                    product["info_embedding"] = text_embedding
                    stats["text_embeddings"] += 1
                    logger.debug(f"  [{i+1}/{len(products)}] Text embed: {product.get('title', '')[:40]}")
            else:
                if existing and existing.get("info_embedding"):
                    product["info_embedding"] = existing["info_embedding"]

            # Progress logging
            if (i + 1) % 10 == 0:
                logger.info(
                    f"  Embedding progress: {i+1}/{len(products)} "
                    f"(front={stats['front_embeddings']}, text={stats['text_embeddings']})"
                )

    return products, stats
