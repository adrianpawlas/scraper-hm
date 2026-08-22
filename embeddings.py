"""Image and text embeddings using SigLIP (768-dim).

Optimized with concurrent image downloads via ThreadPoolExecutor.
"""
import io
import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import requests
import torch
from PIL import Image

try:
    from transformers import SiglipImageProcessorPil as SiglipImageProcessor
except ImportError:
    from transformers import SiglipImageProcessor
from transformers import SiglipModel, SiglipTokenizer

logger = logging.getLogger(__name__)

MODEL_NAME = "google/siglip-base-patch16-384"
EMBEDDING_DIM = 768
DOWNLOAD_WORKERS = 8  # concurrent image downloads

_model = None
_image_processor = None
_tokenizer = None
_device = None


def _get_device():
    global _device
    if _device is None:
        _device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    return _device


def _load_model():
    global _model, _image_processor, _tokenizer
    if _model is None:
        logger.info("Loading SigLIP model %s...", MODEL_NAME)
        _image_processor = SiglipImageProcessor.from_pretrained(MODEL_NAME)
        _tokenizer = SiglipTokenizer.from_pretrained(MODEL_NAME)
        _model = SiglipModel.from_pretrained(MODEL_NAME)
        _model.to(_get_device())
        _model.eval()
    return _model, _image_processor, _tokenizer


def _download_image(image_url: str) -> Optional[Image.Image]:
    """Download and decode an image from URL."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    }
    try:
        resp = requests.get(image_url, timeout=15, headers=headers)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        logger.warning("Failed to download image %s: %s", image_url, e)
        return None


def _embed_image(image: Image.Image) -> Optional[list[float]]:
    """Generate 768-dim embedding for a PIL Image using SigLIP."""
    model, image_processor, _ = _load_model()
    device = _get_device()
    try:
        inputs = image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb_tensor = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            emb_tensor = outputs.last_hidden_state[:, 0, :]
        else:
            emb_tensor = outputs
        return emb_tensor.cpu().float().numpy().flatten().tolist()
    except Exception as e:
        logger.warning("Failed to embed image: %s", e)
        return None


def get_image_embedding(image_url: str) -> Optional[list[float]]:
    """Generate 768-dim embedding for an image URL using SigLIP."""
    image = _download_image(image_url)
    if image is None:
        return None
    return _embed_image(image)


def get_text_embedding(text: str) -> Optional[list[float]]:
    """Generate 768-dim embedding for text using SigLIP text encoder."""
    if not text or not text.strip():
        return None
    model, _, tokenizer = _load_model()
    device = _get_device()
    try:
        inputs = tokenizer(
            text=[text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb_tensor = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            emb_tensor = outputs.last_hidden_state[:, 0, :]
        else:
            emb_tensor = outputs
        return emb_tensor.cpu().float().numpy().flatten().tolist()
    except Exception as e:
        logger.warning("Failed to embed text: %s", e)
        return None


def _build_info_text(product: dict[str, Any]) -> str:
    """Build text representation of product for info embedding."""
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
    try:
        metadata = json.loads(product.get("metadata") or "{}")
        if metadata.get("color_name"):
            parts.append(f"Color: {metadata['color_name']}")
        if metadata.get("source_category"):
            parts.append(f"Style: {metadata['source_category']}")
    except (json.JSONDecodeError, TypeError):
        pass
    return " | ".join(parts)


def _download_batch(urls: list[str]) -> dict[str, Optional[Image.Image]]:
    """Download multiple images concurrently."""
    results = {}
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        future_to_url = {executor.submit(_download_image, url): url for url in urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                results[url] = future.result()
            except Exception as e:
                logger.warning("Download failed for %s: %s", url, e)
                results[url] = None
    return results


def embed_products(
    products: list[dict[str, Any]],
    existing_embeddings: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Generate embeddings for products using local SigLIP model.

    Uses concurrent image downloads for speed.
    """
    if existing_embeddings is None:
        existing_embeddings = {}

    stats = {
        "front_embeddings": 0,
        "back_embeddings": 0,
        "text_embeddings": 0,
        "skipped": 0,
    }

    _load_model()

    # Phase 1: Determine which images need downloading
    to_download = []  # (index, url, type) where type is 'front' or 'back'
    for i, product in enumerate(products):
        product_url = product.get("product_url", "")
        existing = existing_embeddings.get(product_url, {})

        image_url = product.get("image_url", "")
        existing_image_url = existing.get("image_url", "")
        if image_url and (not existing or image_url != existing_image_url):
            to_download.append((i, image_url, "front"))

        back_url = product.get("back_image_url")
        existing_back_url = existing.get("back_image_url")
        if back_url and (not existing or back_url != existing_back_url):
            to_download.append((i, back_url, "back"))

    # Phase 2: Download all needed images concurrently
    if to_download:
        download_urls = list(set(url for _, url, _ in to_download))
        logger.info("Downloading %d images with %d workers...", len(download_urls), DOWNLOAD_WORKERS)
        downloaded = _download_batch(download_urls)
        logger.info("Downloaded %d/%d images", sum(1 for v in downloaded.values() if v is not None), len(download_urls))
    else:
        downloaded = {}

    # Phase 3: Process embeddings sequentially (model inference is CPU/GPU bound)
    for i, product in enumerate(products):
        product_url = product.get("product_url", "")
        existing = existing_embeddings.get(product_url, {})

        image_url = product.get("image_url", "")
        existing_image_url = existing.get("image_url", "")

        # Front embedding
        if image_url and (not existing or image_url != existing_image_url):
            image = downloaded.get(image_url)
            if image:
                embedding = _embed_image(image)
                if embedding:
                    product["image_embedding"] = embedding
                    product["embedding_version"] = 2
                    stats["front_embeddings"] += 1
                else:
                    stats["skipped"] += 1
        else:
            if existing and existing.get("image_embedding"):
                product["image_embedding"] = existing["image_embedding"]

        # Back embedding
        back_url = product.get("back_image_url")
        existing_back_url = existing.get("back_image_url")
        if back_url and (not existing or back_url != existing_back_url):
            image = downloaded.get(back_url)
            if image:
                back_embedding = _embed_image(image)
                if back_embedding:
                    product["back_image_embedding"] = back_embedding
                    stats["back_embeddings"] += 1
        elif not back_url:
            product["back_image_embedding"] = None

        # Info embedding
        info_text = _build_info_text(product)
        existing_info_text = ""
        if existing:
            existing_info_text = _build_info_text(existing)

        if not existing or info_text != existing_info_text:
            text_emb = get_text_embedding(info_text)
            if text_emb:
                product["info_embedding"] = text_emb
                stats["text_embeddings"] += 1
        else:
            if existing and existing.get("info_embedding"):
                product["info_embedding"] = existing["info_embedding"]

        if (i + 1) % 50 == 0:
            logger.info(
                "  Embedding progress: %d/%d (front=%d, text=%d)",
                i + 1,
                len(products),
                stats["front_embeddings"],
                stats["text_embeddings"],
            )

    return products, stats
