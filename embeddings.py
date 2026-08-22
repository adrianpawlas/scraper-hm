"""Image and text embeddings using SigLIP (768-dim).

Optimized with batched model inference and concurrent image downloads.
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
DOWNLOAD_WORKERS = 16
INFERENCE_BATCH_SIZE = 32  # images per forward pass

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
        logger.info("Loading SigLIP model %s (float16)...", MODEL_NAME)
        _image_processor = SiglipImageProcessor.from_pretrained(MODEL_NAME)
        _tokenizer = SiglipTokenizer.from_pretrained(MODEL_NAME)
        _model = SiglipModel.from_pretrained(MODEL_NAME, dtype=torch.float16)
        _model.to(_get_device())
        _model.eval()
    return _model, _image_processor, _tokenizer


def _download_image(image_url: str) -> Optional[Image.Image]:
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


def _embed_images_batch(images: list[Image.Image]) -> list[Optional[list[float]]]:
    """Embed a batch of images through SigLIP vision encoder at once."""
    model, image_processor, _ = _load_model()
    device = _get_device()
    results = [None] * len(images)
    try:
        inputs = image_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb_tensor = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            emb_tensor = outputs.last_hidden_state[:, 0, :]
        else:
            emb_tensor = outputs
        # Convert to list of individual embeddings
        for i in range(emb_tensor.shape[0]):
            results[i] = emb_tensor[i].cpu().float().numpy().flatten().tolist()
    except Exception as e:
        logger.warning("Batch embed failed (size %d): %s", len(images), e)
        # Fallback: try one-by-one
        for i, img in enumerate(images):
            try:
                results[i] = _embed_single(img)
            except Exception:
                pass
    return results


def _embed_single(image: Image.Image) -> Optional[list[float]]:
    """Embed a single image (fallback)."""
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


def get_text_embedding(text: str) -> Optional[list[float]]:
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
    results: dict[str, Optional[Image.Image]] = {}
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
    """Generate embeddings for products using batched SigLIP inference."""
    if existing_embeddings is None:
        existing_embeddings = {}

    stats = {
        "front_embeddings": 0,
        "back_embeddings": 0,
        "text_embeddings": 0,
        "skipped": 0,
    }

    _load_model()

    BATCH_DOWNLOAD = 400

    # Build list of (product_index, url, type) for images needing embedding
    all_needed: list[tuple[int, str, str]] = []
    for i, product in enumerate(products):
        product_url = product.get("product_url", "")
        existing = existing_embeddings.get(product_url, {})

        image_url = product.get("image_url", "")
        existing_image_url = existing.get("image_url", "")
        if image_url and (not existing or image_url != existing_image_url):
            all_needed.append((i, image_url, "front"))

        back_url = product.get("back_image_url")
        existing_back_url = existing.get("back_image_url")
        if back_url and (not existing or back_url != existing_back_url):
            all_needed.append((i, back_url, "back"))

    logger.info("Need to embed %d images total", len(all_needed))

    # Process in download batches
    downloaded_urls: set[str] = set()
    downloaded_images: dict[str, Image.Image] = {}

    total_batches = math.ceil(len(all_needed) / BATCH_DOWNLOAD) if all_needed else 0

    for batch_num, batch_start in enumerate(range(0, len(all_needed), BATCH_DOWNLOAD), 1):
        batch = all_needed[batch_start : batch_start + BATCH_DOWNLOAD]
        batch_urls = list(set(url for _, url, _ in batch if url not in downloaded_urls))

        if batch_urls:
            logger.info(
                "  Batch %d/%d: downloading %d images...",
                batch_num,
                total_batches,
                len(batch_urls),
            )
            batch_images = _download_batch(batch_urls)
            downloaded_images.update(batch_images)
            downloaded_urls.update(batch_urls)
            succeeded = sum(1 for v in batch_images.values() if v is not None)
            logger.info("  Downloaded %d/%d", succeeded, len(batch_urls))

        # Collect images for batched inference
        images_to_embed: list[tuple[int, str, str, Image.Image]] = []
        for idx, url, view_type in batch:
            product = products[idx]
            product_url = product.get("product_url", "")
            existing = existing_embeddings.get(product_url, {})

            if view_type == "front":
                image_url = product.get("image_url", "")
                existing_image_url = existing.get("image_url", "")
                if image_url and (not existing or image_url != existing_image_url):
                    image = downloaded_images.get(image_url)
                    if image:
                        images_to_embed.append((idx, image_url, "front", image))
                    else:
                        stats["skipped"] += 1
                else:
                    if existing and existing.get("image_embedding"):
                        product["image_embedding"] = existing["image_embedding"]

            elif view_type == "back":
                back_url = product.get("back_image_url")
                existing_back_url = existing.get("back_image_url")
                if back_url and (not existing or back_url != existing_back_url):
                    image = downloaded_images.get(back_url)
                    if image:
                        images_to_embed.append((idx, back_url, "back", image))
                elif not back_url:
                    product["back_image_embedding"] = None

        # Batched inference
        for infer_start in range(0, len(images_to_embed), INFERENCE_BATCH_SIZE):
            infer_batch = images_to_embed[infer_start : infer_start + INFERENCE_BATCH_SIZE]
            pil_images = [img for _, _, _, img in infer_batch]

            embeddings = _embed_images_batch(pil_images)

            for (idx, url, view_type, _), embedding in zip(infer_batch, embeddings):
                product = products[idx]
                if embedding:
                    if view_type == "front":
                        product["image_embedding"] = embedding
                        product["embedding_version"] = 2
                        stats["front_embeddings"] += 1
                    else:
                        product["back_image_embedding"] = embedding
                        stats["back_embeddings"] += 1
                else:
                    stats["skipped"] += 1

        # Free memory
        for url in batch_urls:
            downloaded_images.pop(url, None)

        logger.info(
            "  Batch %d done. Totals: front=%d, back=%d, text=%d",
            batch_num,
            stats["front_embeddings"],
            stats["back_embeddings"],
            stats["text_embeddings"],
        )

    # Text embeddings for all products
    logger.info("  Generating text embeddings for %d products...", len(products))
    for i, product in enumerate(products):
        product_url = product.get("product_url", "")
        existing = existing_embeddings.get(product_url, {})

        # Carry forward existing image embeddings
        image_url = product.get("image_url", "")
        existing_image_url = existing.get("image_url", "")
        if not (image_url and (not existing or image_url != existing_image_url)):
            if existing and existing.get("image_embedding"):
                product["image_embedding"] = existing["image_embedding"]

        back_url = product.get("back_image_url")
        existing_back_url = existing.get("back_image_url")
        if not back_url and not (not existing or back_url != existing_back_url):
            if existing and existing.get("back_image_embedding"):
                product["back_image_embedding"] = existing["back_image_embedding"]

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

        if (i + 1) % 500 == 0:
            logger.info(
                "  Text embedding progress: %d/%d",
                i + 1,
                len(products),
            )

    return products, stats
