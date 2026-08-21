"""Supabase client for smart upsert, diffing, and stale cleanup.

Handles:
- Batch upserts (50 products per request)
- Smart field diffing to minimize redundant writes
- Embedding regeneration only when source data changes
- Stale product tracking and deletion
"""

import asyncio
import json
import logging
import time
from typing import Any

import httpx

from config import BATCH_SIZE, EMBEDDING_VERSION, SOURCE, STALE_MISS_THRESHOLD

logger = logging.getLogger(__name__)

UPSERT_COLUMNS = [
    "id", "source", "product_url", "affiliate_url", "image_url",
    "compressed_image_url", "back_image_url", "brand", "title",
    "description", "category", "gender", "price", "sale", "metadata",
    "size", "second_hand", "country", "tags", "additional_images",
    "other", "image_embedding", "back_image_embedding",
    "embedding_version", "info_embedding",
]

# Fields to compare for change detection (excluding embeddings and metadata)
DIFF_FIELDS = [
    "title", "description", "price", "sale", "category", "gender",
    "image_url", "back_image_url", "additional_images", "affiliate_url",
    "size", "other", "compressed_image_url",
]


def _serialize_value(val: Any) -> Any:
    """Serialize a value for Supabase JSON compatibility."""
    if val is None:
        return None
    if isinstance(val, (list, dict)):
        return json.dumps(val)
    return val


def _to_pgvector(val: list[float] | None) -> str | None:
    """Convert a Python list to pgvector string format."""
    if val is None:
        return None
    return "[" + ",".join(f"{v:.8f}" for v in val) + "]"


def _serialize_product_for_upsert(product: dict[str, Any]) -> dict[str, Any]:
    """Convert a product dict to a Supabase-ready row."""
    row = {}
    for col in UPSERT_COLUMNS:
        val = product.get(col)
        if col in ("image_embedding", "back_image_embedding", "info_embedding"):
            row[col] = _to_pgvector(val)
        elif col == "metadata":
            # Ensure metadata is a JSON string
            if isinstance(val, dict):
                row[col] = json.dumps(val)
            elif isinstance(val, str):
                row[col] = val
            else:
                row[col] = val
        elif col == "tags":
            row[col] = val  # Supabase handles array columns
        else:
            row[col] = val
    return row


def _has_changed(existing: dict[str, Any], new: dict[str, Any]) -> bool:
    """Check if any diff-relevant field has changed."""
    for field in DIFF_FIELDS:
        old_val = existing.get(field)
        new_val = new.get(field)
        # Normalize None/empty
        if old_val is None:
            old_val = ""
        if new_val is None:
            new_val = ""
        if str(old_val) != str(new_val):
            return True
    return False


def _should_regen_image_embedding(existing: dict[str, Any], new: dict[str, Any]) -> bool:
    """Determine if image embedding should be regenerated."""
    old_url = existing.get("image_url", "")
    new_url = new.get("image_url", "")
    return bool(new_url) and new_url != old_url


def _should_regen_back_embedding(existing: dict[str, Any], new: dict[str, Any]) -> bool:
    """Determine if back image embedding should be regenerated."""
    old_back = existing.get("back_image_url")
    new_back = new.get("back_image_url")
    return new_back != old_back


def _should_regen_info_embedding(existing: dict[str, Any], new: dict[str, Any]) -> bool:
    """Determine if info embedding should be regenerated."""
    return _has_changed(existing, new)


class SupabaseClient:
    """Smart upsert client for the Finds products table."""

    def __init__(self, url: str, key: str):
        self.url = url.rstrip("/")
        self.key = key
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates",
        }
        self.existing_products: dict[str, dict[str, Any]] = {}

    async def load_existing_products(self, client: httpx.AsyncClient) -> int:
        """Load all existing products for this source into memory.

        Returns the count of loaded products.
        """
        logger.info("Loading existing products from Supabase...")
        offset = 0
        limit = 1000
        all_products = []

        while True:
            try:
                resp = await client.get(
                    f"{self.url}/rest/v1/products",
                    headers=self.headers,
                    params={
                        "select": "*,image_embedding,back_image_embedding,info_embedding",
                        "source": f"eq.{SOURCE}",
                        "offset": str(offset),
                        "limit": str(limit),
                        "order": "created_at.desc",
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                rows = resp.json()

                if not rows:
                    break

                for row in rows:
                    url = row.get("product_url", "")
                    if url:
                        self.existing_products[url] = row

                all_products.extend(rows)
                offset += limit

                if len(rows) < limit:
                    break

            except Exception as e:
                logger.error(f"Error loading existing products: {e}")
                break

        logger.info(f"Loaded {len(all_products)} existing products")
        return len(all_products)

    async def upsert_batch(
        self,
        client: httpx.AsyncClient,
        products: list[dict[str, Any]],
    ) -> dict[str, int]:
        """Upsert a batch of products with smart diffing.

        Returns stats about what was done.
        """
        stats = {
            "new": 0,
            "updated": 0,
            "unchanged": 0,
            "errors": 0,
        }

        rows_to_upsert = []

        for product in products:
            product_url = product.get("product_url", "")
            existing = self.existing_products.get(product_url)

            if not existing:
                # New product - full insert with embeddings
                rows_to_upsert.append(_serialize_product_for_upsert(product))
                stats["new"] += 1
            elif _has_changed(existing, product):
                # Changed product - update fields
                row = _serialize_product_for_upsert(product)

                # Only regenerate embeddings if source changed
                if _should_regen_image_embedding(existing, product):
                    pass  # embedding already set by embed_products
                else:
                    row["image_embedding"] = _to_pgvector(existing.get("image_embedding"))

                if _should_regen_back_embedding(existing, product):
                    pass  # embedding already set by embed_products
                else:
                    row["back_image_embedding"] = _to_pgvector(existing.get("back_image_embedding"))

                if _should_regen_info_embedding(existing, product):
                    pass  # embedding already set by embed_products
                else:
                    row["info_embedding"] = _to_pgvector(existing.get("info_embedding"))

                # Set embedding_version if any embedding was written
                if row.get("image_embedding") and row["image_embedding"] != _to_pgvector(existing.get("image_embedding")):
                    row["embedding_version"] = EMBEDDING_VERSION

                rows_to_upsert.append(row)
                stats["updated"] += 1
            else:
                stats["unchanged"] += 1

        # Batch upsert
        if rows_to_upsert:
            for i in range(0, len(rows_to_upsert), BATCH_SIZE):
                batch = rows_to_upsert[i : i + BATCH_SIZE]
                success = await _upsert_with_retry(client, self.url, self.headers, batch)
                if not success:
                    stats["errors"] += 1

        return stats

    async def cleanup_stale_products(
        self,
        client: httpx.AsyncClient,
        seen_urls: set[str],
    ) -> int:
        """Track and delete stale products.

        Products not seen in 2 consecutive runs are deleted.
        Uses metadata JSON field 'scrape_miss_count' to track misses.
        """
        deleted = 0

        # Get all products for this source
        try:
            resp = await client.get(
                f"{self.url}/rest/v1/products",
                headers=self.headers,
                params={
                    "select": "id,product_url,metadata",
                    "source": f"eq.{SOURCE}",
                    "limit": "5000",
                },
                    timeout=60,
            )
            resp.raise_for_status()
            all_products = resp.json()
        except Exception as e:
            logger.error(f"Error fetching products for cleanup: {e}")
            return 0

        urls_to_delete = []

        for product in all_products:
            product_url = product.get("product_url", "")
            if product_url not in seen_urls:
                # Product not seen this run
                try:
                    metadata = json.loads(product.get("metadata") or "{}")
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

                miss_count = metadata.get("scrape_miss_count", 0)
                miss_count += 1
                metadata["scrape_miss_count"] = miss_count

                if miss_count >= STALE_MISS_THRESHOLD:
                    urls_to_delete.append(product_url)
                    logger.info(f"  Stale product (missed {miss_count}x): {product_url}")
                else:
                    # Update miss count
                    try:
                        await client.patch(
                            f"{self.url}/rest/v1/products",
                            headers=self.headers,
                            params={
                                "source": f"eq.{SOURCE}",
                                "product_url": f"eq.{product_url}",
                            },
                            json={"metadata": json.dumps(metadata)},
                            timeout=30,
                        )
                    except Exception as e:
                        logger.warning(f"Error updating miss count: {e}")
            else:
                # Product seen - reset miss count
                try:
                    metadata = json.loads(product.get("metadata") or "{}")
                    if metadata.get("scrape_miss_count", 0) > 0:
                        metadata["scrape_miss_count"] = 0
                        await client.patch(
                            f"{self.url}/rest/v1/products",
                            headers=self.headers,
                            params={
                                "source": f"eq.{SOURCE}",
                                "product_url": f"eq.{product_url}",
                            },
                            json={"metadata": json.dumps(metadata)},
                            timeout=30,
                        )
                except Exception:
                    pass

        # Delete stale products in batches
        if urls_to_delete:
            for i in range(0, len(urls_to_delete), BATCH_SIZE):
                batch = urls_to_delete[i : i + BATCH_SIZE]
                try:
                    await client.delete(
                        f"{self.url}/rest/v1/products",
                        headers=self.headers,
                        params={
                            "source": f"eq.{SOURCE}",
                            "product_url": f"in.({','.join(batch)})",
                        },
                        timeout=60,
                    )
                    deleted += len(batch)
                    logger.info(f"  Deleted {len(batch)} stale products")
                except Exception as e:
                    logger.error(f"Error deleting stale products: {e}")

        return deleted


async def _upsert_with_retry(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    rows: list[dict[str, Any]],
    max_retries: int = 3,
) -> bool:
    """Upsert a batch with exponential backoff retry."""
    from config import MAX_RETRIES

    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.post(
                f"{url}/rest/v1/products",
                headers=headers,
                json=rows,
                timeout=120,
            )
            if resp.status_code in (200, 201):
                return True

            # 409 conflict = upsert merge issue, usually non-fatal
            if resp.status_code == 409:
                logger.warning(f"Upsert conflict (attempt {attempt+1}), retrying...")
                await asyncio.sleep(2 ** attempt)
                continue

            logger.error(f"Upsert failed with status {resp.status_code}: {resp.text[:200]}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)

        except Exception as e:
            logger.error(f"Upsert error (attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)

    logger.error(f"Upsert failed after {MAX_RETRIES} attempts for {len(rows)} rows")
    return False
