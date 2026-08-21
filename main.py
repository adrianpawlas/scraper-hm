"""H&M Product Scraper - Main Entry Point.

Scrapes all H&M products via their listing API, generates embeddings,
and upserts into Supabase with smart diffing and batching.

Usage:
    python main.py                    # Full scrape
    python main.py --dry-run          # Scrape without DB writes
    python main.py --category ladies  # Scrape specific category only
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from config import BATCH_SIZE, HF_RATE_LIMIT_DELAY, REQUEST_DELAY, SOURCE
from embeddings import embed_products
from scraper import CATEGORIES, scrape_all_categories
from supabase_client import SupabaseClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/scraper.log", mode="a"),
    ],
)
logger = logging.getLogger("hm_scraper")


def _print_summary(stats: dict[str, Any], duration: float):
    """Print run summary."""
    print("\n" + "=" * 60)
    print("SCRAPER RUN SUMMARY")
    print("=" * 60)
    print(f"Duration: {duration:.1f}s")
    print(f"New products added: {stats.get('new', 0)}")
    print(f"Products updated: {stats.get('updated', 0)}")
    print(f"Products unchanged (skipped): {stats.get('unchanged', 0)}")
    print(f"Front embeddings generated: {stats.get('front_embeddings', 0)}")
    print(f"Back embeddings generated: {stats.get('back_embeddings', 0)}")
    print(f"Text embeddings generated: {stats.get('text_embeddings', 0)}")
    print(f"Stale products deleted: {stats.get('stale_deleted', 0)}")
    print(f"Errors / failures: {stats.get('errors', 0)}")
    print("=" * 60)


async def run_scraper(dry_run: bool = False, category_filter: str | None = None):
    """Main scraper execution."""
    start_time = time.time()

    # Stats
    stats = {
        "new": 0,
        "updated": 0,
        "unchanged": 0,
        "front_embeddings": 0,
        "back_embeddings": 0,
        "text_embeddings": 0,
        "stale_deleted": 0,
        "errors": 0,
    }

    logger.info("Starting H&M scraper run")
    logger.info(f"Source: {SOURCE}")
    logger.info(f"Dry run: {dry_run}")

    # Step 1: Scrape all categories
    logger.info("=" * 40)
    logger.info("STEP 1: Scraping H&M product listings")
    logger.info("=" * 40)

    if category_filter:
        from scraper import CATEGORIES as ALL_CATS
        filtered = [c for c in ALL_CATS if category_filter.lower() in c["category_name"].lower()]
        if not filtered:
            logger.error(f"No categories match filter: {category_filter}")
            return
        logger.info(f"Filtering to categories: {[c['category_name'] for c in filtered]}")
        all_products = await scrape_all_categories(filtered)
    else:
        all_products = await scrape_all_categories()

    if not all_products:
        logger.error("No products scraped!")
        return

    logger.info(f"Total unique products scraped: {len(all_products)}")

    # Step 2: Load existing products from Supabase
    existing_map: dict[str, dict[str, Any]] = {}
    seen_urls: set[str] = set()

    if not dry_run:
        logger.info("=" * 40)
        logger.info("STEP 2: Loading existing products from Supabase")
        logger.info("=" * 40)

        from config import SUPABASE_KEY, SUPABASE_URL
        if not SUPABASE_URL or not SUPABASE_KEY:
            logger.error("SUPABASE_URL and SUPABASE_KEY must be set")
            return

        sb = SupabaseClient(SUPABASE_URL, SUPABASE_KEY)
        async with httpx.AsyncClient() as client:
            await sb.load_existing_products(client)
            existing_map = sb.existing_products

    # Track all seen URLs
    for product in all_products:
        seen_urls.add(product["product_url"])

    logger.info(f"Existing products in DB: {len(existing_map)}")
    logger.info(f"Products to process: {len(all_products)}")

    # Step 3: Generate embeddings
    logger.info("=" * 40)
    logger.info("STEP 3: Generating embeddings")
    logger.info("=" * 40)

    all_products, embed_stats = await embed_products(all_products, existing_map)
    stats["front_embeddings"] = embed_stats["front_embeddings"]
    stats["back_embeddings"] = embed_stats["back_embeddings"]
    stats["text_embeddings"] = embed_stats["text_embeddings"]

    # Step 4: Upsert to Supabase
    if not dry_run:
        logger.info("=" * 40)
        logger.info("STEP 4: Upserting to Supabase")
        logger.info("=" * 40)

        async with httpx.AsyncClient() as client:
            # Process in batches
            for i in range(0, len(all_products), BATCH_SIZE):
                batch = all_products[i : i + BATCH_SIZE]
                batch_stats = await sb.upsert_batch(client, batch)
                stats["new"] += batch_stats["new"]
                stats["updated"] += batch_stats["updated"]
                stats["unchanged"] += batch_stats["unchanged"]
                stats["errors"] += batch_stats["errors"]

                if (i + BATCH_SIZE) % (BATCH_SIZE * 5) == 0:
                    logger.info(
                        f"  Upsert progress: {min(i + BATCH_SIZE, len(all_products))}/{len(all_products)}"
                    )

            # Step 5: Stale product cleanup
            logger.info("=" * 40)
            logger.info("STEP 5: Cleaning up stale products")
            logger.info("=" * 40)
            stats["stale_deleted"] = await sb.cleanup_stale_products(client, seen_urls)
    else:
        logger.info("DRY RUN - Skipping DB upsert and cleanup")
        # Count what would happen
        for product in all_products:
            if product["product_url"] not in existing_map:
                stats["new"] += 1
            else:
                stats["unchanged"] += 1

    # Step 6: Log failed products if any
    if stats["errors"] > 0:
        log_path = "logs/failed_products.log"
        with open(log_path, "a") as f:
            f.write(f"\n--- Run {datetime.now(timezone.utc).isoformat()} ---\n")
            f.write(f"Errors: {stats['errors']}\n")
        logger.warning(f"Errors logged to {log_path}")

    duration = time.time() - start_time
    _print_summary(stats, duration)

    return stats


def main():
    parser = argparse.ArgumentParser(description="H&M Product Scraper")
    parser.add_argument("--dry-run", action="store_true", help="Scrape without DB writes")
    parser.add_argument("--category", type=str, help="Filter to specific category")
    args = parser.parse_args()

    asyncio.run(run_scraper(dry_run=args.dry_run, category_filter=args.category))


if __name__ == "__main__":
    main()
