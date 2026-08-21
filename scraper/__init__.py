"""H&M product scraper using their search/listing API.

The H&M website (www2.hm.com) is protected by Akamai CDN and blocks all
direct HTTP and headless browser requests on product detail pages. However,
the search API at api.hm.com/search-services/v1/en_IE/listing/resultpage
returns structured JSON product data without any anti-bot protection.

This module uses that API to fetch product listings with full metadata
including images, prices, sizes, colors, and availability.
"""

import asyncio
import json
import logging
import time
from typing import Any
from urllib.parse import quote

import httpx

from config import PAGE_SIZE, REQUEST_DELAY

logger = logging.getLogger(__name__)

# Category configuration: (pageId, categoryId, gender)
# pageId is the URL path without locale prefix (e.g., /ladies/accessories/view-all)
# categoryId is used as the category identifier in the API
CATEGORIES: list[dict[str, str]] = [
    {
        "url": "https://www2.hm.com/en_ie/ladies/shop-by-product/view-all.html",
        "page_id": "/ladies/shop-by-product/view-all",
        "category_id": "ladies_viewall",
        "gender": "Ladies",
        "category_name": "Ladies All",
    },
    {
        "url": "https://www2.hm.com/en_ie/ladies/accessories/view-all.html",
        "page_id": "/ladies/accessories/view-all",
        "category_id": "ladies_accessories_all",
        "gender": "Ladies",
        "category_name": "Ladies Accessories",
    },
    {
        "url": "https://www2.hm.com/en_ie/ladies/shoes/view-all.html",
        "page_id": "/ladies/shoes/view-all",
        "category_id": "ladies_shoes_all",
        "gender": "Ladies",
        "category_name": "Ladies Shoes",
    },
    {
        "url": "https://www2.hm.com/en_ie/beauty/shop-by-product/view-all.html",
        "page_id": "/beauty/shop-by-product/view-all",
        "category_id": "beauty_viewall",
        "gender": "Beauty",
        "category_name": "Beauty All",
    },
    {
        "url": "https://www2.hm.com/en_ie/ladies/sport/view-all.html",
        "page_id": "/ladies/sport/view-all",
        "category_id": "ladies_sport_all",
        "gender": "Ladies",
        "category_name": "Ladies Sport",
    },
    {
        "url": "https://www2.hm.com/en_ie/men/shop-by-product/view-all.html",
        "page_id": "/men/shop-by-product/view-all",
        "category_id": "men_viewall",
        "gender": "Men",
        "category_name": "Men All",
    },
    {
        "url": "https://www2.hm.com/en_ie/men/accessories/view-all.html",
        "page_id": "/men/accessories/view-all",
        "category_id": "men_accessories_all",
        "gender": "Men",
        "category_name": "Men Accessories",
    },
    {
        "url": "https://www2.hm.com/en_ie/men/shoes/view-all.html",
        "page_id": "/men/shoes/view-all",
        "category_id": "men_shoes_all",
        "gender": "Men",
        "category_name": "Men Shoes",
    },
    {
        "url": "https://www2.hm.com/en_ie/beauty/shop-by-product/men.html",
        "page_id": "/beauty/shop-by-product/men",
        "category_id": "beauty_men",
        "gender": "Beauty",
        "category_name": "Beauty Men",
    },
    {
        "url": "https://www2.hm.com/en_ie/men/sport/view-all.html",
        "page_id": "/men/sport/view-all",
        "category_id": "men_sport_all",
        "gender": "Men",
        "category_name": "Men Sport",
    },
]

API_BASE = "https://api.hm.com/search-services/v1/en_IE/listing/resultpage"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-IE,en;q=0.9",
}


def _format_category_name(main_cat_code: str) -> str:
    """Convert H&M mainCatCode to human-readable category name.

    e.g., 'ladies_tops_longsleeve' -> 'Ladies Tops Longsleeve'
    """
    if not main_cat_code:
        return ""
    parts = main_cat_code.replace("ladies_", "").replace("men_", "").replace("sportswear_", "")
    return " ".join(word.capitalize() for word in parts.split("_"))


def _extract_product(api_product: dict[str, Any], category: dict[str, str]) -> dict[str, Any]:
    """Extract structured product data from H&M API response.

    Returns a dict with fields ready for Supabase upsert.
    """
    article_id = api_product.get("id", "")
    product_url = f"https://www2.hm.com{api_product.get('url', '')}"

    # Front packshot (productImage is the DescriptiveStillLife packshot)
    image_url = api_product.get("productImage", "")
    if not image_url:
        image_url = api_product.get("modelImage", "")

    # Title
    title = api_product.get("productName", "")

    # Price parsing
    prices = api_product.get("prices", [])
    original_price = None
    sale_price = None
    price_str = ""
    sale_str = ""

    for p in prices:
        ptype = p.get("priceType", "")
        formatted = p.get("formattedPrice", "")
        numeric = p.get("price", 0)
        currency = ""
        if "€" in formatted or "EUR" in formatted:
            currency = "EUR"
        elif "$" in formatted or "USD" in formatted:
            currency = "USD"
        elif "£" in formatted or "GBP" in formatted:
            currency = "GBP"
        else:
            currency = "EUR"  # default for IE

        price_val = f"{numeric:.2f}{currency}"

        if ptype == "whitePrice":
            original_price = price_val
        elif ptype == "redPrice":
            sale_price = price_val

    if original_price:
        price_str = original_price
    if sale_price:
        sale_str = sale_price

    # Sizes
    sizes = api_product.get("sizes", [])
    size_labels = [s.get("label", "") for s in sizes if s.get("label")]
    size_str = ", ".join(size_labels) if size_labels else None

    # Colors from swatches
    swatches = api_product.get("swatches", [])
    color_names = [s.get("colorName", "") for s in swatches if s.get("colorName")]
    color_str = ", ".join(color_names) if color_names else None

    # Gallery images (additional_images)
    images = api_product.get("images", [])
    gallery_urls = []
    for img in images:
        url = img.get("url", "")
        asset_type = img.get("assetType", "")
        if url and url != image_url:  # exclude primary front image
            gallery_urls.append(url)
    additional_images = " , ".join(gallery_urls) if gallery_urls else None

    # Category
    main_cat_code = api_product.get("mainCatCode", "")
    category_name = _format_category_name(main_cat_code)

    # Gender from API category
    gender = category.get("gender", "")

    # Metadata JSON
    metadata = {
        "article_id": article_id,
        "main_cat_code": main_cat_code,
        "color_name": api_product.get("colorName", ""),
        "color_code": api_product.get("colors", ""),
        "color_with_names": api_product.get("colorWithNames", ""),
        "availability": api_product.get("availability", {}),
        "has_video": api_product.get("hasVideo", False),
        "is_online": api_product.get("isOnline", True),
        "new_arrival": api_product.get("newArrival", False),
        "sizes": sizes,
        "swatches": [
            {
                "article_id": s.get("articleId", ""),
                "url": s.get("url", ""),
                "color_name": s.get("colorName", ""),
                "color_code": s.get("colorCode", ""),
                "product_image": s.get("productImage", ""),
            }
            for s in swatches
        ],
        "source_category": category.get("category_name", ""),
        "source_category_url": category.get("url", ""),
        "scrape_source": "hm_listing_api",
    }

    return {
        "id": f"hm_{article_id}",
        "source": "scraper-hm",
        "product_url": product_url,
        "affiliate_url": None,
        "image_url": image_url,
        "compressed_image_url": None,
        "back_image_url": None,
        "brand": "H&M",
        "title": title,
        "description": None,
        "category": category_name,
        "gender": gender,
        "price": price_str if price_str else None,
        "sale": sale_str if sale_str else None,
        "metadata": json.dumps(metadata),
        "size": size_str,
        "second_hand": False,
        "country": "IE",
        "tags": None,
        "additional_images": additional_images,
        "other": color_str,
    }


async def _fetch_page(
    client: httpx.AsyncClient,
    page_id: str,
    category_id: str,
    page_num: int,
) -> dict[str, Any] | None:
    """Fetch a single page of products from the H&M listing API."""
    params = {
        "pageSource": "PLP",
        "page": str(page_num),
        "sort": "RELEVANCE",
        "pageId": page_id,
        "page-size": str(PAGE_SIZE),
        "touchPoint": "MOBILE",
        "categoryId": category_id,
    }
    try:
        resp = await client.get(API_BASE, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        logger.warning(f"HTTP {e.response.status_code} for {page_id} page={page_num}")
        return None
    except Exception as e:
        logger.warning(f"Error fetching {page_id} page={page_num}: {e}")
        return None


async def scrape_category(
    client: httpx.AsyncClient,
    category: dict[str, str],
) -> list[dict[str, Any]]:
    """Scrape all products from a single category.

    Paginates until a page returns 0 products.
    """
    page_id = category["page_id"]
    category_id = category["category_id"]
    all_products = []
    page_num = 1

    logger.info(f"Scraping category: {category['category_name']} ({page_id})")

    while True:
        data = await _fetch_page(client, page_id, category_id, page_num)
        if data is None:
            logger.warning(f"Failed to fetch page {page_num} for {category['category_name']}")
            break

        product_list = data.get("plpList", {}).get("productList", [])
        if not product_list:
            logger.info(f"  Page {page_num}: 0 products - done with {category['category_name']}")
            break

        for api_product in product_list:
            product = _extract_product(api_product, category)
            all_products.append(product)

        pagination = data.get("pagination", {})
        total_pages = pagination.get("totalPages", 0)
        logger.info(
            f"  Page {page_num}/{total_pages}: {len(product_list)} products "
            f"(total so far: {len(all_products)})"
        )

        if page_num >= total_pages:
            break

        page_num += 1
        await asyncio.sleep(REQUEST_DELAY)

    logger.info(f"Category {category['category_name']}: {len(all_products)} total products")
    return all_products


async def scrape_all_categories(categories: list[dict[str, str]] | None = None) -> list[dict[str, Any]]:
    """Scrape all configured H&M categories.

    Args:
        categories: List of category dicts to scrape. If None, scrapes all.

    Returns a deduplicated list of products (by product_url).
    """
    if categories is None:
        categories = CATEGORIES
    all_products: dict[str, dict[str, Any]] = {}

    async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True) as client:
        for category in categories:
            products = await scrape_category(client, category)
            for product in products:
                key = product["product_url"]
                if key not in all_products:
                    all_products[key] = product
                else:
                    # Merge category info if product already seen
                    existing = all_products[key]
                    existing_meta = __import__("json").loads(existing.get("metadata") or "{}")
                    new_meta = __import__("json").loads(product.get("metadata") or "{}")
                    existing_cats = existing_meta.get("source_category", "")
                    new_cat = new_meta.get("source_category", "")
                    if new_cat and new_cat not in existing_cats:
                        existing_meta["source_category"] = f"{existing_cats}, {new_cat}".strip(", ")
                        existing["metadata"] = __import__("json").dumps(existing_meta)

            await asyncio.sleep(REQUEST_DELAY)

    logger.info(f"Total unique products: {len(all_products)}")
    return list(all_products.values())
