# H&M Product Scraper

Production-grade scraper for H&M (Ireland) that fetches all products via H&M's listing API, generates 768-dimensional image and text embeddings, and upserts into a Supabase database with smart diffing.

## How It Works

### Data Source
H&M's website (`www2.hm.com`) is protected by Akamai CDN and blocks all direct HTTP/headless browser requests on product pages. However, their search/listing API at `api.hm.com/search-services/v1/en_IE/listing/resultpage` returns structured JSON product data without anti-bot protection.

The scraper uses this API to fetch:
- Product titles, prices, and availability
- Front packshot images (productImage field)
- Gallery images with asset types
- Color swatches and size/stock data
- Category codes and metadata

### Embeddings
- **Image embeddings**: Google SigLIP (`google/siglip-base-patch16-384`) via HuggingFace Inference API
- **Text embeddings**: Gemini `embedding-001` via Google API
- All vectors are L2-normalized (768 dimensions)
- Embeddings are only regenerated when source data changes

### Smart Upsert
- Compares all fields before writing to minimize redundant DB operations
- Embeddings only regenerated when `image_url` or text fields change
- Stale products (missed 2 consecutive runs) are automatically deleted
- Batch upserts (50 products per request) reduce API calls

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variables (or use `.env`):
   ```bash
   export SUPABASE_URL=https://your-project.supabase.co
   export SUPABASE_KEY=your-service-role-key
   export HF_API_TOKEN=hf_your_token_here
   export GEMINI_API_KEY=your-gemini-key-here
   ```

3. Run:
   ```bash
   python main.py                    # Full scrape
   python main.py --dry-run          # Scrape without DB writes
   python main.py --category ladies  # Filter specific categories
   ```

## Categories Scraped

| Category | URL | Approx Products |
|----------|-----|-----------------|
| Ladies All | `/ladies/shop-by-product/view-all.html` | ~12,500 |
| Ladies Accessories | `/ladies/accessories/view-all.html` | ~880 |
| Ladies Shoes | `/ladies/shoes/view-all.html` | ~660 |
| Beauty All | `/beauty/shop-by-product/view-all.html` | ~430 |
| Ladies Sport | `/ladies/sport/view-all.html` | ~700 |
| Men All | `/men/shop-by-product/view-all.html` | ~2,550 |
| Men Accessories | `/men/accessories/view-all.html` | ~200 |
| Men Shoes | `/men/shoes/view-all.html` | ~85 |
| Beauty Men | `/beauty/shop-by-product/men.html` | ~1 |
| Men Sport | `/men/sport/view-all.html` | ~240 |

## GitHub Actions

Runs automatically 3x/week (Sunday, Tuesday, Friday at 7:15 UTC). Can also be triggered manually via workflow_dispatch.

### Required Secrets
- `SUPABASE_URL`
- `SUPABASE_KEY`
- `HF_API_TOKEN`
- `GEMINI_API_KEY`

## Architecture

```
main.py              # Entry point and orchestrator
config.py            # Environment-based configuration
scraper/
  __init__.py        # H&M API client and product extraction
embeddings.py        # SigLIP image + Gemini text embeddings
supabase_client.py   # Smart upsert, diffing, and stale cleanup
```
