"""
datasets.py - download helpers for public URL feeds.

  * OpenPhish community feed: currently-active phishing URLs (changes daily).
  * Tranco top list: ranked popular domains, used as benign examples.

NOTE: Tranco contains bare domains, so benign URLs built from it are homepages
without paths. See the README ("Known data bias") for why that matters.
"""
import io
import zipfile

import httpx

OPENPHISH_FEED = "https://openphish.com/feed.txt"
TRANCO_TOP1M = "https://tranco-list.eu/top-1m.csv.zip"


def fetch_openphish(timeout=30):
    """Return the list of URLs in the current OpenPhish community feed."""
    r = httpx.get(OPENPHISH_FEED, timeout=timeout, follow_redirects=True)
    r.raise_for_status()
    return [line.strip() for line in r.text.splitlines() if line.strip().startswith("http")]


def fetch_tranco(limit=100_000, timeout=120):
    """Return [(rank, domain)] for the top `limit` Tranco domains."""
    r = httpx.get(TRANCO_TOP1M, timeout=timeout, follow_redirects=True)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = z.namelist()[0]
        rows = []
        with z.open(name) as f:
            for line in io.TextIOWrapper(f, encoding="utf-8"):
                rank, _, domain = line.strip().partition(",")
                if rank.isdigit() and domain:
                    rows.append((int(rank), domain))
                if len(rows) >= limit:
                    break
    return rows
