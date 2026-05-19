"""
rag/embeddings.py
=================
Azure OpenAI embeddings with two-level caching:
  - Query cache  (in-memory, TTL 5 min) — same query asked twice → no API call
  - Batch embed  — index all tools in ONE API call at startup

Requires .env:
  AZURE_OPENAI_API_KEY
  AZURE_OPENAI_ENDPOINT
  AZURE_OPENAI_API_VERSION        (default: 2024-02-01)
  AZURE_OPENAI_EMBEDDING_DEPLOYMENT  (default: text-embedding-3-small)
"""

import os
import time
import hashlib
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
from crm_logger import log

load_dotenv()

_client: AsyncAzureOpenAI | None = None

# In-memory query embedding cache — {hash: (vector, timestamp)}
_QUERY_CACHE: dict[str, tuple[list[float], float]] = {}
_CACHE_TTL = 300  # 5 minutes


def _get_client() -> AsyncAzureOpenAI:
    global _client
    if _client is None:
        _client = AsyncAzureOpenAI(
            api_key      = os.getenv("AZURE_OPENAI_API_KEY"),
            api_version  = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
    return _client


def _deployment() -> str:
    return os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")


async def embed_query(text: str) -> list[float]:
    """
    Embed a single user query.
    Result is cached in memory for 5 minutes — same query = zero API cost.
    """
    h   = hashlib.md5(text.strip().lower().encode()).hexdigest()
    now = time.time()

    if h in _QUERY_CACHE:
        vec, ts = _QUERY_CACHE[h]
        if now - ts < _CACHE_TTL:
            log("rag", f"embed cache HIT for '{text[:40]}'")
            return vec

    client = _get_client()
    resp   = await client.embeddings.create(input=[text], model=_deployment())
    vec    = resp.data[0].embedding

    _QUERY_CACHE[h] = (vec, now)
    _evict_stale_cache(now)
    log("rag", f"embedded query '{text[:40]}' → {len(vec)}d")
    return vec


async def embed_tools(texts: list[str]) -> list[list[float]]:
    """
    Embed multiple tool description strings in ONE API call.
    Used only at session start — results stored in Chroma, not the query cache.
    """
    if not texts:
        return []
    client = _get_client()
    resp   = await client.embeddings.create(input=texts, model=_deployment())
    resp.data.sort(key=lambda x: x.index)
    log("rag", f"batch embedded {len(texts)} tool docs")
    return [item.embedding for item in resp.data]


def _evict_stale_cache(now: float):
    stale = [k for k, (_, ts) in _QUERY_CACHE.items() if now - ts > _CACHE_TTL]
    for k in stale:
        del _QUERY_CACHE[k]
