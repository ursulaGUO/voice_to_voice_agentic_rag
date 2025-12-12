import os
import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from fastmcp import FastMCP
from mcp.server.stdio import stdio_server

# Initialize ChromaDB and embedder
project_root = Path(__file__).resolve().parents[1]
CHROMA_PATH = project_root / "data" / "chroma_amazon_clean"

client = PersistentClient(path=str(CHROMA_PATH))
collection = client.get_or_create_collection(name="amazon_products")

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed_text(text: str) -> np.ndarray:
    return embedder.encode([text], convert_to_numpy=True)[0]


def apply_metadata_filters(
    results: Dict[str, List[Any]],
    brand: Optional[str] = None,
    category: Optional[str] = None,
    max_price: Optional[float] = None,
    min_price: Optional[float] = None,
    must_contain: Optional[str] = None,
):
    filtered_ids = []
    filtered_docs = []
    filtered_metas = []
    filtered_dists = []

    total_before = len(results.get("documents", [[]])[0]) if results.get("documents") else 0
    brand_filtered = 0
    category_filtered = 0
    price_filtered = 0
    must_contain_filtered = 0

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):

        # Brand filter: check both brand field and document text (which includes title)
        if brand:
            brand_lower = brand.lower()
            brand_field = str(meta.get("brand", "")).lower()
            title = str(meta.get("title", "")).lower()
            doc_text = doc.lower() if doc else ""
            
            # Check if brand appears in brand field, title, or document text
            brand_found = (
                brand_lower in brand_field or
                brand_lower in title or
                brand_lower in doc_text
            )
            
            if not brand_found:
                brand_filtered += 1
                continue

        if category and category.lower() not in str(meta.get("category", "")).lower():
            category_filtered += 1
            continue

        price_str = meta.get("price", "")
        try:
            price = float(price_str) if price_str not in ("", None) else None
        except:
            price = None

        if max_price is not None and price is not None and price > max_price:
            price_filtered += 1
            continue

        if min_price is not None and price is not None and price < min_price:
            price_filtered += 1
            continue

        if must_contain and must_contain.lower() not in doc.lower():
            must_contain_filtered += 1
            continue

        filtered_docs.append(doc)
        filtered_metas.append(meta)
        filtered_dists.append(dist)
        filtered_ids.append(meta.get("uniq_id", "") or meta.get("doc_id", ""))
    
    # Debug filter statistics
    if total_before > 0:
        print(f"DEBUG MCP FILTER: {total_before} total, filtered: brand={brand_filtered}, category={category_filtered}, price={price_filtered}, must_contain={must_contain_filtered}, passed={len(filtered_docs)}", file=sys.stderr)

    return {
        "documents": filtered_docs,
        "metadatas": filtered_metas,
        "distances": filtered_dists,
        "ids": filtered_ids
    }


def optional_rerank(
    query: str,
    results: Dict[str, List[Any]]
):
    if not results["documents"]:
        return results

    q_emb = embed_text(query)
    doc_embs = embedder.encode(results["documents"], convert_to_numpy=True)

    sims = np.dot(doc_embs, q_emb) / (
        np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(q_emb)
    )
    order = np.argsort(sims)[::-1]

    reranked = {
        "documents": [results["documents"][i] for i in order],
        "metadatas": [results["metadatas"][i] for i in order],
        "distances": [results["distances"][i] for i in order],
        "ids": [results["ids"][i] for i in order],
    }
    return reranked


app = FastMCP("rag-mcp-server")


@app.tool()
def rag_search(
    query: str,
    n_results: int = 5,
    brand: str | None = None,
    category: str | None = None,
    max_price: float | None = None,
    min_price: float | None = None,
    must_contain: str | None = None,
    rerank: bool = True,
):
    """Search the private product catalog using RAG."""
    # Use stderr for debug output since stdout is used for MCP communication
    print(f"DEBUG MCP: RAG search called with query='{query}', n_results={n_results}, brand={brand}, category={category}, max_price={max_price}, min_price={min_price}, must_contain={must_contain}", file=sys.stderr)
    
    query_embedding = embed_text(query)
    print(f"DEBUG MCP: Query embedding created, shape: {query_embedding.shape}", file=sys.stderr)

    raw_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results * 2
    )
    num_raw_docs = len(raw_results.get('documents', [[]])[0]) if raw_results.get('documents') else 0
    print(f"DEBUG MCP: Raw query returned {num_raw_docs} documents", file=sys.stderr)
    
    if num_raw_docs == 0:
        print(f"DEBUG MCP: WARNING - ChromaDB returned 0 documents! Collection might be empty or query failed.", file=sys.stderr)
        return {"query": query, "results": []}

    filtered = apply_metadata_filters(
        raw_results,
        brand=brand,
        category=category,
        max_price=max_price,
        min_price=min_price,
        must_contain=must_contain,
    )
    num_filtered = len(filtered.get('documents', []))
    print(f"DEBUG MCP: After filtering, {num_filtered} documents remain (filters: brand={brand}, category={category}, max_price={max_price}, min_price={min_price}, must_contain={must_contain})", file=sys.stderr)
    
    if num_raw_docs > 0 and num_filtered == 0:
        print(f"DEBUG MCP: WARNING - All {num_raw_docs} documents were filtered out! Filters might be too restrictive.", file=sys.stderr)

    if rerank:
        filtered = optional_rerank(query, {
            "documents": filtered["documents"],
            "metadatas": filtered["metadatas"],
            "distances": filtered["distances"],
            "ids": filtered["ids"]
        })

    # Limit to requested number of results
    num_docs = len(filtered.get("documents", []))
    num_results = min(n_results, num_docs)
    print(f"DEBUG MCP: Returning {num_results} results (requested {n_results}, available {num_docs})", file=sys.stderr)
    
    result_list = []
    if num_docs > 0:
        for doc, meta, dist in zip(
            filtered["documents"][:num_results],
            filtered["metadatas"][:num_results],
            filtered["distances"][:num_results]
        ):
            result_list.append({
                "uniq_id": meta.get("uniq_id", "") or meta.get("doc_id", ""),  # Prefer uniq_id, fallback to doc_id
                "doc_id": meta.get("doc_id", ""),  # Keep doc_id for backward compatibility
                "title": meta.get("title", ""),
                "brand": meta.get("brand", ""),
                "category": meta.get("category", ""),
                "price": meta.get("price", ""),
                "ingredients": meta.get("ingredients", ""),
                "score": float(dist),
                "snippet": doc[:300] if doc else ""
            })
            print(f"DEBUG MCP: Added result: {meta.get('title', 'N/A')} (uniq_id: {meta.get('uniq_id', 'N/A')}, brand: {meta.get('brand', 'N/A')}, category: {meta.get('category', 'N/A')})", file=sys.stderr)
    else:
        print(f"DEBUG MCP: No documents to return - check if filters are too restrictive", file=sys.stderr)
    
    return {
        "query": query,
        "results": result_list
    }


async def main():
    # app.build() converts FastMCP into a raw MCP application
    async with stdio_server(app.build()):
        await asyncio.Future()


if __name__ == "__main__":
    app.run(transport="stdio")