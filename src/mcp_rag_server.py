import os
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

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):

        if brand and brand.lower() not in str(meta.get("brand", "")).lower():
            continue

        if category and category.lower() not in str(meta.get("category", "")).lower():
            continue

        price_str = meta.get("price", "")
        try:
            price = float(price_str) if price_str not in ("", None) else None
        except:
            price = None

        if max_price is not None and price is not None and price > max_price:
            continue

        if min_price is not None and price is not None and price < min_price:
            continue

        if must_contain and must_contain.lower() not in doc.lower():
            continue

        filtered_docs.append(doc)
        filtered_metas.append(meta)
        filtered_dists.append(dist)
        filtered_ids.append(meta.get("uniq_id", "") or meta.get("doc_id", ""))

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
    print(f"DEBUG MCP: RAG search called with query='{query}', n_results={n_results}, brand={brand}, category={category}, max_price={max_price}, min_price={min_price}, must_contain={must_contain}")
    
    query_embedding = embed_text(query)
    print(f"DEBUG MCP: Query embedding created, shape: {query_embedding.shape}")

    raw_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results * 2
    )
    print(f"DEBUG MCP: Raw query returned {len(raw_results.get('documents', [[]])[0])} documents")

    filtered = apply_metadata_filters(
        raw_results,
        brand=brand,
        category=category,
        max_price=max_price,
        min_price=min_price,
        must_contain=must_contain,
    )
    print(f"DEBUG MCP: After filtering, {len(filtered.get('documents', []))} documents remain")

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
    print(f"DEBUG MCP: Returning {num_results} results (requested {n_results}, available {num_docs})")
    
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
            print(f"DEBUG MCP: Added result: {meta.get('title', 'N/A')} (uniq_id: {meta.get('uniq_id', 'N/A')})")
    else:
        print(f"DEBUG MCP: No documents to return - check if filters are too restrictive")
    
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
