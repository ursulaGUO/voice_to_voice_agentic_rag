import asyncio
from fastmcp import FastMCP
from mcp.server.stdio import stdio_server
from rag_search import rag_search as _rag_search

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
    return _rag_search(
        query=query,
        n_results=n_results,
        brand=brand,
        category=category,
        max_price=max_price,
        min_price=min_price,
        must_contain=must_contain,
        rerank=rerank,
    )


async def main():
    # app.build() converts FastMCP into a raw MCP application
    async with stdio_server(app.build()):
        await asyncio.Future()


if __name__ == "__main__":
    app.run(transport="stdio")

