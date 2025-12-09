import os
from dotenv import load_dotenv
import asyncio
from fastmcp import FastMCP
from mcp.server.stdio import stdio_server
from tavily import TavilyClient

# Load API key
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("Missing TAVILY_API_KEY in environment variables")

tavily = TavilyClient(api_key=TAVILY_API_KEY)

app = FastMCP("web-search-mcp")


@app.tool()
def web_search(
    query: str,
    n_results: int = 5,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    recency_days: int | None = None,
):
    """
    Perform Tavily web search and return structured MCP results.
    """

    response = tavily.search(
        query=query,
        max_results=n_results,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        days=recency_days,
    )

    results = []
    for r in response.get("results", []):
        results.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
                "score": r.get("score", None),
            }
        )

    return {
        "query": query,
        "results": results,
    }


async def main():
    async with stdio_server(app.build()):
        await asyncio.Future()


if __name__ == "__main__":
    app.run(transport="stdio")
