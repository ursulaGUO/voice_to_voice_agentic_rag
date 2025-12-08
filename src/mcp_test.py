import asyncio
from fastmcp import Client

async def main():
    client = Client("./src/mcp_server.py")
    async with client:
        tools = await client.list_tools()
        print("Available tools:", tools)
        resp = await client.call_tool("rag_search", {
            "query": "eco friendly stainless steel cleaner", 
            "n_results": 5
        })
        print("RAG results:", resp)

if __name__ == "__main__":
    asyncio.run(main())
