import asyncio
from fastmcp import Client


async def test_web_server():
    print("\n=== Testing Web Search MCP Server ===")

    # Spawn the web MCP server (same pattern as RAG)
    client = Client("./src/mcp_web_server.py")


    async with client:
        tools = await client.list_tools()
        print("Available tools:", tools)

        resp = await client.call_tool("web_search", {
            "query": "best stainless steel cleaner eco friendly",
            "n_results": 1
        })

        print("\nWeb Search Results:")
        print(resp)


async def test_rag_server():
    print("\n=== Testing RAG Search MCP Server ===")
    client = Client("./src/mcp_rag_server.py")
    async with client:
        tools = await client.list_tools()
        print("Available tools:", tools)
        resp = await client.call_tool("rag_search", {
            "query": "eco friendly stainless steel cleaner", 
            "n_results": 1
        })
        print("RAG results:", resp)


async def main():
    await test_rag_server()
    await test_web_server()

if __name__ == "__main__":
    asyncio.run(main())