import asyncio
from graph import build_graph

workflow = build_graph()

async def main():
    result = await workflow.ainvoke({"user_input": "recommend eco-friendly steel cleaner under 15 dollars"})
    print(result["final_answer"])

asyncio.run(main())
