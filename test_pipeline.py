import asyncio
import json
import sys
from pathlib import Path

src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from graph import build_graph


def print_section(title, char="=", width=80):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}\n")


def print_state(state, show_all=False):
    """Print the state at each stage of the pipeline."""
    if "router" in state:
        router = state["router"]
        print("ROUTER OUTPUT:")
        print(f"  Route: {router.get('route', 'N/A')}")
        print(f"  Task: {router.get('task', 'N/A')}")
        print(f"  Extracted Query: {router.get('extracted_query', 'N/A')}")
        print(f"  Constraints: {json.dumps(router.get('constraints', {}), indent=4)}")
        print(f"  Safety Flags: {router.get('safety_flags', [])}")
        print()
    
    if "planner" in state:
        planner = state["planner"]
        plan = planner.get("plan", {})
        print("PLANNER OUTPUT:")
        print(f"  Sources: {plan.get('sources', [])}")
        print(f"  Fields to Retrieve: {plan.get('fields_to_retrieve', [])}")
        print(f"  Comparison Criteria: {plan.get('comparison_criteria', [])}")
        print(f"  Search Params: {json.dumps(plan.get('search_params', {}), indent=4)}")
        if plan.get("use_web_for"):
            print(f"  Web Search Purpose: {plan.get('use_web_for')}")
        print()
    
    if "retriever" in state:
        retriever = state["retriever"]
        results = retriever.get("retrieval_results", {})
        print("RETRIEVER OUTPUT:")
        print(f"  Number of Results: {len(results.get('results', []))}")
        if results.get("conflicts"):
            print(f"  Conflicts Found: {len(results.get('conflicts', []))}")
        if results.get("recommendations"):
            print(f"  Recommendations: {len(results.get('recommendations', []))}")
        if show_all and results.get("results"):
            print("  Top Results:")
            for i, result in enumerate(results["results"][:5], 1):
                source = result.get('source', 'unknown')
                if source == "local_corpus":
                    uniq_id = result.get('uniq_id', '') or result.get('doc_id', 'N/A')
                    print(f"    {i}. [{source}] {result.get('title', 'N/A')} - ${result.get('price', 'N/A')} [uniq_id: {uniq_id}] (no URL)")
                elif source == "web_search":
                    url = result.get('url', 'N/A')
                    print(f"    {i}. [{source}] {result.get('title', 'N/A')} - ${result.get('price', 'N/A')} [URL: {url}] (no uniq_id)")
                else:
                    print(f"    {i}. {result.get('title', 'N/A')} - ${result.get('price', 'N/A')}")
        print()
    
    if "final_answer" in state:
        print("FINAL ANSWER:")
        answer = state['final_answer']
        # Print multi-line answers with proper indentation
        for line in answer.split('\n'):
            print(f"  {line}")
        if "citations" in state and state.get("citations"):
            print(f"\n  Citations (doc_ids): {state.get('citations', [])}")
        if "web_urls" in state and state.get("web_urls"):
            print(f"\n  🔗 Web Sources:")
            for url in state.get("web_urls", [])[:5]:  # Show up to 5 URLs
                print(f"    - {url}")
        if "grounded" in state:
            grounded_status = "Grounded" if state.get('grounded', True) else "⚠️  Partially Grounded"
            print(f"\n  {grounded_status}")
        print()


async def test_pipeline(user_input, test_name):
    """Test the pipeline with a given user input."""
    print_section(f"TEST: {test_name}", "=")
    print(f"User Input: \"{user_input}\"")
    print()
    
    workflow = build_graph()
    
    try:
        result = await workflow.ainvoke({"user_input": user_input})
        print_state(result, show_all=True)
        return result
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """test cases."""
    print_section("AGENT PIPELINE TEST SUITE", "=")
    
    # Test Case 1
    test1 = "I want to buy a Barbie doll."
    await test_pipeline(test1, "Product search with brand name.")
    
    """
    # Test Case 2
    test2 = "What is stainless steel?"
    await test_pipeline(test2, "General knowledge query")
    

    # Test Case 3
    test3 = "Compare different skate boards. I want to spend less than 300 dollars. "
    await test_pipeline(test3, "Product comparison with budget limit")"""

    # Test Case 4 waterproof quality

    

    
    print_section("ALL TESTS COMPLETED", "=")


if __name__ == "__main__":
    asyncio.run(main())

