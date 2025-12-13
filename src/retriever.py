import os
import json
from dotenv import load_dotenv
from fastmcp import Client

load_dotenv()


async def call_rag_search(
    query: str,
    n_results: int = 5,
    brand: str | None = None,
    category: str | None = None,
    max_price: float | None = None,
    min_price: float | None = None,
    must_contain: str | None = None,
    rerank: bool = True,
):
    """Call the RAG search MCP server."""
    try:
        print(f"DEBUG: Calling RAG MCP with query: {query}, n_results: {n_results}")
        client = Client("./src/mcp_rag_server.py")
        async with client:
            resp = await client.call_tool("rag_search", {
                "query": query,
                "n_results": n_results,
                "brand": brand,
                "category": category,
                "max_price": max_price,
                "min_price": min_price,
                "must_contain": must_contain,
                "rerank": rerank,
            })
            print(f"DEBUG: RAG MCP response type: {type(resp)}")
            print(f"DEBUG: RAG MCP response: {resp}")
            
            # Extract content from CallToolResult object
            if hasattr(resp, 'content'):
                content = resp.content
                print(f"DEBUG: Response has content attribute, type: {type(content)}")
                if isinstance(content, list) and len(content) > 0:
                    text_content = content[0]
                    print(f"DEBUG: Content is list, first item type: {type(text_content)}")
                    if hasattr(text_content, 'text'):
                        result = json.loads(text_content.text)
                        print(f"DEBUG: Parsed JSON from text, got {len(result.get('results', []))} results")
                        return result
                    elif isinstance(text_content, str):
                        result = json.loads(text_content)
                        print(f"DEBUG: Parsed JSON from string, got {len(result.get('results', []))} results")
                        return result
                elif isinstance(content, str):
                    result = json.loads(content)
                    print(f"DEBUG: Parsed JSON from content string, got {len(result.get('results', []))} results")
                    return result
                elif isinstance(content, dict):
                    print(f"DEBUG: Content is dict, got {len(content.get('results', []))} results")
                    return content
            # If it's already a dict, return as is
            if isinstance(resp, dict):
                print(f"DEBUG: Response is dict, got {len(resp.get('results', []))} results")
                return resp
            # Fallback: try to get dict representation
            try:
                result = dict(resp) if hasattr(resp, '__dict__') else {"error": "Could not parse RAG search results"}
                print(f"DEBUG: Fallback conversion, result: {result}")
                return result
            except Exception as e2:
                print(f"DEBUG: Fallback failed: {e2}")
                return {"error": "Could not parse RAG search results", "raw_response": str(resp)}
    except Exception as e:
        print(f"RAG search error: {e}")
        import traceback
        traceback.print_exc()
        return None


async def call_web_search(query: str, n_results: int = 5):
    """Call the web search MCP server."""
    try:
        client = Client("./src/mcp_web_server.py")
        async with client:
            resp = await client.call_tool("web_search", {
                "query": query,
                "n_results": n_results
            })
            # Extract content from CallToolResult object
            # FastMCP returns CallToolResult which has content attribute
            if hasattr(resp, 'content'):
                content = resp.content
                # Content is usually a list of TextContent objects
                if isinstance(content, list) and len(content) > 0:
                    # Get the first text content and parse as JSON
                    text_content = content[0]
                    if hasattr(text_content, 'text'):
                        return json.loads(text_content.text)
                    elif isinstance(text_content, str):
                        return json.loads(text_content)
                elif isinstance(content, str):
                    return json.loads(content)
                elif isinstance(content, dict):
                    return content
            # If it's already a dict, return as is
            if isinstance(resp, dict):
                return resp
            # Fallback: try to get dict representation
            try:
                return dict(resp) if hasattr(resp, '__dict__') else {"error": "Could not parse web search results"}
            except:
                return {"error": "Could not parse web search results"}
    except Exception as e:
        print(f"Web search error: {e}")
        import traceback
        traceback.print_exc()
        return None


async def retriever_node(state):
    """Retriever node that executes the retrieval plan.
    
    This node focuses on execution - it takes the plan from the planner
    and retrieves data from the specified sources, then reconciles conflicts.
    """
    planner_output = state.get("planner", {})
    plan = planner_output.get("plan", {})
    
    if not plan:
        # Fallback if planner didn't provide a plan
        router_output = state.get("router", {})
        extracted_query = router_output.get("extracted_query", "")
        constraints = router_output.get("constraints", {})
        
        plan = {
            "sources": ["private"],
            "fields_to_retrieve": ["title", "brand", "price", "category"],
            "comparison_criteria": [],
            "search_params": {
                "query": extracted_query,
                "brand": constraints.get("brand"),
                "category": constraints.get("category"),
                "max_price": constraints.get("max_price"),
                "min_price": constraints.get("min_price"),
                "must_contain": constraints.get("must_contain"),
                "n_results": 5,
                "rerank": True
            },
            "use_web_for": ""
        }
    
    search_params = plan.get("search_params", {})
    sources = plan.get("sources", ["private"])
    comparison_criteria = plan.get("comparison_criteria", [])
    use_web_for = plan.get("use_web_for", "")
    
    # Retrieve from private catalog using MCP server
    private_results = None
    if "private" in sources:
        private_results = await call_rag_search(
            query=search_params.get("query", ""),
            n_results=search_params.get("n_results", 5),
            brand=search_params.get("brand"),
            category=search_params.get("category"),
            max_price=search_params.get("max_price"),
            min_price=search_params.get("min_price"),
            must_contain=search_params.get("must_contain"),
            rerank=search_params.get("rerank", True)
        )
        # Ensure private_results is a dict (from MCP response)
        if private_results and not isinstance(private_results, dict):
            # Try to extract dict from CallToolResult if still not converted
            if hasattr(private_results, 'content'):
                content = private_results.content
                if isinstance(content, list) and len(content) > 0:
                    text_content = content[0]
                    if hasattr(text_content, 'text'):
                        private_results = json.loads(text_content.text)
                    elif isinstance(text_content, str):
                        private_results = json.loads(text_content)
                elif isinstance(content, str):
                    private_results = json.loads(content)
        
        # Mark all private results with source and ensure they have uniq_id and product URL
        if private_results and isinstance(private_results, dict) and private_results.get("results"):
            # Import here to avoid circular imports
            from comparison_table import get_product_url
            
            for result in private_results["results"]:
                result["source"] = "local_corpus"
                # Ensure uniq_id exists (it should from RAG MCP server)
                if not result.get("uniq_id") and result.get("doc_id"):
                    result["uniq_id"] = result["doc_id"]
                
                # Add product URL from CSV lookup using uniq_id
                uniq_id = result.get("uniq_id", "")
                if uniq_id:
                    product_url = get_product_url(uniq_id)
                    if product_url:
                        result["url"] = product_url
                    elif "url" in result:
                        # Keep existing URL if lookup didn't find one
                        pass
                elif "url" in result:
                    # Remove URL if no uniq_id (shouldn't happen for local corpus)
                    del result["url"]
            # Debug: print RAG search results
            print(f"DEBUG: Local corpus search (via MCP) found {len(private_results['results'])} results")
            for i, r in enumerate(private_results["results"][:3], 1):
                url_info = f"URL: {r.get('url', 'N/A')}" if r.get("url") else "no URL"
                print(f"  Local Corpus Result {i}: {r.get('title', 'N/A')} - uniq_id: {r.get('uniq_id', 'N/A')} ({url_info})")
    
    # Retrieve from web if needed
    web_results = None
    if "live" in sources:
        web_query = search_params.get("query", "")
        if use_web_for:
            web_query = f"{web_query} {use_web_for}"
        web_results = await call_web_search(web_query, n_results=3)
        # Mark all web results with source and ensure they have url, no uniq_id
        if web_results and isinstance(web_results, dict) and web_results.get("results"):
            for result in web_results["results"]:
                result["source"] = "web_search"
                # Remove any uniq_id if present (web search shouldn't have uniq_id)
                if "uniq_id" in result:
                    del result["uniq_id"]
                if "doc_id" in result:
                    del result["doc_id"]
            print(f"DEBUG: Web search found {len(web_results['results'])} results")
            for i, r in enumerate(web_results["results"][:3], 1):
                print(f"  Web Search Result {i}: {r.get('title', 'N/A')} - URL: {r.get('url', 'N/A')} (no uniq_id)")
    
    # Reconcile conflicts if both sources used
    reconciled_results = private_results
    conflicts = []
    recommendations = []
    
    if private_results and web_results:
        # Ensure web_results is a dict before JSON serialization
        if not isinstance(web_results, dict):
            # Try to extract dict from CallToolResult if still not converted
            if hasattr(web_results, 'content'):
                content = web_results.content
                if isinstance(content, list) and len(content) > 0:
                    text_content = content[0]
                    if hasattr(text_content, 'text'):
                        web_results = json.loads(text_content.text)
                    elif isinstance(text_content, str):
                        web_results = json.loads(text_content)
                elif isinstance(content, str):
                    web_results = json.loads(content)
        
        # Use LLM to reconcile conflicts
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Only proceed if web_results is now a dict
        if isinstance(web_results, dict):
            reconcile_prompt = f"""Reconcile product search results from private catalog and web search.
Private catalog results (source: local_corpus, have uniq_id, NO urls): {json.dumps(private_results.get('results', [])[:3], indent=2)}
Web search results (source: web_search, have urls, NO uniq_id): {json.dumps(web_results, indent=2)}

Compare prices, availability, and other information. Return a JSON object with:
- reconciled_results: array of best products with reconciled information
- conflicts: array of any conflicts found (price differences, availability issues, etc.)
- recommendations: array of recommendations based on the comparison

CRITICAL REQUIREMENTS:
1. For products from local_corpus: MUST include 'uniq_id' field, MUST NOT include 'url' field, set source="local_corpus"
2. For products from web_search: MUST include 'url' field, MUST NOT include 'uniq_id' or 'doc_id' fields, set source="web_search"
3. If a product appears in both sources, create TWO separate entries - one for local_corpus (with uniq_id) and one for web_search (with url)
4. Do NOT merge uniq_id and url into the same result - keep them separate by source

Focus on: {', '.join(comparison_criteria) if comparison_criteria else 'price and quality'}"""

            try:
                reconcile_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a product information reconciler. Compare and reconcile product data from different sources."},
                        {"role": "user", "content": reconcile_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2
                )
                reconciled = json.loads(reconcile_response.choices[0].message.content)
                reconciled_results_list = reconciled.get("reconciled_results", [])
                
                # Post-process to ensure proper source separation
                private_results_list = private_results.get("results", [])
                web_results_list = web_results.get("results", [])
                
                # Create lookup maps
                private_data_by_title = {r.get("title", ""): r for r in private_results_list}
                web_data_by_title = {r.get("title", ""): r for r in web_results_list}
                
                # Clean up reconciled results to ensure proper source separation
                # Import here to avoid circular imports
                from comparison_table import get_product_url
                
                cleaned_results = []
                for result in reconciled_results_list:
                    title = result.get("title", "")
                    source = result.get("source", "")
                    
                    if source == "local_corpus" or (result.get("uniq_id") and not result.get("url")):
                        # This is a local corpus result
                        if title in private_data_by_title:
                            private_data = private_data_by_title[title]
                            # Ensure it has uniq_id and product URL from CSV
                            result["uniq_id"] = result.get("uniq_id") or private_data.get("uniq_id") or private_data.get("doc_id", "")
                            result["source"] = "local_corpus"
                            
                            # Add product URL from CSV lookup
                            uniq_id = result.get("uniq_id", "")
                            if uniq_id:
                                product_url = get_product_url(uniq_id)
                                if product_url:
                                    result["url"] = product_url
                                elif "url" in result:
                                    # Remove URL if lookup didn't find one and it's not from CSV
                                    del result["url"]
                            elif "url" in result:
                                del result["url"]
                            
                            if "doc_id" not in result:
                                result["doc_id"] = private_data.get("doc_id", "")
                        cleaned_results.append(result)
                    elif source == "web_search" or (result.get("url") and not result.get("uniq_id")):
                        # This is a web search result
                        if title in web_data_by_title:
                            web_data = web_data_by_title[title]
                            # Ensure it has url and no uniq_id
                            result["url"] = result.get("url") or web_data.get("url", "")
                            result["source"] = "web_search"
                            if "uniq_id" in result:
                                del result["uniq_id"]
                            if "doc_id" in result:
                                del result["doc_id"]
                        cleaned_results.append(result)
                
                # If reconciliation didn't work well, combine both sources separately
                if not cleaned_results:
                    # Import here to avoid circular imports
                    from comparison_table import get_product_url
                    
                    # Add all local corpus results
                    for r in private_results_list[:5]:
                        r_copy = r.copy()
                        r_copy["source"] = "local_corpus"
                        # Add product URL from CSV lookup if not already present
                        uniq_id = r_copy.get("uniq_id", "")
                        if uniq_id:
                            product_url = get_product_url(uniq_id)
                            if product_url:
                                r_copy["url"] = product_url
                            elif "url" in r_copy:
                                del r_copy["url"]
                        elif "url" in r_copy:
                            del r_copy["url"]
                        cleaned_results.append(r_copy)
                    # Add all web search results
                    for r in web_results_list[:3]:
                        r_copy = r.copy()
                        r_copy["source"] = "web_search"
                        if "uniq_id" in r_copy:
                            del r_copy["uniq_id"]
                        if "doc_id" in r_copy:
                            del r_copy["doc_id"]
                        cleaned_results.append(r_copy)
                
                reconciled_results = {
                    "query": private_results.get("query", ""),
                    "results": cleaned_results,
                }
                conflicts = reconciled.get("conflicts", [])
                recommendations = reconciled.get("recommendations", [])
            except Exception as e:
                print(f"Reconciliation error: {e}")
                # Fall back to private results
    
    # If web_results exist but weren't reconciled, combine them properly
    if not reconciled_results:
        # Import here to avoid circular imports
        from comparison_table import get_product_url
        
        final_results = {"results": []}
        
        # Add local corpus results (with uniq_id and product URL from CSV)
        if private_results and private_results.get("results"):
            for result in private_results["results"]:
                result_copy = result.copy()
                result_copy["source"] = "local_corpus"
                # Add product URL from CSV lookup
                uniq_id = result_copy.get("uniq_id", "")
                if uniq_id:
                    product_url = get_product_url(uniq_id)
                    if product_url:
                        result_copy["url"] = product_url
                    elif "url" in result_copy:
                        del result_copy["url"]
                elif "url" in result_copy:
                    del result_copy["url"]
                final_results["results"].append(result_copy)
        
        # Add web search results (with url, no uniq_id)
        if web_results and isinstance(web_results, dict) and web_results.get("results"):
            for result in web_results["results"]:
                result_copy = result.copy()
                result_copy["source"] = "web_search"
                if "uniq_id" in result_copy:
                    del result_copy["uniq_id"]
                if "doc_id" in result_copy:
                    del result_copy["doc_id"]
                final_results["results"].append(result_copy)
        
        final_results["query"] = private_results.get("query", "") if private_results else web_results.get("query", "")
    else:
        final_results = reconciled_results
    
    return {
        "retriever": {
            "retrieval_results": {
                **final_results,
                "conflicts": conflicts,
                "recommendations": recommendations
            }
        }
    }
