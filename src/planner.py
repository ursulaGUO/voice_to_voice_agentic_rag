import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


async def planner_node(state):
    """Planner node that uses LLM to create a detailed retrieval strategy.
    
    Takes router output (task, constraints) and creates a plan for:
    - Which sources to query (private catalog vs live web search)
    - Which fields to retrieve
    - Comparison criteria for ranking products
    - Search parameters (query refinement, filters, etc.)
    """
    router_output = state.get("router", {})
    task = router_output.get("task", "")
    constraints = router_output.get("constraints", {})
    extracted_query = router_output.get("extracted_query", "")
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    system_prompt = """You are a retrieval strategist for a product recommendation system.
Given a task and constraints from the router, create a detailed retrieval plan.

Available sources:
- "private": Internal product catalog with structured data (title, brand, category, price, ingredients, description)
- "live": Web search for current prices, availability, reviews, comparisons, deals

Available fields in private catalog:
- title, brand, category, price, ingredients, description, uniq_id, doc_id

IMPORTANT: The private catalog always includes 'uniq_id' - this is a unique identifier for products in the local corpus.
Web search results will have 'url' links but NO uniq_id.

Your job is to decide:
1. Which source(s) to use based on the task
2. What fields are needed to answer the query (ALWAYS include 'uniq_id' if using private catalog)
3. How to compare/rank products (comparison criteria)
4. Search parameters (refined query, filters, number of results)

Return a JSON object with:
- sources: array of "private" and/or "live"
- fields_to_retrieve: array of field names needed (MUST include 'uniq_id' if "private" is in sources)
- comparison_criteria: array of criteria for ranking (e.g., ["price", "eco-friendliness", "brand reputation"])
- search_params: object with:
  - query: refined search query
  - brand: brand filter if specified
  - category: category filter if specified
  - max_price: maximum price if budget constraint exists
  - min_price: minimum price if specified
  - must_contain: keywords that must be in results
  - n_results: number of results to retrieve (default 5)
  - rerank: whether to rerank results (default true)
- use_web_for: description of what web search should be used for (if "live" in sources)

Example:
{
  "sources": ["private", "live"],
  "fields_to_retrieve": ["title", "brand", "price", "category", "ingredients", "uniq_id"],
  "comparison_criteria": ["price", "eco-friendliness", "brand reputation"],
  "search_params": {
    "query": "eco-friendly stainless steel cleaner",
    "max_price": 20.0,
    "must_contain": "eco-friendly",
    "n_results": 5,
    "rerank": true
  },
  "use_web_for": "Verify current prices and check for better deals or promotions"
}

Note: If "private" is in sources, "uniq_id" MUST be included in fields_to_retrieve."""

    constraints_str = json.dumps(constraints, indent=2)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Task: {task}\n\nOriginal query: {extracted_query}\n\nConstraints:\n{constraints_str}\n\nCreate a retrieval plan."}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        plan = json.loads(response.choices[0].message.content)
        
        # Extract search params with defaults
        search_params = plan.get("search_params", {})
        if not search_params.get("query"):
            search_params["query"] = extracted_query
        if "n_results" not in search_params:
            search_params["n_results"] = 5
        if "rerank" not in search_params:
            search_params["rerank"] = True
        
        fields_to_retrieve = plan.get("fields_to_retrieve", ["title", "brand", "price", "category"])
        sources = plan.get("sources", ["private"])
        
        # Ensure uniq_id is included if using private catalog
        if "private" in sources and "uniq_id" not in fields_to_retrieve:
            fields_to_retrieve.append("uniq_id")
        
        return {
            "planner": {
                "plan": {
                    "sources": sources,
                    "fields_to_retrieve": fields_to_retrieve,
                    "comparison_criteria": plan.get("comparison_criteria", []),
                    "search_params": search_params,
                    "use_web_for": plan.get("use_web_for", "")
                }
            }
        }
    except Exception as e:
        print(f"Planner error: {e}")
        # Fallback to basic plan
        return {
            "planner": {
                "plan": {
                    "sources": ["private"],
                    "fields_to_retrieve": ["title", "brand", "price", "category", "uniq_id"],
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
            }
        }
