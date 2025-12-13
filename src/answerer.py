import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


async def answerer_node(state):
    """Answerer node that uses LLM to synthesize concise, cited recommendations with grounding and safety."""
    retrieval_results = state.get("retriever", {}).get("retrieval_results", {})
    results = retrieval_results.get("results", [])
    router_output = state.get("router", {})
    task = router_output.get("task", "")
    safety_flags = router_output.get("safety_flags", [])
    retrieval_strategy = state.get("retriever", {}).get("retrieval_strategy", {})
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    if not results:
        return {
            "final_answer": "I couldn't find any suitable products matching your criteria. Try adjusting your filters or search terms."
        }
    
    # Check safety flags
    if safety_flags:
        return {
            "final_answer": f"I cannot assist with this request due to safety concerns: {', '.join(safety_flags)}"
        }
    
    system_prompt = """You are a product recommendation assistant. Synthesize a concise, well-cited recommendation based on search results.

Requirements:
1. Be concise and actionable
2. Cite specific products with their uniq_id (or doc_id) from the private database
3. Ground all claims in the provided data - don't make up information
4. Highlight key features, prices, and why you're recommending them
5. If there are conflicts or limitations, mention them
6. Ensure safety - don't recommend harmful or inappropriate products
7. ALWAYS include web links (URLs) when available - format as [Source: <url>] or include clickable links

Format your response as a clear, natural recommendation with citations:
- Private DB products: [uniq_id: <id>] or [doc_id: <id>], and include product URL if available
- Web search results: [Source: <url>] or include the full URL
- Both local and web products may have URLs - always include them when present"""

    results_str = json.dumps(results[:5], indent=2)  # Limit to top 5 for context
    conflicts = retrieval_results.get("conflicts", [])
    recommendations = retrieval_results.get("recommendations", [])
    
    user_prompt = f"""Task: {task}

Product search results:
{results_str}

"""
    
    if conflicts:
        user_prompt += f"Conflicts found: {json.dumps(conflicts, indent=2)}\n\n"
    
    if recommendations:
        user_prompt += f"Additional recommendations: {json.dumps(recommendations, indent=2)}\n\n"
    
    user_prompt += """Synthesize a concise recommendation. Include:
- Your top recommendation(s) with specific product names
- Key features (brand, category, price)
- Why you're recommending them
- Citations: [uniq_id: <id>] or [doc_id: <id>] for private DB products, [Source: <url>] for web search results
- Web links (URLs) MUST be included for ANY products that have them (both local catalog and web search products can have URLs)
- Format URLs as clickable links: [Product Name](url) or [Source: url]
- Any important limitations or considerations"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Verify grounding - check if answer references the actual results
        grounding_check = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a fact-checker. Verify if the recommendation is grounded in the provided data."},
                {"role": "user", "content": f"Recommendation:\n{answer}\n\nAvailable products:\n{results_str}\n\nDoes the recommendation only use information from the available products? Return JSON: {{\"grounded\": true/false, \"issues\": [array of any ungrounded claims]}}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        grounding_result = json.loads(grounding_check.choices[0].message.content)
        
        if not grounding_result.get("grounded", True):
            issues = grounding_result.get("issues", [])
            answer += f"\n\n[Note: Some claims may not be fully verified. Issues: {', '.join(issues)}]"
        
        # Collect all citations including URLs
        citations = [r.get("uniq_id", "") or r.get("doc_id", "") for r in results if r.get("uniq_id") or r.get("doc_id")]
        web_urls = [r.get("url", "") for r in results if r.get("url")]
        
        # If answer doesn't include URLs but we have them, append them
        if web_urls and "http" not in answer.lower():
            answer += "\n\n**Web Sources:**\n"
            for url in web_urls[:5]:  # Limit to top 5 URLs
                answer += f"- {url}\n"
        
        return {
            "final_answer": answer,
            "citations": citations,
            "web_urls": web_urls,
            "grounded": grounding_result.get("grounded", True)
        }
    except Exception as e:
        print(f"Answerer error: {e}")
        # Fallback to simple template
        top = results[0]
        answer = f"My top recommendation is **{top.get('title', 'Product')}**.\n- Brand: {top.get('brand', 'N/A')}\n- Category: {top.get('category', 'N/A')}\n- Price: ${top.get('price', 'N/A')}"
        
        uniq_id = top.get("uniq_id", "") or top.get("doc_id", "")
        if uniq_id:
            answer += f"\n\nCitation (private DB uniq_id): {uniq_id}"
        
        if top.get("url"):
            answer += f"\n\nSource: {top.get('url')}"
        
        uniq_id = top.get("uniq_id", "") or top.get("doc_id", "")
        citations = [uniq_id] if uniq_id else []
        web_urls = [top.get("url", "")] if top.get("url") else []
        
        return {
            "final_answer": answer,
            "citations": citations,
            "web_urls": web_urls,
            "grounded": True
        }