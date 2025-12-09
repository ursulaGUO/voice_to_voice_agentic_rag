import os
import json
from typing import Literal
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class RouterOutput(BaseModel):
    route: Literal["search", "general", "unsafe"]
    extracted_query: str
    task: str
    constraints: dict
    safety_flags: list[str]


async def router_node(state):
    """Router node that uses LLM to classify intent, extract task, constraints, and safety flags."""
    user_input = state.get("user_input", "")
    
    if not user_input:
        # Fallback if user_input is missing
        return {
            "route": "general",
            "extracted_query": "",
            "task": "",
            "constraints": {},
            "safety_flags": []
        }
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    system_prompt = """You are an intent classifier for a product recommendation system. 
Analyze the user's query and extract:
1. The main task (what they want to do)
2. Constraints: budget (max_price, min_price), material, brand, category, or other filters
3. Safety flags: any concerning content (inappropriate requests, harmful products, etc.)
4. Route: "search" for product queries, "general" for conversational queries, "unsafe" for flagged content

Return a JSON object with:
- route: "search" | "general" | "unsafe"
- extracted_query: cleaned search query
- task: brief description of the task
- constraints: object with budget, brand, material, category, etc. (use null for missing values)
- safety_flags: array of any safety concerns (empty array if none)

Example:
{
  "route": "search",
  "extracted_query": "eco-friendly stainless steel cleaner under $20",
  "task": "Find eco-friendly stainless steel cleaner within budget",
  "constraints": {
    "max_price": 20.0,
    "material": "stainless steel",
    "must_contain": "eco-friendly"
  },
  "safety_flags": []
}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Validate route
        route = result.get("route", "general")
        if route not in ["search", "general", "unsafe"]:
            route = "general"
        
        return {
            "router": {
                "route": route,
                "extracted_query": result.get("extracted_query", user_input),
                "task": result.get("task", ""),
                "constraints": result.get("constraints", {}),
                "safety_flags": result.get("safety_flags", [])
            }
        }
    except Exception as e:
        # Fallback to safe defaults on error
        return {
            "router": {
                "route": "general",
                "extracted_query": user_input,
                "task": "",
                "constraints": {},
                "safety_flags": []
            }
        }
