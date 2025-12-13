from typing import TypedDict, Any
from langgraph.graph import StateGraph, END

from router import router_node, safety_response_node
from planner import planner_node
from retriever import retriever_node
from answerer import answerer_node

class GraphState(TypedDict, total=False):
    user_input: str
    router: dict[str, Any]
    planner: dict[str, Any]
    retriever: dict[str, Any]
    final_answer: str
    citations: list[str]
    grounded: bool

def should_route_to_safety(state):
    """Conditional function to check if route is unsafe.
    Returns a key that maps to the next node."""
    router_output = state.get("router", {})
    if not router_output:
        return "safe"  # Default to safe if router output is missing
    
    route = router_output.get("route", "general")
    # Return a key that will be used in the mapping
    # Only "unsafe" route goes to safety_response, all others go to planner
    if route == "unsafe":
        return "unsafe"
    else:
        return "safe"

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("router", router_node)
    graph.add_node("safety_response", safety_response_node)
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("answerer", answerer_node)

    graph.set_entry_point("router")

    # Conditional routing: if unsafe, go to safety_response, otherwise continue to planner
    graph.add_conditional_edges(
        "router",
        should_route_to_safety,
        {
            "unsafe": "safety_response",
            "safe": "planner"
        }
    )
    
    graph.add_edge("safety_response", END)
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "answerer")
    graph.add_edge("answerer", END)

    return graph.compile()
