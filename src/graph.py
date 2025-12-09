from typing import TypedDict, Any
from langgraph.graph import StateGraph, END

from router import router_node
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

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("router", router_node)
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("answerer", answerer_node)

    graph.set_entry_point("router")

    graph.add_edge("router", "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "answerer")
    graph.add_edge("answerer", END)

    return graph.compile()
