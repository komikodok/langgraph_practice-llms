from langgraph.graph import StateGraph, START, END
from node import (
    State,
    retrieve,
    documents_grader,
    decide_to_generate,
    create_context,
    generate,
    hallucinations_grader,
    transform_query,
    # web_search
)


workflow = StateGraph(State)

workflow.add_node("retrieve", retrieve)
workflow.add_node("documents_grader", documents_grader)
workflow.add_node("create_context", create_context)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
# workflow.add_node("web_search", web_search)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "documents_grader")
workflow.add_conditional_edges(
    "documents_grader",
    decide_to_generate,
    {
        "generate": "generate",
        "create_context": "create_context"
    }
)
workflow.add_edge("create_context", "generate")
workflow.add_conditional_edges(
    "generate",
    hallucinations_grader,
    {
        "transform_query": "transform_query",
        "__end__": END
    }
)
workflow.add_edge("transform_query", "generate")# "web_search")
# workflow.add_edge("web_search", "generate")

