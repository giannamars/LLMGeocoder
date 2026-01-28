# src/graph.py 

from langgraph.graph import StateGraph, END
from .nodes import State, analyze_query, retrieve_em, detect_accession_numbers

def build_llm_graph(llm, prompt):
    graph = StateGraph(State)

    graph.add_node("analyze_query", analyze_query)

    # Wrapper that injects the LLM and the PromptTemplate into the retrieve node
    async def async_retrieve(state):
        return await retrieve_em(state, llm, prompt)

    graph.add_node("retrieve", async_retrieve)
    graph.add_node("detect_accession", detect_accession_numbers)

    graph.set_entry_point("analyze_query")
    graph.add_edge("analyze_query", "retrieve") 
    graph.add_edge("retrieve", "detect_accession")
    graph.add_edge("detect_accession", END)

    return graph.compile()