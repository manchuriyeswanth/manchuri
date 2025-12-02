from langgraph.graph import StateGraph, START,END
from src.state.state import State
from src.nodes.llmnode import LLMNode
class Graph:
    def __init__(self, model):
        self.llm = model
        self.graph = StateGraph(State)
    
    def add_graph(self):
        llmnode = LLMNode(self.llm)
        self.graph.add_node("llmnode",llmnode.process)
        self.graph.add_edge(START,"llmnode")
        self.graph.add_edge("llmnode",END)
    
    def build_graph(self):
        self.add_graph()
        return self.graph.compile()


