
from langgraph.graph import StateGraph, START, END
from src.langgraph_agenticai.state.state import State

from src.langgraph_agenticai.Node.basic_chatbot_node import Basic_Chatbot_Node


class GraphBuilder:
    def __init__(self,model):
        self.llm = model
        self.graph_builder = StateGraph(State)
    
    def basic_chatbot_build_graph(self):
        """
        Builds a basic Chatbot using  Langgraph
        This method initializes a chatbot node using the 'BasicChatbotNode' class
        and integrates it into the graph. The Chat bot node is set as entry and exit of the graph
        """
        self.basic_chatbot = Basic_Chatbot_Node(self.llm)
        self.graph_builder.add_node("chatbot",self.basic_chatbot.process)
        self.graph_builder.add_edge(START,"chatbot")
        self.graph_builder.add_edge("chatbot",END)
    
    def setupgraph(self, usecase :str):

        if usecase == "Basic Chatbot":
            self.basic_chatbot_build_graph()
        return self.graph_builder.compile()
