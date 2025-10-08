
from langgraph.graph import StateGraph, START, END
from src.langgraph_agenticai.state.state import State

from src.langgraph_agenticai.Node.basic_chatbot_node import Basic_Chatbot_Node
from src.langgraph_agenticai.tools.search_tool import get_tools, create_tool_node
from src.langgraph_agenticai.Node.chatbot_with_tool_node import Chatbot_with_tool
from langgraph.prebuilt import tools_condition


class GraphBuilder:
    def __init__(self,model):
        self.llm = model
        self.graph_builder = StateGraph(State)
    
    def chatbot_with_tools(self):
        """ Builds a chatbot with tools integration.
        This method creates a chatbot with tool node also included.
        """
        tools = get_tools()
        tool_node = create_tool_node()

        llm = self.llm
        chatbot_With_tool_object = Chatbot_with_tool(llm)
        chatbot_tools_node = chatbot_With_tool_object.create_chatbot(tools)
        self.graph_builder.add_node("chatbot",chatbot_tools_node)
        self.graph_builder.add_node("tools",tool_node)
        ## Define conditional and edges 
        self.graph_builder.add_edge(START,"chatbot")
        self.graph_builder.add_conditional_edges("chatbot",tools_condition)
        self.graph_builder.add_edge("tools","chatbot")
        self.graph_builder.add_edge("chatbot",END)



    
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
        if usecase == "Chatbot with Web":
            self.chatbot_with_tools()
        return self.graph_builder.compile()
    
