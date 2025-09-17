from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode

def get_tools():
    """Return the list of tools with chatbot"""
    tools = [TavilySearchResults(max_results=2)] # List of tools 
    return tools


def create_tool_node():
    """
    Create and returns a tool node for the graph 
    """
    return ToolNode(tools=get_tools())