
from src.langgraph_agenticai.state.state import State

class Basic_Chatbot_Node:
    """Basic Chatbot Logic Implementation"""
    def __init__(self,model):
        self.llm = model
    
    def process(self,state:State)->dict:
        """Process a input state and generate the LLM response """
        return {"messages":self.llm.invoke(state['messages'])}