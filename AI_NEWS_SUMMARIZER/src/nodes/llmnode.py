from src.llm.groqllm import GroqLLM
from src.state.state import State
class LLMNode:
    def __init__(self,model):
        self.llm = model
        #self.llm = self.llm.groq_llm_model()
    
    def process(self, state:State)->dict:
        return {"messages":self.llm.invoke(state["messages"])}


