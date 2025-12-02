import streamlit as st

from src.ui.streamlit.displayui import LoadSteamlitUI
from src.ui.streamlit.display_result import Display_Result
from src.nodes.llmnode import LLMNode
from src.llm.groqllm import GroqLLM
from src.graph.graphbuilder import Graph
def AI_news_summarizer():
    UI = LoadSteamlitUI()

    user_input = UI.load_streamlit()
    print(user_input)
    if not user_input:
        st.error("Error: Failed to load user input from AI")
        return
    
    user_message = st.chat_input("Enter the text:")
    print(user_message)
    if user_message:
        try:
            llm_model = GroqLLM(model = user_input["SELECTED_GROQ_MODEL"], api_key = user_input["GROQ_API_KEY"] ).groq_llm_model()
            if not llm_model:
                st.error("LLM Loading Failed")
                return

            graph_builder = Graph(llm_model)
            try:
                graph = graph_builder.build_graph()
                Display_Result(user_message,graph,user_input["SELECTED_USECASE"]).display()
            except Exception as e:
                st.error(f"Error: Graph Failed: {e}")
                return
        except Exception as e:
            st.error(f"No user input: {e}")
            return


