import streamlit as st

from src.langgraph_agenticai.UI.streamlit_ui.loadui import LoadStreamLitUI
from src.langgraph_agenticai.LLM.groqllm import GroqLLM
from src.langgraph_agenticai.graph.graph_builder import GraphBuilder
from src.langgraph_agenticai.UI.streamlit_ui.display_result import DisplayResultStreamlit

def load_langgraph_agenticai_app():
    """ Loads and runs the Agentic AI application using Streamlit 
    This function initializes UI, handles user input and configures LLM Model"""

    ui = LoadStreamLitUI()   # Class
    user_input = ui.load_streamlit_ui()    ## Class Function

    if not user_input:
        st.error("Error: Failed to load userinput from UI")
        return 
    
    user_message = st.chat_input("Enter your Message:")
    print(user_message)
    if user_message:
        try :
            ## Configure the LLM 
            obj_llm_config = GroqLLM(user_input)
            model = obj_llm_config.get_llm_model()

            if not model :
                st.error("LLM model couldnt be initialized")
                return 
            
            ## Initialize Graph based on use Case

            use_case = user_input["selected_usecase"]

            if not use_case:
                st.error("No use case selected")
                return
            
            graph_builder = GraphBuilder(model)

            try :
                graph = graph_builder.setupgraph(use_case)
                print(user_message)
                DisplayResultStreamlit(use_case,graph, user_message).display_result_on_ui()

            except Exception as e:
                st.error(f"Error Graph Setup Failed: {e}")
                return 
        
        except Exception as e:
            st.error(f"No user input: {e}")
            return
            

