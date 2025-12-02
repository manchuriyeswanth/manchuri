from src.ui.loaduiconfig import Config
import streamlit as st
import os 
class LoadSteamlitUI:
    def __init__(self):
        self.user_controls = {}
        self.config = Config()
    
    def load_streamlit(self):

        st.set_page_config(page_title=" "+self.config.get_page_title(),layout="wide")
        st.header(""+self.config.get_page_title())

        with st.sidebar:

            llm_options = self.config.get_llm_options()
            usecase_optiions = self.config.get_use_case()

            self.user_controls["SELECTED_LLM"] = st.selectbox("Select LLM", llm_options)

            if self.user_controls["SELECTED_LLM"] == "GROQ":
                model_options = self.config.get_llm_model()
                self.user_controls["SELECTED_GROQ_MODEL"] = st.selectbox("Select Groq Model", model_options)
                os.environ["GROQ_API_KEY"]=self.user_controls["GROQ_API_KEY"]= st.session_state['GROQ_API_KEY']=st.text_input("API_KEY", type="password")
            self.user_controls["SELECTED_USECASE"] = st.selectbox("Select UseCase", usecase_optiions)
        


        return self.user_controls

