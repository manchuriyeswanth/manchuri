from langchain_groq import ChatGroq
import os
import streamlit as st
class GroqLLM:
    def __init__(self,model,api_key):
        self.model = model
        self.api_key = api_key
    
    def groq_llm_model(self):
        try :
            if self.model=="" or self.api_key=="" or os.environ['GROQ_API_KEY']=="":
                st.error("Error: Groq Model or API_KEY is not present")
                return
            llm =  ChatGroq(model=self.model, api_key = self.api_key)
        except Exception as e:
            st.error(f"Groq LLM model Load Failed:{e}")
            return
        print(llm)
        return llm