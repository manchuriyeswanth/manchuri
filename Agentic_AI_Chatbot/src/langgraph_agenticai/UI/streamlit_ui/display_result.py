import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import json


class DisplayResultStreamlit:
    def __init__(self,usecase, graph, user_message):
        self.usecase = usecase
        self.graph = graph
        self.user_message = user_message

    def display_result_on_ui(self):
        usecase = self.usecase
        graph = self.graph
        user_message = self.user_message
        print(user_message)
        if usecase == "Basic Chatbot":
            for event in graph.stream ({'messages':("user",user_message)}):
                print(event.values())
                for value in event.values():
                    print(value['messages'])
                    with st.chat_message("user"):
                        st.write(user_message)
                    with st.chat_message("assistant"):
                        st.write(value["messages"].content)
        
        elif usecase== "Chatbot with Web":
            initial_state = {"messages":user_message}
            res = graph.invoke(initial_state)
            for mes in res['messages']:
                if type(mes) == HumanMessage:
                    with st.chat_message("user"):
                        st.write(mes.content)
                elif type(mes)== ToolMessage:
                    with st.chat_message("ai"):
                        st.write("Tool Call Start")
                        st.write(mes.content)
                        st.write("Tool Call END")
                elif type(mes)== AIMessage and mes.content :
                    with st.chat_message("assistant"):
                        st.write(mes.content)
