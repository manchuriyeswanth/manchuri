import streamlit as st

class Display_Result:
    def __init__(self,user_input,graph,use_case):
        self.user_input = user_input
        self.graph = graph
        self.use_case = use_case
    
    def display(self):
        if self.use_case == "NEWS SUMMARIZER":
            for event in self.graph.stream({"messages":("user",self.user_input)}):
                print(event.values())
                for value in event.values():
                    print(value["messages"])
                with st.chat_message("user"):
                    st.write(self.user_input)
                with st.chat_message("assistant"):
                    st.write(value["messages"].content)