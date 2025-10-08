
from src.state.state import BlogState, Blog
from langchain_core.messages import HumanMessage, SystemMessage

class BlogNode:
    def __init__(self,llm):
        self.llm = llm
    
    def title_creation(self,state:BlogState):

        if "topic" in state and state["topic"]:
            prompt = """ 
                        You are an expert Blog Content Writer. Use MarkDown Formatting. Generate a Blog title for the {topic}.
                        This title should be creative and SEO friendly.
                        """

            system_message = prompt.format(topic = state["topic"])
            response = self.llm.invoke(system_message)
            return {"blog":{"title":response.content}}
        
    def content_generation(self,state:BlogState):
        if "topic" in state and state["topic"]:
            prompt = """ 
                        You are an expert Blog Content Writer. Use MarkDown Formatting. 
                        Generate detailed blog content for the {topic}.
                        """

            system_message = prompt.format(topic = state["topic"])
            response = self.llm.invoke(system_message)
            return {"blog":{"title":state["blog"]["title"],"content":response.content}}
    
    def translation(self,state:BlogState):
        prompt = """
                    Translate the following into {current_language}
                    -Maintain original tone , style and formatting
                    - Adapt cultural references and idioms appropriate for {current_language}
                    
                    ORIGINAL_CONTENT = 
                    {blog_content}
                    """
        
        blog_content = state["blog"]["content"]
        messages = prompt.format(current_language= state["curr_language"], blog_content = blog_content)
        translated_content = self.llm.with_structured_output(Blog).invoke(messages)
        return {"blog":{"title":state["blog"]["title"],"content":translated_content.content}}


    def route(self,state:BlogState):
        return {"curr_language":state["curr_language"]}
    
    def route_decision(self,state:BlogState):
        if state["curr_language"]=="hindi":
            return "hindi"
        elif state["curr_language"]=="french":
            return "french"
        else :
            return state["curr_language"]

        
        

                
    