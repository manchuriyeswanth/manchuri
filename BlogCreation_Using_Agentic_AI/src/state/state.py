from typing import TypedDict
from pydantic import BaseModel, Field


class Blog(BaseModel):
    title : str = Field(description= "Title of Blog post")
    content : str = Field(description = "Main Content of Blog Post")


class BlogState(TypedDict):
    topic : str 
    blog : Blog
    curr_language : str 
