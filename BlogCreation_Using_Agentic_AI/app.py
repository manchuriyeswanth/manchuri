import uvicorn
from fastapi import FastAPI, Request

from src.graph.graphbuilder import GraphBuilder
from src.llm.groqllm import GroqLLM

import os 

from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

os.environ["LANGSMITH_API_KEY"]= os.getenv("LANGCHAIN_API_KEY")

## API 

@app.post("/blogs")
async def create_blogs (request : Request):
    data = await request.json()
    topic = data.get("topic","")
    language = data.get("language","")

    ## get llm

    groqllm = GroqLLM()
    llm = groqllm.get_llm()

    graph_builder = GraphBuilder(llm)
    if topic and language :
        graph = graph_builder.setup_graph(usecase="topic")
        state = graph.invoke({"topic":topic,"curr_language":language.lower()})
    elif topic: 
        graph = graph_builder.setup_graph(usecase="topic")
        state = graph.invoke({"topic":topic})
    
    
    return {"data":state}


if __name__=="__main__":
    uvicorn.run("app:app",host="0.0.0.0",reload="True",port=8000)