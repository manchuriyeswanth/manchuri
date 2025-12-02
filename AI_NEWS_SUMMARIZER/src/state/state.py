from typing_extensions import Annotated, List
from typing import TypedDict
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages:Annotated[List, add_messages]
