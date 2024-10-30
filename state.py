from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from typing import (
    TypedDict, 
    Union,
    List,
    Literal
)

class State(TypedDict):

    documents: List[Document]
    question: str
    answer: str
    chat_history: List[Union[HumanMessage, AIMessage]]
    query_search: Literal["on-topic", "off-topic", None] = None