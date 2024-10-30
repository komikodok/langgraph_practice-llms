from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_community.tools.tavily_search import TavilySearchResults

from typing import Literal
from state import State
from database_client import (
    collection_documents,
    collection_chat_history
)
from dotenv import find_dotenv, load_dotenv
import os

load_dotenv(find_dotenv())

llm = ChatOllama(model=os.getenv("LLMs"))
embeddings = OllamaEmbeddings(model=os.getenv("LLMs"))

string_parser = StrOutputParser()

vectorstore = MongoDBAtlasVectorSearch(
    embedding=embeddings, 
    collection=collection_documents, 
    index_name="vector_index"
)

retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def retrieve(state: State):

    print("---RETRIEVE NODE---")
    question = state["question"]
    chat_history = state["chat_history"]
    
    if chat_history is None:
        chat_history = []
        chat_history.append(HumanMessage(content="Hello"))
        chat_history.append(AIMessage(content="Hello!"))

    docs = retriever.invoke(question)

    print(f"Here's the documents retrieved: \n{docs}")
    print()

    return {
        "documents": docs,
        "question": question,
        "chat_history": chat_history
    }

def documents_grader(state: State):

    print("---DOCUMENTS GRADER NODE---")
    documents = state["documents"]
    question = state["question"]
    query_search = state["query_search"]
    
    class DocumentsGrader(BaseModel):

        score: Literal["yes", "no"] = Field(
            description="Consider whether the answer is relevance or no, given the existing context, if relevance consider it 'yes'"
        )


    json_parser = JsonOutputParser(pydantic_object=DocumentsGrader)

    documents_grader_node_template = """
                You are expert AI assistant that can consider wheter the answer is relevance or no. \
                Answer with given the existing context, if relevance consider it 'yes', if is not relevance consider it 'no'. \
                
                Format instructions: {format_instructions}. \
                Example instructions: {example_instructions}. \
    """
    documents_grader_node_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", documents_grader_node_template),
            ("human", "Context: {context}. Question: {question}")
        ]
    )
    format_prompt = documents_grader_node_prompt.partial(
        format_instructions=json_parser.get_format_instructions(),
        example_instructions="""{"score": "yes"} or {"score": "no"}"""
    )

    grade_documents = (
        format_prompt
        | llm
        | json_parser
    )
    grade = grade_documents.invoke({"question": question, "context": documents})
    score = grade["score"]

    if score == "yes":
        query_search = "on-topic"
    else:
        query_search = "off-topic"

    print(f"Is query search on-topic or off-topic? \nHere the result: {query_search}")
    print()

    return {
        "documents": documents,
        "question": question,
        "query_search": query_search
    }

def decide_to_generate(state: State):
    
    print("---DECIDE TO GENERATE---")
    query_search = state["query_search"]

    next_step = 'generate' if query_search == 'on-topic' else 'create_context'
    print(f"Decide to next step, generate node or create new context before generate? \nHere the result: {next_step}")
    print()

    return next_step
    
def create_context(state: State):

    print("---CREATE CONTEXT NODE---")
    documents = state["documents"]
    question = state["question"]

    create_context_node_template = """
            You are expert AI assistant for give a context about the question. \ 
            Provide the context contained in the user's question. \
    """
    create_context_node_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", create_context_node_template),
            ("human", "{question}")
        ]
    )
    create_context_chain = (
        create_context_node_prompt
        | llm
        | string_parser
    )
    context = create_context_chain.invoke({"question": question})
    docs = Document(page_content=context)

    print("If query search is off-topic, then create new context before generate")
    print(f"Here's new context: {docs}")
    print()

    return {
        "documents": [docs],
        "question": question
    }

def generate(state: State):

    print("---GENERATE NODE---")
    documents = state["documents"]
    question = state["question"]
    answer = state["answer"]
    chat_history = state["chat_history"]

    generate_node_template = """
                You're helpful AI assistant. \
                You're an AI assistant for answer-question task. \
                Use the following pieces of retrieved context to answer the question. \
                Answer only based on existing context in retriever. \
                If the user question asks in English, the answer must be answered in English. \
                If the user question asks in Indonesian, the answer must be answered in Indonesian. \
                
    """
    generate_node_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", generate_node_template),
            MessagesPlaceholder("chat_history"),
            ("human", "Context: {context}. Question: {question}")
        ]
    )

    qa_chain = (
        generate_node_prompt 
        | llm 
        | string_parser
    )

    answer = qa_chain.invoke({"context": documents, "question": question, "chat_history": chat_history})

    human_message = HumanMessage(content=question)
    ai_message = AIMessage(content=answer)
    
    chat_history.append(human_message)
    chat_history.append(ai_message)

    print(f"LLM generation: {answer}")
    print()

    return {
        "documents": documents,
        "question": question,
        "answer": answer,
        "chat_history": chat_history,
    }

def hallucinations_grader(state: State):

    print("---HALLUCINATIONS GRADER---")
    documents = state["documents"]
    answer = state["answer"]

    class HallucinationsGrader(BaseModel):

        score: Literal["yes", "no"] = Field( 
            description="Consider it without calling external APIs for additional informations. Answer is suppoerted only by the facts, answer it 'yes' or 'no'"
        )


    json_parser = JsonOutputParser(pydantic_object=HallucinationsGrader)

    hallucinations_grader_node_template = """
                You are grader assesing wheter an LLM generation is supported by a set of retrieved facts. \
                Restricts yourself to give a binary score, either 'yes' or 'no'. If answer supported or partially supported by set of facts, consider it 'yes'. \
                Do not consider calling external APIs for additional informations as consistent with the facts. \
                
                Format instructions: {format_instructions}. \
                Example instructions: {example_instructions}. \
    """
    hallucinations_grader_node_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", hallucinations_grader_node_template),
            ("human", "Here set the facts: {facts}, Here the generation: {generation}")
        ]
    )
    format_prompt = hallucinations_grader_node_prompt.partial(
        format_instructions=json_parser.get_format_instructions(),
        example_instructions="""{"score": "yes"} or {"score": "no"}"""
    )

    grade_hallucinations = (
        format_prompt
        | llm
        | json_parser
    )
    
    grade = grade_hallucinations.invoke({"facts": documents, "generation": answer})
    score = grade["score"]

    print("If the generation gives hallucinations, then it will go to the transform_query node")
    print("If the generation doesn't gives hallucinations, then it will be end")
    print(f"Is generation doesn't gives hallucinations and supported/partially supported by set of the documents? {score}")
    print()

    return "transform_query" if score == 'no' else "__end__"

def transform_query(state: State):

    print("---TRANSFORM QUERY NODE---")
    question = state["question"]

    class TransformQuestion(BaseModel):

        transform_question: str = Field(
            description="Re-write the question that converts an input question to a better version that is optimized"
        )

    json_parser = JsonOutputParser(pydantic_object=TransformQuestion)

    transform_query_template = """
            You are question re-writer that converts an input question to a better version that is optimized. \
            Look at the input and try to reason about the underlying semantic intent or meaning. \
            Do not answer the question, then formulate a standalone question. \
            Just re-reformulate it if needed and otherwise return it as is. \
            
            Format instructions: {format_instructions}. \
            Example instructions: {example_instructions}. \
    """
    transform_query_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", transform_query_template),
            ("human", "{question}")
        ]
    )

    format_prompt = transform_query_prompt.partial(
        format_instructions=json_parser.get_format_instructions(),
        example_instructions="""{"transform_question": "<here the new question that is optimized>"}"""
    )

    transform_query_chain = (
        format_prompt
        | llm
        | json_parser
    )

    rewrite_question = transform_query_chain.invoke({"question": question})
    new_question = rewrite_question["transform_question"]

    print(f"Here news question transform: {new_question}")
    print()

    return {
        "question": new_question
    }

# def web_search(state: State):

#     print("---WEB SEARCH NODE---")
#     question = state["question"]

#     tool = TavilySearchResults()
#     result = tool.invoke({"query": question})
#     documents = [Document(page_content=docs.content) for docs in result]

#     print(f"Web search documents: {documents}")
#     print()

#     return {
#         "documents": documents,
#         "question": question
#     }