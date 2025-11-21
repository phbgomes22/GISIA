from langgraph.graph import MessagesState, StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from typing import List
from dotenv import load_dotenv
import os
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

def format_docs(docs: List[Document]) -> str:
        """
        Joins chunk texts with double‐newline. Used to create a single `context` string.
        """
        return "\n\n".join(doc.page_content for doc in docs)

# (A) Define a custom State with all fields we need
class RAGState(MessagesState):
    query: str = ""
    prompt_template: PromptTemplate 
    retriever_k: int
    filter_list: List[str]

    raw_chunks: List[Document] = []
    formatted_context: str = ""
    rag_stream: any = None
    sources: List[str] = []
    answer:str = ""
    vectorstore: Chroma = None

    def __init__(self, query: str = "", prompt_template: PromptTemplate = None, retriever_k: int = 4, filter_list: List[str] = None, vectorstore: Chroma = None):
        super().__init__()
        self.query = query
        self.prompt_template = prompt_template
        self.retriever_k = retriever_k
        self.filter_list = filter_list if filter_list is not None else []
        self.raw_chunks = []
        self.formatted_context = ""
        self.rag_stream = None
        self.sources = []
        self.answer = ""
        self.vectorstore = vectorstore

# (1) Node: retrieve top‐K chunks from Chroma
def retrieve_chunks(state: RAGState) -> List[Document]:
    retriever = state["vectorstore"].as_retriever(
        search_kwargs={"k": state["retriever_k"]}
    )

    docs: List[Document] = retriever.invoke(state["query"])
    state["raw_chunks"] = docs
    return state

# (2) Node: format those chunks into one big string

def format_context(state: RAGState) -> str:
    ctxt = format_docs(state["raw_chunks"])
    state["formatted_context"] = ctxt
    return ctxt

# (3) Node: call the LLM in streaming mode

def call_llm_stream(state: RAGState):
    if state["messages"]:
        hist_lines = []
        for turn in state["messages"]:
            if (type(turn) is HumanMessage):
                role = "user"
            elif (type(turn) is AIMessage):
                role = "assistant"
            # print("turn", turn)
            cont = turn.content
            # e.g. "User: How does X work?"
            hist_lines.append(f"{role.capitalize()}: {cont}")
        history_str = "\n".join(hist_lines) + "\n\n"
    else:
        history_str = ""

    # (b) Combine history + retrieved-docs
    formatted_context = history_str + state["formatted_context"]

    state["formatted_context"] = formatted_context

    rag_chain = (RunnablePassthrough.assign(context=(lambda x: state["formatted_context"]))
                 | state["prompt_template"]
                 | ChatOpenAI(model_name="gpt-4o-mini", api_key=openai_key)
                 | StrOutputParser()
    )
    
    answer: str = rag_chain.invoke({"question": state["query"], "context": state["formatted_context"]})
    state["answer"] = answer


    stream_gen = rag_chain.stream({"question": state["query"]})
    state["rag_stream"] = stream_gen
    return state

# (4) Node: collect “sources” from metadata

def collect_sources(state: RAGState) -> List[str]:
    sources: List[str] = []
    for chunk in state["raw_chunks"]:
        src = chunk.metadata.get("source", "")
        if "page" in chunk.metadata:
            src += f"\n\nPage {chunk.metadata['page']}"
        sources.append(src)
    state["sources"] = sources
    return state

# (5) Node: run them all in order
def run_pipeline(state: RAGState) -> RAGState:
    _ = retrieve_chunks(state)
    _ = format_context(state)
    _ = call_llm_stream(state)
    _ = collect_sources(state)
    return state

graph = StateGraph(RAGState)
graph.add_node("run_pipeline", run_pipeline)
graph.add_edge("run_pipeline", "__end__")
graph.set_entry_point("run_pipeline")
# memory = MemorySaver()
# config = {"configurable": {"thread_id": "abc123"}}
app = graph.compile()
