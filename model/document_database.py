from typing import List, Dict
import os
import re
import json
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langgraph.graph import MessagesState, StateGraph
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

import config
from model.database import Database

load_dotenv(override=True)
openai_key = os.getenv("OPENAI_API_KEY")


class DocumentDatabase(Database):
    """
    A Database subclass that:
      1. Builds/loads a Chroma vectorstore of PDF chunks.
      2. Constructs a small LangGraph graph to:
         - Receive a user query,
         - Retrieve top-k chunks,
         - Format them,
         - Send them (with the prompt template) into a ChatOpenAI call,
         - Stream back "rag_stream" plus "sources."
    """

    # def __init__(self, chroma_db: Chroma = None, file_path: str = "data/"):
    #     """
    #     If `chroma_db` is provided, we reuse it; otherwise, build a new one from `file_path`.
    #     Then we build the LangGraph pipeline.
    #     """
    #     super().__init__()
    #     print("chromadb", chroma_db)
    #     self.file_path = file_path
    #     self._initialize(chroma_db)
    #     self._build_graph()

    def _initialize(self, chroma_db: Chroma, file_path):
        """
        — If an existing Chroma is passed in, reuse it.
        — Otherwise, either load from disk or build from scratch.
        """
        if chroma_db:
            self.vectorstore = chroma_db
            existing_metadatas = self.vectorstore.get()["metadatas"]
            existing_docs = {meta["source"] for meta in existing_metadatas if "source" in meta}
            print("✅ Document Database instantiated with existing document count:", len(existing_docs))
        else:
            self._create_chroma_db(file_path=file_path)
        self._build_graph()
        self.file_path = file_path


    def _create_chroma_db(self, file_path="data/", text_splitter=None, loader=None):
        # Load existing database if it exists
        if os.path.exists(config.PERSIST_DIRECTORY):
            print("Loading existing vector database...")
            self.vectorstore = Chroma(persist_directory=config.PERSIST_DIRECTORY, embedding_function=OpenAIEmbeddings())
            existing_metadatas = self.vectorstore.get()["metadatas"]
            existing_docs = {meta["source"] for meta in existing_metadatas if "source" in meta}  # Use source as ID
        else:
            print("No existing database found. Creating a new one...")
            self.vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory=config.PERSIST_DIRECTORY)
            existing_docs = set()

        print(f"Existing document count: {len(existing_docs)}")

        # Get all PDF paths
        all_documents = [
            os.path.join(root, file)
            for root, _, files in os.walk(file_path)
            for file in files if file.endswith(".pdf")
        ]

        # Filter out ones already indexed
        new_documents = [doc for doc in all_documents if doc not in existing_docs]

        print(f"Found {len(all_documents)} PDFs in total.")
        print(f"+++ New PDFs to process: {len(new_documents)}")

        if not new_documents:
            print("No new documents to add.")
            return

        new_splits = []
        subjects = [f.path for f in os.scandir(file_path) if f.is_dir()]

        for i, document_path in enumerate(new_documents):
            print(f"+++ Processing document {i+1}/{len(new_documents)}: {document_path}")
            loader = PDFPlumberLoader(file_path=document_path)
            docs = loader.load()
            for doc in docs:
                for subject in subjects:
                    if subject in document_path:
                        doc.metadata['subject'] = subject
                doc.metadata["source"] = document_path  # Track source

            if text_splitter is None:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

            # new_splits.extend(text_splitter.split_documents(docs))

        # Persist the updated topics.json for filtering in UI
        self._save_topics_json()

        # Add new chunks into Chroma and persist
        print(f"Adding {len(new_splits)} new documents to the vector database...")
        self.vectorstore.add_documents(new_splits)
        self.vectorstore.persist()
        print("Vector database update complete.")

    def _save_topics_json(self, output_folder: str = "./chroma_db"):
        """
        Write out a JSON file listing all subfolder names under self.file_path.
        This powers the Streamlit filter UI.
        """
        topics = [f.path for f in os.scandir(self.file_path) if f.is_dir()]
        topics_clean = [re.search(r"[^/]+$", topic).group() for topic in topics]
        topics_json_path = os.path.join(output_folder, "topics.json")

        if os.path.exists(topics_json_path):
            with open(topics_json_path, "r", encoding="utf-8") as f:
                existing_topics = json.load(f)
        else:
            existing_topics = []

        updated_topics = list(set(existing_topics + topics_clean))
        with open(topics_json_path, "w", encoding="utf-8") as f:
            json.dump(updated_topics, f, ensure_ascii=False, indent=4)

        print(f"✅ Topics updated and saved to {topics_json_path}")

    def format_docs(self, docs: List[Document]) -> str:
        """
        Joins chunk texts with double‐newline. Used to create a single `context` string.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def _build_graph(self, prompt_tpl, retriever_k, filter_list):
        """
        Construct a LangGraph graph that:
          • Takes a user query + user‐edited prompt template fields,
          • Retrieves top‐K chunks from Chroma,
          • Joins them into one big “context” string,
          • Sends that into ChatOpenAI (gpt‐4o‐mini),
          • Streams back a “rag_stream” plus “sources.”
        """

         # (A) Define a custom State with all fields we need
        class RAGState(MessagesState):
            query: str = ""
            prompt_tpl: PromptTemplate = None
            retriever_k: int = 1
            filter_list: List[str] = []

            raw_chunks: List[Document] = []
            formatted_context: str = ""
            rag_stream: any = None
            sources: List[str] = []
            answer = ""
            history = []

        self.graph = StateGraph(RAGState)
        
        # (1) Node: retrieve top‐K chunks from Chroma
        def retrieve_chunks(state: RAGState) -> List[Document]:
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": retriever_k}
            )
            # If you want filtering by “subject,” uncomment below:
            # retriever = self.vectorstore.as_retriever(
            #     search_kwargs={
            #         "k": state.retriever_k,
            #         "filter": {"subject": {"$in": state.filter_list}}
            #     }
            # )
            docs: List[Document] = retriever(state.query)
            state.raw_chunks = docs
            return docs

        # (2) Node: format those chunks into one big string
        
        def format_context(state: RAGState) -> str:
            ctxt = self.format_docs(state.raw_chunks)
            state.formatted_context = ctxt
            return ctxt

        # (3) Node: call the LLM in streaming mode
        
        def call_llm_stream(state: RAGState):

            if state.history:
                hist_lines = []
                for turn in state.history:
                    role = turn["role"]
                    cont = turn["content"]
                    # e.g. "User: How does X work?"
                    hist_lines.append(f"{role.capitalize()}: {cont}")
                history_str = "\n".join(hist_lines) + "\n\n"
            else:
                history_str = ""

            # (b) Combine history + retrieved-docs
            combined_context = history_str + state.formatted_context

            state.formatted_context = combined_context

            rag_chain = (
                RunnablePassthrough.assign(context=(lambda x: state.formatted_context))
                | prompt_tpl
                | ChatOpenAI(model_name="gpt-4o-mini", api_key=openai_key)
                | StrOutputParser()
            )

            full_answer: str = rag_chain.invoke({"question": state.query})
            state.answer = full_answer


            stream_gen = rag_chain.stream({"question": state.query})
            state.rag_stream = stream_gen
            return stream_gen

        # (4) Node: collect “sources” from metadata
        
        def collect_sources(state: RAGState) -> List[str]:
            sources: List[str] = []
            for chunk in state.raw_chunks:
                src = chunk.metadata.get("source", "")
                if "page" in chunk.metadata:
                    src += f"\n\nPage {chunk.metadata['page']}"
                sources.append(src)
            state.sources = sources
            return sources

        # (5) Node: run them all in order
        def run_pipeline(state: RAGState):
            _ = retrieve_chunks(state)
            _ = format_context(state)
            _ = call_llm_stream(state)
            _ = collect_sources(state)
            return state
        
        
        self.graph.add_node('pipeline', run_pipeline)

        self.graph.set_entry_point('pipeline')

    def run_rag(self,
                query: str,
                prompt_tpl: PromptTemplate,
                retriever_k: int = 1,
                filter_list: List[str] = None) -> Dict:
        """
        Public method that any controller/UI can call:
          • It takes the raw user query + a PromptTemplate instance
          • It triggers LangGraph, returns a dict with:
                { "query": query,
                  "rag_stream": <generator>,
                  "sources": [list of source‐strings]
                }
        """
        if filter_list is None:
            filter_list = []

        # Initialize a fresh state
        initial_state = {
            "query": query,
            "prompt_tpl": prompt_tpl,
            "retriever_k": retriever_k,
            "filter_list": filter_list,
        }

        # invoke the graph (this will run the `generate_rag` function)
        app = self.graph.compile()
        # new_state = self.graph.invoke(initial_state)
        new_state = app.invoke(initial_state)

        
        return {
            "query": new_state.query,
            "rag_stream": new_state.rag_stream,
            "rag_text": new_state.answer, 
            "sources": new_state.sources
        }
