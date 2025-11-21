from typing import List, Dict
import os
import re
import json
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import config
from model.database import Database
from model.graph_chatbot import app

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

    def _initialize(self, chroma_db: Chroma, file_path, prompt_template: PromptTemplate = None, retriever_k: int = 1, filter_list: List[str] = None):
        """
        — If an existing Chroma is passed in, reuse it.
        — Otherwise, either load from disk or build from scratch.
        """
        self.file_path = file_path
        self.prompt_template = prompt_template
        self.retriever_k = retriever_k
        self.filter_list = filter_list

        if chroma_db:
            self.vectorstore = chroma_db
            existing_metadatas = self.vectorstore.get()["metadatas"]
            existing_docs = {meta["source"] for meta in existing_metadatas if "source" in meta}
            print("✅ Document Database instantiated with existing document count:", len(existing_docs))
        else:
            self._create_chroma_db(file_path=file_path)
        self._build_graph()

    def _create_chroma_db(self, file_path="data/", text_splitter=None, loader=None):
        # Load existing database if it exists
        if os.path.exists(config.PERSIST_DIRECTORY):
            print("Loading existing vector database...")
            self.vectorstore = Chroma(persist_directory=config.PERSIST_DIRECTORY, embedding_function=OpenAIEmbeddings())
            existing_metadatas = self.vectorstore.get()["metadatas"]
            existing_docs = {meta["source"] for meta in existing_metadatas if "source" in meta}  # Use source as ID
            print(existing_docs)
            return
        else:
            print("No existing database found. Creating a new one...")
            self.vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory=config.PERSIST_DIRECTORY)
            existing_docs = set()

        print(f"Existing document count: {len(existing_docs)}")

        # Get all PDF paths
        new_documents = [
            os.path.join(root, file)
            for root, _, files in os.walk(file_path)
            for file in files if file.endswith(".txt")
        ]

        print(f"+++ New PDFs to process: {len(new_documents)}")

        if not new_documents:
            print("No new documents to add.")
            return

        new_splits = []
        subjects = [f.path for f in os.scandir(file_path) if f.is_dir()]

        # Process .txt documents instead of PDFs
        for i, document_path in enumerate(new_documents):
            print(f"+++ Processing document {i+1}/{len(new_documents)}: {document_path}")
            
            loader = TextLoader(file_path=document_path)
            docs = loader.load()
            for doc in docs:
                for subject in subjects:
                    if subject in document_path:
                        doc.metadata['subject'] = subject
                doc.metadata["source"] = document_path  # Track source

            if text_splitter is None:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

            new_splits.extend(text_splitter.split_documents(docs))


        # Add new chunks into Chroma and persist
        print(f"Adding {len(new_splits)} new documents to the vector database...")
        self.vectorstore = Chroma.from_documents(documents=new_splits, embedding=OpenAIEmbeddings(),
                                 persist_directory=config.PERSIST_DIRECTORY)

        # Persist the updated topics.json for filtering in UI
        self._save_topics_json()
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

    def _build_graph(self):
        """
        Construct a LangGraph graph that:
          • Takes a user query + user‐edited prompt template fields,
          • Retrieves top‐K chunks from Chroma,
          • Joins them into one big “context” string,
          • Sends that into ChatOpenAI (gpt‐4o‐mini),
          • Streams back a “rag_stream” plus “sources.”
        """
        self.graph = app

    def run_rag(self,
                query: str, messages: List[str]) -> Dict:
        """
        Public method that any controller/UI can call:
          • It takes the raw user query + a PromptTemplate instance
          • It triggers LangGraph, returns a dict with:
                { "query": query,
                  "rag_stream": <generator>,
                  "sources": [list of source‐strings]
                }
        """
        # if filter_list is None:
        #     filter_list = []

        # Initialize a fresh state
        initial_state = {
            "query": query,
            "prompt_template": self.prompt_template,
            "retriever_k": self.retriever_k,
            "filter_list": self.filter_list,
            "raw_chunks": [],
            "formatted_context": "",
            "rag_stream": None,
            "sources": [],
            "answer": "",
            "combined_context": "",
            # "history": [],
            "vectorstore": self.vectorstore,
            "messages": messages
        }


        app = self.graph
        # new_state = self.graph.invoke(initial_state)
        new_state = app.invoke(initial_state)
        

        return {
            "query": new_state["query"],
            "rag_stream": new_state["rag_stream"],
            "rag_text": new_state["answer"], 
            "sources": new_state["sources"]
        }
