from typing import List
# import promptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import config
import os
from model.database import Database
import re
import json
# from database import Database

from dotenv import load_dotenv
load_dotenv(override=True)

openai_key = os.getenv("OPENAI_API_KEY")

class DocumentDatabase(Database):

    paths = []

    def format_docs(self, docs: List[Document]):
        return "\n\n".join(doc.page_content for doc in docs)

    def _initialize(self, load=True, file_path="data/", text_splitter=None, loader=None):
        self.file_path = file_path
        # Load existing database if it exists
        if load:
            self.vectorstore = Chroma(persist_directory=config.PERSIST_DIRECTORY, embedding_function=OpenAIEmbeddings())
            return
        elif os.path.exists(config.PERSIST_DIRECTORY) and load:
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

        subjects = [f.path for f in os.scandir(file_path) if f.is_dir()]

         # Filter out already indexed documents
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

            new_splits.extend(text_splitter.split_documents(docs))

        # Saves the topics for filtering in the interface
        self._save_topics_json(file_path)

        # Add new documents to the vector store
        print(f"Adding {len(new_splits)} new documents to the vector database...")
        self.vectorstore.add_documents(new_splits)
        self.vectorstore.persist()  # Save the updated DB

        print("Vector database update complete.")


    def _save_topics_json(self, file_path, output_folder="./chroma_db"):
        topics = [f.path for f in os.scandir(file_path) if f.is_dir()]
        topics_clean = [re.search(r'[^/]+$', topic).group() for topic in topics]

        topics_json_path = os.path.join(output_folder, "topics.json")

        # Load existing topics if the file exists
        if os.path.exists(topics_json_path):
            with open(topics_json_path, "r", encoding="utf-8") as f:
                existing_topics = json.load(f)
        else:
            existing_topics = []

        # Combine old and new topics, removing duplicates
        updated_topics = list(set(existing_topics + topics_clean))

        # Save back to JSON
        with open(topics_json_path, "w", encoding="utf-8") as f:
            json.dump(updated_topics, f, ensure_ascii=False, indent=4)

        print(f"✅ Topics updated and saved to {topics_json_path}")

        
    def _setup_rag(self, chain_params, prompt: PromptTemplate, **kwargs):
        
        if "filter_dict" in kwargs.keys():
            print("Filter dict in setup: ", kwargs["filter_dict"])
            filter_dict = kwargs["filter_dict"]
            for i in range(len(filter_dict["filters"])):
                full_filter = self.file_path + "/" + filter_dict["filters"][i]
                filter_dict["filters"][i] = full_filter
            # print("Full Filter dict in setup: ", filter_dict)
            retriever = self.vectorstore.as_retriever(
                
                search_kwargs={
                    "k": chain_params["retriever_k"],
                    "filter": {"subject": {"$in":filter_dict["filters"]}},
                }
            )
        else:
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": chain_params["retriever_k"]}
            )

        llm = ChatOpenAI(model_name="gpt-4o-mini", api_key=openai_key)

        rag_chain_from_docs = ( RunnablePassthrough.assign(
            context=(lambda x: self.format_docs(x["context"])))
            | prompt
            | llm
            | StrOutputParser()
        )

        rag_chain_with_source = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

        return rag_chain_with_source

    def ask_rag(self, query, prompt: PromptTemplate, debug=False, *args, **kwargs) -> dict:
        print("args = ",args)
        # kwargs = len(args) > 0
        chain_params = {}
        output_format = "string"
        if len(args) > 0:
            chain_params = args[0]
            if len(args) > 1:
                output_format = args[1]
        filter_dict = {}
        if "filter_dict" in kwargs:
            filter_dict = kwargs["filter_dict"]
            # print("Filter dict: ", filter_dict)
            rag_chain = self._setup_rag(chain_params, prompt, filter_dict=filter_dict)
        else:
            rag_chain = self._setup_rag(chain_params, prompt)
        llm = ChatOpenAI(model_name="gpt-4o-mini", api_key=openai_key)
        if debug:
            fake_docs = [Document(page_content="CONTEXT", metadata={"source":"SOURCE"+str(i)}) for i in range(1, chain_params["retriever_k"]+1)]
            responses = {"query": query, "llm": "LLM ANSWER", "rag": {"answer":"RAG ANSWER", "context": fake_docs}} 
            return responses
        if output_format == "stream":
            # Get sources
            sources = []
            context = rag_chain.invoke(query)["context"]
            if len(context) == 0:
                print("No context found!!!!!!")
            answer_chain = rag_chain.pick("answer")
            for i in range(len(context)):
                print("Context: ",context[i].metadata["source"])
                sources.append(context[i].metadata["source"])
                if "page" in context[i].metadata:
                    sources[i] += "\n\n Pagina " + str(context[i].metadata["page"])

            # llm.stream(query) - not asking llm
            responses = {"query": query, "llm_stream": "", "rag_stream": answer_chain.stream(query), "sources": sources}
        else:
            # llm.invoke(query).content - not asking llm
            responses = {"query": query, "llm": "", "rag": rag_chain.invoke(query)}
        return responses
