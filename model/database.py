from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, SeleniumURLLoader, PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


from abc import ABC, abstractmethod

import os
from dotenv import load_dotenv
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

class Database(ABC):
    _instance = None

    # Flag to track if the instance has been initialized
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
        return cls._instance

    def __init__(self, *args, **kwargs):
        if not type(self)._initialized:
            # Only initialize once
            self._initialize(*args, **kwargs)
            type(self)._initialized = True

    @abstractmethod
    def _initialize(self,*args, **kwargs):
        """Method for initialization logic, must be implemented by the subclass."""
        pass

    # @abstractmethod
    # def _setup_rag(self):
    #     raise NotImplementedError

    # @abstractmethod
    # def ask_rag(self, query, *args) -> dict:
    #     raise NotImplementedError
