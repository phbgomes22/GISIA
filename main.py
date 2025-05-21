__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from model import DocumentDatabase
from view import *
from auxiliary import document_loader, vector_db_loader
from controller import ChatController
import subprocess
import streamlit as st
from st_files_connection import FilesConnection
from langchain_chroma import Chroma
import os
from langchain_openai import OpenAIEmbeddings


@st.cache_resource
def get_chroma_db():
    """Downloads the ChromaDB SQLite database from S3 and stores it in a folder."""
    
    S3_CHROMADB_KEY = os.getenv("S3_CHROMADB_KEY")
    LOCAL_DB_FOLDER = "chroma_db"
    LOCAL_DB_PATH = os.path.join(LOCAL_DB_FOLDER, "chroma.sqlite3")

    os.makedirs(LOCAL_DB_FOLDER, exist_ok=True)  # Ensure folder exists

    print("📥 Downloading ChromaDB from S3...")

    s3 = st.connection('s3', type=FilesConnection)
    # Open S3 file and read its contents
    with s3.open(S3_CHROMADB_KEY, mode="rb") as downloaded_file:
        file_data = downloaded_file.read()  # Read as bytes
    
    # Write the file to the local directory
    with open(LOCAL_DB_PATH, "wb") as f:
        f.write(file_data)


    print("✅ ChromaDB Loaded from Cache!")
    return Chroma(persist_directory=LOCAL_DB_FOLDER, embedding_function=OpenAIEmbeddings())

def main():

    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    
    # Load the database once and use it persistently
    chroma_db = get_chroma_db()

    view = ChatView(file_path="")


    # Initialize MVC components
    model = DocumentDatabase(chroma_db=chroma_db, file_path="data/Dominios sobre impacto socioambiental positivo")
    controller = ChatController(model, view)

    # Run the chat interface
    controller.run()

# Run the app
if __name__ == "__main__":
    main()