__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from model import DocumentDatabase
from view import *
from aux import document_loader, vector_db_loader
from controller import ChatController
import subprocess
import streamlit as st



def load_chromadb():
    print("++++ Downloading Vector Database from remote folder...")
    vector_db_loader.download_vector_database()

def download_documents():
    print("++++ Downloading Original Dataset Files from remote folder...")
    document_loader.download_dataset()

def main():
    # Check if the script and download tasks have already been done
    if "chromadb_loaded" not in st.session_state:
        # Load ChromaDB from remote source
        load_chromadb()
        st.session_state.chromadb_loaded = True  # Mark it as done

    if "documents_downloaded" not in st.session_state:
        # Download Original Documents
        download_documents()
        st.session_state.documents_downloaded = True  # Mark it as done

    # Initialize MVC components
    model = DocumentDatabase(file_path="data/Dominios sobre impacto socioambiental positivo")
    view = ChatView(file_path="data/Dominios sobre impacto socioambiental positivo")
    controller = ChatController(model, view)

    # Run the chat interface
    controller.run()

# Run the app
if __name__ == "__main__":
    main()