__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# from model import Database
from model import DocumentDatabase
from view import *
from controller import ChatController
import subprocess



if __name__ == "__main__":
    # Load ChromaDB from remote source
    try:
        result = subprocess.run(["./loader_vector_db.sh", "--load"], check=True, text=True, capture_output=True)
        print("****** Script output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("****** Error:", e.stderr)

    # Initialize MVC components
    model = DocumentDatabase(file_path="data/Dominios sobre impacto socioambiental positivo")
    view = ChatView(file_path="data/Dominios sobre impacto socioambiental positivo")
    controller = ChatController(model, view)

    # Run the chat interface
    controller.run()
    # view.display()