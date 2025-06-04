from model import DocumentDatabase
from view import ChatView  # assuming chat_view.py lives under view/
# from document_database import DocumentDatabase


class ChatController:
    """
    Orchestrates between Streamlit (ChatView) and our LangGraph‐powered DocumentDatabase.
    """

    def __init__(self, db: DocumentDatabase, view: ChatView):
        self.db = db
        self.view = view
        self.last_input = ""
        self.history = []
        self.prompt = self.view.get_edited_prompt()
        self.retriever_k = self.view.retriever_k
        self.filters = self.view.get_search_filters()
        self.user_edited_prompt = self.view.get_edited_prompt()
        self._update_query()


    def _update_query(self):
        self.db.prompt_template = self.prompt
        self.db.retriever_k = self.retriever_k
        self.db.filter_list = self.filters


    def run(self, debug: bool = False):
        """
        Main loop: whenever the user submits a new question, we:
          1. Collect: user_input, edited_prompt (PromptTemplate), retriever_k, filters.
          2. Pass them into `db.run_rag(...)`.
          3. Get back a streaming generator + sources → hand off to the view.
        """
        while True:
            user_input = self.view.get_text()
            self.user_edited_prompt = self.view.get_edited_prompt()
            self.retriever_k = self.view.retriever_k
            self.filter_dict = {"filters": self.view.get_search_filters()}

            # Only proceed if the user typed something new and nonempty
            if user_input and user_input != self.last_input:
                self.last_input = user_input

                self.history.append({"role": "user", "content": user_input})

                # Unpack our filter list:
                flist = self.filter_dict["filters"]

                # Ask our LangGraph‐powered RAG engine to stream an answer:
                rag_response = self.db.run_rag(
                    query=user_input,
                )

                # rag_response = { "query": ..., "rag_stream": <generator>, "sources": [...] }
                # Hand that off to the view for display:
                self.view.display(responses=rag_response)

                assistant_text = rag_response["rag_text"]
                self.history.append({"role": "assistant", "content": assistant_text})

            else:
                # If user_input is empty, or unchanged, just break out
                break
