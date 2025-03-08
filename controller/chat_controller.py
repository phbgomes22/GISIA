from model.database import Database
from view import *

class ChatController:
    def __init__(self, db: Database, view: ChatView):
        self.db = db
        self.view = view

    def run(self, debug=False):
        print("Running chat controller")
        last_input = None
        while True:
            user_input = self.view.get_text()
            user_edited_prompt = self.view.get_edited_prompt()
            retriever_k = self.view.retriever_k
            filter_dict = {"filters": self.view.get_search_filters()}
            if user_input != last_input and user_input != "" and user_input != '':
                # self.db.add_user_input(user_input)
                query_par = {"retriever_k": retriever_k}
                output_format = "stream"
                responses = self.db.ask_rag(user_input, user_edited_prompt, debug, query_par, output_format, filter_dict=filter_dict)
                last_input = user_input
                self.view.display(responses=responses)
            else:
                break