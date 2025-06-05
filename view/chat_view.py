import os
import re
import json
import urllib.parse

import streamlit as st
from langchain.prompts import PromptTemplate

from streamlit_chat import message
from dotenv import load_dotenv

import config


class ChatView:
    """
    Exactly the same Streamlit‐based UI you already had,
    except now it expects `responses` to be a dict containing:
      • "rag_stream"  → a streaming generator from LangGraph
      • "sources"     → a list of source‐strings to show as links
    """

    def __init__(self, file_path: str = None):
        load_dotenv()

        # Top‐left logo
        st.columns(1)[0].image("view/images/iarisLogo.jpeg", width=80)

        # Sidebar: slider + multi‐select of topics
        with st.sidebar:
            st.title("⚙️ Chatbot Settings")

            with st.expander("🔍 Search Parameters", expanded=False):
                self.values = st.slider("Search in how many documents", 1, 10)
                topics = self._load_topics_json()
                self.search_filter = st.multiselect(
                    "Filter by", topics, default=topics, format_func=self._format_topic
                )

            # Let user tweak the PromptTemplate
            self.user_prompt = st.text_area(
                "Edit the chatbot's prompt template",
                '''You are IAris, a concise chatbot expert in the social, cultural and environment impact that will advice leaders that want to build social businesses.
## The client asks you the following question: "{question}"
## You have to provide an answer based on the following documents: "{context}"
Your answer should only be based on the documents provided.
Be provocative and ask one follow-up inquiry to question the client, making sure that gaps are considered.''',
                height=400,
            )

        self.promptTemplate = PromptTemplate.from_template(self.user_prompt)
        self.retriever_k = 1
        self.key = 0

        # Placeholders for messages
        self.human_message = st.chat_message("🙋")
        self.rag_message = st.chat_message("🤖")
        self.sources_tab = st.empty()

        # Text input form
        with st.form(key="input_form", border=False):
            self.user_input = st.text_area("", "", key="input", placeholder="Pergunte alguma coisa")
            self.submit_button = st.form_submit_button(label="Enviar")

        # Initialize session state for caching messages
        if "user_input" not in st.session_state:
            self._init_session_state()

    def _init_session_state(self):
        st.session_state["user_input"] = []
        st.session_state["rag_stream"] = None
        st.session_state["rag_generated"] = []
        st.session_state["sources"] = None

    @staticmethod
    def _format_topic(topic: str) -> str:
        return topic.replace("_", " ").title()

    def _load_topics_json(self):
        topics_json_path = config.TOPICS_FILE
        if not os.path.exists(topics_json_path):
            print("⚠️ topics.json not found. Check remote S3 ChromaDB.")
            return []
        with open(topics_json_path, "r", encoding="utf-8") as f:
            topics_clean = json.load(f)
        print("✅ Topics loaded for filtering.")
        return topics_clean

    def get_text(self) -> str:
        # update retriever_k from slider
        self.retriever_k = self.values
        return self.user_input

    def get_edited_prompt(self) -> PromptTemplate:
        # user may have modified the prompt text
        return PromptTemplate.from_template(self.user_prompt)

    def get_search_filters(self):
        return self.search_filter

    def display(self, responses: dict = None):
        """
        Show the new human message + RAG answer (streaming) + sources.
        responses is expected to contain:
           { "query": str,
             "rag_stream": <generator>,
             "sources": [file‐path, ...]
           }
        """
        if self.user_input:
            st.session_state["user_input"].append(self.user_input)
            if responses:
                # print("Responses received:", responses)
                self._handle_responses(responses)

        # Always re‐render the entire chat history + streams
        with self.human_message.container():
            if st.session_state["user_input"]:
                self.human_message.markdown(st.session_state["user_input"][-1])

        with self.rag_message.container():
            # If there’s a streaming generator, use write_stream to show partial tokens
            if st.session_state["rag_stream"]:
                st.write_stream(st.session_state["rag_stream"])
            elif st.session_state["rag_generated"]:
                self.rag_message.markdown(st.session_state["rag_generated"][-1])

        self._display_sources()
        self.key += 1

    def _handle_responses(self, responses: dict):
        # Grab the streaming generator
        if "rag_stream" in responses:
            st.session_state["rag_stream"] = responses["rag_stream"]
            st.session_state["sources"] = responses["sources"]

        # If user decides to not stream (in future), you could also handle a non‐streaming “rag_text”
        if "rag_text" in responses:
            # Append the generated text to the session state
            st.session_state["rag_generated"].append(responses["rag_text"])            
            st.session_state["sources"] = responses["sources"]

    def _display_sources(self):
        """
        If we have a list of sources, show them horizontally as clickable links.
        """
        with self.sources_tab.container():
            if st.session_state["sources"]:
                st.markdown("📚 **Fontes**")

                cols = st.columns(len(st.session_state['sources']))  # Create one column per source
                
                for col, source_path in zip(cols, st.session_state['sources']):
                    cleaned_path = re.match(r"^(.+?\.pdf)\b", source_path)
                    if cleaned_path:
                        file_name = os.path.basename(source_path)
                        # [TODO]: fix this, find a way to present file (maybe storing documents somewhere else? [not s3])
                        file_url = urllib.parse.quote(cleaned_path.group(1), safe=":/")
                        with col:
                            st.link_button(label=f"{file_name}", url=file_url)


    def _load_topics_json(self):
        topics_json_path = config.TOPICS_FILE

        if not os.path.exists(topics_json_path):
            print("⚠️ topics.json not found. Check remote S3 ChromaDB.")
            return []

        with open(topics_json_path, "r", encoding="utf-8") as f:
            topics_clean = json.load(f)
        print("✅ Topics loaded for filtering.")
        return topics_clean
    

    def generate_context(self):
        # If any history exists
        context = []
        if st.session_state['rag_generated']:
            # Add the last three exchanges
            EXCHANGE_LIMIT = 3
            size = len(st.session_state['rag_generated'])
            # print("Current size of history:", size)
            for i in range(max(size-EXCHANGE_LIMIT, 0), size):
                context.append(
                    {'role': 'user', 'content': st.session_state['user_input'][i]}
                )
                context.append(
                    {'role': 'assistant', 'content': st.session_state["rag_generated"][i]}
                )
        return context
