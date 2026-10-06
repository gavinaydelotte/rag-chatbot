import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@st.cache_resource
def get_query_engine():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found. Add it to your .env file and restart the app.")
        st.stop()

    Settings.llm = GoogleGenAI(model="gemini-2.5-flash", api_key=api_key)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine()

st.title("Bare Bones Rag Chatbot")
query_engine = get_query_engine()
prompt = st.chat_input("Ask me anything...")
if prompt:
    st.write(f"You: {prompt}")
    response = query_engine.query(prompt)
    bot_response = response.response
    st.write(f"Bot: {bot_response}")
