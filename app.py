import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google.genai.errors import APIError
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from pypdf import PdfReader
from pypdf.errors import PdfReadError

load_dotenv()

DATA_DIR = Path("data")
LLM_MODEL = "gemini-3.5-flash-lite"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def get_api_key():
    """Return the Gemini API key, or stop the app if it isn't set."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error(
            "GEMINI_API_KEY is not set. Add a line like `GEMINI_API_KEY=your-key` "
            "to the `.env` file in the project folder, then restart the app."
        )
        st.stop()
    return api_key


def check_data_dir():
    """Stop the app if DATA_DIR is missing or has no visible files."""
    if not DATA_DIR.is_dir():
        st.error(f"Expected a data folder at `{DATA_DIR.resolve()}`, but it doesn't exist.")
        st.stop()

    # Ignore hidden files like .DS_Store so they don't count as documents.
    visible_files = [
        path for path in DATA_DIR.iterdir()
        if path.is_file() and not path.name.startswith(".")
    ]
    if not visible_files:
        st.error(f"The data folder `{DATA_DIR.resolve()}` is empty. Add a PDF to it and restart the app.")
        st.stop()


@st.cache_resource
def get_query_engine(api_key):
    """Load the handbook PDF, index it, and return a query engine."""
    Settings.llm = GoogleGenAI(model=LLM_MODEL, api_key=api_key)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    reader = PdfReader(DATA_DIR / "handbook.pdf")
    documents = [
        Document(text=page.extract_text() or "", metadata={"page": i + 1})
        for i, page in enumerate(reader.pages)
    ]
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine()


st.title("Bare Bones Rag Chatbot")

api_key = get_api_key()
check_data_dir()

try:
    query_engine = get_query_engine(api_key)
except PdfReadError as e:
    st.error(f"Couldn't read the handbook PDF in `{DATA_DIR}`. It may be corrupt. Details: {e}")
    st.stop()
except APIError as e:
    st.error(f"Gemini rejected the request. Check that GEMINI_API_KEY in `.env` is valid. Details: {e}")
    st.stop()
except OSError as e:
    st.error(
        "Couldn't load the handbook or the embedding model. "
        f"Check that the file exists and that you're online. Details: {e}"
    )
    st.stop()
except Exception as e:
    st.error(f"Couldn't build the search index: {e}")
    st.stop()

prompt = st.chat_input("Ask me anything...")
if prompt:
    st.write(f"User: {prompt}")
    try:
        response = query_engine.query(prompt)
    except APIError as e:
        st.error(f"The Gemini API returned an error (you may be rate limited). Please try again. Details: {e}")
    except Exception as e:
        st.error(f"Something went wrong answering your question. Please try again. Details: {e}")
    else:
        with st.chat_message("assistant"):
            st.write(f"Bot response: {response.response}")
