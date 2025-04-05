import os
import streamlit as st
from streamlit_chat import message
from langchain_chroma import Chroma
from langchain_community.llms.ollama import Ollama
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import difflib
import re

# Set GPU acceleration for Ollama
os.environ["OLLAMA_ACCELERATE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Paths
CHROMA_PATH = "chroma_db"
DATA_PATH = "data_brihaspathi"
FORM_PATH = "form.html"  # Path to the quotation form

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class SentenceTransformerEmbedding:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

def get_embedding_function():
    return SentenceTransformerEmbedding()

# Initialize ChromaDB
embedding_function = get_embedding_function()
chroma_client = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

def query_rag(query_text: str):
    identity_questions = ["who are you", "what is your name", "introduce yourself", "tell me about yourself", "who am I chatting with"]
    best_match = difflib.get_close_matches(query_text.lower().strip(), identity_questions, n=1, cutoff=0.8)
    
    if best_match:
        return (
            "I am a Support Executive at Brihaspathi Industries Pvt Ltd. "
            "You can reach me at 9676021111, 9676031111, or support@brihaspathi.com. "
            "For technical assistance, please call our Toll-Free number 1800 296 8899."
        )

    quotation_keywords = ["price", "quotation", "cost", "quote", "pricing", "estimate"]
    product_keywords = ["cctv", "solar", "home automation", "sales"]

    if any(q in query_text.lower() for q in quotation_keywords) and any(p in query_text.lower() for p in product_keywords):
        st.session_state["show_form"] = True  # Trigger the form display inside Streamlit
        return "Please fill in the form below to get a quotation."

    query_embedding = embedding_model.encode([query_text])[0]
    results = chroma_client.similarity_search_with_score(query_text, k=5)
    context_text = "\n".join([doc.page_content for doc, _ in results])

    prompt = f"""
    You are a business assistant. Your **only** source of truth is the provided context.

    Context:
    {context_text}

    Answer the question based **strictly** on the context above.
    - If you don't find an answer, **say you don't have enough information** instead of guessing.
    - If the user is asking for quotations, redirect them to the quotation form.
    - Avoid generic definitions or answering beyond the provided data.
    - Provide answers in a direct and informative way.

    Question: {query_text}
    Response:
    """
    model = Ollama(model="llama3.2")
    response_text = model.invoke(prompt)
    return response_text.strip()

# Streamlit UI
st.image("new logo blue.png", width=600)
# st.title("Brihaspathi Chatbot")

# Initialize session state
if "message_history" not in st.session_state:
    st.session_state.message_history = []
if "show_form" not in st.session_state:
    st.session_state.show_form = False
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False

# User input field
user_input = st.text_input("Ask me anything:")

if st.button("Get Answer") and user_input:
    with st.spinner("Fetching answer..."):
        response = query_rag(user_input)
        st.session_state.message_history.append({"text": user_input, "is_user": True})
        st.session_state.message_history.append({"text": response, "is_user": False})

# Display chat messages
for i, chat_message in enumerate(st.session_state.message_history):
    message(chat_message["text"], is_user=chat_message["is_user"], key=f"chat_{i}")

# Show embedded form if quotation is requested
if st.session_state.show_form and not st.session_state.form_submitted:
    st.subheader("Quotation Form")
    with st.form("quotation_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Email Address")
        phone = st.text_input("Mobile Number", max_chars=10)
        interest = st.selectbox("Interest", ["CCTV/Camera", "Software", "Computer/Laptop", "All"])
        submit = st.form_submit_button("Submit")

        if submit:
            st.success("✅ Your request has been submitted! We will get back to you soon.")
            st.session_state.form_submitted = True  # Hide form after submission

# Clear chat button
if st.button("Clear Conversation"):
    st.session_state.message_history = []
    st.session_state.show_form = False
    st.session_state.form_submitted = False
