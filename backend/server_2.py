from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from typing import List

from datetime import datetime
import pytz 
import os
from langchain_chroma import Chroma
from langchain_community.llms.ollama import Ollama
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import difflib

# 🌐 Initialize FastAPI
app = FastAPI()

# 🔄 Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (modify for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Function to convert timestamp to IST before saving
def get_ist_time():
    utc_now = datetime.utcnow()
    ist = pytz.timezone('Asia/Kolkata')
    return utc_now.astimezone(ist).strftime('%d-%m-%Y %I:%M %p')  # Format: DD-MM-YYYY hh:mm AM/PM



# ⚡ Enable GPU acceleration for Ollama
os.environ["OLLAMA_ACCELERATE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 📂 Paths for Vector DB
CHROMA_PATH = "chroma_db"
DATA_PATH = "data_brihaspathi"

# 🔍 Load embedding model
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

# 🔍 Initialize ChromaDB
embedding_function = get_embedding_function()
chroma_client = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# 📌 Class for user query
class UserQuery(BaseModel):
    message: str



class ChatMessage(BaseModel):
    type: str  # "user" or "bot"
    text: str
    timestamp: str  # Store timestamp as a string (formatted)

    def to_dict(self):
        """ Convert to dictionary for MongoDB storage. """
        return {"type": self.type, "text": self.text, "timestamp": self.timestamp}

class ChatHistory(BaseModel):
    userId: str  # Ensure this is a string
    chatHistory: List[ChatMessage]  # Ensure this is a **list of objects**
    lastMessageTime: str  # Store human-readable IST timestamp

# 📌 Class for storing chat history
# class ChatHistory(BaseModel):
#     userId: str
#     chatHistory: list  # List of user and bot messages
#     lastMessageTime: int  # Timestamp of last message

# 📡 MongoDB Atlas Connection (Replace `<your-cluster-url>` with actual URL)
MONGO_URI = "mongodb+srv://sahasra:4Jan%401998@cluster0.8yacy.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["chatbot"]  # Database Name
chat_collection = db["chat_history"]  # Collection Name

import difflib

def query_rag(query_text: str):
    """Processes user queries and returns appropriate responses."""

    # Normalize query for consistent matching
    normalized_query = query_text.lower().strip()

    # ✅ **Explicit Handling for MD-related Queries (Overrides RAG Search)**
    md_queries = ["md", "who is md", "who is the md", "managing director", "md of brihaspathi", "who is the md of brihaspathi"]
    if any(q in normalized_query for q in md_queries):
        return {
            "response": "<b>Mr. Rajasekhar Papolu</b> is the Managing Director of Brihaspathi Technologies Limited.", 
            "show_form": False
        }


    # ✅ **Predefined Hardcoded Responses for Specific Buttons**
    predefined_responses = {
    "careers": """<b>🔹 Careers at Brihaspathi Technologies Limited</b><br>
        
        For career opportunities, please visit our official careers page or contact our HR department at <a href='mailto:HR@Brihaspathi.com'>HR@Brihaspathi.com</a>.""",

    "products": """<b>🛠 Our Products & Services at Brihaspathi Technologies Limited</b><br>
            We offer a wide range of <b>security solutions</b>, including:<br><br>
            <b>🔹 Security & Surveillance Solutions:</b><br>
            1️⃣ CCTV Surveillance<br>
            2️⃣ Biometric Time & Attendance / Access Control<br>
            3️⃣ IP PBX / EPABX<br>
            4️⃣ Home Automation<br>
            5️⃣ Entry Gate Solutions<br>
            6️⃣ Servers & Networking<br>
            7️⃣ VTS / GPS Tracking<br>
            8️⃣ Fire Alarm / Intrusion Alarm<br>
            9️⃣ Burglar Alarm System<br><br>
            
            <b>🔹 Additional Technological Services:</b><br>
            1️⃣ E-Security Solutions<br>
            2️⃣ App Development<br>
            3️⃣ Software Development<br>
            4️⃣ IoT Services<br>
            5️⃣ IT Products<br>
            6️⃣ AI Solutions<br>
            7️⃣ ELV Solutions<br><br>

            We have served over <b>12,000+ clients</b> across various industries.<br><br>
            For product inquiries or quotations, please reach us out at <a href='mailto:sales@Brihaspathi.com'>sales@Brihaspathi.com</a>.""",
    "about us": """<b>🏢 About Brihaspathi Technologies Limited</b><br><br>

        <b>🚀 A Solutions-Driven Company for Custom E-Security & IT Innovations</b><br><br>

        Brihaspathi Technologies Limited is a trusted leader in IT products and solutions since 2006, headquartered in Hyderabad. We specialize in custom e-security solutions, tailoring cutting-edge technologies to meet the unique needs of businesses and institutions worldwide.<br><br>

        <b>🔹 What Sets Us Apart:</b><br>
        ✔️ A visionary leadership team and an exceptional workforce driving innovation and excellence.<br>
        ✔️ Expertise in Software Development, Mobile Applications, E-Communications, E-Security Systems, and Global Positioning Systems (GPS).<br>
        ✔️ Recognized for delivering high-impact, scalable, and future-ready solutions that empower businesses across industries.<br><br>

        <b>🔹 Key Milestones:</b><br>
        ✔️ Successfully implemented the <b>Kurnool Smart City Project</b>, enhancing urban security and surveillance.<br>
        ✔️ Providing Web Development, IT Infrastructure, and AI-driven Security Solutions to corporate, educational, and government institutions.<br>
        ✔️ Continually evolving to meet the complex demands of the digital era with innovative and robust solutions.<br><br>

        <b>🚀 Our Commitment:</b><br>
        We are dedicated to delivering reliable, efficient, and secure solutions that help businesses stay ahead in a rapidly advancing technological landscape.<br><br>

        📧 For inquiries, reach out at <a href='mailto:info@brihaspathi.com'>info@brihaspathi.com</a><br>
        ☎️ Contact us: <b>+91 9676021111, +91 9676031111</b><br>""",

}

    # ✅ **Check for Hardcoded Responses**
    if normalized_query in predefined_responses:
        return {"response": predefined_responses[normalized_query], "show_form": False}

    # ✅ **Identity-Based Questions Handling**
    identity_questions = [
        "who are you", "what is your name", "introduce yourself", "tell me about yourself", "who am I chatting with", "who r u"
    ]
    best_match = difflib.get_close_matches(normalized_query, identity_questions, n=1, cutoff=0.8)

    if best_match:
        return {
            "response": (
                "I am a Support Executive at Brihaspathi Technologies Limited. "
                "You can reach me at 9676021111, 9676031111, or support@brihaspathi.com. "
                "For technical assistance, please call our Toll-Free number 1800 296 8899."
            ),
            "show_form": False
        }

    # ✅ **Refined Keyword Matching for Quotations**
    quotation_phrases = [
        "I need a quotation", "Can you give me a quote?", "How much does it cost?", 
        "What is the price?", "Can I get an estimate?, Quotation please"
    ]
    detected_quotation_request = any(phrase in normalized_query for phrase in quotation_phrases)

    product_keywords = ["cctv", "solar", "home automation", "sales", "price", "quotation", "cost", "quote", "pricing", "estimate", "Quotation", "Quote"]
    product_request = any(p in normalized_query for p in product_keywords)

    if detected_quotation_request:
        return {
            "response": "Please fill in the form below to get a quotation.",
            "show_form": True
        }

    # ✅ **Use Vector Search for General Queries**
    query_embedding = embedding_model.encode([query_text])[0]
    results = chroma_client.similarity_search_with_score(query_text, k=5)

    if not results:
        return {
            "response": "I don’t have enough information to answer that. Please contact us directly for further assistance at 9676021111, 9676031111, or support@brihaspathi.com.",
            "show_form": False
        }

    context_text = "\n".join([doc.page_content for doc, _ in results])

    # ✅ **LLM Query Processing**
    prompt = f"""
    You are a business assistant and company chatbot. Your **only** source of truth is the provided context. **Never** use placeholders such as [Insert Quotation Form]. Provide only factual, direct responses.  

    Context:
    {context_text}

    Answer the question based **strictly** on the context above.
    - If you don't find an answer, **say you don't have enough information** instead of guessing.
    - If the user is asking for quotations, redirect them to the quotation form.
    - Avoid generic definitions or answering beyond the provided data.
    - Provide answers in a direct and informative way and use bullet points if there is a list of items.
    - **Do not generate responses with placeholders such as [insert link], [company contact details], or [your name]. Instead, directly give the quotation form instead of any fake links or fake link placeholders.**
    - **Prevent hallucination by only responding with factual, verified information from the given data. Do not invent details**
    - **DO NOT make up company names or information. Only refer to Brihaspathi Technologies Limited.**
    - Provide a direct and professional response. Do not include phrases like "According to the provided context" or "Based on the given data."


    Question: {query_text}
    Response:
    """
    
    model = Ollama(model="llama3.2", temperature=0.2, top_k=40)
    response_text = model.invoke(prompt).strip()

    # ✅ **Remove Unwanted Phrases**
    unwanted_phrases = [
        "According to the provided context,", 
        "According to context,", 
        "Comptech",
        "vasavi",
        "Based on the given data,"
    ]

    for phrase in unwanted_phrases:
        response_text = response_text.replace(phrase, "").strip()

    return {"response": response_text, "show_form": False}

@app.post("/save-chat-history")
async def save_chat_history(chat_data: ChatHistory):
    print("✅ Received chat history:", chat_data.dict())  # Debugging output

    # Convert each ChatMessage object to a dictionary
    chat_history_dicts = [message.to_dict() for message in chat_data.chatHistory]

    # Store chat history in MongoDB (upsert logic)
    chat_collection.update_one(
        {"userId": chat_data.userId},
        {
            "$set": {"lastMessageTime": chat_data.lastMessageTime},
            "$push": {"chatHistory": {"$each": chat_history_dicts}}  # Store as list of dicts
        },
        upsert=True
    )

    return {"message": "✅ Chat history stored successfully!"}

### ✅ **API to Retrieve Chat History for a User**
@app.get("/get-chat-history")
async def get_chat_history():
    """ Retrieve all chat histories from the database. """
    try:
        chats = chat_collection.find({})
        history = []
        
        for chat in chats:
            history.append({
                "userId": chat.get("userId", "Unknown"),
                "chatHistory": chat.get("chatHistory", []),
                "lastMessageTime": chat.get("lastMessageTime", "Unknown"),
            })
        
        return {"chat_history": history}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {str(e)}")


# API endpoint to handle user messages
@app.post("/chat")
async def chat(query: UserQuery):
    return query_rag(query.message)

# API endpoint for form submission
@app.post("/submit_form")
async def submit_form(name: str = Form(...), email: str = Form(...), phone: str = Form(...), interest: str = Form(...)):
    return {"message": "✅ Your request has been submitted! We will get back to you soon."}

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
