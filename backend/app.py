from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from typing import List
from datetime import datetime
import pytz
import os
import difflib
from langchain_chroma import Chroma
from langchain_community.llms.ollama import Ollama
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from fastapi.responses import FileResponse

# 🌐 Initialize FastAPI
app = FastAPI()

# 🔄 Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ⏳ Convert timestamp to IST before saving
def get_ist_time():
    utc_now = datetime.utcnow()
    ist = pytz.timezone('Asia/Kolkata')
    return utc_now.astimezone(ist).strftime('%d-%m-%Y %I:%M %p')

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

# 🔍 Initialize ChromaDB (ONLY ONCE)
embedding_function = get_embedding_function()
chroma_client = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

def load_and_process_pdfs():
    """Loads and processes multiple PDFs for vector storage (if not already indexed)."""
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()

    # Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    print(f"✅ Loaded {len(texts)} text chunks from multiple PDFs.")
    return texts

def store_documents_in_chroma():
    """Embeds and stores documents from multiple PDFs in ChromaDB (ONLY if not already stored)."""
    if len(chroma_client.get()["documents"]) > 0:
        print("✅ Documents already exist in ChromaDB. Skipping reprocessing.")
        return  # Prevent duplicate processing

    docs = load_and_process_pdfs()
    
    # Convert document chunks into embeddings and store in ChromaDB
    chroma_client.add_documents(docs)
    print("✅ Multi-PDF documents stored successfully in ChromaDB!")

# Run document storage process **ONLY IF CHROMADB IS EMPTY**
store_documents_in_chroma()

# 📡 MongoDB Atlas Connection 
MONGO_URI = "mongodb+srv://sahasra:4Jan%401998@cluster0.8yacy.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["chatbot"]
chat_collection = db["chat_history"]

# 📌 Class for user query
class UserQuery(BaseModel):
    message: str

class ChatMessage(BaseModel):
    type: str  
    text: str
    timestamp: str  

    def to_dict(self):
        """ Convert to dictionary for MongoDB storage. """
        return {"type": self.type, "text": self.text, "timestamp": self.timestamp}

class ChatHistory(BaseModel):
    userId: str  
    chatHistory: List[ChatMessage]  
    lastMessageTime: str  



def query_rag(query_text: str):
    """Processes user queries and returns appropriate responses by prioritizing vector search first, 
    then falling back to grouped query handles via a loop structure (including fuzzy matching), and finally showing a fallback message."""

    normalized_query = query_text.lower().strip().replace("?", "")

    query_groups = [
        {
            "patterns": ["md", "who is md", "who is the md", "managing director","director","Director"],
            "response": "<b>Mr. Rajasekhar Papolu</b> is the Managing Director of Brihaspathi Technologies Limited."
        },
        {
            "patterns": ["unique solutions", "different solutions", "non-hardware solutions", "ai solutions", "iot solutions", "custom solutions", "software solutions", "any unique solutions", "special solutions", "ai iot solutions"],
            "response": """<b>🚀 Unique AI & IoT-Based Solutions at Brihaspathi Technologies Limited</b>
            <p>We offer **custom-built AI and IoT-driven solutions** beyond traditional hardware, tailored for various industries:</p>
            <ul>
                <li>🤖 <b>AI-Based Solutions:</b> VMS (Video Management Systems), Intelligent Chatbots, AI based Visitot management system/ Access Control System</li>
                <li>⚖️ <b>E-Court Solutions:</b> Smart Case Management & Digital Documentation</li>
                <li>🔆 <b>Solar Smart Pole:</b> Energy-efficient smart poles integrated with surveillance & IoT sensors</li>
                <li>💼 <b>HRMS Solutions:</b> Employee management and automation tool</li>
                <li>📊 <b>Visitor Management System:</b> AI-driven insights for predictive system for monitoring visitors</li>
            </ul>
            <p>Our focus is on <b>AI & IoT innovation</b> to create intelligent, scalable, and automation-driven solutions.</p>
            <p>For more details, reach out at 📧 <a href='mailto:info@brihaspathi.com'>info@brihaspathi.com</a></p>""",
            "use_fuzzy": True  # added indicator to use difflib for fuzzy matching
        },
        {
            "patterns": ["how many branches", "branch locations", "where are your branches", "service centers", "office locations", "company units", "where are you located"],
            "response": """<b>🏢 Brihaspathi Technologies Limited - Branches & Service Centers</b>
            <p>We have multiple branches and service centers across **India** to serve our customers efficiently.</p>

            <b>📍 **Corporate & Registered Offices**</b>
            <b>✔️ Registered Address:</b>  
            7-1-621/259, 5th Floor, Sahithi Arcade, SR Nagar, Hyderabad – 500038.<br>
            <b>✔️ Corporate Address:</b>  
            #501, #508-510, Shangrila Plaza, Road No. 2, Park View Enclave, Banjara Hills, Hyderabad, Telangana – 500034.<br><br>

            <b>🌍 **Branch Offices & Service Centers**</b>
            <ul>
                <li>📌 <b>Kurnool:</b> H.No. 45/204-A1-3, Near KNR High School, Venkatramana Colony Road, Mamatha Cafe, Ashok Nagar, Kurnool, Andhra Pradesh – 518001.</li>
                <li>📌 <b>Vijayawada:</b> Durga Enclave, Flat No A1, 1st Floor, ex C M Road, Old toll gate, behind HP Petrol Bunk, Tadepalle, Kunchanapalli, Andhra Pradesh 522501.</li>
                <li>📌 <b>Visakhapatnam:</b> 94-6-11/F, Near Sai Baba Temple, Santhipuram, Dwaraka Nagar, Ramakrishna Nagar, Visakhapatnam, Andhra Pradesh – 530016.</li>
                <li>📌 <b>Bhopal:</b> LIG-G-01, Bawadia Kala, Fortune Glory Ext 2, Rohit Nagar, Bhopal, MP-462026.</li>
                <li>📌 <b>Guwahati:</b> Bishnu Rabha Byelane, Dalbari, Satgaon, P.O.: Udayan Vihar, Guwahati, Assam – 781171.</li>
                <li>📌 <b>Pune:</b> Kharadi road, Sainath Nagar Chowk, Pune, Maharashtra – 411014.</li>
                <li>📌 <b>Delhi:</b> 201, A Block, Naurang House, KG Marg, Cannought Place New Delhi -110001.</li>
                <li>📌 <b>West Bengal:</b> 85 SP Road Kolkata West Bengal, 700026.</li>
                <li>📌 <b>Tamil Nadu:</b> 1/1A UR Anna Nagar Chennai Tamil Nadu.</li>
                <li>📌 <b>Bangalore:</b> A52 RPR Complex, Kamadenu Nagar, B Narayanpura, Mahadevpura, Bangalore -560016.</li>
                <li>📌 <b>Patna:</b> 1st Floor, A/3, PC Colony, Kankarbagh Road, Patna.</li>
            </ul>

            <p>📞 For support & service inquiries, please contact us at:  
            ☎️ <b>+91 9676021111, +91 9676031111</b> or  
            📧 <a href='mailto:support@brihaspathi.com'>support@brihaspathi.com</a></p>"""
        }
    ]

    for group in query_groups:
        if group.get("use_fuzzy"):
            best_match = difflib.get_close_matches(normalized_query, group["patterns"], n=1, cutoff=0.8)
            if best_match:
                return {"response": group["response"], "show_form": False}
        else:
            if any(pattern in normalized_query for pattern in group["patterns"]):
                return {"response": group["response"], "show_form": False}

    predefined_responses = {
    "careers": """
        <b>🔹 Careers at Brihaspathi Technologies Limited</b>
        <p>For career opportunities, please visit our official careers page or contact our HR department:</p>
        <p>📧 <a href='mailto:HR@Brihaspathi.com'>HR@Brihaspathi.com</a></p>
    """,

    "products": """
        <b>🛠 Our Products & Services at Brihaspathi Technologies Limited</b>
        <p>We offer a wide range of <b>security solutions</b>, including:</p>

        <b>🔹 Security & Surveillance Solutions:</b>
        <ul>
            <li>📷 CCTV Surveillance</li>
            <li>🛂 Biometric Time & Attendance / Access Control</li>
            <li>📞 IP PBX / EPABX</li>
            <li>🏡 Home Automation</li>
            <li>🚪 Entry Gate Solutions</li>
            <li>🌐 Servers & Networking</li>
            <li>🚗 VTS / GPS Tracking</li>
            <li>🔥 Fire Alarm / Intrusion Alarm</li>
            <li>🚨 Burglar Alarm System</li>
        </ul>

        <b>🔹 Additional Technological Services:</b>
        <ul>
            <li>🔐 E-Security Solutions</li>
            <li>📱 App Development</li>
            <li>💻 Software Development</li>
            <li>🔗 IoT Services</li>
            <li>🖥️ IT Products</li>
            <li>🤖 AI and VMS-based Solutions</li>
            <li>💡 ELV Solutions</li>
        </ul>

        <p>We have served over <b>12,000+ clients</b> across various industries.</p>
        <p>For product inquiries or quotations, please contact:</p>
        <p>📧 <a href='mailto:sales@Brihaspathi.com'>sales@Brihaspathi.com</a></p>
    """,

    "about us": """
        <b>🏢 About Brihaspathi Technologies Limited</b>
        <p><b>🚀 A Solutions-Driven Company for Custom E-Security & IT Innovations</b></p>
        <p>Brihaspathi Technologies Limited is a trusted leader in IT products and solutions since 2006, headquartered in Hyderabad.</p>
        <p>We specialize in custom e-security solutions, tailoring cutting-edge technologies to meet the unique needs of businesses and institutions worldwide.</p>
        <p>📧 <a href='mailto:info@brihaspathi.com'>info@brihaspathi.com</a></p>
        <p>☎️ Contact us: <b>+91 9676021111, +91 9676031111</b></p>
    """,

    "positions": """
        <b> Company Hierarchy at Brihaspathi Technologies Limited</b>
        <p>Our company operates with the following leadership structure:</p>

        <ul>
            <li>🔹 <b>Managing Director (MD):</b> Mr. Rajasekhar Papolu</li>
            <li>🔹 <b>Director - Administration:</b> Mrs. Hymavathi Rajasekhar Papolu</li>
            <li>🔹 <b>Executive Director: (MD):</b> Mr. Murali Krishna</li>
            <li>🔹 <b>Chief Operating Officer(Retail Sales): (COO):</b> Mr. Madhu Kuppani</li>
            <li>🔹 <b>Chief Financial Officer: (CFO):</b> Mr. Govardhan Chawla</li>
            <li>🔹 <b>Chief Sales Officer(Institutional Sales): (CSO):</b> Mr. Saketh Addepalli</li>
            <li>🔹 <b>Vice President - Corporate & Legal:</b>Ms.Vrinda Jaiswal</li>
        </ul>

        <p>For official inquiries, please contact:</p>
        <p>📧 <a href='mailto:info@brihaspathi.com'>info@brihaspathi.com</a></p>
    """,
    "Sales": """
        <b>📊 Sales Divisions at Brihaspathi Technologies Limited</b>
        <p>We operate with two specialized sales verticals to cater to diverse market segments: <b>Retail Sales</b> and <b>Institutional Sales</b>.</p>

        <b>🛒 Retail Sales</b>
        <p>Retail sales refer to the process of selling products or services directly to individual consumers for personal use. These sales typically occur through brick-and-mortar stores, e-commerce platforms, or direct sales representatives.</p>

        <b>Key Characteristics:</b>
        <ul>
            <li>🎯 Directly target individual consumers</li>
            <li>💸 Smaller transaction values but high frequency</li>
            <li>🛍️ Includes in-store purchases, online shopping, and direct sales</li>
            <li>⭐ Focus on branding, customer experience, and promotions</li>
        </ul>

        <b>📞 Contact – Retail Sales:</b>
        <ul>
            <li>🔹 <b>Mr. Madhu Kuppani</b> – +91 96762 12345</li>
            <li>🔹 <b>Mr. Moses</b> – +91 82473 21053</li>
        </ul>

        <br>

        <b>🏢 Institutional Sales</b>
        <p>Institutional sales involve selling products or services in bulk to businesses, government bodies, or educational institutions. These sales require strategic planning and long-term relationship management.</p>

        <b>Key Characteristics:</b>
        <ul>
            <li>🏛️ Target businesses, institutions, and organizations</li>
            <li>💼 Larger transaction values and long-term commitments</li>
            <li>📃 Involves customized pricing, contracts, and service agreements</li>
            <li>🤝 Requires strong relationship management and strategic sales efforts</li>
        </ul>

        <b>📞 Contact – Institutional Sales:</b>
        <ul>
            <li>🔹 <b>Mr. Saketh Addepalli</b> – +91 96408 33333</li>
        </ul>
"""
}  # Add all predefined long-form HTML responses here.

    if normalized_query in predefined_responses:
        return {"response": predefined_responses[normalized_query], "show_form": False}

    quotation_phrases = ["i need a quotation", "can you give me a quote?", "how much does it cost?", "what is the price?", "can i get an estimate?", "quotation please","price","cctv", "solar", "home automation","quotation", "cost", "quote", "pricing", "estimate", "Quotation", "Quote"]
    if any(phrase in normalized_query for phrase in quotation_phrases):
        return {"response": "Please fill in the form below to get a quotation or click on contact us above.", "show_form": True}
            

    # ✅ 1️⃣ Handle simple greetings and conversational queries first
    if normalized_query in ["hi", "hello", "hey"]:
        return {"response": f"{query_text.capitalize()}! How can I assist you today?", "show_form": False}

    if any(p in normalized_query for p in ["who are you", "what is your name", "who r u", "introduce yourself", "who am i chatting with"]):
        return {"response": "I am a Support Executive at Brihaspathi Technologies Limited. You can reach me at 9676021111, 9676031111, or support@brihaspathi.com. For technical assistance, please call our Toll-Free number 1800 296 8899.", "show_form": False}

     

    # ✅ 1️⃣ Perform vector search first
    results = chroma_client.similarity_search_with_score(query_text, k=10)
    high_confidence_results = [r for r in results if r[1] > 0.7]

    if high_confidence_results:
        context_text = "\n".join([doc.page_content for doc, _ in high_confidence_results])

        prompt = f"""
        You are a business assistant chatbot for Brihaspathi Technologies Limited. Answer strictly using the context below.

        Context:
        {context_text}

        Answer the question based **strictly** on the context above.
    - If the user message is exactly "hi", "hello", or "hey", respond with the same greeting and follow up with: "How can I assist you today?"
    - If the user says "thank you" or "thanks", reply with: "You're welcome! Is there anything else I can help you with?"
    - If you don't find an answer, **say Sorry and you don't have enough information** instead of guessing.
    - If the user is asking for quotations, redirect them to the quotation form, but do not use place holders such as [quotation form]
    - Avoid generic definitions or answering beyond the provided data.
    - Provide answers in a direct and informative way and use bullet points if there is a list of items.
    - **Do not generate responses with placeholders such as [insert link], [company contact details], or [your name]. Instead, directly give the quotation form instead of any fake links or fake link placeholders.**
    - **Prevent hallucination by only responding with factual, verified information from the given data. Do not invent details.**
    - **DO NOT make up company names or information. Only refer to Brihaspathi Technologies Limited.**
    - Provide a direct and professional response.

        Question: {query_text}
        Response:
        """

        model = Ollama(model="llama3.2", temperature=0.2, top_k=40)
        response_text = model.invoke(prompt).strip()

        # Clean LLM output
        unwanted_phrases = [
            "According to the provided context,", 
            "According to context,", 
            "Comptech",
            "vasavi",
            "Based on the given data,"
        ]

        for phrase in unwanted_phrases:
            response_text = response_text.replace(phrase, "").strip()

        if "i don't have enough information" not in response_text.lower():
            return {"response": response_text, "show_form": False}


    
    # ✅ 3️⃣ Final fallback:
    return {"response": "Sorry, I don’t have enough information to answer that. Please contact us at 9676021111 or support@brihaspathi.com.", "show_form": False}

@app.post("/save-chat-history")
async def save_chat_history(chat_data: ChatHistory):
    print("✅ Received chat history:", chat_data.dict()) 

    chat_history_dicts = [message.to_dict() for message in chat_data.chatHistory]

    chat_collection.update_one(
        {"userId": chat_data.userId},
        {
            "$set": {"lastMessageTime": chat_data.lastMessageTime},
            "$push": {"chatHistory": {"$each": chat_history_dicts}}  
        },
        upsert=True
    )

    return {"message": "✅ Chat history stored successfully!"}

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

@app.post("/chat")
async def chat(query: UserQuery):
    if query.message.strip().lower() == "start":
        return {"response": "Hi, I am Briha 🤖, how may I help you?"}

    chat_history = query_rag(query.message)
    return chat_history

@app.post("/submit_form")
async def submit_form(name: str = Form(...), email: str = Form(...), phone: str = Form(...), interest: str = Form(...)):
    return {"message": "✅ Your request has been submitted! We will get back to you soon."}


@app.get("/download-brochure")
async def download_brochure():
    brochure_path = "/home/sahasra/Downloads/GenAI/rag-tutorial-v2/data_brihaspathi/updated_brochure_BTL_25-01-2025.pdf"
    if os.path.exists(brochure_path):
        return FileResponse(brochure_path, media_type='application/pdf', filename="updated_brochure_BTL_25-01-2025.pdf")
    else:
        raise HTTPException(status_code=404, detail="Brochure not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
