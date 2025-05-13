from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
from pymongo import MongoClient
from transformers import AutoModelForCausalLM, AutoTokenizer
import os, logging

app = FastAPI()

# MongoDB Setup
client = MongoClient("mongodb+srv://sahasra:4Jan%401998@cluster0.8yacy.mongodb.net/?retryWrites=true&w=majority&tls=true&tlsAllowInvalidCertificates=true")
db = client["chatbot"]
chat_collection = db["chat_history"]

# Model setup (DistilGPT2 for CPU)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Schema models
class Message(BaseModel):
    role: str
    content: str
    def to_dict(self):
        return {"role": self.role, "content": self.content}

class ChatHistory(BaseModel):
    userId: str
    chatHistory: List[Message]
    lastMessageTime: str

class UserQuery(BaseModel):
    userId: str
    message: str

# Helper: LLM Fallback
def generate_response_with_distilgpt2(prompt: str) -> str:
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

# RAG/LLM hybrid response logic
def query_rag(query_text: str):
    normalized_query = query_text.strip().lower()

    predefined_responses = {
        "contact": """
        📧 <a href='mailto:sales@Brihaspathi.com'>sales@Brihaspathi.com</a><br>
        📞 <b>+91 9676021111, +91 9676031111</b>
        """,
        "about us": """
        <b>🏢 Brihaspathi Technologies Limited</b><br>
        🚀 A leader in custom e-security & IT solutions since 2006.<br>
        📧 <a href='mailto:info@brihaspathi.com'>info@brihaspathi.com</a><br>
        📞 <b>+91 9676021111, +91 9676031111</b>
        """
    }

    if normalized_query in predefined_responses:
        return {"response": predefined_responses[normalized_query], "show_form": False}

    if any(kw in normalized_query for kw in ["quotation", "quote", "price", "estimate", "cost", "pricing"]):
        return {"response": "📝 Please fill out the form to receive a quotation.", "show_form": True}

    if normalized_query in ["hi", "hello", "hey"]:
        return {"response": f"{query_text.capitalize()}! How can I assist you today?", "show_form": False}

    if any(p in normalized_query for p in ["who are you", "your name", "introduce yourself"]):
        return {"response": "I am Briha 🤖 from Brihaspathi Technologies. How can I assist you?", "show_form": False}

    return {"response": "Sorry, I don’t have enough information to answer that. Please contact us at 9676021111 or support@brihaspathi.com.", "show_form": False}

@app.get("/")
def root():
    return {"status": "✅ Backend is working!"}

@app.post("/save-chat-history")
async def save_chat_history(chat_data: ChatHistory):
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
    try:
        if query.message.strip().lower() == "start":
            return {"response": "Hi, I am Briha 🤖, how may I help you?"}

        result = query_rag(query.message)

        # Save user & assistant message
        chat_collection.insert_many([
            {"userId": query.userId, "role": "user", "content": query.message},
            {"userId": query.userId, "role": "assistant", "content": result["response"]}
        ])
        return result
    except Exception as e:
        logging.exception("Chat error")
        raise HTTPException(status_code=500, detail="Chat failed")

@app.post("/submit_form")
async def submit_form(name: str = Form(...), email: str = Form(...), phone: str = Form(...), interest: str = Form(...)):
    return {"message": "✅ Your request has been submitted! We will get back to you soon."}

@app.get("/download-brochure")
async def download_brochure():
    # Construct the absolute path using the appropriate OS separator
    brochure_path = os.path.join(os.getcwd(), 'data_brihaspathi', 'new updated brochure BTL 25-01-2025.pdf')


    # Log the current working directory and the brochure path
    logging.info(f"Current working directory: {os.getcwd()}")
    logging.info(f"Brochure path: {brochure_path}")

    # Check if the brochure exists at the specified path
    if not os.path.exists(brochure_path):
        logging.error(f"Brochure not found at {brochure_path}")
        raise HTTPException(status_code=404, detail=f"Brochure not found at {brochure_path}")

    logging.info("Brochure found. Preparing to download.")
    
    # Return the file as a response
    return FileResponse(
    brochure_path,
    filename="new updated brochure BTL 25-01-2025.pdf",
    media_type="application/pdf"
)

# ✅ Uvicorn launcher for local dev
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
