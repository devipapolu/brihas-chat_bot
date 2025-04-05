import streamlit as st
from streamlit_chat import message
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import re
import uuid
import os
os.environ["OLLAMA_ACCELERATE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from get_embedding_function import get_embedding_function

COMPANY_EMAIL = "contact@brihaspathi.com"
COMPANY_PHONE_NUMBER = "9493531598"
SALES_EMAIL = "sales@brihaspathi.com"
SALES_PHONE = "9087654321"
PROJECT_DIRECTOR_EMAIL = "director@brihaspathi.com"
PROJECT_DIRECTOR_PHONE = "6789054321"
COMPANY_WEBSITE = "https://www.brihaspathi.com"
# Constants
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Generate a precise and professional response based on the following details:

Context: {context}

Question: {question}

Guidelines:
- Provide responses that are precise and directly address the question.
- Avoid repetitive phrases like 'based on the provided context'.
- Avoid email-style formats such as greetings, sign-offs, or subject headers unless specifically requested by the user.
- Respond in a direct, conversational tone without formal email elements like greetings or sign-offs.
- Avoid greetings such as 'Dear User' and sign-offs such as 'Best regards'.
- Do not use email formatting, including headers or footers.
- Provide information directly and succinctly without additional fluff or padding.
- Provide clear, direct answers using bullet points or direct statements.
- If the query goes beyond the data, suggest visiting our website for more details instead of emailing or calling.
- Use bullet points for summaries or lists where appropriate.
- Refer to the company as 'we' or 'our', not 'their', to maintain a unified and internal perspective and to refer to the company and provide specific contact details without saying 'contact us at'.
- If the question is outside the available data, guide the user to contact the sales team for more information, including the company contact details.
- Explicitly use given contact details such as emails, phone numbers, and website URLs in the response, avoiding generic terms like "[company contact details]" or anything in square brackets, instead give direct responses.
- Provide direct and detailed responses that thoroughly address the user's query.
- Use professional and engaging language appropriate for business communications.
- Avoid using placeholders like '[Your Name]' or '[Your Position]' or any brackets. Use generic sign-offs if personalization is not available.
- Include actual contact details, website URLs, and avoid vague phrases.
- If detailed product information is requested, provide key features and direct links or contact info for further inquiries.
- Always format the response to be ready for immediate use without requiring additional editing for placeholders.
- Adapt responses based on the user's context or previous interactions to provide a more personalized experience.
- Encourage users to provide feedback on responses, which can be used to improve future interactions.
- Ensure all responses comply with relevant legal and privacy regulations, especially when processing personal data.
- Consider cultural and regional differences when crafting responses, especially for global audiences, to ensure relevance and sensitivity.

Examples:
Q: What is the range of your solar products?
A: Our solar range includes Solar Smart Poles that feature CCTV, Wi-Fi capabilities, and more. For more details, visit our website at www.example.com.

Q: How can I contact customer support?
A: You can reach our customer support at support@example.com or by calling +1234567890. They are available 24/7 to assist you.

Q: Whom to contact for getting info about solar?
A: For more details about our Solar Smart Poles, please contact our sales team at sales@brihaspathi.com or call us at +91-123-456-7890. Visit our website at www.brihaspathi.com for comprehensive product information.

Example of desired response:
Q: Can I know more about solar?
A: Our Solar Smart Poles harness solar power for surveillance. Features include CCTV, Wi-Fi, and more. Visit our website or contact us directly at +91-123-456-7890 for more details.

Example of undesired response:
Q: Can I know more about solar?
A: Dear User, Thank you for your inquiry about solar solutions...

Response:
"""



import re

def clean_response(response):
    # Remove unwanted email formatting phrases
    unwanted_phrases = [
        'Dear User,', 'Best regards,', 'Thank you for reaching out', 'Warm regards', 
        'Sincerely', '|', 'Regards,', 'Kind regards,'
    ]
    for phrase in unwanted_phrases:
        response = response.replace(phrase, '')

    # Remove placeholders or unwanted patterns
    response = re.sub(r'\[.*?\]', '', response)  # Removes anything inside square brackets
    response = re.sub(r'\(.*?\)', '', response)  # Removes anything inside parentheses
    response = re.sub(r'XXXX+', '123-456-7890', response)  # Replace placeholder numbers
    response = re.sub(r'\+91-XXX-XXXXXXX', '+91-123-456-7890', response)  # Fix phone placeholders
    response = re.sub(r'www\.\[website\]\.com', 'www.brihaspathi.com', response)  # Fix website placeholders

    # Remove generic contact details (only keep actual details)
    response = '\n'.join([
        line for line in response.split('\n') 
        if 'contact our sales team at' not in line and 'visit our website at' not in line
    ])

    return response.strip()  # Remove extra spaces or blank lines


def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = " ".join([doc.page_content for doc, _score in results])  # Concatenate all relevant contexts
    # Format the prompt with specific instructions
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)
    model = Ollama(model="mistral")  # Assuming 'mistral' is your configured model
    # print(f"📝 Debugging Prompt:\n{prompt}")
    response_text = model.invoke(prompt)
     # Insert specific contact details dynamically
    response_text = response_text.replace("[company contact details]", f"{SALES_EMAIL} or {SALES_PHONE}")
    response_text = response_text.replace("[website URL]", COMPANY_WEBSITE)
    response_text = clean_response(response_text)
        # Enhance response for out-of-scope queries
    if "please contact the sales team" in response_text.lower():
        response_text += f"\n\nFor more detailed information, please contact our sales team at: {COMPANY_EMAIL} or call us at {COMPANY_PHONE_NUMBER}."

    
    return response_text


# Initialize session state for message history if it doesn't exist
if 'message_history' not in st.session_state:
    st.session_state.message_history = []

# st.title("Brihaspathi Query Bot")

# Display the logo at the top of the chatbot
logo_path = '/home/sahasra/Downloads/GenAI/rag-tutorial-v2/new logo blue.png'  # Replace with your logo file path or URL
st.image(logo_path)  # Makes the image responsive to container width

# Input field for new message
with st.form("chat", clear_on_submit=True):  # This clears the input after submission
    user_input = st.text_input("How can I help you?", key="user_input")
    submit_button = st.form_submit_button("Get Answer")

if submit_button and user_input:
    # Append user message first
    st.session_state.message_history.append({"text": user_input, "is_user": True})
    
    # Fetch and append bot response
    with st.spinner("Fetching answer..."):
        bot_response = query_rag(user_input)
        st.session_state.message_history.append({"text": bot_response, "is_user": False})

# Reverse the display order of messages
for i, chat_message in enumerate(st.session_state.message_history):
    unique_key = f"chat_{i}_{hash(chat_message['text'])}"  # Ensure unique keys
    message(chat_message['text'], is_user=chat_message['is_user'], key=unique_key)


# Optionally, add a button to clear the conversation
if st.button("Clear Conversation"):
    st.session_state.message_history = []










# from flask import Flask, request, jsonify

# app = Flask(__name__)

# # Sample pricing (you can fetch this from a database)
# SOLAR_PRICING = {
#     "100W": 5000,
#     "300W": 12000,
#     "500W": 18000,
# }

# CCTV_PRICING = {
#     "2MP": 2500,
#     "5MP": 4500,
#     "8MP": 7000,
# }

# @app.route('/get_quotation', methods=['POST'])
# def get_quotation():
#     data = request.json
#     product_type = data.get("product_type")
    
#     if product_type == "solar":
#         panel = data.get("panel_wattage", "300W")
#         inverter = data.get("inverter_size", "5kW")
#         battery = data.get("battery_backup", False)
#         total_cost = SOLAR_PRICING.get(panel, 12000) + (5000 if battery else 0)
#         return jsonify({"quotation": f"Total cost for {panel} solar panel with inverter: ₹{total_cost}"})
    
#     elif product_type == "cctv":
#         resolution = data.get("resolution", "2MP")
#         quantity = int(data.get("quantity", 2))
#         total_cost = CCTV_PRICING.get(resolution, 2500) * quantity
#         return jsonify({"quotation": f"Total cost for {quantity} {resolution} CCTV cameras: ₹{total_cost}"})
    
#     return jsonify({"error": "Invalid product type"}), 400

# if __name__ == '__main__':
#     app.run(debug=True)
