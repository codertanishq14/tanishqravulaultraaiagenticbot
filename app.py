# tanishqravulasuperaibot-main/app.py
# Updated Code: Refactored for Robustness, Stability, and Simplicity.
# Removed streaming (yield) in favor of a standard request-response model.
# Added comprehensive error handling to all routes.

import os
import json
import uuid
import re
import io
from flask import Flask, request, jsonify, render_template, Response
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

# PDF Export Imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors

# LangChain Imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# --- UTILS (assuming they exist from the original code) ---
# It's better to keep utility functions in a separate file (e.g., utils.py)
# For this example, I'll stub them here if they are not provided.
# import utils # In a real scenario, you'd import your utils file.
class Utils:
    def initialize_mistral(self, api_key):
        # Placeholder for Mistral initialization
        print("Mistral initialized (placeholder).")

    def generate_images_from_prompt(self, prompt, num_images):
        # Placeholder for image generation
        return {"success": True, "message": f"Generated {num_images} images for: {prompt}"}

    def generate_video_from_prompt(self, prompt):
        # Placeholder for video generation
        return {"success": True, "message": f"Generated video for: {prompt}"}

    def process_large_uploaded_files(self, files):
         # Placeholder for large file processing
        return "This is extracted text from large files. " * 100

    def process_uploaded_file(self, file):
        # Placeholder for standard file processing
        # Check for image type, etc.
        if file.mimetype.startswith('image/'):
            return Image.open(file.stream)
        return "This is extracted text from a standard file."

    def get_youtube_transcript(self, url):
        # Placeholder for YouTube transcript
        return f"Transcript for YouTube video at {url}."

    def get_website_content(self, url):
        # Placeholder for website scraping
        return f"Scraped content from website at {url}."

utils = Utils() # Instantiate the placeholder class

# --- INITIALIZATION ---
load_dotenv()
app = Flask(__name__)

# --- API CONFIGURATION ---
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    if not GOOGLE_API_KEY or not MISTRAL_API_KEY:
        raise ValueError("API keys for Google and Mistral must be set in the .env file.")

    genai.configure(api_key=GOOGLE_API_KEY)
    utils.initialize_mistral(api_key=MISTRAL_API_KEY)

    # Initialize Generative Models
    text_model = genai.GenerativeModel('gemini-1.5-flash')
    vision_model = genai.GenerativeModel('gemini-1.5-flash') # Use a model that supports vision

except Exception as e:
    # This will prevent the app from even starting if keys are missing
    print(f"FATAL ERROR during initialization: {e}")
    exit()


# --- CONSTANTS ---
CHATS_DIR = "chats"
os.makedirs(CHATS_DIR, exist_ok=True)

# System instruction to force Markdown output (for model's persona)
SYSTEM_INSTRUCTION = {
    "role": "user",
    "parts": ["You are a helpful AI assistant. You must format all of your responses using Markdown. For tabular data, use Markdown tables. For code, use fenced code blocks with the language identifier (e.g., ```python)."]
}
SYSTEM_RESPONSE = {
    "role": "model",
    "parts": ["Okay, I understand. I will format all my responses in Markdown, using tables for data and fenced code blocks for code snippets."]
}


# --- HELPER FUNCTIONS ---

def is_valid_uuid(val):
    """Check if a string is a valid UUID."""
    try:
        # A simple regex is faster than uuid.UUID() for validation
        return bool(re.match(r'^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$', val))
    except (TypeError, AttributeError):
        return False

def get_user_chat_dir(user_id):
    """Get the chat directory for a given user, ensuring the user_id is valid."""
    if not is_valid_uuid(user_id):
        return None
    return os.path.join(CHATS_DIR, user_id)

def get_conversational_chain():
    """Create and return a LangChain question-answering chain."""
    prompt_template = """
    You are an expert analyst. Answer the question as detailed as possible from the provided context.
    Make sure to provide all the details, elaborate on every point, and cite the source file if possible.
    Provide examples, explanations, and any relevant information. If the answer is not in the
    provided context, just say, "The answer is not available in the provided documents."
    Do not provide incorrect information.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=1.0)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# --- CHAT HISTORY MANAGEMENT ---

def get_chat_history_list(user_id):
    """Load a list of chat titles and IDs for a user."""
    user_chat_dir = get_user_chat_dir(user_id)
    if not user_chat_dir or not os.path.exists(user_chat_dir):
        return []

    histories = []
    files = [f for f in os.listdir(user_chat_dir) if f.endswith(".json")]
    for filename in files:
        try:
            filepath = os.path.join(user_chat_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
            histories.append({
                "id": filename.replace(".json", ""),
                "title": data.get("title", "Untitled Chat")
            })
        except Exception as e:
            print(f"Error loading chat title from {filename}: {e}")
            # Skip corrupted files
    # Sort by modification time, newest first
    histories.sort(key=lambda x: os.path.getmtime(os.path.join(user_chat_dir, f"{x['id']}.json")), reverse=True)
    return histories

def load_chat_conversation(user_id, chat_id):
    """Load a specific chat conversation from a JSON file."""
    if not is_valid_uuid(user_id) or not is_valid_uuid(chat_id):
        return None
    user_chat_dir = get_user_chat_dir(user_id)
    filepath = os.path.join(user_chat_dir, f"{chat_id}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading or decoding chat file {filepath}: {e}")
            return None
    return None

def save_chat_conversation(user_id, chat_id, conversation_data):
    """Save a chat conversation to a JSON file."""
    if not is_valid_uuid(user_id) or not is_valid_uuid(chat_id):
        raise ValueError("Invalid user_id or chat_id for saving conversation.")
    user_chat_dir = get_user_chat_dir(user_id)
    os.makedirs(user_chat_dir, exist_ok=True)
    filepath = os.path.join(user_chat_dir, f"{chat_id}.json")
    with open(filepath, 'w') as f:
        json.dump(conversation_data, f, indent=2)

# --- FLASK ROUTES ---

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

@app.route('/api/user/new', methods=['GET'])
def create_new_user():
    """Generate a new unique user ID."""
    return jsonify({"user_id": str(uuid.uuid4())})

# --- HISTORY ROUTES ---

@app.route('/api/history', methods=['GET'])
def list_history():
    """Get the list of all chat histories for a user."""
    try:
        user_id = request.headers.get('X-User-ID')
        if not user_id:
            return jsonify({"error": "User ID is required in X-User-ID header"}), 401
        return jsonify(get_chat_history_list(user_id))
    except Exception as e:
        print(f"ERROR in /api/history: {e}")
        return jsonify({"error": "An internal server error occurred while fetching history."}), 500

@app.route('/api/history/<chat_id>', methods=['GET'])
def get_history_by_id(chat_id):
    """Get a specific chat conversation by its ID."""
    try:
        user_id = request.headers.get('X-User-ID')
        if not user_id:
            return jsonify({"error": "User ID is required in X-User-ID header"}), 401
        conversation = load_chat_conversation(user_id, chat_id)
        if conversation:
            return jsonify(conversation)
        return jsonify({"error": "Chat not found"}), 404
    except Exception as e:
        print(f"ERROR in /api/history/<chat_id>: {e}")
        return jsonify({"error": "An internal server error occurred while fetching the chat."}), 500


@app.route('/api/history/<chat_id>/export', methods=['GET'])
def export_history_to_pdf(chat_id):
    """Export a chat history to a PDF file."""
    try:
        user_id = request.headers.get('X-User-ID')
        if not user_id:
            return jsonify({"error": "User ID is required"}), 401

        conversation = load_chat_conversation(user_id, chat_id)
        if not conversation:
            return jsonify({"error": "Chat not found"}), 404

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)

        # PDF styling
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='TitleStyle', fontSize=18, leading=22, spaceAfter=20))
        styles.add(ParagraphStyle(name='UserStyle', backColor=colors.HexColor('#e0f7fa'), borderColor=colors.HexColor('#b2ebf2'), borderWidth=1, padding=8, borderRadius=5, spaceBefore=10, spaceAfter=5))
        styles.add(ParagraphStyle(name='BotStyle', backColor=colors.whitesmoke, borderColor=colors.lightgrey, borderWidth=1, padding=8, borderRadius=5, spaceBefore=5, spaceAfter=10))
        styles.add(ParagraphStyle(name='CodeStyle', fontName='Courier', fontSize=9, leading=11, textColor=colors.darkblue, backColor=colors.HexColor('#f0f0f0'), padding=8, borderRadius=3, spaceBefore=5, spaceAfter=5, leftIndent=12, rightIndent=12, allowSplitting=1))

        story = []
        title = conversation.get('title', 'Chat History')
        story.append(Paragraph(f"Chat History: {title}", styles['TitleStyle']))

        for msg in conversation.get('messages', []):
            role = msg.get('role')
            content = "".join(msg.get('parts', [])).strip()
            if not content: continue

            if role == 'user':
                p_text = f"<b>You:</b> {content.replace('/n', '<br/>')}"
                story.append(Paragraph(p_text, styles['UserStyle']))
            elif role == 'model':
                story.append(Paragraph("<b>Assistant:</b>", styles['Normal']))
                # Handle code blocks separately for proper rendering in PDF
                code_blocks = re.split(r'(```[\s\S]*?```)', content)
                for i, block in enumerate(code_blocks):
                    if block.strip():
                        if i % 2 == 1: # Code block
                            code_content = block.strip().replace('```', '')
                            # Remove language identifier if present
                            code_content = re.sub(r'^\w+\n', '', code_content, count=1)
                            story.append(Preformatted(code_content, styles['CodeStyle']))
                        else: # Regular text
                            p_text = block.replace('\n', '<br/>')
                            story.append(Paragraph(p_text, styles['BotStyle']))
                story.append(Spacer(1, 12))

        doc.build(story)
        buffer.seek(0)
        return Response(
            buffer,
            mimetype='application/pdf',
            headers={'Content-Disposition': f'attachment;filename=chat-history-{chat_id}.pdf'}
        )
    except Exception as e:
        print(f"ERROR generating PDF for chat {chat_id}: {e}")
        return jsonify({"error": "Failed to generate PDF. An internal server error occurred."}), 500

# --- CHAT & MEDIA GENERATION ROUTES ---

@app.route('/api/generate/image', methods=['POST'])
def generate_image_route():
    try:
        data = request.json
        prompt = data.get('prompt')
        if not prompt:
            return jsonify({"success": False, "error": "A description prompt is required."}), 400
        num_images = data.get('num_images', 1)
        result = utils.generate_images_from_prompt(prompt, num_images)
        return jsonify(result), 200 if result.get("success") else 500
    except Exception as e:
        print(f"ERROR in /api/generate/image: {e}")
        return jsonify({"success": False, "error": "An unexpected server error occurred."}), 500

@app.route('/api/generate/video', methods=['POST'])
def generate_video_route():
    try:
        data = request.json
        prompt = data.get('prompt')
        if not prompt:
            return jsonify({"success": False, "error": "A description prompt is required."}), 400
        result = utils.generate_video_from_prompt(prompt)
        return jsonify(result), 200 if result.get("success") else 500
    except Exception as e:
        print(f"ERROR in /api/generate/video: {e}")
        return jsonify({"success": False, "error": "An unexpected server error occurred."}), 500

# --- CORE CHAT LOGIC HELPERS ---

def _handle_large_file_query(prompt, files):
    """Processes a query that involves large file uploads (RAG)."""
    raw_text = utils.process_large_uploaded_files(files)
    if not raw_text or "ERROR" in raw_text:
        raise ValueError(raw_text or "No text could be extracted from the large files.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    text_chunks = text_splitter.split_text(raw_text)
    if not text_chunks:
        raise ValueError("Could not split the document content into text chunks.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # Adjusted model
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    docs = vector_store.similarity_search(prompt, k=5)

    if not docs:
        return "I couldn't find any relevant information in the uploaded documents to answer your question."

    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs, "question": prompt}, return_only_outputs=True)
    return response['output_text']

def _handle_standard_query(prompt, conversation, regular_file, youtube_url, website_url):
    """Processes a standard query with optional small attachments or URLs."""
    model_input = [prompt or "Describe the content provided."]
    context_text = ""
    is_vision_request = False

    if regular_file:
        processed_content = utils.process_uploaded_file(regular_file)
        if isinstance(processed_content, Image.Image):
            model_input.append(processed_content)
            is_vision_request = True
        elif "ERROR:" in str(processed_content):
            raise ValueError(processed_content)
        else:
            context_text += f"File Content ({regular_file.filename}):\n{processed_content}\n\n"

    if youtube_url:
        context_text += f"YouTube Video Transcript:\n{utils.get_youtube_transcript(youtube_url)}\n\n"

    if website_url:
        context_text += f"Website Content:\n{utils.get_website_content(website_url)}\n\n"

    if context_text:
        model_input[0] = f"Based on the following context:\n\n---\n{context_text}---\n\nUser query: {model_input[0]}"

    model_to_use = vision_model if is_vision_request else text_model
    gemini_history = [SYSTEM_INSTRUCTION, SYSTEM_RESPONSE] + [
        msg for msg in conversation.get("messages", []) if 'parts' in msg and msg['parts']
    ]

    chat_session = model_to_use.start_chat(history=gemini_history)
    response = chat_session.send_message(model_input)
    return response.text

# --- MAIN CHAT ROUTE (NON-STREAMING) ---

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        # --- 1. Get and Validate Inputs ---
        user_id = request.headers.get('X-User-ID')
        if not user_id:
            return jsonify({"error": "User ID is required in X-User-ID header."}), 401

        prompt = request.form.get('prompt', '')
        chat_id = request.form.get('chat_id')
        upload_type = request.form.get('upload_type')
        large_files = request.files.getlist('large_files[]')
        regular_file = request.files.get('file')
        youtube_url = request.form.get('youtube_url')
        website_url = request.form.get('website_url')

        if not any([prompt, regular_file, youtube_url, website_url, large_files]):
            return jsonify({"error": "A prompt or an attachment is required."}), 400

        # --- 2. Load or Create Conversation ---
        is_new_chat = not chat_id or chat_id in ['null', 'undefined']
        if is_new_chat:
            chat_id = str(uuid.uuid4())
            title_prompt = f"Create a very short, concise title (4-5 words max) for this user prompt: '{prompt[:100]}'"
            title_response = text_model.generate_content(title_prompt)
            chat_title = title_response.text.strip().replace('"', '')
            conversation = {"title": chat_title, "messages": []}
        else:
            conversation = load_chat_conversation(user_id, chat_id)
            if not conversation:
                # If chat_id is provided but not found, treat as an error or create new
                 return jsonify({"error": f"Chat with ID {chat_id} not found."}), 404

        # --- 3. Process Request and Generate Response ---
        final_response = ""
        if upload_type == 'large' and large_files:
            final_response = _handle_large_file_query(prompt, large_files)
        else:
            final_response = _handle_standard_query(prompt, conversation, regular_file, youtube_url, website_url)

        # --- 4. Save Conversation and Return Response ---
        user_message = {"role": "user", "parts": [prompt]} # Simplified for history
        model_message = {"role": "model", "parts": [final_response]}
        conversation["messages"].extend([user_message, model_message])

        save_chat_conversation(user_id, chat_id, conversation)
        
        response_data = {
            "response": final_response,
            "chatId": chat_id
        }
        if is_new_chat:
            response_data["title"] = conversation["title"]

        return jsonify(response_data)

    except Exception as e:
        # Generic catch-all for any unexpected errors
        print(f"FATAL ERROR in /api/chat: {type(e).__name__} - {e}")
        # Consider logging the full traceback here for debugging
        # import traceback
        # traceback.print_exc()
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500


# --- APP STARTUP ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    # debug=False is recommended for production
    app.run(host="0.0.0.0", port=port, debug=True)
