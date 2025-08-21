# app.py (Updated for Multi-User Support)

import os
import json
import uuid
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import utils

# --- INITIALIZATION ---
load_dotenv()
app = Flask(__name__)

# Configure APIs
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not GOOGLE_API_KEY or not MISTRAL_API_KEY:
    raise ValueError("API keys for Google and Mistral are not set in .env file.")

genai.configure(api_key=GOOGLE_API_KEY)
utils.initialize_mistral(api_key=MISTRAL_API_KEY)

# Models
text_model = genai.GenerativeModel('gemini-2.5-flash')
vision_model = genai.GenerativeModel('gemini-2.5-flash')

# NEW: Main directory for all user chats
CHATS_DIR = "user_chats"
os.makedirs(CHATS_DIR, exist_ok=True)

# System instruction for Markdown formatting (no changes here)
SYSTEM_INSTRUCTION = {
    "role": "user",
    "parts": ["You are a helpful AI assistant. You must format all of your responses using Markdown. For tabular data, use Markdown tables. For code, use fenced code blocks with the language identifier (e.g., ```python)."]
}
SYSTEM_RESPONSE = {
    "role": "model",
    "parts": ["Okay, I understand. I will format all my responses in Markdown, using tables for data and fenced code blocks for code snippets."]
}


# --- NEW: USER-SPECIFIC CHAT HISTORY MANAGEMENT ---

def get_user_chat_dir(user_guid):
    """Gets the path to a user's chat directory, creating it if it doesn't exist."""
    if not user_guid or not isinstance(user_guid, str) or '..' in user_guid:
        raise ValueError("Invalid user GUID.")
    user_dir = os.path.join(CHATS_DIR, user_guid)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def get_chat_history_list(user_guid):
    """Lists chat history for a specific user."""
    user_dir = get_user_chat_dir(user_guid)
    files = os.listdir(user_dir)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(user_dir, x)), reverse=True)
    histories = []
    for filename in files:
        if filename.endswith(".json"):
            try:
                with open(os.path.join(user_dir, filename), 'r') as f:
                    data = json.load(f)
                    histories.append({
                        "id": filename.replace(".json", ""),
                        "title": data.get("title", "Untitled Chat")
                    })
            except Exception as e:
                print(f"Error loading chat title for user {user_guid} from {filename}: {e}")
    return histories

def load_chat_conversation(user_guid, chat_id):
    """Loads a specific chat conversation for a user."""
    user_dir = get_user_chat_dir(user_guid)
    filepath = os.path.join(user_dir, f"{chat_id}.json")
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def save_chat_conversation(user_guid, chat_id, conversation_data):
    """Saves a chat conversation for a user."""
    user_dir = get_user_chat_dir(user_guid)
    filepath = os.path.join(user_dir, f"{chat_id}.json")
    with open(filepath, 'w') as f:
        json.dump(conversation_data, f, indent=2)

# --- FLASK ROUTES (UPDATED FOR USER GUID) ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/history', methods=['GET'])
def list_history():
    user_guid = request.headers.get('X-User-GUID')
    if not user_guid:
        return jsonify({"error": "User GUID is required."}), 400
    return jsonify(get_chat_history_list(user_guid))

@app.route('/api/history/<chat_id>', methods=['GET'])
def get_history_by_id(chat_id):
    user_guid = request.headers.get('X-User-GUID')
    if not user_guid:
        return jsonify({"error": "User GUID is required."}), 400
    conversation = load_chat_conversation(user_guid, chat_id)
    if conversation:
        return jsonify(conversation)
    return jsonify({"error": "Chat not found"}), 404

@app.route('/api/chat', methods=['POST'])
def chat():
    # --- 1. GET USER GUID AND PARSE DATA ---
    user_guid = request.headers.get('X-User-GUID')
    if not user_guid:
        return jsonify({"error": "User GUID is required."}), 400

    prompt = request.form.get('prompt', '')
    chat_id = request.form.get('chat_id')
    file = request.files.get('file')
    youtube_url = request.form.get('youtube_url')
    website_url = request.form.get('website_url')

    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    # --- 2. PREPARE MODEL INPUT (No changes) ---
    model_input = [prompt]
    context_text = ""
    is_vision_request = False

    if file:
        processed_content = utils.process_uploaded_file(file)
        if isinstance(processed_content, Image.Image):
            model_input.append(processed_content)
            is_vision_request = True
        else:
            context_text += processed_content + "\n\n"

    if youtube_url:
        context_text += f"YouTube Video Transcript for {youtube_url}:\n\n{utils.get_youtube_transcript(youtube_url)}\n\n"

    if website_url:
        context_text += f"Website Content for {website_url}:\n\n{utils.get_website_content(website_url)}\n\n"

    if context_text:
        model_input[0] = f"Based on the following context:\n{context_text}\n\nUser query: {prompt}"

    # --- 3. HANDLE CHAT HISTORY (UPDATED for user_guid) ---
    if not chat_id or chat_id == 'null':
        chat_id = str(uuid.uuid4())
        title_prompt = f"Create a very short, concise title (4-5 words max) for this user prompt: '{prompt}'"
        title_response = text_model.generate_content(title_prompt)
        chat_title = title_response.text.strip().replace('"', '')
        conversation = {"title": chat_title, "messages": []}
    else:
        conversation = load_chat_conversation(user_guid, chat_id) or {"title": "Chat", "messages": []}

    user_message = {"role": "user", "parts": [prompt]}
    conversation["messages"].append(user_message)

    # --- 4. STREAM RESPONSE FROM GEMINI (UPDATED for user_guid) ---
    model = vision_model if is_vision_request else text_model

    def generate_and_stream():
        full_response = ""
        try:
            gemini_history = [SYSTEM_INSTRUCTION, SYSTEM_RESPONSE] + [msg for msg in conversation["messages"] if 'parts' in msg]
            chat_session = model.start_chat(history=gemini_history[:-1])
            stream = chat_session.send_message(model_input, stream=True)

            for chunk in stream:
                if chunk.text:
                    full_response += chunk.text
                    yield chunk.text

            conversation["messages"].append({"role": "model", "parts": [full_response]})
            save_chat_conversation(user_guid, chat_id, conversation)

        except Exception as e:
            print(f"Error during generation: {e}")
            error_message = f"Sorry, an error occurred: {e}"
            yield error_message
            conversation["messages"].append({"role": "model", "parts": [error_message]})
            save_chat_conversation(user_guid, chat_id, conversation)

    response = Response(stream_with_context(generate_and_stream()), mimetype='text/plain')
    response.headers['X-Chat-Id'] = chat_id
    return response

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
