# app.py (Mobile-Specific Fixes for Tab Switching and Screen Off)

import os
import json
import uuid
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import utils
import time
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Main directory for all user chats
CHATS_DIR = "user_chats"
os.makedirs(CHATS_DIR, exist_ok=True)

# System instruction for Markdown formatting
SYSTEM_INSTRUCTION = {
    "role": "user",
    "parts": ["You are a helpful AI assistant. You must format all of your responses using Markdown. For tabular data, use Markdown tables. For code, use fenced code blocks with the language identifier (e.g., ```python)."]
}
SYSTEM_RESPONSE = {
    "role": "model",
    "parts": ["Okay, I understand. I will format all my responses in Markdown, using tables for data and fenced code blocks for code snippets."]
}

# Session management with extended timeout
active_sessions = {}
# Track ongoing generations to resume if interrupted
ongoing_generations = {}
# Lock for thread-safe operations
session_lock = threading.Lock()

# --- USER-SPECIFIC CHAT HISTORY MANAGEMENT ---

def get_user_chat_dir(user_guid):
    if not user_guid or not isinstance(user_guid, str) or '..' in user_guid:
        raise ValueError("Invalid user GUID.")
    user_dir = os.path.join(CHATS_DIR, user_guid)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def get_chat_history_list(user_guid):
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
                logger.error(f"Error loading chat title for user {user_guid} from {filename}: {e}")
    return histories

def load_chat_conversation(user_guid, chat_id):
    user_dir = get_user_chat_dir(user_guid)
    filepath = os.path.join(user_dir, f"{chat_id}.json")
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def save_chat_conversation(user_guid, chat_id, conversation_data):
    user_dir = get_user_chat_dir(user_guid)
    filepath = os.path.join(user_dir, f"{chat_id}.json")
    with open(filepath, 'w') as f:
        json.dump(conversation_data, f, indent=2)

# --- SESSION MANAGEMENT ---

def update_session_activity(user_guid):
    """Update session activity with thread safety"""
    with session_lock:
        active_sessions[user_guid] = time.time()

def is_session_active(user_guid):
    """Check if session is still active with generous timeout"""
    with session_lock:
        if user_guid not in active_sessions:
            return False
        # 24-hour session timeout
        return time.time() - active_sessions[user_guid] < 86400

def cleanup_inactive_sessions():
    """Clean up inactive sessions periodically"""
    with session_lock:
        current_time = time.time()
        inactive_users = [user for user, last_active in active_sessions.items() 
                         if current_time - last_active > 86400]  # 24 hours
        for user in inactive_users:
            active_sessions.pop(user, None)
            # Also clean up any ongoing generations for this user
            if user in ongoing_generations:
                ongoing_generations.pop(user, None)

# --- MOBILE-SPECIFIC FIXES ---

def is_mobile_request():
    """Check if the request is from a mobile device"""
    user_agent = request.headers.get('User-Agent', '').lower()
    mobile_keywords = ['mobile', 'android', 'iphone', 'ipad', 'ipod', 'blackberry', 'windows phone']
    return any(keyword in user_agent for keyword in mobile_keywords)

def handle_mobile_interruption(user_guid, chat_id, prompt, model_input, is_vision):
    """Handle mobile-specific interruptions by resuming generation"""
    try:
        # Load the conversation
        conversation = load_chat_conversation(user_guid, chat_id) or {"title": "Chat", "messages": []}
        
        # Check if we already have a partial response
        last_message = conversation["messages"][-1] if conversation["messages"] else None
        if last_message and last_message.get("role") == "model" and last_message.get("parts"):
            # We already have a response, no need to regenerate
            return True, last_message["parts"][0]
        
        # Regenerate the response
        model = vision_model if is_vision else text_model
        gemini_history = [SYSTEM_INSTRUCTION, SYSTEM_RESPONSE] + [
            msg for msg in conversation["messages"] if 'parts' in msg and msg['role'] != 'model'
        ]
        
        chat_session = model.start_chat(history=gemini_history)
        response = chat_session.send_message(model_input)
        
        if response and response.text:
            # Save the response
            conversation["messages"].append({"role": "model", "parts": [response.text]})
            save_chat_conversation(user_guid, chat_id, conversation)
            return True, response.text
    
    except Exception as e:
        logger.error(f"Error handling mobile interruption: {e}")
    
    return False, "⚠️ Sorry, something went wrong while generating your response. Please try again."

# --- FLASK ROUTES ---

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

@app.route('/api/keepalive', methods=['POST'])
def keepalive():
    """Endpoint to keep session alive with periodic heartbeats"""
    user_guid = request.headers.get('X-User-GUID')
    if user_guid:
        update_session_activity(user_guid)
        return jsonify({"status": "ok"})
    return jsonify({"error": "User GUID is required."}), 400

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

    # Update session activity
    update_session_activity(user_guid)

    # --- 2. PREPARE MODEL INPUT ---
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

    # --- 3. HANDLE CHAT HISTORY ---
    if not chat_id or chat_id == 'null':
        chat_id = str(uuid.uuid4())
        try:
            title_prompt = f"Create a very short, concise title (4-5 words max) for this user prompt: '{prompt}'"
            title_response = text_model.generate_content(title_prompt)
            chat_title = (title_response.text or "New Chat").strip().replace('"', '')
        except Exception:
            chat_title = "New Chat"
        conversation = {"title": chat_title, "messages": []}
    else:
        conversation = load_chat_conversation(user_guid, chat_id) or {"title": "Chat", "messages": []}

    user_message = {"role": "user", "parts": [prompt]}
    conversation["messages"].append(user_message)

    # --- 4. STREAM RESPONSE FROM GEMINI (WITH MOBILE-SPECIFIC HANDLING) ---
    model = vision_model if is_vision_request else text_model

    def generate_and_stream():
        full_response = ""
        try:
            # Check if session is still active with generous timeout
            if not is_session_active(user_guid):
                error_message = "⚠️ Session expired due to long inactivity. Please refresh the page."
                yield error_message
                conversation["messages"].append({"role": "model", "parts": [error_message]})
                save_chat_conversation(user_guid, chat_id, conversation)
                return

            gemini_history = [SYSTEM_INSTRUCTION, SYSTEM_RESPONSE] + [
                msg for msg in conversation["messages"] if 'parts' in msg
            ]
            chat_session = model.start_chat(history=gemini_history[:-1])

            # Track this generation
            with session_lock:
                ongoing_generations[user_guid] = {
                    "chat_id": chat_id,
                    "prompt": prompt,
                    "model_input": model_input,
                    "is_vision": is_vision_request,
                    "start_time": time.time()
                }

            # For mobile devices, use a different approach to handle interruptions
            if is_mobile_request():
                # For mobile, we'll generate the complete response first, then stream it
                # This prevents interruptions from breaking the generation
                response = chat_session.send_message(model_input)
                if response and response.text:
                    full_response = response.text
                    # Stream the response in chunks to simulate real-time generation
                    words = full_response.split()
                    for i in range(0, len(words), 3):  # Send 3 words at a time
                        if not is_session_active(user_guid):
                            break
                        chunk = " ".join(words[i:i+3]) + " "
                        yield chunk
                        time.sleep(0.05)  # Small delay to simulate streaming
                else:
                    yield "⚠️ Sorry, I couldn't generate a response. Please try again."
            else:
                # For desktop, use regular streaming
                stream = chat_session.send_message(model_input, stream=True)
                for chunk in stream:
                    # Check if client is still connected
                    if not is_session_active(user_guid):
                        break
                        
                    if hasattr(chunk, "text") and chunk.text:
                        full_response += chunk.text
                        yield chunk.text

            # Save response
            conversation["messages"].append({"role": "model", "parts": [full_response]})
            save_chat_conversation(user_guid, chat_id, conversation)

            # Clean up ongoing generation tracking
            with session_lock:
                if user_guid in ongoing_generations:
                    ongoing_generations.pop(user_guid, None)

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            
            # For mobile devices, try to handle the interruption gracefully
            if is_mobile_request():
                success, result = handle_mobile_interruption(user_guid, chat_id, prompt, model_input, is_vision_request)
                if success:
                    yield result
                else:
                    yield result
            else:
                error_message = "⚠️ Sorry, something went wrong while generating your response. Please try again."
                yield error_message
                conversation["messages"].append({"role": "model", "parts": [error_message]})
                save_chat_conversation(user_guid, chat_id, conversation)
            
            # Clean up ongoing generation tracking
            with session_lock:
                if user_guid in ongoing_generations:
                    ongoing_generations.pop(user_guid, None)

    response = Response(stream_with_context(generate_and_stream()), mimetype='text/plain')
    response.headers['X-Chat-Id'] = chat_id
    response.headers['X-Is-Mobile'] = str(is_mobile_request()).lower()
    return response

@app.route('/api/mobile-resume', methods=['POST'])
def mobile_resume():
    """Special endpoint for mobile devices to resume interrupted generations"""
    user_guid = request.headers.get('X-User-GUID')
    if not user_guid:
        return jsonify({"error": "User GUID is required."}), 400
        
    chat_id = request.form.get('chat_id')
    prompt = request.form.get('prompt')
    
    if not chat_id or not prompt:
        return jsonify({"error": "Chat ID and prompt are required."}), 400
        
    # Try to resume the generation
    conversation = load_chat_conversation(user_guid, chat_id) or {"title": "Chat", "messages": []}
    
    # Check if we already have a response
    if conversation.get("messages") and conversation["messages"][-1].get("role") == "model":
        return jsonify({
            "status": "completed", 
            "response": conversation["messages"][-1]["parts"][0]
        })
    
    # Try to generate a new response
    try:
        model = text_model  # Default to text model for resuming
        gemini_history = [SYSTEM_INSTRUCTION, SYSTEM_RESPONSE] + [
            msg for msg in conversation["messages"] if 'parts' in msg and msg['role'] != 'model'
        ]
        
        chat_session = model.start_chat(history=gemini_history)
        response = chat_session.send_message(prompt)
        
        if response and response.text:
            # Save the response
            conversation["messages"].append({"role": "model", "parts": [response.text]})
            save_chat_conversation(user_guid, chat_id, conversation)
            
            return jsonify({
                "status": "completed", 
                "response": response.text
            })
        else:
            return jsonify({
                "status": "error", 
                "message": "Failed to generate response"
            }), 500
            
    except Exception as e:
        logger.error(f"Error in mobile resume: {e}")
        return jsonify({
            "status": "error", 
            "message": "Internal server error"
        }), 500

# Clean up inactive sessions periodically
@app.before_request
def cleanup_sessions_before_request():
    cleanup_inactive_sessions()

# Background thread for periodic cleanup
def periodic_cleanup():
    while True:
        time.sleep(3600)  # Clean up every hour
        cleanup_inactive_sessions()

# Start cleanup thread
cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, threaded=True)
