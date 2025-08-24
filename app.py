# app.py (Fixed Mobile Tab Switching, Screen Off, and Response Generation Issues)

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
                print(f"Error loading chat title for user {user_guid} from {filename}: {e}")
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

    # --- 4. STREAM RESPONSE FROM GEMINI (WITH ROBUST ERROR HANDLING) ---
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
                    "start_time": time.time()
                }

            def try_send_message(input_text, attempt=1):
                """Try sending to Gemini, retry with improved prompt if empty."""
                nonlocal full_response
                got_valid_text = False

                try:
                    stream = chat_session.send_message(input_text, stream=True)
                    for chunk in stream:
                        # Check if client is still connected
                        if not is_session_active(user_guid):
                            return
                            
                        if hasattr(chunk, "text") and chunk.text:
                            got_valid_text = True
                            full_response += chunk.text
                            yield chunk.text

                    # Retry with improved prompt if Gemini returned nothing
                    if not got_valid_text and attempt <= 3:  # Try up to 3 times
                        if attempt == 1:
                            retry_input = f"Please provide a helpful, safe, and complete response to this user request: {input_text}"
                        elif attempt == 2:
                            retry_input = f"I didn't receive a response. Please answer this query in detail: {input_text}"
                        else:
                            retry_input = f"Provide a comprehensive response to: {input_text}"
                            
                        yield from try_send_message(retry_input, attempt=attempt+1)

                    # Final fallback
                    if not got_valid_text and attempt > 3:
                        safe_msg = "I'm here to help, but I couldn't provide details for that request. Please try rephrasing."
                        full_response = safe_msg
                        yield safe_msg
                except Exception as e:
                    if attempt <= 2:  # Retry on error
                        yield from try_send_message(input_text, attempt=attempt+1)
                    else:
                        # If we're still having issues, try a fresh generation
                        if attempt == 3:
                            try:
                                # Start a fresh chat session
                                fresh_chat = model.start_chat(history=gemini_history[:-1])
                                fresh_stream = fresh_chat.send_message(input_text, stream=True)
                                for chunk in fresh_stream:
                                    if hasattr(chunk, "text") and chunk.text:
                                        got_valid_text = True
                                        full_response += chunk.text
                                        yield chunk.text
                                
                                if not got_valid_text:
                                    raise Exception("Fresh generation also failed")
                            except Exception as retry_error:
                                raise e from retry_error  # Re-raise original error
                        else:
                            raise e

            # Run first attempt
            yield from try_send_message(model_input)

            # Save response
            conversation["messages"].append({"role": "model", "parts": [full_response]})
            save_chat_conversation(user_guid, chat_id, conversation)

            # Clean up ongoing generation tracking
            with session_lock:
                if user_guid in ongoing_generations:
                    ongoing_generations.pop(user_guid, None)

        except Exception as e:
            print(f"Error during generation: {e}")
            error_message = "⚠️ Sorry, something went wrong while generating your response. Please try again."
            
            # If we have a partial response, include it in the error message
            if full_response:
                error_message = f"⚠️ Response generation was interrupted. Partial response:\n\n{full_response}\n\nPlease try again for a complete response."
            
            yield error_message
            conversation["messages"].append({"role": "model", "parts": [error_message]})
            save_chat_conversation(user_guid, chat_id, conversation)
            
            # Clean up ongoing generation tracking
            with session_lock:
                if user_guid in ongoing_generations:
                    ongoing_generations.pop(user_guid, None)

    response = Response(stream_with_context(generate_and_stream()), mimetype='text/plain')
    response.headers['X-Chat-Id'] = chat_id
    return response

@app.route('/api/resume-generation', methods=['POST'])
def resume_generation():
    """Endpoint to resume interrupted generation"""
    user_guid = request.headers.get('X-User-GUID')
    if not user_guid:
        return jsonify({"error": "User GUID is required."}), 400
        
    with session_lock:
        if user_guid not in ongoing_generations:
            return jsonify({"error": "No ongoing generation to resume"}), 404
            
        generation_info = ongoing_generations[user_guid]
        chat_id = generation_info["chat_id"]
        model_input = generation_info["model_input"]
        
        # Remove from ongoing generations
        ongoing_generations.pop(user_guid, None)
    
    # Load the conversation
    conversation = load_chat_conversation(user_guid, chat_id) or {"title": "Chat", "messages": []}
    
    # Continue generation
    model = text_model  # Default to text model for resuming
    
    def resume_and_stream():
        full_response = ""
        try:
            gemini_history = [SYSTEM_INSTRUCTION, SYSTEM_RESPONSE] + [
                msg for msg in conversation["messages"] if 'parts' in msg
            ]
            chat_session = model.start_chat(history=gemini_history[:-1])
            
            # Track this resumed generation
            with session_lock:
                ongoing_generations[user_guid] = {
                    "chat_id": chat_id,
                    "prompt": generation_info["prompt"],
                    "model_input": model_input,
                    "start_time": time.time(),
                    "resumed": True
                }
            
            # Continue from where we left off
            stream = chat_session.send_message(model_input, stream=True)
            for chunk in stream:
                if not is_session_active(user_guid):
                    return
                    
                if hasattr(chunk, "text") and chunk.text:
                    full_response += chunk.text
                    yield chunk.text
            
            # Save the completed response
            conversation["messages"].append({"role": "model", "parts": [full_response]})
            save_chat_conversation(user_guid, chat_id, conversation)
            
            # Clean up ongoing generation tracking
            with session_lock:
                if user_guid in ongoing_generations:
                    ongoing_generations.pop(user_guid, None)
                    
        except Exception as e:
            print(f"Error during resumed generation: {e}")
            error_message = "⚠️ Sorry, something went wrong while resuming your response."
            
            # If we have a partial response, include it
            if full_response:
                error_message = f"⚠️ Resumed generation was interrupted. Partial response:\n\n{full_response}\n\nPlease try again for a complete response."
            
            yield error_message
            conversation["messages"].append({"role": "model", "parts": [error_message]})
            save_chat_conversation(user_guid, chat_id, conversation)
            
            # Clean up ongoing generation tracking
            with session_lock:
                if user_guid in ongoing_generations:
                    ongoing_generations.pop(user_guid, None)
    
    return Response(stream_with_context(resume_and_stream()), mimetype='text/plain')

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
