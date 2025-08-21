# tanishqravulasuperaibot-main/app.py
# Updated and consolidated code

import os
import json
import uuid
import re
import time
import io
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import utils

# PDF Export Imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors

# LangChain Imports for Large File Processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

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

text_model = genai.GenerativeModel('gemini-2.5-flash')
vision_model = genai.GenerativeModel('gemini-2.5-flash')

# Chat history storage
CHATS_DIR = "chats"
os.makedirs(CHATS_DIR, exist_ok=True)

# System instruction to force Markdown output
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
    try:
        return bool(re.match(r'^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$', val))
    except (TypeError, AttributeError):
        return False

def get_user_chat_dir(user_id):
    if not is_valid_uuid(user_id): return None
    return os.path.join(CHATS_DIR, user_id)

def get_conversational_chain():
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
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# --- CHAT HISTORY MANAGEMENT ---
def get_chat_history_list(user_id):
    user_chat_dir = get_user_chat_dir(user_id)
    if not user_chat_dir or not os.path.exists(user_chat_dir): return []
    files = [f for f in os.listdir(user_chat_dir) if f.endswith(".json")]
    histories = []
    for filename in files:
        try:
            filepath = os.path.join(user_chat_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                histories.append({"id": filename.replace(".json", ""), "title": data.get("title", "Untitled Chat")})
        except Exception as e:
            print(f"Error loading chat title from {filename}: {e}")
    histories.sort(key=lambda x: os.path.getmtime(os.path.join(user_chat_dir, f"{x['id']}.json")), reverse=True)
    return histories

def load_chat_conversation(user_id, chat_id):
    user_chat_dir = get_user_chat_dir(user_id)
    if not user_chat_dir or not is_valid_uuid(chat_id): return None
    filepath = os.path.join(user_chat_dir, f"{chat_id}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f: return json.load(f)
        except json.JSONDecodeError: return None
    return None

def save_chat_conversation(user_id, chat_id, conversation_data):
    user_chat_dir = get_user_chat_dir(user_id)
    if not user_chat_dir or not is_valid_uuid(chat_id): raise ValueError("Invalid user_id or chat_id")
    os.makedirs(user_chat_dir, exist_ok=True)
    filepath = os.path.join(user_chat_dir, f"{chat_id}.json")
    with open(filepath, 'w', encoding='utf-8') as f: json.dump(conversation_data, f, indent=2)

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/user/new', methods=['GET'])
def create_new_user():
    return jsonify({"user_id": str(uuid.uuid4())})

@app.route('/api/history', methods=['GET'])
def list_history():
    user_id = request.headers.get('X-User-ID')
    if not user_id: return jsonify({"error": "User ID is required"}), 401
    return jsonify(get_chat_history_list(user_id))

@app.route('/api/history/<chat_id>', methods=['GET'])
def get_history_by_id(chat_id):
    user_id = request.headers.get('X-User-ID')
    if not user_id: return jsonify({"error": "User ID is required"}), 401
    conversation = load_chat_conversation(user_id, chat_id)
    if conversation: return jsonify(conversation)
    return jsonify({"error": "Chat not found"}), 404

@app.route('/api/history/<chat_id>/export', methods=['GET'])
def export_history_to_pdf(chat_id):
    user_id = request.headers.get('X-User-ID')
    if not user_id:
        return jsonify({"error": "User ID is required"}), 401

    conversation = load_chat_conversation(user_id, chat_id)
    if not conversation:
        return jsonify({"error": "Chat not found"}), 404

    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        styles = getSampleStyleSheet()
        story = []
        
        # Add Title
        title = conversation.get('title', 'Chat History').encode('utf-8').decode('latin-1')
        story.append(Paragraph(f"Chat History: {title}", styles['h1']))
        story.append(Spacer(1, 24))

        # Add Messages
        for msg in conversation.get('messages', []):
            role = msg.get('role')
            content = "".join(msg.get('parts', [])).strip()
            if not content: continue

            if role == 'user':
                story.append(Paragraph(f"<b>You:</b> {content.replace('/n', '<br/>')}", styles['Normal']))
            elif role == 'model':
                story.append(Paragraph("<b>Assistant:</b>", styles['Normal']))
                p_content = content.replace('\n', '<br/>')
                story.append(Paragraph(p_content, styles['BodyText']))
            story.append(Spacer(1, 12))
            
        doc.build(story)
        buffer.seek(0)
        return Response(
            buffer,
            mimetype='application/pdf',
            headers={'Content-Disposition': f'attachment;filename=chat-history-{chat_id}.pdf'}
        )
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return jsonify({"error": f"Failed to generate PDF: {str(e)}"}), 500

# --- MEDIA GENERATION ROUTES ---
@app.route('/api/generate/image', methods=['POST'])
def generate_image_route():
    try:
        data = request.json
        prompt = data.get('prompt')
        num_images = data.get('num_images', 1)
        if not prompt: return jsonify({"success": False, "error": "A description prompt is required."}), 400
        result = utils.generate_images_from_prompt(prompt, num_images)
        return jsonify(result), 200 if result.get("success") else 500
    except Exception as e:
        return jsonify({"success": False, "error": f"An unexpected server error occurred: {e}"}), 500

@app.route('/api/generate/video', methods=['POST'])
def generate_video_route():
    try:
        data = request.json
        prompt = data.get('prompt')
        if not prompt: return jsonify({"success": False, "error": "A description prompt is required."}), 400
        result = utils.generate_video_from_prompt(prompt)
        return jsonify(result), 200 if result.get("success") else 500
    except Exception as e:
        return jsonify({"success": False, "error": f"An unexpected server error occurred: {e}"}), 500

# --- CHAT ROUTE ---
@app.route('/api/chat', methods=['POST'])
def chat():
    user_id = request.headers.get('X-User-ID')
    if not user_id:
        def error_stream(): yield f"data: {json.dumps({'type': 'error', 'content': 'User ID is required.'})}\n\n"
        return Response(error_stream(), mimetype='text/event-stream')

    prompt = request.form.get('prompt', '')
    chat_id = request.form.get('chat_id')
    upload_type = request.form.get('upload_type')
    large_files = request.files.getlist('large_files[]')
    regular_file = request.files.get('file')
    website_url = request.form.get('website_url')

    if not prompt and not regular_file and not website_url and not large_files:
        def error_stream(): yield f"data: {json.dumps({'type': 'error', 'content': 'Prompt or an attachment is required.'})}\n\n"
        return Response(error_stream(), mimetype='text/event-stream')

    def generate_and_stream():
        nonlocal chat_id
        def yield_event(data):
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(0.01)

        try:
            if not chat_id or chat_id == 'null' or chat_id == 'undefined':
                chat_id = str(uuid.uuid4())
                title_prompt = f"Create a very short, concise title (4-5 words max) for this user prompt: '{prompt or 'New Media Chat'}'"
                title_response = text_model.generate_content(title_prompt)
                chat_title = title_response.text.strip().replace('"', '')
                conversation = {"title": chat_title, "messages": []}
                yield from yield_event({"type": "new_chat_info", "chatId": chat_id, "title": chat_title})
            else:
                conversation = load_chat_conversation(user_id, chat_id) or {"title": "Chat", "messages": []}
            
            # --- USER MESSAGE HANDLING ---
            user_parts = [prompt] if prompt else []
            # Append file name to prompt for context if not a vision request
            if regular_file: user_parts.append(f"(File Attached: {regular_file.filename})")
            if large_files: user_parts.append(f"({len(large_files)} Large Files Attached)")
            if website_url: user_parts.append(f"(URL Attached: {website_url})")
            user_message_content = " ".join(user_parts)
            user_message = {"role": "user", "parts": [user_message_content]}


            # --- LARGE FILE (RAG) PROCESSING ---
            if upload_type == 'large' and large_files:
                yield from yield_event({"type": "status", "content": f"Processing {len(large_files)} large file(s)..."})
                raw_text = utils.process_large_uploaded_files(large_files)
                if "ERROR" in raw_text or not raw_text.strip():
                    raise ValueError(raw_text or "No text could be extracted from the large files.")
                
                yield from yield_event({"type": "status", "content": "Creating text chunks..."})
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
                text_chunks = text_splitter.split_text(raw_text)
                if not text_chunks: raise ValueError("Could not split documents into text chunks.")

                yield from yield_event({"type": "status", "content": "Building vector store (in-memory)..."})
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")
                vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
                
                yield from yield_event({"type": "status", "content": "Searching for relevant documents..."})
                docs = vector_store.similarity_search(prompt, k=5)
                
                if not docs:
                    final_response = "I couldn't find any relevant information in the uploaded documents to answer your question."
                    yield from yield_event({"type": "chunk", "content": final_response})
                else:
                    yield from yield_event({"type": "status", "content": "Generating response from documents..."})
                    chain = get_conversational_chain()
                    response_data = chain.invoke({"input_documents": docs, "question": prompt}, return_only_outputs=True)
                    final_response = response_data['output_text']
                    yield from yield_event({"type": "chunk", "content": final_response})

            # --- REGULAR/VISION/URL PROCESSING ---
            else:
                model_input = [prompt] if prompt else ["Describe the provided content."]
                context_text = ""
                is_vision_request = False
                if regular_file:
                    yield from yield_event({"type": "status", "content": f"Processing file: {regular_file.filename}..."})
                    processed_content = utils.process_uploaded_file(regular_file)
                    if isinstance(processed_content, Image.Image):
                        model_input.append(processed_content)
                        is_vision_request = True
                    elif "ERROR:" in processed_content:
                        raise ValueError(processed_content)
                    else:
                        context_text += processed_content + "\n\n"
                if website_url:
                    youtube_transcript = utils.get_youtube_transcript(website_url)
                    if "ERROR" not in youtube_transcript:
                         yield from yield_event({"type": "status", "content": f"Fetching YouTube transcript..."})
                         context_text += f"YouTube Video Transcript for {website_url}:\n\n{youtube_transcript}\n\n"
                    else:
                         yield from yield_event({"type": "status", "content": f"Scraping website content..."})
                         context_text += f"Website Content for {website_url}:\n\n{utils.get_website_content(website_url)}\n\n"

                yield from yield_event({"type": "status", "content": "Context ready. Generating response..."})
                if context_text.strip():
                    model_input = f"Based on the following context:\n\n---\n{context_text}---\n\nUser query: {model_input}"
                
                model_to_use = vision_model if is_vision_request else text_model
                
                # Construct history, excluding potentially malformed messages
                gemini_history = [SYSTEM_INSTRUCTION, SYSTEM_RESPONSE] + \
                                 [msg for msg in conversation.get("messages", []) if 'parts' in msg and msg['parts']]

                chat_session = model_to_use.start_chat(history=gemini_history)
                stream = chat_session.send_message(model_input, stream=True)
                final_response = ""
                for chunk in stream:
                    if chunk.text:
                        final_response += chunk.text
                        yield from yield_event({"type": "chunk", "content": chunk.text})

            # Save conversation
            conversation["messages"].append(user_message)
            conversation["messages"].append({"role": "model", "parts": [final_response]})
            save_chat_conversation(user_id, chat_id, conversation)
            yield from yield_event({"type": "done", "content": "Stream finished."})

        except Exception as e:
            print(f"Error during generation stream: {e}")
            error_message = f"Sorry, an error occurred: {str(e)}"
            yield from yield_event({"type": "error", "content": error_message})

    return Response(stream_with_context(generate_and_stream()), mimetype='text/event-stream')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
