# tanishqravulasuperaibot-main/utils.py
# FINAL CORRECTED VERSION: Fixes the 'tuple' error and improves stability.

import os
import io
import time
import uuid
import math
import json
import requests
import pandas as pd
import openpyxl
from PIL import Image
import google.generativeai as genai
import tempfile
import shutil
import traceback
from PyPDF2 import PdfReader
import re # Import 're' for robust YouTube ID extraction

# Media Generation Imports
from g4f.client import Client as G4FClient
from gtts import gTTS
from moviepy.editor import ImageSequenceClip, AudioFileClip

# Document Processing Imports
import fitz  # PyMuPDF
from docx import Document as PyDocxDocument
from pptx import Presentation
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
import zipfile
import tarfile
from mistralai import Mistral, DocumentURLChunk
from pathlib import Path

# --- INITIALIZATION ---
mistral_client = None
g4f_client = G4FClient()

GENERATED_MEDIA_DIR = "static/generated_media"
os.makedirs(GENERATED_MEDIA_DIR, exist_ok=True)

def initialize_mistral(api_key):
    global mistral_client
    mistral_client = Mistral(api_key=api_key)

# --- MEDIA GENERATION ---
def generate_images_from_prompt(prompt, num_images=4, download_dir=None):
    if download_dir is None:
        download_dir = GENERATED_MEDIA_DIR
    image_urls, local_paths = [], []
    print(f"⏳ Generating {num_images} image(s) for prompt: {prompt}")
    for i in range(num_images):
        try:
            response = g4f_client.images.generate(model="dall-e-3", prompt=prompt, n=1, response_format="url")
            if response and hasattr(response, "data") and response.data:
                image_urls.append(response.data[0].url)
        except Exception as e:
            print(f"❌ [{i+1}/{num_images}] Error generating image: {e}")
    if not image_urls: return {"success": False, "error": "Failed to generate any image URLs."}
    
    print(f"⏳ Downloading {len(image_urls)} generated images...")
    for i, url in enumerate(image_urls):
        try:
            image_response = requests.get(url, timeout=45)
            if image_response.status_code == 200:
                file_name = f"image_{uuid.uuid4().hex[:8]}.jpg"
                file_path = os.path.join(download_dir, file_name)
                with open(file_path, "wb") as f: f.write(image_response.content)
                local_paths.append(file_path)
        except Exception as e: print(f"❌ Error downloading image from {url}: {e}")

    if not local_paths: return {"success": False, "error": "Failed to download generated images."}
    return {"success": True, "paths": local_paths}

def generate_video_from_prompt(prompt):
    temp_work_dir = tempfile.mkdtemp()
    try:
        script_model = genai.GenerativeModel('gemini-2.5-flash')
        script_response = script_model.generate_content(f"Create a short, engaging voiceover script (about 60-80 seconds, 11-16 sentences) for a video about: '{prompt}'. The script should be descriptive and suitable for a slideshow-style video.")
        script_text = script_response.text.replace("*", "").strip()
        if not script_text: raise Exception("Failed to generate script.")

        audio_path_local = os.path.join(temp_work_dir, f"audio_{uuid.uuid4().hex[:8]}.mp3")
        gTTS(text=script_text, lang='en', slow=False).save(audio_path_local)

        with AudioFileClip(audio_path_local) as audio_clip: audio_duration = audio_clip.duration
        if audio_duration <= 0: raise Exception("Generated audio has invalid duration.")
        num_images = max(5, min(math.ceil(audio_duration / 3.5), 15))

        image_result = generate_images_from_prompt(f"A series of high-quality, cinematic, varied images for a video about '{prompt}'.", num_images, download_dir=temp_work_dir)
        if not image_result["success"]: return image_result
        temp_image_files_local = image_result["paths"]

        for img_path in temp_image_files_local:
            with Image.open(img_path) as img:
                img.convert('RGB').resize((1280, 720), Image.Resampling.LANCZOS).save(img_path)

        fps = len(temp_image_files_local) / audio_duration
        video_clip = ImageSequenceClip(temp_image_files_local, fps=fps).set_audio(AudioFileClip(audio_path_local))
        
        video_filename = f"video_{uuid.uuid4().hex[:8]}.mp4"
        temp_video_path = os.path.join(temp_work_dir, video_filename)
        video_clip.write_videofile(temp_video_path, codec="libx264", audio_codec="aac")

        final_video_path = os.path.join(GENERATED_MEDIA_DIR, video_filename)
        shutil.move(temp_video_path, final_video_path)
        
        return {"success": True, "path": f"/{final_video_path.replace(os.path.sep, '/')}"}

    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": f"An unexpected error during video generation: {e}"}
    finally:
        if os.path.exists(temp_work_dir): shutil.rmtree(temp_work_dir)

# --- LARGE FILE-HANDLING (FOR RAG) ---
def extract_text_with_pypdf2(pdf_bytes):
    text = ""
    try:
        with io.BytesIO(pdf_bytes) as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        return f"ERROR: PyPDF2 failed. Details: {str(e)}"
    return text

def process_large_uploaded_files(files):
    all_text = ""
    for file_storage in files:
        filename = file_storage.filename
        # ### CRITICAL FIX ### The bug was here.
        # OLD: file_ext = os.path.splitext(filename).lower() -> This is a TUPLE.
        # NEW: Get the second element [1] which is the extension string.
        file_ext = os.path.splitext(filename)[1].lower()
        all_text += f"\n\n--- Content from {filename} ---\n\n"
        try:
            file_bytes = file_storage.read()
            file_storage.seek(0)
            if file_ext == '.pdf':
                all_text += extract_text_with_pypdf2(file_bytes)
            elif file_ext in ['.xlsx', '.xls']:
                all_text += pd.read_excel(io.BytesIO(file_bytes)).to_string()
            elif file_ext == '.csv':
                all_text += pd.read_csv(io.BytesIO(file_bytes)).to_string()
            else:
                all_text += file_bytes.decode('utf-8', errors='ignore')
        except Exception as e:
            all_text += f"ERROR processing file {filename}: {e}\n"
    return all_text

# --- REGULAR FILE PROCESSING ---
def process_pdf_with_mistral(file_data: bytes, file_name: str) -> str:
    # This remains largely the same, it was not the source of the crash.
    if not mistral_client: raise ValueError("Mistral client not initialized.")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name
        with open(tmp_path, "rb") as f:
            uploaded_file = mistral_client.files.upload(file={"file_name": Path(tmp_path).stem, "content": f.read()}, purpose="ocr")
        signed_url = mistral_client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
        response = mistral_client.ocr.process(document=DocumentURLChunk(document_url=signed_url.url), model="mistral-ocr-latest", include_image_base64=False)
        return "\n\n".join([page.markdown for page in response.pages])
    except Exception as e:
        traceback.print_exc()
        return f"ERROR: Mistral PDF processing failed. Details: {str(e)}"
    finally:
        if tmp_path and os.path.exists(tmp_path): os.unlink(tmp_path)

def extract_text_from_docx(docx_file):
    doc = PyDocxDocument(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_ppt(ppt_file):
    presentation = Presentation(ppt_file)
    return "".join(shape.text + "\n" for slide in presentation.slides for shape in slide.shapes if hasattr(shape, "text"))

def get_youtube_transcript(video_url):
    # ### ROBUSTNESS IMPROVEMENT ### Handles more YouTube URL types.
    try:
        youtube_regex = (r'(https?://)?(www\.)?'
                         r'(youtube|youtu|youtube-nocookie)\.(com|be)/'
                         r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
        match = re.search(youtube_regex, video_url)
        if not match:
            return "ERROR: Not a valid YouTube URL."
        video_id = match.group(6)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return ' '.join([item['text'] for item in transcript])
    except Exception as e:
        # This will catch if the video has disabled transcripts, etc.
        return f"ERROR: Could not fetch YouTube transcript: {str(e)}"

def get_website_content(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.extract()
        return soup.get_text(separator='\n', strip=True)
    except Exception as e:
        return f"ERROR: Failed to fetch website content: {str(e)}"

def extract_file_content(file_obj, filename):
    try:
        if filename.endswith((".xlsx", ".xls")):
            return pd.read_excel(file_obj).to_string()
        return file_obj.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return f"Error reading content from {filename}: {str(e)}"

def process_uploaded_file(file_storage):
    filename = file_storage.filename
    # ### CRITICAL FIX ### The same bug was also here.
    # OLD: file_ext = os.path.splitext(filename).lower() -> This is a TUPLE.
    # NEW: Get the second element [1] which is the extension string.
    file_ext = os.path.splitext(filename)[1].lower()

    try:
        file_bytes = file_storage.read()
        file_buffer = io.BytesIO(file_bytes)
        
        content = ""
        if file_ext == '.pdf':
            content = process_pdf_with_mistral(file_bytes, filename)
        elif file_ext == '.docx':
            content = extract_text_from_docx(file_buffer)
        elif file_ext == '.pptx':
            content = extract_text_from_ppt(file_buffer)
        elif file_ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif']:
            # For vision requests, we return the PIL Image object directly
            return Image.open(file_buffer)
        else:
            # All other files are treated as text-based
            file_buffer.seek(0)
            content = extract_file_content(file_buffer, filename)
        
        if content and content.startswith("ERROR:"):
             return content
        
        return f"Content from {filename}:\n\n{content}"
        
    except Exception as e:
        traceback.print_exc()
        return f"ERROR: An unexpected failure occurred while processing the file {filename}: {str(e)}"
