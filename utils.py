# tanishqravulasuperaibot-main/utils.py
# Updated Code

import os
import io
import time
import uuid
import math
import json
import requests
import pandas as pd
import openpyxl # Added for excel processing
from PIL import Image
import google.generativeai as genai
import tempfile
import shutil
import traceback 

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

# ### NEW LARGE FILE-HANDLING ###: Import PyPDF2 for large PDF handling
from PyPDF2 import PdfReader
# ### END OF NEW CODE ###


# --- INITIALIZATION ---
mistral_client = None
g4f_client = G4FClient()

GENERATED_MEDIA_DIR = "static/generated_media"
os.makedirs(GENERATED_MEDIA_DIR, exist_ok=True)

def initialize_mistral(api_key):
    global mistral_client
    mistral_client = Mistral(api_key=api_key)

# --- MEDIA GENERATION (Unchanged) ---
def generate_images_from_prompt(prompt, num_images=4, download_dir=None):
    if download_dir is None:
        download_dir = GENERATED_MEDIA_DIR
    image_urls, local_paths = [], []
    print(f"⏳ Generating {num_images} image(s) for prompt: {prompt}")
    for i in range(num_images):
        try:
            response = g4f_client.images.generate(model="midjourney", prompt=prompt, n=1, response_format="url")
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
                file_name, file_path = f"image_{uuid.uuid4().hex[:8]}.jpg", os.path.join(download_dir, file_name)
                with open(file_path, "wb") as f: f.write(image_response.content)
                local_paths.append(file_path)
        except Exception as e: print(f"❌ Error downloading image from {url}: {e}")
    if not local_paths: return {"success": False, "error": "Failed to download generated images."}
    return {"success": True, "paths": local_paths}

def generate_video_from_prompt(prompt):
    temp_work_dir = tempfile.mkdtemp()
    print(f"✅ Created temporary working directory: {temp_work_dir}")
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
                img_rgb = img.convert('RGB') if img.mode != 'RGB' else img
                img_rgb.resize((1280, 720), Image.Resampling.LANCZOS).save(img_path)

        fps = len(temp_image_files_local) / audio_duration
        video_clip = ImageSequenceClip(temp_image_files_local, fps=fps)

        with AudioFileClip(audio_path_local) as audio_clip_for_video:
            video_clip = video_clip.set_audio(audio_clip_for_video)
            video_filename = f"video_{uuid.uuid4().hex[:8]}.mp4"
            temp_video_path = os.path.join(temp_work_dir, video_filename)
            video_clip.write_videofile(temp_video_path, codec="libx264", audio_codec="aac")

        final_video_path_local = os.path.join(GENERATED_MEDIA_DIR, video_filename)
        shutil.move(temp_video_path, final_video_path_local)
        public_video_path = f"/{final_video_path_local.replace(os.path.sep, '/')}"
        return {"success": True, "path": public_video_path}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": f"An unexpected error occurred during video generation: {e}"}
    finally:
        if os.path.exists(temp_work_dir): shutil.rmtree(temp_work_dir)

# --- NEW LARGE FILE-HANDLING FUNCTIONS ---
def extract_text_with_pypdf2(pdf_bytes):
    """Extracts text from PDF bytes using PyPDF2."""
    text = ""
    try:
        pdf_file = io.BytesIO(pdf_bytes)
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        return f"ERROR: Could not extract text with PyPDF2. Details: {str(e)}"
    return text

def process_large_uploaded_files(files):
    """
    Extracts text from a list of large uploaded files (FileStorage objects).
    Uses PyPDF2 for PDFs and other extractors for different file types.
    """
    all_text = ""
    for file_storage in files:
        filename = file_storage.filename
        file_ext = os.path.splitext(filename)[1].lower()
        all_text += f"\n\n--- Content from {filename} ---\n\n"

        try:
            file_bytes = file_storage.read()
            # Reset stream position for potential re-reads
            file_storage.seek(0) 

            if file_ext == '.pdf':
                all_text += extract_text_with_pypdf2(file_bytes)
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(io.BytesIO(file_bytes))
                all_text += df.to_string()
            elif file_ext == '.csv':
                df = pd.read_csv(io.BytesIO(file_bytes))
                all_text += df.to_string()
            elif file_ext in ['.zip', '.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tar.xz']:
                # For archives, we extract content from all inner files
                archive_content_dict = {}
                if file_ext == '.zip':
                    archive_content_dict = extract_content_from_zip(io.BytesIO(file_bytes))
                else: # tar files
                    archive_content_dict = extract_content_from_tar(io.BytesIO(file_bytes))

                for path, text in archive_content_dict.items():
                    all_text += f"\n--- Inner File: {path} ---\n{text}\n"
            else: # Assume text-based files (code, txt, md, etc.)
                all_text += file_bytes.decode('utf-8', errors='ignore')

        except Exception as e:
            all_text += f"ERROR processing file {filename}: {e}\n"

    return all_text
# ### END OF NEW CODE ###


# --- EXISTING FILE PROCESSING FUNCTIONS (Unchanged logic, but used by new functions) ---
def process_pdf_with_mistral(file_data: bytes, file_name: str) -> str:
    # This is for the *regular* file upload, not the large file one.
    if not mistral_client: raise ValueError("Mistral client not initialized.")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name
        with open(tmp_path, "rb") as f:
            uploaded_file = mistral_client.files.upload(file={"file_name": Path(tmp_path).stem, "content": f.read()}, purpose="ocr")
        signed_url = mistral_client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
        response = mistral_client.ocr.process(document=DocumentURLChunk(document_url=signed_url.url), model="mistral-ocr-latest", include_image_base64=False)
        os.unlink(tmp_path)
        return "\n\n".join([page.markdown for page in response.pages])
    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path): os.unlink(tmp_path)
        return f"ERROR: Could not process the PDF with Mistral. Details: {str(e)}"

def extract_text_from_docx(docx_file):
    doc = PyDocxDocument(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_ppt(ppt_file):
    presentation = Presentation(ppt_file)
    text = ""
    for slide in presentation.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
    return text

def get_youtube_transcript(video_url):
    try:
        video_id = video_url.split("=")[1].split("&")[0]
        return ' '.join([item['text'] for item in YouTubeTranscriptApi.get_transcript(video_id)])
    except Exception as e:
        return f"ERROR: Error fetching YouTube transcript: {str(e)}"

def get_website_content(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style"]): script.extract()
        return soup.get_text(separator='\n', strip=True)
    except Exception as e:
        return f"ERROR: Error fetching website content: {str(e)}"

def extract_file_content(file_obj, filename):
    file_content = ""
    try:
        code_exts = (  ".py", ".cs", ".java", ".cpp", ".c", ".js", ".ts", ".rb", ".go", ".swift",
            ".kt", ".dart", ".lua", ".sh", ".pl", ".scala", ".r", ".m", ".asm",
            ".vb", ".f90", ".rs", ".clj", ".jl", ".groovy",
            ".html", ".htm", ".css", ".php", ".xml", ".ejs", ".jsx", ".tsx",
            ".h", ".hpp", ".hh", ".hxx", ".cxx", ".cc", ".mak", ".make", ".mk",
            ".ini", ".cfg", ".conf", ".env", ".toml", ".yaml", ".yml", ".properties",
            ".json", ".csv", ".tsv", ".md", ".rst", ".txt",
            ".sql", ".db", ".sqlite", ".pgsql", ".psql",
            ".bat", ".cmd", ".ps1", ".Dockerfile", ".jenkinsfile", ".gitignore", ".gitattributes", ".editorconfig",
            ".log", ".out", ".err",
            ".edmx", ".proto", ".thrift", ".ipynb", ".vbproj", ".csproj", ".sln", ".xaml"
        )
        if filename.endswith(code_exts) or ".log" in filename:
            file_content = file_obj.read().decode("utf-8", errors="ignore")
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_obj)
            file_content = df.to_string()
        else:
            file_content = file_obj.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return f"Error reading file {filename}: {str(e)}"
    return file_content

def extract_content_from_zip(zip_file, parent_name=""):
    all_file_content = {}
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.is_dir(): continue
            file_key = f"{parent_name}/{file_info.filename}" if parent_name else file_info.filename
            with zip_ref.open(file_info) as file_obj:
                if file_info.filename.endswith(".zip"):
                    all_file_content.update(extract_content_from_zip(file_obj, file_key))
                else:
                    all_file_content[file_key] = extract_file_content(file_obj, file_info.filename)
    return all_file_content

def extract_content_from_tar(tar_file, parent_name=""):
    all_file_content = {}
    with tarfile.open(fileobj=tar_file, mode='r:*') as tar_ref:
        for member in tar_ref.getmembers():
            if member.isdir(): continue
            file_key = f"{parent_name}/{member.name}" if parent_name else member.name
            extracted_file = tar_ref.extractfile(member)
            if extracted_file:
                if any(member.name.endswith(ext) for ext in [".tar", ".gz", ".tgz", ".bz2", ".xz"]):
                    all_file_content.update(extract_content_from_tar(io.BytesIO(extracted_file.read()), file_key))
                else:
                    all_file_content[file_key] = extract_file_content(extracted_file, member.name)
    return all_file_content

def process_uploaded_file(file_storage):
    filename = file_storage.filename
    file_bytes = file_storage.read()
    file_buffer = io.BytesIO(file_bytes)
    file_ext = os.path.splitext(filename)[1].lower()
    try:
        if file_ext == '.pdf':
            content = process_pdf_with_mistral(file_bytes, filename)
        elif file_ext == '.docx':
            content = extract_text_from_docx(file_buffer)
        elif file_ext == '.pptx':
            content = extract_text_from_ppt(file_buffer)
        elif file_ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif']:
            return Image.open(file_buffer)
        elif file_ext == '.zip':
            content_dict = extract_content_from_zip(file_buffer)
            content = "\n".join([f"\n--- File: {path} ---\n{text}" for path, text in content_dict.items()])
        elif file_ext in ['.tar', '.gz', '.tgz', '.bz2', '.xz']:
            content_dict = extract_content_from_tar(file_buffer)
            content = "\n".join([f"\n--- File: {path} ---\n{text}" for path, text in content_dict.items()])
        else:
            file_buffer.seek(0)
            content = extract_file_content(file_buffer, filename)

        if content.startswith("ERROR:"): return content
        return f"Content from {filename}:\n\n{content}"
    except Exception as e:
        traceback.print_exc()
        return f"ERROR: An unexpected error occurred while processing file {filename}: {str(e)}"
