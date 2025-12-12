import os
import io
import tempfile
from pathlib import Path
from typing import Optional
import whisper
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize Whisper model (load once, reuse)
_whisper_model = None

def get_whisper_model():
    """Load Whisper model (lazy loading, cached)."""
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper model...")
        _whisper_model = whisper.load_model("base")
        print("Whisper model loaded.")
    return _whisper_model


def audio_to_text(audio_file_path: str) -> str:
    """Convert audio file to text using Whisper."""
    model = get_whisper_model()
    result = model.transcribe(audio_file_path)
    return result["text"].strip()


def text_to_speech(text: str, output_path: Optional[str] = None) -> bytes:
    """Convert text to speech using OpenAI TTS API."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Use OpenAI's TTS API (tts-1 model)
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",  # Options: alloy, echo, fable, onyx, nova, shimmer
        input=text
    )
    
    audio_bytes = response.content
    
    if output_path:
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
    
    return audio_bytes


def process_voice_input(audio_bytes: bytes) -> str:
    """Process audio bytes to text."""
    # Save to temporary file for Whisper
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name
    
    try:
        text = audio_to_text(tmp_path)
        return text
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def generate_voice_response(text: str) -> bytes:
    """Generate audio response from text."""
    return text_to_speech(text)
