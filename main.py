import base64
import os
import uuid
from typing import Literal
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

app = FastAPI(title="AI Voice Detector API")

# --- Configuration ---
VALID_API_KEY = "sk_test_123456789"
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

# --- Data Models ---
class VoiceDetectionRequest(BaseModel):
    language: Literal["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    audioFormat: Literal["mp3"]
    audioBase64: str

class VoiceDetectionResponse(BaseModel):
    status: str
    language: str
    classification: Literal["AI_GENERATED", "HUMAN"]
    confidenceScore: float
    explanation: str

# --- Helper Functions ---
def detect_ai_voice(file_path: str, language: str):
    """
    Yahan aapka actual ML model logic aayega. 
    Abhi ke liye ye dummy logic use kar raha hai.
    """
    # Example Logic: 
    # Real life mein aap 'librosa' use karke features extract karenge
    # ya kisi pre-trained model (Wav2Vec2) ka use karenge.
    
    # Dummy Detection Logic
    is_ai = len(file_path) % 2 == 0 # Just a placeholder logic
    
    if is_ai:
        return "AI_GENERATED", 0.91, "Unnatural pitch consistency and robotic speech patterns detected"
    else:
        return "HUMAN", 0.95, "Natural breath patterns and emotional nuances detected"

# --- API Endpoint ---
@app.post("/api/voice-detection", response_model=VoiceDetectionResponse)
async def voice_detection(
    request: VoiceDetectionRequest, 
    x_api_key: str = Header(None)
):
    # 1. API Key Validation
    if x_api_key != VALID_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key or malformed request")

    try:
        # 2. Decode Base64 Audio
        audio_data = base64.b64decode(request.audioBase64)
        
        # 3. Save as temporary MP3 file
        temp_filename = f"temp_{uuid.uuid4()}.mp3"
        with open(temp_filename, "wb") as f:
            f.write(audio_data)

        # 4. Perform Detection
        classification, score, explanation = detect_ai_voice(temp_filename, request.language)

        # 5. Cleanup temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

        # 6. Return Success Response
        return {
            "status": "success",
            "language": request.language,
            "classification": classification,
            "confidenceScore": score,
            "explanation": explanation
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)