from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os, tempfile, subprocess
from dotenv import load_dotenv
from openai import OpenAI
from sarvamai import SarvamAI
import uvicorn

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Clients ───────────────────────────────────────────────────────────────────

sarvam_client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = (
    "You are a calm, intelligent voice assistant. "
    "Keep responses short and natural — one or two sentences only."
)

# ── STT ───────────────────────────────────────────────────────────────────────

@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    # Write incoming webm blob to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_in:
        tmp_in.write(audio_bytes)
        webm_path = tmp_in.name

    # Sarvam saaras:v3 needs a proper WAV (16kHz, mono).
    # Browser MediaRecorder gives us webm/opus — convert with ffmpeg.
    wav_path = webm_path.replace(".webm", ".wav")

    try:
        ff = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", webm_path,
                "-ar", "16000",   # 16 kHz sample rate
                "-ac", "1",       # mono
                "-f", "wav",
                wav_path
            ],
            capture_output=True,
            timeout=30
        )

        if ff.returncode != 0:
            raise HTTPException(500, f"ffmpeg conversion failed: {ff.stderr.decode()}")

        with open(wav_path, "rb") as f:
            result = sarvam_client.speech_to_text.transcribe(
                file=f, model="saaras:v3", language_code="en-IN"
            )

        return {"transcript": result.transcript}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        if os.path.exists(webm_path): os.unlink(webm_path)
        if os.path.exists(wav_path):  os.unlink(wav_path)

# ── LLM ───────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    text: str

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": req.text}
            ],
            temperature=0.3,
            timeout=30
        )
        return {"reply": resp.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(500, str(e))

# ── TTS ───────────────────────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str

@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    try:
        resp = sarvam_client.text_to_speech.convert(
            text=req.text,
            model="bulbul:v3",
            target_language_code="en-IN",
            speaker="sophia"
        )
        audio_b64 = resp.audios[0]
        return {"audio": audio_b64}
    except Exception as e:
        raise HTTPException(500, str(e))

# ── SERVE FRONTEND ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index2.html", encoding="utf-8") as f:
        return f.read()

# ── SERVE STATIC FILES (videos + assets) ─────────────────────────────────────
# Mount AFTER all route definitions so named routes take priority.
# Project layout:
#   main.py
#   index.html
#   static/
#     idle.mp4
#     talk.mp4

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8005, reload=True)