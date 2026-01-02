"""
Transcription Service
=====================

FastAPI microservice for audio/video transcription:
- Accept video/audio files or paths
- Extract audio from video if needed
- Transcribe using Whisper (local GPU)
- Return timestamped segments with word-level timing
"""

import os
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
import aiofiles
import ffmpeg
import whisper
import torch

app = FastAPI(
    title="Transcription Service",
    description="Transcribe audio/video using Whisper",
    version="1.0.0"
)

# Configuration
MEDIA_BASE = Path(os.environ.get("MEDIA_BASE", "/home/gamtec1000/cn_media"))
AUDIO_DIR = MEDIA_BASE / "audio"
TRANSCRIPTS_DIR = MEDIA_BASE / "transcripts"
JOBS_DIR = MEDIA_BASE / ".jobs"

# Ensure directories exist
for d in [AUDIO_DIR, TRANSCRIPTS_DIR, JOBS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Device detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Model loading (lazy)
_model = None
_model_name = os.environ.get("WHISPER_MODEL", "base")


def get_model():
    """Lazy load Whisper model."""
    global _model
    if _model is None:
        print(f"Loading Whisper model: {_model_name}")
        _model = whisper.load_model(_model_name, device=DEVICE)
        print(f"Model loaded on {DEVICE}")
    return _model


# Job storage
jobs: Dict[str, Dict[str, Any]] = {}


class TranscribeRequest(BaseModel):
    """Request to transcribe a file."""
    file_path: str
    language: Optional[str] = None  # Auto-detect if None
    job_id: Optional[str] = None
    word_timestamps: bool = True


class TranscribeResponse(BaseModel):
    """Response from transcription."""
    job_id: str
    status: str
    message: str


class Segment(BaseModel):
    """Transcription segment."""
    id: int
    start: float
    end: float
    text: str
    words: Optional[List[Dict]] = None


class TranscriptionResult(BaseModel):
    """Complete transcription result."""
    job_id: str
    file_path: str
    language: str
    duration: float
    text: str
    segments: List[Segment]
    transcribed_at: str


class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: str
    progress: float = 0.0
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def generate_job_id() -> str:
    """Generate unique job ID."""
    import uuid
    return str(uuid.uuid4())[:8]


def update_job(job_id: str, **kwargs):
    """Update job status."""
    if job_id not in jobs:
        jobs[job_id] = {"job_id": job_id, "status": "unknown"}
    jobs[job_id].update(kwargs)

    # Persist to disk
    job_file = JOBS_DIR / f"transcribe_{job_id}.json"
    with open(job_file, "w") as f:
        json.dump(jobs[job_id], f, indent=2, default=str)


def get_job(job_id: str) -> Optional[Dict]:
    """Get job status."""
    if job_id in jobs:
        return jobs[job_id]

    # Try to load from disk
    job_file = JOBS_DIR / f"transcribe_{job_id}.json"
    if job_file.exists():
        with open(job_file) as f:
            jobs[job_id] = json.load(f)
            return jobs[job_id]

    return None


async def extract_audio(video_path: Path, output_path: Path) -> bool:
    """Extract audio from video file."""
    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(str(output_path), acodec='pcm_s16le', ar=16000, ac=1)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        return True
    except Exception as e:
        print(f"Audio extraction failed: {e}")
        return False


def is_video_file(path: Path) -> bool:
    """Check if file is a video."""
    video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv'}
    return path.suffix.lower() in video_extensions


async def transcribe_file(job_id: str, file_path: str, language: Optional[str], word_timestamps: bool):
    """Background task to transcribe file."""
    try:
        input_path = Path(file_path)

        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        audio_path = input_path

        # Extract audio if video
        if is_video_file(input_path):
            update_job(job_id, status="extracting", message="Extracting audio from video...")
            audio_path = AUDIO_DIR / f"{job_id}.wav"
            success = await extract_audio(input_path, audio_path)
            if not success:
                raise ValueError("Failed to extract audio from video")

        update_job(job_id, status="transcribing", message="Transcribing audio...", progress=0.1)

        # Load model and transcribe
        model = get_model()

        # Transcription options
        options = {
            "language": language,
            "word_timestamps": word_timestamps,
            "verbose": False,
        }

        # Remove None values
        options = {k: v for k, v in options.items() if v is not None}

        update_job(job_id, status="transcribing", message="Running Whisper...", progress=0.2)

        result = model.transcribe(str(audio_path), **options)

        update_job(job_id, status="processing", message="Processing results...", progress=0.9)

        # Build segments
        segments = []
        for i, seg in enumerate(result.get("segments", [])):
            segment = Segment(
                id=i,
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
                words=seg.get("words") if word_timestamps else None
            )
            segments.append(segment)

        # Calculate duration
        duration = segments[-1].end if segments else 0.0

        # Build result
        transcription = TranscriptionResult(
            job_id=job_id,
            file_path=file_path,
            language=result.get("language", "unknown"),
            duration=duration,
            text=result["text"].strip(),
            segments=[s.model_dump() for s in segments],
            transcribed_at=datetime.utcnow().isoformat()
        )

        # Save transcript to file
        transcript_file = TRANSCRIPTS_DIR / f"{job_id}.json"
        with open(transcript_file, "w") as f:
            json.dump(transcription.model_dump(), f, indent=2)

        update_job(
            job_id,
            status="complete",
            message="Transcription complete",
            progress=1.0,
            result=transcription.model_dump()
        )

        # Cleanup temp audio if extracted
        if audio_path != input_path and audio_path.exists():
            audio_path.unlink()

    except Exception as e:
        update_job(job_id, status="error", error=str(e), message=f"Transcription failed: {e}")


# ============== API Endpoints ==============

@app.get("/")
async def root():
    """Service health check."""
    return {
        "service": "transcription",
        "status": "healthy",
        "version": "1.0.0",
        "device": DEVICE,
        "model": _model_name,
        "model_loaded": _model is not None
    }


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: TranscribeRequest, background_tasks: BackgroundTasks):
    """
    Transcribe audio/video file.

    Returns job_id to track progress.
    """
    job_id = request.job_id or generate_job_id()

    update_job(job_id, status="queued", message="Transcription queued", file_path=request.file_path)

    background_tasks.add_task(
        transcribe_file,
        job_id,
        request.file_path,
        request.language,
        request.word_timestamps
    )

    return TranscribeResponse(
        job_id=job_id,
        status="queued",
        message="Transcription job queued"
    )


@app.post("/transcribe/upload", response_model=TranscribeResponse)
async def transcribe_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: Optional[str] = None,
    word_timestamps: bool = True
):
    """
    Upload and transcribe file.

    Returns job_id to track progress.
    """
    job_id = generate_job_id()

    # Save uploaded file
    file_ext = Path(file.filename).suffix or ".wav"
    file_path = AUDIO_DIR / f"{job_id}_upload{file_ext}"

    update_job(job_id, status="uploading", message="Receiving upload...")

    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    update_job(job_id, status="queued", message="Upload complete, transcription queued")

    background_tasks.add_task(transcribe_file, job_id, str(file_path), language, word_timestamps)

    return TranscribeResponse(
        job_id=job_id,
        status="queued",
        message=f"File uploaded, transcription queued: {file.filename}"
    )


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Get job status by ID."""
    job = get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatus(
        job_id=job_id,
        status=job.get("status", "unknown"),
        progress=job.get("progress", 0),
        message=job.get("message", ""),
        result=job.get("result"),
        error=job.get("error")
    )


@app.get("/transcript/{job_id}")
async def get_transcript(job_id: str):
    """Get transcript by job ID."""
    job = get_job(job_id)

    if not job or job.get("status") != "complete":
        raise HTTPException(status_code=404, detail="Transcript not available")

    return job.get("result", {})


@app.get("/models")
async def list_models():
    """List available Whisper models."""
    return {
        "current": _model_name,
        "available": ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        "loaded": _model is not None
    }


@app.post("/warmup")
async def warmup():
    """Pre-load the Whisper model."""
    try:
        model = get_model()
        return {"status": "ok", "message": f"Model {_model_name} loaded on {DEVICE}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
