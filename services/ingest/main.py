"""
Video Ingest Service
====================

FastAPI microservice for video ingestion:
- Accept video uploads or URLs (YouTube, direct links)
- Extract metadata (duration, resolution, codec, fps)
- Generate thumbnail previews
- Store in media directory
"""

import os
import uuid
import json
import asyncio
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
import aiofiles
import ffmpeg

# Try to import yt-dlp
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False

app = FastAPI(
    title="Video Ingest Service",
    description="Ingest videos from uploads or URLs, extract metadata and thumbnails",
    version="1.0.0"
)

# Configuration
MEDIA_BASE = Path(os.environ.get("MEDIA_BASE", "/home/gamtec1000/cn_media"))
RAW_DIR = MEDIA_BASE / "raw"
THUMBNAILS_DIR = MEDIA_BASE / "thumbnails"
JOBS_DIR = MEDIA_BASE / ".jobs"

# Ensure directories exist
for d in [RAW_DIR, THUMBNAILS_DIR, JOBS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Job storage (in production, use Redis)
jobs: Dict[str, Dict[str, Any]] = {}


class IngestURLRequest(BaseModel):
    """Request to ingest video from URL."""
    url: str
    job_id: Optional[str] = None


class IngestResponse(BaseModel):
    """Response from ingest operation."""
    job_id: str
    status: str  # queued, downloading, processing, complete, error
    message: str


class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: str
    progress: float = 0.0
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class VideoMetadata(BaseModel):
    """Extracted video metadata."""
    job_id: str
    filename: str
    file_path: str
    file_size: int
    duration: float
    width: int
    height: int
    fps: float
    codec: str
    audio_codec: Optional[str] = None
    bitrate: Optional[int] = None
    thumbnail_path: Optional[str] = None
    source_url: Optional[str] = None
    ingested_at: str


def generate_job_id() -> str:
    """Generate unique job ID."""
    return str(uuid.uuid4())[:8]


def update_job(job_id: str, **kwargs):
    """Update job status."""
    if job_id not in jobs:
        jobs[job_id] = {"job_id": job_id, "status": "unknown"}
    jobs[job_id].update(kwargs)

    # Persist to disk
    job_file = JOBS_DIR / f"{job_id}.json"
    with open(job_file, "w") as f:
        json.dump(jobs[job_id], f, indent=2, default=str)


def get_job(job_id: str) -> Optional[Dict]:
    """Get job status."""
    if job_id in jobs:
        return jobs[job_id]

    # Try to load from disk
    job_file = JOBS_DIR / f"{job_id}.json"
    if job_file.exists():
        with open(job_file) as f:
            jobs[job_id] = json.load(f)
            return jobs[job_id]

    return None


async def extract_metadata(video_path: Path) -> Dict[str, Any]:
    """Extract video metadata using ffprobe."""
    try:
        probe = ffmpeg.probe(str(video_path))

        video_stream = next(
            (s for s in probe["streams"] if s["codec_type"] == "video"),
            None
        )
        audio_stream = next(
            (s for s in probe["streams"] if s["codec_type"] == "audio"),
            None
        )

        if not video_stream:
            raise ValueError("No video stream found")

        # Calculate FPS
        fps_parts = video_stream.get("r_frame_rate", "0/1").split("/")
        fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 and float(fps_parts[1]) > 0 else 0

        metadata = {
            "duration": float(probe["format"].get("duration", 0)),
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "fps": round(fps, 2),
            "codec": video_stream.get("codec_name", "unknown"),
            "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
            "bitrate": int(probe["format"].get("bit_rate", 0)) if probe["format"].get("bit_rate") else None,
            "file_size": int(probe["format"].get("size", 0)),
        }

        return metadata
    except Exception as e:
        raise ValueError(f"Failed to extract metadata: {e}")


async def generate_thumbnail(video_path: Path, output_path: Path, time_offset: float = 1.0):
    """Generate thumbnail from video."""
    try:
        (
            ffmpeg
            .input(str(video_path), ss=time_offset)
            .filter("scale", 320, -1)
            .output(str(output_path), vframes=1)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        return True
    except Exception as e:
        print(f"Thumbnail generation failed: {e}")
        return False


async def download_from_url(url: str, job_id: str) -> Path:
    """Download video from URL using yt-dlp."""
    if not YT_DLP_AVAILABLE:
        raise ImportError("yt-dlp not installed")

    output_template = str(RAW_DIR / f"{job_id}_%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [lambda d: update_job(job_id, progress=float(d.get("downloaded_bytes", 0)) / max(d.get("total_bytes", 1), 1) * 100 if d.get("total_bytes") else 0)],
    }

    update_job(job_id, status="downloading", message="Downloading video...")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        return Path(filename)


async def process_video(job_id: str, video_path: Path, source_url: Optional[str] = None):
    """Process video: extract metadata and generate thumbnail."""
    try:
        update_job(job_id, status="processing", message="Extracting metadata...")

        # Extract metadata
        metadata = await extract_metadata(video_path)

        # Generate thumbnail
        thumbnail_name = f"{job_id}_thumb.jpg"
        thumbnail_path = THUMBNAILS_DIR / thumbnail_name

        update_job(job_id, status="processing", message="Generating thumbnail...")
        thumb_time = min(metadata["duration"] / 4, 5.0)  # Thumbnail at 25% or 5s
        await generate_thumbnail(video_path, thumbnail_path, thumb_time)

        # Build result
        result = VideoMetadata(
            job_id=job_id,
            filename=video_path.name,
            file_path=str(video_path),
            file_size=metadata["file_size"],
            duration=metadata["duration"],
            width=metadata["width"],
            height=metadata["height"],
            fps=metadata["fps"],
            codec=metadata["codec"],
            audio_codec=metadata["audio_codec"],
            bitrate=metadata["bitrate"],
            thumbnail_path=str(thumbnail_path) if thumbnail_path.exists() else None,
            source_url=source_url,
            ingested_at=datetime.now().isoformat()
        )

        update_job(
            job_id,
            status="complete",
            message="Video ingested successfully",
            result=result.model_dump()
        )

    except Exception as e:
        update_job(job_id, status="error", error=str(e), message=f"Processing failed: {e}")


async def ingest_from_url_task(job_id: str, url: str):
    """Background task to ingest video from URL."""
    try:
        video_path = await download_from_url(url, job_id)
        await process_video(job_id, video_path, source_url=url)
    except Exception as e:
        update_job(job_id, status="error", error=str(e), message=f"Ingest failed: {e}")


async def ingest_upload_task(job_id: str, file_path: Path):
    """Background task to process uploaded video."""
    try:
        await process_video(job_id, file_path)
    except Exception as e:
        update_job(job_id, status="error", error=str(e), message=f"Ingest failed: {e}")


# ============== API Endpoints ==============

@app.get("/")
async def root():
    """Service health check."""
    return {
        "service": "video-ingest",
        "status": "healthy",
        "version": "1.0.0",
        "yt_dlp_available": YT_DLP_AVAILABLE
    }


@app.post("/ingest/url", response_model=IngestResponse)
async def ingest_from_url(request: IngestURLRequest, background_tasks: BackgroundTasks):
    """
    Ingest video from URL (YouTube, direct link, etc.)

    Returns job_id to track progress.
    """
    job_id = request.job_id or generate_job_id()

    update_job(job_id, status="queued", message="Job queued", source_url=request.url)

    background_tasks.add_task(ingest_from_url_task, job_id, request.url)

    return IngestResponse(
        job_id=job_id,
        status="queued",
        message=f"Video ingest queued from URL"
    )


@app.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Ingest video from file upload.

    Returns job_id to track progress.
    """
    job_id = generate_job_id()

    # Save uploaded file
    file_ext = Path(file.filename).suffix or ".mp4"
    safe_filename = f"{job_id}_{hashlib.md5(file.filename.encode()).hexdigest()[:8]}{file_ext}"
    file_path = RAW_DIR / safe_filename

    update_job(job_id, status="uploading", message="Receiving upload...")

    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    update_job(job_id, status="queued", message="Upload complete, processing queued")

    background_tasks.add_task(ingest_upload_task, job_id, file_path)

    return IngestResponse(
        job_id=job_id,
        status="queued",
        message=f"Video upload received: {file.filename}"
    )


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Get job status by ID."""
    job = get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Handle progress that might be a string like "100.0%"
    progress = job.get("progress", 0)
    if isinstance(progress, str):
        progress = float(progress.replace("%", "")) if progress else 0

    return JobStatus(
        job_id=job_id,
        status=job.get("status", "unknown"),
        progress=float(progress),
        message=job.get("message", ""),
        result=job.get("result"),
        error=job.get("error")
    )


@app.get("/thumbnail/{job_id}")
async def get_thumbnail(job_id: str):
    """Get thumbnail for ingested video."""
    job = get_job(job_id)

    if not job or job.get("status") != "complete":
        raise HTTPException(status_code=404, detail="Thumbnail not available")

    result = job.get("result", {})
    thumbnail_path = result.get("thumbnail_path")

    if not thumbnail_path or not Path(thumbnail_path).exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    return FileResponse(thumbnail_path, media_type="image/jpeg")


@app.get("/jobs")
async def list_jobs(limit: int = 20):
    """List recent jobs."""
    job_files = sorted(JOBS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)

    result = []
    for job_file in job_files[:limit]:
        with open(job_file) as f:
            result.append(json.load(f))

    return {"jobs": result, "total": len(job_files)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
