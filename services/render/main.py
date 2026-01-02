"""
Render Service
==============

FastAPI microservice for video rendering:
- Apply cuts from SmartCut analysis
- Concatenate video segments
- Support multiple output formats and codecs
- GPU acceleration when available
- Progress tracking
"""

import os
import json
import uuid
import asyncio
import subprocess
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import aiofiles

app = FastAPI(
    title="Render Service",
    description="Video rendering with cuts and format conversion",
    version="1.0.0"
)

# Configuration
MEDIA_BASE = Path(os.environ.get("MEDIA_BASE", "/home/gamtec1000/cn_media"))
RENDERED_DIR = MEDIA_BASE / "rendered"
TEMP_DIR = MEDIA_BASE / ".temp"
JOBS_DIR = MEDIA_BASE / ".jobs"

# Ensure directories exist
for d in [RENDERED_DIR, TEMP_DIR, JOBS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class OutputFormat(str, Enum):
    MP4 = "mp4"
    WEBM = "webm"
    MOV = "mov"
    MKV = "mkv"


class VideoCodec(str, Enum):
    H264 = "h264"
    H265 = "h265"
    VP9 = "vp9"
    COPY = "copy"  # Stream copy, no re-encoding


class AudioCodec(str, Enum):
    AAC = "aac"
    OPUS = "opus"
    MP3 = "mp3"
    COPY = "copy"


class Preset(str, Enum):
    ULTRAFAST = "ultrafast"
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    QUALITY = "veryslow"


class Segment(BaseModel):
    """Video segment to include."""
    start: float
    end: float


class RenderRequest(BaseModel):
    """Request to render video with cuts."""
    video_path: str
    segments: List[Segment]  # Segments to KEEP
    output_name: Optional[str] = None

    # Output settings
    format: OutputFormat = OutputFormat.MP4
    video_codec: VideoCodec = VideoCodec.H264
    audio_codec: AudioCodec = AudioCodec.AAC
    preset: Preset = Preset.FAST

    # Quality settings
    crf: int = 23  # Constant Rate Factor (0-51, lower = better)
    resolution: Optional[str] = None  # e.g., "1920x1080", None = original
    fps: Optional[float] = None  # None = original

    # Audio settings
    audio_bitrate: str = "128k"

    # Options
    use_gpu: bool = True


class RenderResponse(BaseModel):
    """Response from render request."""
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: str
    progress: float = 0.0
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class RenderResult(BaseModel):
    """Render result."""
    job_id: str
    input_path: str
    output_path: str
    original_duration: float
    rendered_duration: float
    file_size: int
    format: str
    video_codec: str
    audio_codec: str
    rendered_at: str


# Job storage
jobs: Dict[str, Dict[str, Any]] = {}


def generate_job_id() -> str:
    """Generate unique job ID."""
    return str(uuid.uuid4())[:8]


def update_job(job_id: str, **kwargs):
    """Update job status."""
    if job_id not in jobs:
        jobs[job_id] = {"job_id": job_id, "status": "unknown"}
    jobs[job_id].update(kwargs)

    job_file = JOBS_DIR / f"render_{job_id}.json"
    with open(job_file, "w") as f:
        json.dump(jobs[job_id], f, indent=2, default=str)


def get_job(job_id: str) -> Optional[Dict]:
    """Get job status."""
    if job_id in jobs:
        return jobs[job_id]

    job_file = JOBS_DIR / f"render_{job_id}.json"
    if job_file.exists():
        with open(job_file) as f:
            jobs[job_id] = json.load(f)
            return jobs[job_id]

    return None


def check_nvidia_gpu() -> bool:
    """Check if NVIDIA GPU is available for encoding."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True
        )
        return "h264_nvenc" in result.stdout
    except Exception:
        return False


def get_video_duration(video_path: str) -> float:
    """Get video duration using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def get_codec_params(request: RenderRequest, has_gpu: bool) -> Dict[str, Any]:
    """Get ffmpeg codec parameters based on request."""
    params = {}

    # Video codec
    if request.video_codec == VideoCodec.COPY:
        params["vcodec"] = "copy"
    elif request.video_codec == VideoCodec.H264:
        if has_gpu and request.use_gpu:
            params["vcodec"] = "h264_nvenc"
            params["preset"] = "p4" if request.preset == Preset.FAST else "p7"
            params["rc"] = "vbr"
            params["cq"] = request.crf
        else:
            params["vcodec"] = "libx264"
            params["preset"] = request.preset.value
            params["crf"] = request.crf
    elif request.video_codec == VideoCodec.H265:
        if has_gpu and request.use_gpu:
            params["vcodec"] = "hevc_nvenc"
            params["preset"] = "p4" if request.preset == Preset.FAST else "p7"
            params["rc"] = "vbr"
            params["cq"] = request.crf
        else:
            params["vcodec"] = "libx265"
            params["preset"] = request.preset.value
            params["crf"] = request.crf
    elif request.video_codec == VideoCodec.VP9:
        params["vcodec"] = "libvpx-vp9"
        params["crf"] = request.crf
        params["b:v"] = "0"

    # Audio codec
    if request.audio_codec == AudioCodec.COPY:
        params["acodec"] = "copy"
    elif request.audio_codec == AudioCodec.AAC:
        params["acodec"] = "aac"
        params["b:a"] = request.audio_bitrate
    elif request.audio_codec == AudioCodec.OPUS:
        params["acodec"] = "libopus"
        params["b:a"] = request.audio_bitrate
    elif request.audio_codec == AudioCodec.MP3:
        params["acodec"] = "libmp3lame"
        params["b:a"] = request.audio_bitrate

    return params


async def create_segment_files(job_id: str, video_path: str, segments: List[Segment]) -> List[str]:
    """Extract segments to temporary files."""
    segment_files = []

    for i, seg in enumerate(segments):
        segment_file = TEMP_DIR / f"{job_id}_seg_{i:03d}.ts"

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(seg.start),
            "-i", video_path,
            "-t", str(seg.end - seg.start),
            "-c", "copy",
            "-bsf:v", "h264_mp4toannexb",
            "-f", "mpegts",
            str(segment_file)
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()

        if segment_file.exists():
            segment_files.append(str(segment_file))

    return segment_files


async def render_with_concat(
    job_id: str,
    video_path: str,
    segments: List[Segment],
    output_path: str,
    request: RenderRequest,
    has_gpu: bool
):
    """Render video using concat demuxer."""
    # Create concat file
    concat_file = TEMP_DIR / f"{job_id}_concat.txt"

    # Extract segments first
    update_job(job_id, status="extracting", message="Extracting segments...", progress=0.1)
    segment_files = await create_segment_files(job_id, video_path, segments)

    if not segment_files:
        raise ValueError("No segments extracted")

    # Write concat file
    with open(concat_file, "w") as f:
        for seg_file in segment_files:
            f.write(f"file '{seg_file}'\n")

    update_job(job_id, status="rendering", message="Rendering video...", progress=0.3)

    # Get codec params
    codec_params = get_codec_params(request, has_gpu)

    # Build ffmpeg command
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
    ]

    # Add codec params
    for key, value in codec_params.items():
        cmd.extend([f"-{key}", str(value)])

    # Add resolution if specified
    if request.resolution:
        cmd.extend(["-s", request.resolution])

    # Add fps if specified
    if request.fps:
        cmd.extend(["-r", str(request.fps)])

    # Output
    cmd.append(output_path)

    # Run ffmpeg with progress monitoring
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await process.communicate()

    # Cleanup temp files
    for seg_file in segment_files:
        try:
            Path(seg_file).unlink()
        except Exception:
            pass
    try:
        concat_file.unlink()
    except Exception:
        pass

    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {stderr.decode()[-500:]}")

    return output_path


async def render_with_filter(
    job_id: str,
    video_path: str,
    segments: List[Segment],
    output_path: str,
    request: RenderRequest,
    has_gpu: bool
):
    """Render video using filter_complex for precise cuts."""
    update_job(job_id, status="rendering", message="Building filter graph...", progress=0.2)

    # Build filter complex for segment selection
    filter_parts = []
    concat_inputs = []

    for i, seg in enumerate(segments):
        # Trim each segment
        filter_parts.append(
            f"[0:v]trim=start={seg.start}:end={seg.end},setpts=PTS-STARTPTS[v{i}]"
        )
        filter_parts.append(
            f"[0:a]atrim=start={seg.start}:end={seg.end},asetpts=PTS-STARTPTS[a{i}]"
        )
        concat_inputs.append(f"[v{i}][a{i}]")

    # Concat all segments
    concat_filter = "".join(concat_inputs) + f"concat=n={len(segments)}:v=1:a=1[outv][outa]"
    filter_complex = ";".join(filter_parts) + ";" + concat_filter

    # Get codec params
    codec_params = get_codec_params(request, has_gpu)

    # Build ffmpeg command
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "[outa]",
    ]

    # Add codec params
    for key, value in codec_params.items():
        cmd.extend([f"-{key}", str(value)])

    # Add resolution if specified
    if request.resolution:
        cmd.extend(["-s", request.resolution])

    # Add fps if specified
    if request.fps:
        cmd.extend(["-r", str(request.fps)])

    cmd.append(output_path)

    update_job(job_id, status="rendering", message="Encoding video...", progress=0.4)

    # Run ffmpeg
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {stderr.decode()[-500:]}")

    return output_path


async def render_video(job_id: str, request: RenderRequest):
    """Background task to render video."""
    try:
        video_path = request.video_path

        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if not request.segments:
            raise ValueError("No segments provided")

        update_job(job_id, status="preparing", message="Preparing render...", progress=0.05)

        # Check GPU availability
        has_gpu = check_nvidia_gpu() if request.use_gpu else False

        # Get original duration
        original_duration = get_video_duration(video_path)

        # Calculate rendered duration
        rendered_duration = sum(seg.end - seg.start for seg in request.segments)

        # Generate output filename
        if request.output_name:
            output_name = request.output_name
        else:
            input_stem = Path(video_path).stem
            output_name = f"{input_stem}_rendered_{job_id}"

        output_path = str(RENDERED_DIR / f"{output_name}.{request.format.value}")

        # Choose render method based on codec
        if request.video_codec == VideoCodec.COPY and request.audio_codec == AudioCodec.COPY:
            # Use concat for stream copy (faster)
            await render_with_concat(
                job_id, video_path, request.segments,
                output_path, request, has_gpu
            )
        else:
            # Use filter_complex for re-encoding (more precise)
            await render_with_filter(
                job_id, video_path, request.segments,
                output_path, request, has_gpu
            )

        update_job(job_id, status="finalizing", message="Finalizing...", progress=0.95)

        # Get output file size
        file_size = Path(output_path).stat().st_size

        # Build result
        result = RenderResult(
            job_id=job_id,
            input_path=video_path,
            output_path=output_path,
            original_duration=original_duration,
            rendered_duration=rendered_duration,
            file_size=file_size,
            format=request.format.value,
            video_codec=request.video_codec.value,
            audio_codec=request.audio_codec.value,
            rendered_at=datetime.now().isoformat()
        )

        update_job(
            job_id,
            status="complete",
            message="Render complete",
            progress=1.0,
            result=result.model_dump()
        )

    except Exception as e:
        update_job(job_id, status="error", error=str(e), message=f"Render failed: {e}")


# ============== API Endpoints ==============

@app.get("/")
async def root():
    """Service health check."""
    has_gpu = check_nvidia_gpu()
    return {
        "service": "render",
        "status": "healthy",
        "version": "1.0.0",
        "gpu_available": has_gpu,
        "formats": [f.value for f in OutputFormat],
        "video_codecs": [c.value for c in VideoCodec],
        "audio_codecs": [c.value for c in AudioCodec]
    }


@app.post("/render", response_model=RenderResponse)
async def render(request: RenderRequest, background_tasks: BackgroundTasks):
    """
    Render video with specified cuts.

    Accepts segments to KEEP and produces output video.
    Returns job_id to track progress.
    """
    job_id = generate_job_id()

    update_job(
        job_id,
        status="queued",
        message="Render job queued",
        video_path=request.video_path,
        segments_count=len(request.segments)
    )

    background_tasks.add_task(render_video, job_id, request)

    return RenderResponse(
        job_id=job_id,
        status="queued",
        message=f"Render job queued with {len(request.segments)} segments"
    )


@app.post("/render-from-smartcut")
async def render_from_smartcut(
    video_path: str,
    smartcut_job_id: str,
    background_tasks: BackgroundTasks,
    format: OutputFormat = OutputFormat.MP4,
    video_codec: VideoCodec = VideoCodec.H264,
    preset: Preset = Preset.FAST
):
    """
    Render video using SmartCut analysis results.

    Fetches keep_segments from SmartCut job and renders.
    """
    # Load SmartCut job result
    smartcut_file = JOBS_DIR / f"smartcut_{smartcut_job_id}.json"

    if not smartcut_file.exists():
        raise HTTPException(status_code=404, detail=f"SmartCut job {smartcut_job_id} not found")

    with open(smartcut_file) as f:
        smartcut_job = json.load(f)

    if smartcut_job.get("status") != "complete":
        raise HTTPException(status_code=400, detail="SmartCut job not complete")

    result = smartcut_job.get("result", {})
    keep_segments = result.get("keep_segments", [])

    if not keep_segments:
        raise HTTPException(status_code=400, detail="No segments to render")

    # Convert to Segment objects
    segments = [Segment(start=s["start"], end=s["end"]) for s in keep_segments]

    # Create render request
    request = RenderRequest(
        video_path=video_path,
        segments=segments,
        format=format,
        video_codec=video_codec,
        preset=preset
    )

    job_id = generate_job_id()

    update_job(
        job_id,
        status="queued",
        message="Render from SmartCut queued",
        video_path=video_path,
        smartcut_job_id=smartcut_job_id,
        segments_count=len(segments)
    )

    background_tasks.add_task(render_video, job_id, request)

    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Rendering {len(segments)} segments from SmartCut analysis",
        "original_duration": result.get("original_duration"),
        "expected_duration": result.get("original_duration", 0) - result.get("cut_duration", 0)
    }


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


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """Get render result by job ID."""
    job = get_job(job_id)

    if not job or job.get("status") != "complete":
        raise HTTPException(status_code=404, detail="Result not available")

    return job.get("result", {})


@app.get("/presets")
async def list_presets():
    """List available encoding presets with descriptions."""
    return {
        "presets": {
            "ultrafast": "Fastest encoding, larger file size",
            "fast": "Good balance of speed and quality",
            "medium": "Default preset, balanced",
            "slow": "Better compression, slower encoding",
            "veryslow": "Best compression, slowest encoding"
        },
        "crf_guide": {
            "18": "Visually lossless",
            "23": "Default, good quality",
            "28": "Smaller file, acceptable quality",
            "35": "Very small file, lower quality"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
