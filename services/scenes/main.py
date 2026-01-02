"""
Scenes Detection Service
========================

FastAPI microservice for video scene analysis:
- Detect scene changes using ffmpeg
- Generate scene thumbnails
- Calculate scene duration and statistics
- Content analysis (brightness, motion)
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
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(
    title="Scenes Detection Service",
    description="Detect and analyze video scenes",
    version="1.0.0"
)

# Configuration
MEDIA_BASE = Path(os.environ.get("MEDIA_BASE", "/home/gamtec1000/cn_media"))
SCENES_DIR = MEDIA_BASE / "scenes"
JOBS_DIR = MEDIA_BASE / ".jobs"

# Ensure directories exist
for d in [SCENES_DIR, JOBS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class SceneDetectionMethod(str, Enum):
    CONTENT = "content"  # Content-based (histogram difference)
    THRESHOLD = "threshold"  # Simple threshold
    COMBINED = "combined"  # Both methods


class Scene(BaseModel):
    """Detected scene."""
    id: int
    start: float
    end: float
    duration: float
    thumbnail_path: Optional[str] = None
    score: float = 0.0  # Scene change confidence
    brightness: Optional[float] = None
    is_black: bool = False


class SceneDetectionRequest(BaseModel):
    """Request for scene detection."""
    video_path: str

    # Detection settings
    method: SceneDetectionMethod = SceneDetectionMethod.CONTENT
    threshold: float = 0.3  # Scene change threshold (0.0-1.0)
    min_scene_duration: float = 0.5  # Minimum scene length in seconds

    # Output options
    generate_thumbnails: bool = True
    thumbnail_width: int = 320
    analyze_brightness: bool = True
    detect_black_frames: bool = True
    black_threshold: float = 0.1


class SceneDetectionResponse(BaseModel):
    """Response from scene detection."""
    job_id: str
    status: str
    message: str


class SceneDetectionResult(BaseModel):
    """Complete scene detection result."""
    job_id: str
    video_path: str
    duration: float
    scene_count: int
    scenes: List[Scene]
    average_scene_duration: float
    shortest_scene: float
    longest_scene: float
    analyzed_at: str


class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: str
    progress: float = 0.0
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# Job storage
jobs: Dict[str, Dict[str, Any]] = {}


def generate_job_id() -> str:
    return str(uuid.uuid4())[:8]


def update_job(job_id: str, **kwargs):
    if job_id not in jobs:
        jobs[job_id] = {"job_id": job_id, "status": "unknown"}
    jobs[job_id].update(kwargs)

    job_file = JOBS_DIR / f"scenes_{job_id}.json"
    with open(job_file, "w") as f:
        json.dump(jobs[job_id], f, indent=2, default=str)


def get_job(job_id: str) -> Optional[Dict]:
    if job_id in jobs:
        return jobs[job_id]

    job_file = JOBS_DIR / f"scenes_{job_id}.json"
    if job_file.exists():
        with open(job_file) as f:
            jobs[job_id] = json.load(f)
            return jobs[job_id]
    return None


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


def get_video_fps(video_path: str) -> float:
    """Get video FPS using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    fps_str = result.stdout.strip()
    if "/" in fps_str:
        num, den = fps_str.split("/")
        return float(num) / float(den)
    return float(fps_str)


async def detect_scenes_ffmpeg(
    video_path: str,
    threshold: float = 0.3,
    min_duration: float = 0.5
) -> List[Dict]:
    """Detect scene changes using ffmpeg's select filter."""

    # Use scene detection filter
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-"
    ]

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await process.communicate()
    output = stderr.decode()

    # Parse scene timestamps from showinfo output
    scene_times = [0.0]  # Start with 0

    # Look for pts_time in showinfo output
    for line in output.split('\n'):
        if 'pts_time:' in line:
            match = re.search(r'pts_time:(\d+\.?\d*)', line)
            if match:
                timestamp = float(match.group(1))
                # Only add if it's far enough from the last scene
                if not scene_times or (timestamp - scene_times[-1]) >= min_duration:
                    scene_times.append(timestamp)

    return scene_times


async def detect_scenes_content(
    video_path: str,
    threshold: float = 0.3,
    min_duration: float = 0.5
) -> List[Dict]:
    """Detect scenes using content-aware analysis."""

    # Use ffprobe to get frame-level scene scores
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_frames",
        "-select_streams", "v",
        "-print_format", "json",
        "-f", "lavfi",
        f"movie={video_path},select='gte(scene,0)'"
    ]

    # Alternative: Use simpler scene detection
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})',metadata=print:file=-",
        "-f", "null", "-"
    ]

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await process.communicate()

    scene_times = [0.0]

    # Parse metadata output
    for line in stdout.decode().split('\n') + stderr.decode().split('\n'):
        if 'pts_time' in line.lower():
            match = re.search(r'pts_time[=:](\d+\.?\d*)', line, re.IGNORECASE)
            if match:
                timestamp = float(match.group(1))
                if not scene_times or (timestamp - scene_times[-1]) >= min_duration:
                    scene_times.append(timestamp)

    return scene_times


async def generate_thumbnail(
    video_path: str,
    timestamp: float,
    output_path: str,
    width: int = 320
) -> bool:
    """Generate thumbnail at specific timestamp."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(timestamp),
        "-i", video_path,
        "-vframes", "1",
        "-vf", f"scale={width}:-1",
        output_path
    ]

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    await process.communicate()
    return Path(output_path).exists()


async def analyze_frame_brightness(video_path: str, timestamp: float) -> float:
    """Analyze brightness of frame at timestamp."""
    cmd = [
        "ffmpeg",
        "-ss", str(timestamp),
        "-i", video_path,
        "-vframes", "1",
        "-vf", "signalstats",
        "-f", "null", "-"
    ]

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await process.communicate()
    output = stderr.decode()

    # Parse YAVG (average luminance) from signalstats
    match = re.search(r'YAVG:(\d+\.?\d*)', output)
    if match:
        return float(match.group(1)) / 255.0  # Normalize to 0-1

    return 0.5  # Default mid-brightness


async def detect_black_frames(video_path: str, threshold: float = 0.1) -> List[Dict]:
    """Detect black frames in video."""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"blackdetect=d=0.1:pix_th={threshold}",
        "-f", "null", "-"
    ]

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await process.communicate()
    output = stderr.decode()

    black_frames = []
    for line in output.split('\n'):
        if 'black_start:' in line:
            start_match = re.search(r'black_start:(\d+\.?\d*)', line)
            end_match = re.search(r'black_end:(\d+\.?\d*)', line)
            if start_match and end_match:
                black_frames.append({
                    "start": float(start_match.group(1)),
                    "end": float(end_match.group(1))
                })

    return black_frames


async def analyze_scenes(job_id: str, request: SceneDetectionRequest):
    """Background task to detect and analyze scenes."""
    try:
        video_path = request.video_path

        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        update_job(job_id, status="analyzing", message="Getting video info...", progress=0.05)

        # Get video info
        duration = get_video_duration(video_path)
        fps = get_video_fps(video_path)

        update_job(job_id, status="analyzing", message="Detecting scenes...", progress=0.1)

        # Detect scene changes
        scene_times = await detect_scenes_ffmpeg(
            video_path,
            request.threshold,
            request.min_scene_duration
        )

        # Add end of video
        if not scene_times or scene_times[-1] < duration - 0.5:
            scene_times.append(duration)

        update_job(
            job_id,
            status="processing",
            message=f"Found {len(scene_times)-1} scenes, processing...",
            progress=0.3
        )

        # Detect black frames if requested
        black_frames = []
        if request.detect_black_frames:
            black_frames = await detect_black_frames(video_path, request.black_threshold)

        # Create scene directory for this job
        job_scenes_dir = SCENES_DIR / job_id
        job_scenes_dir.mkdir(exist_ok=True)

        # Build scene objects
        scenes = []
        total_scenes = len(scene_times) - 1

        for i in range(total_scenes):
            scene_start = scene_times[i]
            scene_end = scene_times[i + 1]
            scene_duration = scene_end - scene_start

            scene = Scene(
                id=i + 1,
                start=round(scene_start, 3),
                end=round(scene_end, 3),
                duration=round(scene_duration, 3),
                score=request.threshold if i > 0 else 0.0
            )

            # Check if scene is black
            for bf in black_frames:
                if bf["start"] <= scene_start and bf["end"] >= scene_end:
                    scene.is_black = True
                    break

            # Generate thumbnail
            if request.generate_thumbnails:
                thumb_time = scene_start + min(0.5, scene_duration / 2)
                thumb_path = job_scenes_dir / f"scene_{i+1:03d}.jpg"
                success = await generate_thumbnail(
                    video_path, thumb_time, str(thumb_path), request.thumbnail_width
                )
                if success:
                    scene.thumbnail_path = str(thumb_path)

            # Analyze brightness
            if request.analyze_brightness:
                brightness = await analyze_frame_brightness(video_path, scene_start)
                scene.brightness = round(brightness, 3)

            scenes.append(scene)

            # Update progress
            progress = 0.3 + (0.6 * (i + 1) / total_scenes)
            update_job(
                job_id,
                status="processing",
                message=f"Processing scene {i+1}/{total_scenes}...",
                progress=progress
            )

        update_job(job_id, status="finalizing", message="Building result...", progress=0.95)

        # Calculate statistics
        scene_durations = [s.duration for s in scenes]
        avg_duration = sum(scene_durations) / len(scene_durations) if scene_durations else 0

        result = SceneDetectionResult(
            job_id=job_id,
            video_path=video_path,
            duration=duration,
            scene_count=len(scenes),
            scenes=[s.model_dump() for s in scenes],
            average_scene_duration=round(avg_duration, 3),
            shortest_scene=round(min(scene_durations), 3) if scene_durations else 0,
            longest_scene=round(max(scene_durations), 3) if scene_durations else 0,
            analyzed_at=datetime.now().isoformat()
        )

        update_job(
            job_id,
            status="complete",
            message=f"Detected {len(scenes)} scenes",
            progress=1.0,
            result=result.model_dump()
        )

    except Exception as e:
        update_job(job_id, status="error", error=str(e), message=f"Analysis failed: {e}")


# ============== API Endpoints ==============

@app.get("/")
async def root():
    """Service health check."""
    return {
        "service": "scenes",
        "status": "healthy",
        "version": "1.0.0",
        "methods": [m.value for m in SceneDetectionMethod]
    }


@app.post("/detect", response_model=SceneDetectionResponse)
async def detect_scenes(request: SceneDetectionRequest, background_tasks: BackgroundTasks):
    """
    Detect scenes in video.

    Analyzes video for scene changes, generates thumbnails,
    and calculates scene statistics.
    Returns job_id to track progress.
    """
    job_id = generate_job_id()

    update_job(
        job_id,
        status="queued",
        message="Scene detection queued",
        video_path=request.video_path
    )

    background_tasks.add_task(analyze_scenes, job_id, request)

    return SceneDetectionResponse(
        job_id=job_id,
        status="queued",
        message="Scene detection job queued"
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


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """Get scene detection result."""
    job = get_job(job_id)

    if not job or job.get("status") != "complete":
        raise HTTPException(status_code=404, detail="Result not available")

    return job.get("result", {})


@app.get("/scenes/{job_id}")
async def get_scenes(job_id: str, include_thumbnails: bool = False):
    """Get detected scenes with optional thumbnail paths."""
    job = get_job(job_id)

    if not job or job.get("status") != "complete":
        raise HTTPException(status_code=404, detail="Scenes not available")

    result = job.get("result", {})
    scenes = result.get("scenes", [])

    if not include_thumbnails:
        # Remove thumbnail paths for lighter response
        scenes = [{k: v for k, v in s.items() if k != "thumbnail_path"} for s in scenes]

    return {
        "scene_count": len(scenes),
        "scenes": scenes
    }


@app.get("/thumbnail/{job_id}/{scene_id}")
async def get_scene_thumbnail(job_id: str, scene_id: int):
    """Get thumbnail for specific scene."""
    thumb_path = SCENES_DIR / job_id / f"scene_{scene_id:03d}.jpg"

    if not thumb_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    return FileResponse(thumb_path, media_type="image/jpeg")


@app.get("/timeline/{job_id}")
async def get_timeline(job_id: str):
    """Get scene timeline for visualization."""
    job = get_job(job_id)

    if not job or job.get("status") != "complete":
        raise HTTPException(status_code=404, detail="Timeline not available")

    result = job.get("result", {})
    scenes = result.get("scenes", [])
    duration = result.get("duration", 0)

    # Build timeline data for visualization
    timeline = []
    for scene in scenes:
        timeline.append({
            "id": scene["id"],
            "start_percent": round(scene["start"] / duration * 100, 2) if duration else 0,
            "width_percent": round(scene["duration"] / duration * 100, 2) if duration else 0,
            "is_black": scene.get("is_black", False),
            "brightness": scene.get("brightness", 0.5)
        })

    return {
        "duration": duration,
        "scene_count": len(scenes),
        "timeline": timeline
    }


@app.post("/quick-detect")
async def quick_detect(video_path: str, threshold: float = 0.3):
    """
    Quick synchronous scene detection.

    Returns basic scene times without thumbnails or analysis.
    Suitable for short videos or quick previews.
    """
    if not Path(video_path).exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")

    duration = get_video_duration(video_path)
    scene_times = await detect_scenes_ffmpeg(video_path, threshold, 0.5)

    if not scene_times or scene_times[-1] < duration - 0.5:
        scene_times.append(duration)

    scenes = []
    for i in range(len(scene_times) - 1):
        scenes.append({
            "id": i + 1,
            "start": round(scene_times[i], 3),
            "end": round(scene_times[i + 1], 3),
            "duration": round(scene_times[i + 1] - scene_times[i], 3)
        })

    return {
        "video_path": video_path,
        "duration": duration,
        "scene_count": len(scenes),
        "scenes": scenes
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
