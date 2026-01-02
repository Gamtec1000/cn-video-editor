"""
SmartCut Service
================

FastAPI microservice for intelligent video cutting:
- Detect silences in audio
- Identify filler words from transcripts
- Suggest cuts based on pauses and sentence boundaries
- Generate EDL (Edit Decision List) for video editors
"""

import os
import json
import uuid
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import numpy as np

app = FastAPI(
    title="SmartCut Service",
    description="AI-powered video cutting suggestions",
    version="1.0.0"
)

# Configuration
MEDIA_BASE = Path(os.environ.get("MEDIA_BASE", "/home/gamtec1000/cn_media"))
CUTS_DIR = MEDIA_BASE / "cuts"
JOBS_DIR = MEDIA_BASE / ".jobs"

# Ensure directories exist
for d in [CUTS_DIR, JOBS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Common filler words to detect
FILLER_WORDS = {
    "um", "uh", "er", "ah", "like", "you know", "basically",
    "actually", "literally", "right", "so", "well", "i mean",
    "kind of", "sort of", "okay so", "yeah so"
}

# Job storage
jobs: Dict[str, Dict[str, Any]] = {}


class SmartCutRequest(BaseModel):
    """Request for smart cut analysis."""
    video_path: str
    transcript_path: Optional[str] = None
    transcript_data: Optional[Dict] = None

    # Cut detection settings
    silence_threshold_db: float = -40.0  # dB threshold for silence
    min_silence_duration: float = 0.5    # Minimum silence to cut (seconds)
    min_speech_duration: float = 0.3     # Minimum speech to keep (seconds)
    detect_fillers: bool = True          # Detect filler words
    padding: float = 0.1                 # Padding around cuts (seconds)


class CutSegment(BaseModel):
    """A segment to cut or keep."""
    start: float
    end: float
    duration: float
    type: str  # "silence", "filler", "keep", "speech"
    action: str  # "cut" or "keep"
    reason: Optional[str] = None
    confidence: float = 1.0


class SmartCutResult(BaseModel):
    """Result of smart cut analysis."""
    job_id: str
    video_path: str
    original_duration: float
    cut_duration: float
    saved_duration: float
    saved_percent: float
    segments: List[CutSegment]
    keep_segments: List[CutSegment]
    cut_segments: List[CutSegment]
    edl_path: Optional[str] = None
    analyzed_at: str


class SmartCutResponse(BaseModel):
    """Response from smart cut request."""
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


def generate_job_id() -> str:
    """Generate unique job ID."""
    return str(uuid.uuid4())[:8]


def update_job(job_id: str, **kwargs):
    """Update job status."""
    if job_id not in jobs:
        jobs[job_id] = {"job_id": job_id, "status": "unknown"}
    jobs[job_id].update(kwargs)

    # Persist to disk
    job_file = JOBS_DIR / f"smartcut_{job_id}.json"
    with open(job_file, "w") as f:
        json.dump(jobs[job_id], f, indent=2, default=str)


def get_job(job_id: str) -> Optional[Dict]:
    """Get job status."""
    if job_id in jobs:
        return jobs[job_id]

    job_file = JOBS_DIR / f"smartcut_{job_id}.json"
    if job_file.exists():
        with open(job_file) as f:
            jobs[job_id] = json.load(f)
            return jobs[job_id]

    return None


def get_audio_duration(video_path: str) -> float:
    """Get duration of audio/video file using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def detect_silences_ffmpeg(video_path: str, threshold_db: float = -40, min_duration: float = 0.5) -> List[Dict]:
    """Detect silences using ffmpeg silencedetect filter."""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-af", f"silencedetect=noise={threshold_db}dB:d={min_duration}",
        "-f", "null", "-"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr = result.stderr

    silences = []
    current_silence = {}

    for line in stderr.split('\n'):
        if 'silence_start:' in line:
            parts = line.split('silence_start:')
            if len(parts) > 1:
                try:
                    start = float(parts[1].strip().split()[0])
                    current_silence = {"start": start}
                except (ValueError, IndexError):
                    pass
        elif 'silence_end:' in line and current_silence:
            parts = line.split('silence_end:')
            if len(parts) > 1:
                try:
                    end_parts = parts[1].strip().split()
                    end = float(end_parts[0])
                    duration = float(end_parts[-1]) if len(end_parts) > 2 else end - current_silence["start"]
                    silences.append({
                        "start": current_silence["start"],
                        "end": end,
                        "duration": duration
                    })
                    current_silence = {}
                except (ValueError, IndexError):
                    pass

    return silences


def find_filler_words(transcript_data: Dict) -> List[Dict]:
    """Find filler words in transcript segments."""
    fillers_found = []

    segments = transcript_data.get("segments", [])

    for segment in segments:
        words = segment.get("words", [])
        text = segment.get("text", "").lower()

        # Check for multi-word fillers in segment text
        for filler in FILLER_WORDS:
            if filler in text and len(filler.split()) > 1:
                # Multi-word filler found in segment
                fillers_found.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "word": filler,
                    "type": "filler_phrase",
                    "confidence": 0.7
                })

        # Check individual words
        for word_info in words:
            word = word_info.get("word", "").lower().strip()
            word_clean = ''.join(c for c in word if c.isalnum())

            if word_clean in FILLER_WORDS:
                fillers_found.append({
                    "start": word_info.get("start", segment["start"]),
                    "end": word_info.get("end", segment["end"]),
                    "word": word,
                    "type": "filler_word",
                    "confidence": word_info.get("probability", 0.5)
                })

    return fillers_found


def merge_overlapping_segments(segments: List[Dict], padding: float = 0.1) -> List[Dict]:
    """Merge overlapping or adjacent segments."""
    if not segments:
        return []

    # Sort by start time
    sorted_segs = sorted(segments, key=lambda x: x["start"])
    merged = [sorted_segs[0].copy()]

    for seg in sorted_segs[1:]:
        last = merged[-1]
        # If overlapping or very close, merge
        if seg["start"] <= last["end"] + padding:
            last["end"] = max(last["end"], seg["end"])
            last["duration"] = last["end"] - last["start"]
            if "reasons" not in last:
                last["reasons"] = [last.get("reason", "")]
            last["reasons"].append(seg.get("reason", ""))
        else:
            merged.append(seg.copy())

    return merged


def generate_edl(job_id: str, keep_segments: List[CutSegment], fps: float = 30.0) -> str:
    """Generate EDL (Edit Decision List) file."""
    edl_path = CUTS_DIR / f"{job_id}.edl"

    def timecode(seconds: float) -> str:
        """Convert seconds to timecode HH:MM:SS:FF"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        f = int((seconds % 1) * fps)
        return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"

    lines = ["TITLE: SmartCut Export", "FCM: NON-DROP FRAME", ""]

    record_start = 0.0
    for i, seg in enumerate(keep_segments, 1):
        src_in = timecode(seg.start)
        src_out = timecode(seg.end)
        rec_in = timecode(record_start)
        record_start += seg.duration
        rec_out = timecode(record_start)

        lines.append(f"{i:03d}  001      V     C        {src_in} {src_out} {rec_in} {rec_out}")

    edl_content = "\n".join(lines)

    with open(edl_path, "w") as f:
        f.write(edl_content)

    return str(edl_path)


async def analyze_video(job_id: str, request: SmartCutRequest):
    """Background task to analyze video for smart cuts."""
    try:
        video_path = request.video_path

        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        update_job(job_id, status="analyzing", message="Getting video duration...", progress=0.1)

        # Get duration
        duration = get_audio_duration(video_path)

        update_job(job_id, status="analyzing", message="Detecting silences...", progress=0.2)

        # Detect silences
        silences = detect_silences_ffmpeg(
            video_path,
            threshold_db=request.silence_threshold_db,
            min_duration=request.min_silence_duration
        )

        # Convert silences to cut segments
        cut_segments = []
        for silence in silences:
            # Add padding - keep a bit of silence at boundaries
            padded_start = silence["start"] + request.padding
            padded_end = silence["end"] - request.padding

            if padded_end > padded_start + 0.1:  # Only if meaningful silence remains
                cut_segments.append({
                    "start": padded_start,
                    "end": padded_end,
                    "duration": padded_end - padded_start,
                    "type": "silence",
                    "reason": f"Silence ({silence['duration']:.1f}s)"
                })

        update_job(job_id, status="analyzing", message="Analyzing transcript...", progress=0.5)

        # Analyze transcript for filler words
        filler_cuts = []
        if request.detect_fillers:
            transcript_data = request.transcript_data

            # Load from file if path provided
            if not transcript_data and request.transcript_path:
                transcript_file = Path(request.transcript_path)
                if transcript_file.exists():
                    with open(transcript_file) as f:
                        transcript_data = json.load(f)

            if transcript_data:
                fillers = find_filler_words(transcript_data)
                for filler in fillers:
                    # Only cut low-confidence filler words
                    if filler["confidence"] < 0.6:
                        filler_cuts.append({
                            "start": filler["start"],
                            "end": filler["end"],
                            "duration": filler["end"] - filler["start"],
                            "type": "filler",
                            "reason": f"Filler word: {filler['word']}"
                        })

        update_job(job_id, status="processing", message="Merging cut regions...", progress=0.7)

        # Combine all cuts
        all_cuts = cut_segments + filler_cuts
        merged_cuts = merge_overlapping_segments(all_cuts, request.padding)

        # Calculate keep segments (inverse of cuts)
        keep_segments = []
        current_pos = 0.0

        for cut in sorted(merged_cuts, key=lambda x: x["start"]):
            if cut["start"] > current_pos + request.min_speech_duration:
                keep_segments.append({
                    "start": current_pos,
                    "end": cut["start"],
                    "duration": cut["start"] - current_pos,
                    "type": "speech",
                    "action": "keep"
                })
            current_pos = cut["end"]

        # Add final segment if needed
        if current_pos < duration - request.min_speech_duration:
            keep_segments.append({
                "start": current_pos,
                "end": duration,
                "duration": duration - current_pos,
                "type": "speech",
                "action": "keep"
            })

        update_job(job_id, status="processing", message="Generating EDL...", progress=0.9)

        # Calculate savings
        cut_duration = sum(seg["duration"] for seg in merged_cuts)
        kept_duration = duration - cut_duration
        saved_percent = (cut_duration / duration * 100) if duration > 0 else 0

        # Generate EDL
        keep_segment_models = [
            CutSegment(
                start=s["start"],
                end=s["end"],
                duration=s["duration"],
                type=s["type"],
                action="keep"
            ) for s in keep_segments
        ]

        edl_path = generate_edl(job_id, keep_segment_models)

        # Build result
        cut_segment_models = [
            CutSegment(
                start=s["start"],
                end=s["end"],
                duration=s["duration"],
                type=s["type"],
                action="cut",
                reason=s.get("reason")
            ) for s in merged_cuts
        ]

        all_segments = sorted(
            keep_segment_models + cut_segment_models,
            key=lambda x: x.start
        )

        result = SmartCutResult(
            job_id=job_id,
            video_path=video_path,
            original_duration=duration,
            cut_duration=cut_duration,
            saved_duration=cut_duration,
            saved_percent=round(saved_percent, 1),
            segments=[s.model_dump() for s in all_segments],
            keep_segments=[s.model_dump() for s in keep_segment_models],
            cut_segments=[s.model_dump() for s in cut_segment_models],
            edl_path=edl_path,
            analyzed_at=datetime.now().isoformat()
        )

        update_job(
            job_id,
            status="complete",
            message="Analysis complete",
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
        "service": "smartcut",
        "status": "healthy",
        "version": "1.0.0",
        "filler_words": len(FILLER_WORDS)
    }


@app.post("/analyze", response_model=SmartCutResponse)
async def analyze(request: SmartCutRequest, background_tasks: BackgroundTasks):
    """
    Analyze video for smart cuts.

    Detects silences and filler words, suggests cut points.
    Returns job_id to track progress.
    """
    job_id = generate_job_id()

    update_job(job_id, status="queued", message="Analysis queued", video_path=request.video_path)

    background_tasks.add_task(analyze_video, job_id, request)

    return SmartCutResponse(
        job_id=job_id,
        status="queued",
        message="Smart cut analysis queued"
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
    """Get analysis result by job ID."""
    job = get_job(job_id)

    if not job or job.get("status") != "complete":
        raise HTTPException(status_code=404, detail="Result not available")

    return job.get("result", {})


@app.get("/edl/{job_id}")
async def get_edl(job_id: str):
    """Get EDL file content for job."""
    job = get_job(job_id)

    if not job or job.get("status") != "complete":
        raise HTTPException(status_code=404, detail="EDL not available")

    result = job.get("result", {})
    edl_path = result.get("edl_path")

    if not edl_path or not Path(edl_path).exists():
        raise HTTPException(status_code=404, detail="EDL file not found")

    with open(edl_path) as f:
        return {"edl": f.read(), "path": edl_path}


@app.post("/preview-cuts")
async def preview_cuts(request: SmartCutRequest):
    """
    Quick preview of cut suggestions without full analysis.
    Synchronous - returns immediately with basic silence detection.
    """
    video_path = request.video_path

    if not Path(video_path).exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")

    # Quick silence detection
    silences = detect_silences_ffmpeg(
        video_path,
        threshold_db=request.silence_threshold_db,
        min_duration=request.min_silence_duration
    )

    duration = get_audio_duration(video_path)
    total_silence = sum(s["duration"] for s in silences)

    return {
        "video_path": video_path,
        "duration": duration,
        "silence_count": len(silences),
        "total_silence": round(total_silence, 2),
        "potential_savings_percent": round(total_silence / duration * 100, 1) if duration > 0 else 0,
        "silences": silences[:10]  # First 10 for preview
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
