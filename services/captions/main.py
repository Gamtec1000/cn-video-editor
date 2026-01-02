"""
Captions Service
================

FastAPI microservice for caption/subtitle generation:
- Generate SRT, VTT, ASS subtitle files from transcripts
- Word-level highlighting for animated captions
- Customizable styling (font, color, position)
- Burn captions into video
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
from fastapi.responses import PlainTextResponse, FileResponse
from pydantic import BaseModel

app = FastAPI(
    title="Captions Service",
    description="Generate and burn subtitles/captions",
    version="1.0.0"
)

# Configuration
MEDIA_BASE = Path(os.environ.get("MEDIA_BASE", "/home/gamtec1000/cn_media"))
CAPTIONS_DIR = MEDIA_BASE / "captions"
JOBS_DIR = MEDIA_BASE / ".jobs"

# Ensure directories exist
for d in [CAPTIONS_DIR, JOBS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class CaptionFormat(str, Enum):
    SRT = "srt"
    VTT = "vtt"
    ASS = "ass"
    JSON = "json"


class CaptionPosition(str, Enum):
    BOTTOM = "bottom"
    TOP = "top"
    MIDDLE = "middle"


class CaptionStyle(BaseModel):
    """Caption styling options."""
    font_name: str = "Arial"
    font_size: int = 24
    primary_color: str = "FFFFFF"  # White (hex without #)
    outline_color: str = "000000"  # Black
    background_color: str = "80000000"  # Semi-transparent black
    outline_width: int = 2
    shadow_depth: int = 1
    position: CaptionPosition = CaptionPosition.BOTTOM
    margin_v: int = 30  # Vertical margin
    margin_h: int = 20  # Horizontal margin
    bold: bool = False
    italic: bool = False


class Segment(BaseModel):
    """Transcript segment."""
    start: float
    end: float
    text: str
    words: Optional[List[Dict]] = None


class GenerateCaptionsRequest(BaseModel):
    """Request to generate captions."""
    transcript_path: Optional[str] = None
    transcript_data: Optional[Dict] = None
    segments: Optional[List[Segment]] = None

    format: CaptionFormat = CaptionFormat.SRT
    style: Optional[CaptionStyle] = None

    # Word-level options
    word_highlight: bool = False  # Highlight current word
    highlight_color: str = "FFFF00"  # Yellow

    # Line breaking
    max_chars_per_line: int = 42
    max_lines: int = 2


class BurnCaptionsRequest(BaseModel):
    """Request to burn captions into video."""
    video_path: str
    caption_path: Optional[str] = None
    caption_data: Optional[str] = None  # Raw SRT/VTT/ASS content

    style: Optional[CaptionStyle] = None
    output_name: Optional[str] = None


class CaptionResponse(BaseModel):
    """Response from caption generation."""
    job_id: str
    status: str
    message: str
    caption_path: Optional[str] = None


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

    job_file = JOBS_DIR / f"captions_{job_id}.json"
    with open(job_file, "w") as f:
        json.dump(jobs[job_id], f, indent=2, default=str)


def get_job(job_id: str) -> Optional[Dict]:
    if job_id in jobs:
        return jobs[job_id]

    job_file = JOBS_DIR / f"captions_{job_id}.json"
    if job_file.exists():
        with open(job_file) as f:
            jobs[job_id] = json.load(f)
            return jobs[job_id]
    return None


def format_timestamp_srt(seconds: float) -> str:
    """Format timestamp for SRT (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format timestamp for VTT (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def format_timestamp_ass(seconds: float) -> str:
    """Format timestamp for ASS (H:MM:SS.cc)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def wrap_text(text: str, max_chars: int = 42, max_lines: int = 2) -> str:
    """Wrap text to fit within constraints."""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        word_len = len(word)
        if current_length + word_len + (1 if current_line else 0) <= max_chars:
            current_line.append(word)
            current_length += word_len + (1 if len(current_line) > 1 else 0)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_length = word_len

            if len(lines) >= max_lines:
                break

    if current_line and len(lines) < max_lines:
        lines.append(" ".join(current_line))

    return "\n".join(lines)


def generate_srt(segments: List[Segment], max_chars: int = 42, max_lines: int = 2) -> str:
    """Generate SRT subtitle content."""
    lines = []

    for i, seg in enumerate(segments, 1):
        start_ts = format_timestamp_srt(seg.start)
        end_ts = format_timestamp_srt(seg.end)
        text = wrap_text(seg.text.strip(), max_chars, max_lines)

        lines.append(str(i))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


def generate_vtt(segments: List[Segment], max_chars: int = 42, max_lines: int = 2) -> str:
    """Generate WebVTT subtitle content."""
    lines = ["WEBVTT", ""]

    for i, seg in enumerate(segments, 1):
        start_ts = format_timestamp_vtt(seg.start)
        end_ts = format_timestamp_vtt(seg.end)
        text = wrap_text(seg.text.strip(), max_chars, max_lines)

        lines.append(str(i))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


def hex_to_ass_color(hex_color: str) -> str:
    """Convert hex color to ASS format (&HBBGGRR)."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
        return f"&H00{b}{g}{r}"
    elif len(hex_color) == 8:
        a, r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6], hex_color[6:8]
        return f"&H{a}{b}{g}{r}"
    return "&H00FFFFFF"


def generate_ass(
    segments: List[Segment],
    style: CaptionStyle,
    max_chars: int = 42,
    max_lines: int = 2,
    word_highlight: bool = False,
    highlight_color: str = "FFFF00"
) -> str:
    """Generate ASS/SSA subtitle content with styling."""

    # ASS header
    header = f"""[Script Info]
Title: Generated Captions
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{style.font_name},{style.font_size},{hex_to_ass_color(style.primary_color)},{hex_to_ass_color(highlight_color)},{hex_to_ass_color(style.outline_color)},{hex_to_ass_color(style.background_color)},{1 if style.bold else 0},{1 if style.italic else 0},0,0,100,100,0,0,1,{style.outline_width},{style.shadow_depth},{2 if style.position == CaptionPosition.BOTTOM else 8 if style.position == CaptionPosition.TOP else 5},{style.margin_h},{style.margin_h},{style.margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events = []

    for seg in segments:
        start_ts = format_timestamp_ass(seg.start)
        end_ts = format_timestamp_ass(seg.end)
        text = seg.text.strip().replace("\n", "\\N")

        if word_highlight and seg.words:
            # Generate word-by-word highlight using karaoke tags
            highlighted_text = ""
            for word_info in seg.words:
                word = word_info.get("word", "").strip()
                if word:
                    # Duration in centiseconds
                    word_start = word_info.get("start", seg.start)
                    word_end = word_info.get("end", seg.end)
                    duration_cs = int((word_end - word_start) * 100)
                    highlighted_text += f"{{\\kf{duration_cs}}}{word} "
            text = highlighted_text.strip()

        events.append(f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{text}")

    return header + "\n".join(events)


def generate_json_captions(segments: List[Segment]) -> str:
    """Generate JSON caption format."""
    data = {
        "captions": [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "words": seg.words
            }
            for seg in segments
        ]
    }
    return json.dumps(data, indent=2)


def load_segments_from_transcript(transcript_data: Dict) -> List[Segment]:
    """Load segments from transcript data."""
    segments = []
    for seg in transcript_data.get("segments", []):
        segments.append(Segment(
            start=seg.get("start", 0),
            end=seg.get("end", 0),
            text=seg.get("text", ""),
            words=seg.get("words")
        ))
    return segments


async def generate_captions(job_id: str, request: GenerateCaptionsRequest) -> str:
    """Generate captions in specified format."""

    # Load segments
    segments = []

    if request.segments:
        segments = request.segments
    elif request.transcript_data:
        segments = load_segments_from_transcript(request.transcript_data)
    elif request.transcript_path:
        transcript_path = Path(request.transcript_path)
        if transcript_path.exists():
            with open(transcript_path) as f:
                transcript_data = json.load(f)
                segments = load_segments_from_transcript(transcript_data)

    if not segments:
        raise ValueError("No segments provided")

    style = request.style or CaptionStyle()

    # Generate captions based on format
    if request.format == CaptionFormat.SRT:
        content = generate_srt(segments, request.max_chars_per_line, request.max_lines)
        ext = "srt"
    elif request.format == CaptionFormat.VTT:
        content = generate_vtt(segments, request.max_chars_per_line, request.max_lines)
        ext = "vtt"
    elif request.format == CaptionFormat.ASS:
        content = generate_ass(
            segments, style,
            request.max_chars_per_line, request.max_lines,
            request.word_highlight, request.highlight_color
        )
        ext = "ass"
    else:  # JSON
        content = generate_json_captions(segments)
        ext = "json"

    # Save to file
    output_path = CAPTIONS_DIR / f"{job_id}.{ext}"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return str(output_path)


async def burn_captions_task(job_id: str, request: BurnCaptionsRequest):
    """Background task to burn captions into video."""
    try:
        video_path = request.video_path

        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        update_job(job_id, status="preparing", message="Preparing captions...", progress=0.1)

        # Get or create caption file
        if request.caption_path:
            caption_path = request.caption_path
        elif request.caption_data:
            # Save caption data to temp file
            caption_path = str(CAPTIONS_DIR / f"{job_id}_temp.srt")
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write(request.caption_data)
        else:
            raise ValueError("No caption source provided")

        if not Path(caption_path).exists():
            raise FileNotFoundError(f"Caption file not found: {caption_path}")

        update_job(job_id, status="rendering", message="Burning captions...", progress=0.3)

        # Determine subtitle filter based on file extension
        caption_ext = Path(caption_path).suffix.lower()

        # Output path
        input_stem = Path(video_path).stem
        output_name = request.output_name or f"{input_stem}_captioned_{job_id}"
        output_path = str(MEDIA_BASE / "rendered" / f"{output_name}.mp4")

        # Build ffmpeg command
        if caption_ext == ".ass":
            # Use ASS filter for styled subtitles
            subtitle_filter = f"ass='{caption_path}'"
        else:
            # Use subtitles filter for SRT/VTT
            style_opts = ""
            if request.style:
                s = request.style
                style_opts = f":force_style='FontName={s.font_name},FontSize={s.font_size},PrimaryColour={hex_to_ass_color(s.primary_color)},OutlineColour={hex_to_ass_color(s.outline_color)},Outline={s.outline_width},Shadow={s.shadow_depth}'"
            subtitle_filter = f"subtitles='{caption_path}'{style_opts}"

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", subtitle_filter,
            "-c:a", "copy",
            output_path
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {stderr.decode()[-500:]}")

        update_job(job_id, status="finalizing", message="Finalizing...", progress=0.9)

        file_size = Path(output_path).stat().st_size

        result = {
            "job_id": job_id,
            "input_path": video_path,
            "caption_path": caption_path,
            "output_path": output_path,
            "file_size": file_size,
            "rendered_at": datetime.now().isoformat()
        }

        update_job(
            job_id,
            status="complete",
            message="Captions burned successfully",
            progress=1.0,
            result=result
        )

    except Exception as e:
        update_job(job_id, status="error", error=str(e), message=f"Burn failed: {e}")


# ============== API Endpoints ==============

@app.get("/")
async def root():
    """Service health check."""
    return {
        "service": "captions",
        "status": "healthy",
        "version": "1.0.0",
        "formats": [f.value for f in CaptionFormat],
        "positions": [p.value for p in CaptionPosition]
    }


@app.post("/generate", response_model=CaptionResponse)
async def generate(request: GenerateCaptionsRequest):
    """
    Generate captions from transcript.

    Supports SRT, VTT, ASS, and JSON formats.
    Synchronous operation - returns immediately with caption file path.
    """
    job_id = generate_job_id()

    try:
        caption_path = await generate_captions(job_id, request)

        return CaptionResponse(
            job_id=job_id,
            status="complete",
            message=f"Captions generated in {request.format.value} format",
            caption_path=caption_path
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/generate-all")
async def generate_all_formats(request: GenerateCaptionsRequest):
    """
    Generate captions in all formats from transcript.

    Returns paths to SRT, VTT, ASS, and JSON files.
    """
    job_id = generate_job_id()
    results = {}

    for fmt in CaptionFormat:
        try:
            request_copy = request.model_copy()
            request_copy.format = fmt
            caption_path = await generate_captions(f"{job_id}_{fmt.value}", request_copy)
            results[fmt.value] = caption_path
        except Exception as e:
            results[fmt.value] = f"Error: {str(e)}"

    return {
        "job_id": job_id,
        "status": "complete",
        "captions": results
    }


@app.post("/burn", response_model=CaptionResponse)
async def burn_captions(request: BurnCaptionsRequest, background_tasks: BackgroundTasks):
    """
    Burn captions into video.

    Takes video and caption file, produces video with embedded subtitles.
    Returns job_id to track progress.
    """
    job_id = generate_job_id()

    update_job(
        job_id,
        status="queued",
        message="Burn job queued",
        video_path=request.video_path
    )

    background_tasks.add_task(burn_captions_task, job_id, request)

    return CaptionResponse(
        job_id=job_id,
        status="queued",
        message="Caption burn job queued"
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


@app.get("/download/{job_id}")
async def download_caption(job_id: str, format: CaptionFormat = CaptionFormat.SRT):
    """Download generated caption file."""
    caption_path = CAPTIONS_DIR / f"{job_id}.{format.value}"

    if not caption_path.exists():
        raise HTTPException(status_code=404, detail="Caption file not found")

    return FileResponse(
        caption_path,
        filename=caption_path.name,
        media_type="text/plain"
    )


@app.get("/preview/{job_id}")
async def preview_caption(job_id: str, format: CaptionFormat = CaptionFormat.SRT):
    """Preview caption content."""
    caption_path = CAPTIONS_DIR / f"{job_id}.{format.value}"

    if not caption_path.exists():
        raise HTTPException(status_code=404, detail="Caption file not found")

    with open(caption_path, encoding="utf-8") as f:
        content = f.read()

    return PlainTextResponse(content)


@app.get("/styles/presets")
async def get_style_presets():
    """Get available style presets."""
    return {
        "default": CaptionStyle().model_dump(),
        "youtube": CaptionStyle(
            font_name="Roboto",
            font_size=28,
            primary_color="FFFFFF",
            outline_color="000000",
            background_color="80000000",
            outline_width=2,
            position=CaptionPosition.BOTTOM,
            margin_v=50
        ).model_dump(),
        "netflix": CaptionStyle(
            font_name="Netflix Sans",
            font_size=32,
            primary_color="FFFFFF",
            outline_color="000000",
            background_color="00000000",
            outline_width=3,
            shadow_depth=2,
            position=CaptionPosition.BOTTOM,
            margin_v=40
        ).model_dump(),
        "tiktok": CaptionStyle(
            font_name="Proxima Nova",
            font_size=36,
            primary_color="FFFFFF",
            outline_color="000000",
            background_color="00000000",
            outline_width=3,
            bold=True,
            position=CaptionPosition.MIDDLE,
            margin_v=0
        ).model_dump(),
        "minimal": CaptionStyle(
            font_name="Helvetica",
            font_size=22,
            primary_color="FFFFFF",
            outline_color="000000",
            background_color="00000000",
            outline_width=1,
            shadow_depth=0,
            position=CaptionPosition.BOTTOM,
            margin_v=20
        ).model_dump()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
