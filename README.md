# CN Video Editor

AI-powered video editing microservices architecture for automated video processing, transcription, smart cutting, and caption generation.

## Services

| Service | Port | Description |
|---------|------|-------------|
| **Ingest** | 8001 | YouTube/URL download, metadata extraction, thumbnail generation |
| **Transcribe** | 8002 | Whisper-based speech-to-text with word-level timestamps |
| **SmartCut** | 8003 | Silence detection, filler word removal, EDL export |
| **Render** | 8004 | Apply cuts, GPU-accelerated encoding, format conversion |
| **Captions** | 8005 | SRT/VTT/ASS subtitle generation, burn-in captions |
| **Scenes** | 8006 | Scene change detection, thumbnails, timeline visualization |

## Architecture

```
Input → Ingest → Transcribe → Scenes → SmartCut → Render → Captions → Output
         │          │           │         │          │         │
       8001       8002        8006      8003       8004      8005
         ↓          ↓           ↓         ↓          ↓         ↓
     metadata    text +      scene     silence    final    subtitled
     + thumb     words      detect     removal    video     video
```

## Quick Start

### Development Mode (without Docker)

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r services/ingest/requirements.txt
pip install -r services/transcribe/requirements.txt
pip install -r services/smartcut/requirements.txt
pip install -r services/render/requirements.txt
pip install -r services/captions/requirements.txt
pip install -r services/scenes/requirements.txt

# Start all services
./scripts/start_dev.sh
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## API Examples

### 1. Ingest Video from YouTube

```bash
curl -X POST http://localhost:8001/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=VIDEO_ID"}'
```

### 2. Transcribe Video

```bash
curl -X POST http://localhost:8002/transcribe \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/video.mp4", "word_timestamps": true}'
```

### 3. Detect Smart Cuts

```bash
curl -X POST http://localhost:8003/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/path/to/video.mp4",
    "transcript_path": "/path/to/transcript.json",
    "detect_fillers": true
  }'
```

### 4. Render with Cuts

```bash
curl -X POST "http://localhost:8004/render-from-smartcut?video_path=/path/to/video.mp4&smartcut_job_id=JOB_ID"
```

### 5. Generate Captions

```bash
curl -X POST http://localhost:8005/generate \
  -H "Content-Type: application/json" \
  -d '{
    "transcript_path": "/path/to/transcript.json",
    "format": "srt"
  }'
```

### 6. Detect Scenes

```bash
curl -X POST http://localhost:8006/detect \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/path/to/video.mp4",
    "threshold": 0.3,
    "generate_thumbnails": true
  }'
```

## Features

### Ingest Service
- YouTube and direct URL downloads via yt-dlp
- File upload support
- Automatic metadata extraction (duration, resolution, codec, fps)
- Thumbnail generation
- Background job processing

### Transcribe Service
- OpenAI Whisper integration
- Word-level timestamps with confidence scores
- Automatic language detection
- GPU acceleration support
- Multiple model sizes (tiny, base, small, medium, large)

### SmartCut Service
- FFmpeg-based silence detection
- Filler word detection from transcripts
- Configurable thresholds
- EDL (Edit Decision List) export
- Cut preview without rendering

### Render Service
- GPU-accelerated encoding (NVENC)
- Multiple output formats (MP4, WebM, MOV, MKV)
- Multiple codecs (H.264, H.265, VP9)
- Quality presets (ultrafast to veryslow)
- Resolution and FPS conversion

### Captions Service
- SRT, VTT, ASS/SSA format support
- Customizable styling (font, color, position)
- Style presets (YouTube, Netflix, TikTok)
- Word-level highlighting for karaoke effect
- Burn-in captions to video

### Scenes Service
- Content-aware scene detection
- Automatic thumbnail generation per scene
- Brightness analysis
- Black frame detection
- Timeline visualization data

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEDIA_BASE` | `/home/gamtec1000/cn_media` | Base directory for media files |
| `WHISPER_MODEL` | `base` | Whisper model size |

### Media Directory Structure

```
cn_media/
├── raw/           # Original ingested videos
├── thumbnails/    # Video thumbnails
├── audio/         # Extracted audio files
├── transcripts/   # Transcription JSON files
├── cuts/          # EDL and cut data
├── rendered/      # Rendered output videos
├── captions/      # Generated subtitle files
├── scenes/        # Scene thumbnails
└── .jobs/         # Job status files
```

## Requirements

- Python 3.11+
- FFmpeg with libx264, libx265, libvpx
- NVIDIA GPU (optional, for acceleration)
- yt-dlp
- OpenAI Whisper

## License

MIT License
