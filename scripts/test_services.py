#!/usr/bin/env python3
"""
Test script for Video Editor services.
Run after docker-compose up to verify services are working.
"""

import httpx
import asyncio
import sys
from pathlib import Path


SERVICES = {
    "ingest": "http://localhost:8001",
    "transcribe": "http://localhost:8002",
}


async def check_service(name: str, url: str) -> bool:
    """Check if a service is healthy."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{url}/", timeout=5.0)
            data = response.json()
            print(f"  {name}: {data.get('status', 'unknown')}")
            return data.get("status") == "healthy"
    except Exception as e:
        print(f"  {name}: FAILED - {e}")
        return False


async def test_ingest_url(url: str, video_url: str) -> dict:
    """Test video ingest from URL."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{url}/ingest/url",
            json={"url": video_url},
            timeout=10.0
        )
        return response.json()


async def test_transcribe(url: str, file_path: str) -> dict:
    """Test transcription of a file."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{url}/transcribe",
            json={"file_path": file_path},
            timeout=10.0
        )
        return response.json()


async def get_job_status(url: str, job_id: str) -> dict:
    """Get job status."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{url}/status/{job_id}", timeout=5.0)
        return response.json()


async def wait_for_job(url: str, job_id: str, max_wait: int = 300) -> dict:
    """Wait for job to complete."""
    for _ in range(max_wait):
        status = await get_job_status(url, job_id)
        if status.get("status") in ["complete", "error"]:
            return status
        await asyncio.sleep(1)
    return {"status": "timeout"}


async def main():
    print("=" * 50)
    print("CN Video Editor - Service Tests")
    print("=" * 50)

    # Check all services
    print("\n1. Checking service health...")
    all_healthy = True
    for name, url in SERVICES.items():
        if not await check_service(name, url):
            all_healthy = False

    if not all_healthy:
        print("\nSome services are not healthy. Start them with:")
        print("  cd /home/gamtec1000/cn_video_editor && docker-compose up -d")
        sys.exit(1)

    print("\nAll services healthy!")

    # Test ingest if URL provided
    if len(sys.argv) > 1:
        video_url = sys.argv[1]
        print(f"\n2. Testing ingest with URL: {video_url}")

        try:
            result = await test_ingest_url(SERVICES["ingest"], video_url)
            job_id = result.get("job_id")
            print(f"   Job ID: {job_id}")
            print(f"   Status: {result.get('status')}")

            print("\n   Waiting for ingest to complete...")
            final = await wait_for_job(SERVICES["ingest"], job_id)
            print(f"   Final status: {final.get('status')}")

            if final.get("status") == "complete":
                result_data = final.get("result", {})
                print(f"   File: {result_data.get('filename')}")
                print(f"   Duration: {result_data.get('duration', 0):.1f}s")
                print(f"   Resolution: {result_data.get('width')}x{result_data.get('height')}")

                # Test transcription
                file_path = result_data.get("file_path")
                if file_path:
                    print(f"\n3. Testing transcription...")
                    trans_result = await test_transcribe(SERVICES["transcribe"], file_path)
                    trans_job_id = trans_result.get("job_id")
                    print(f"   Job ID: {trans_job_id}")

                    print("\n   Waiting for transcription...")
                    trans_final = await wait_for_job(SERVICES["transcribe"], trans_job_id)
                    print(f"   Final status: {trans_final.get('status')}")

                    if trans_final.get("status") == "complete":
                        trans_data = trans_final.get("result", {})
                        print(f"   Language: {trans_data.get('language')}")
                        print(f"   Duration: {trans_data.get('duration', 0):.1f}s")
                        text = trans_data.get("text", "")
                        preview = text[:200] + "..." if len(text) > 200 else text
                        print(f"   Text preview: {preview}")

        except Exception as e:
            print(f"   Error: {e}")

    print("\n" + "=" * 50)
    print("Tests complete!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
