"""Image generation via DeAPI.ai (async: submit → poll → download)."""
from __future__ import annotations

import os
import random
import time
from pathlib import Path

import httpx

DEAPI_SUBMIT_URL = "https://api.deapi.ai/api/v1/client/txt2img"
DEAPI_POLL_URL = "https://api.deapi.ai/api/v1/client/request-status"

STYLE_SUFFIX = (
    ", cinematic digital illustration, detailed scene art, strong composition, "
    "professional youtube visual quality, no text, no captions, no watermark, no logos"
)

DEFAULT_NEGATIVE = (
    "blurry, low quality, watermark, logo, text, title, signature, ugly, grainy, "
    "gore, blood, nudity, child-unsafe"
)


def full_visual_prompt(scene: str, style_suffix: str | None = None) -> str:
    """Combine the scene description with a channel-specific style suffix."""
    return f"{scene.strip()}{(style_suffix or STYLE_SUFFIX)}"


def _deapi_generate(
    prompt: str,
    *,
    api_key: str,
    width: int,
    height: int,
    model: str,
    max_polls: int = 30,
    poll_interval: float = 3.0,
) -> bytes:
    """Submit image job, poll until done, download result."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Step 1: Submit
    payload = {
        "prompt": prompt,
        "model": model,
        "width": width,
        "height": height,
        "steps": 4,
        "seed": random.randint(1, 999999),
    }

    with httpx.Client(timeout=60.0) as client:
        # Submit with retry on 429
        for submit_try in range(5):
            resp = client.post(DEAPI_SUBMIT_URL, json=payload, headers=headers)
            if resp.status_code == 429:
                wait = 15 * (submit_try + 1)
                print(f"      DeAPI 429 on submit — waiting {wait}s (try {submit_try + 1}/5)…")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        else:
            raise RuntimeError("DeAPI: 429 on submit after 5 retries")

        data = resp.json()

        request_id = data.get("data", {}).get("request_id")
        if not request_id:
            raise RuntimeError(f"No request_id in DeAPI response: {data}")
        print(f"      DeAPI submitted (id: {request_id})")

        # Step 2: Poll
        poll_headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }

        for attempt in range(1, max_polls + 1):
            time.sleep(poll_interval)

            poll_resp = client.get(
                f"{DEAPI_POLL_URL}/{request_id}",
                headers=poll_headers,
                timeout=30.0,
            )
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()

            status = poll_data.get("data", {}).get("status", "")

            if status in ("completed", "success", "done"):
                image_url = poll_data["data"].get("result_url")
                if not image_url:
                    raise RuntimeError(f"Completed but no result_url: {poll_data}")

                img_resp = client.get(image_url, timeout=60.0)
                img_resp.raise_for_status()
                print(f"      DeAPI done (polled {attempt}x)")
                return img_resp.content

            if status in ("failed", "error"):
                raise RuntimeError(f"DeAPI image failed: {poll_data}")

            # Still processing — keep polling

        raise RuntimeError(f"DeAPI timed out after {max_polls} polls for {request_id}")


def _pollinations_generate(prompt: str, *, width: int = 768, height: int = 768, max_retries: int = 4) -> bytes:
    """Generate image via Pollinations.ai — free, no API key needed."""
    import urllib.parse
    encoded = urllib.parse.quote(prompt)
    seed = random.randint(1, 999999)
    url = f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&model=flux&seed={seed}&nologo=true"

    with httpx.Client(timeout=90.0, follow_redirects=True) as client:
        for attempt in range(max_retries):
            try:
                resp = client.get(url)
                if resp.status_code == 200 and resp.content:
                    print(f"      Pollinations done (attempt {attempt+1})")
                    return resp.content
                print(f"      Pollinations status {resp.status_code}, retrying…")
                time.sleep(10)
            except Exception as e:
                print(f"      Pollinations error: {e}, retrying…")
                time.sleep(10)

    raise RuntimeError(f"Pollinations image generation failed after {max_retries} retries")


def save_scene_image(
    index: int,
    prompt: str,
    out_path: Path,
    *,
    width: int = 768,
    height: int = 768,
    negative: str = DEFAULT_NEGATIVE,
) -> tuple[str, str]:
    """Generate and save one image. Returns (status, detail).

    Uses DeAPI if DEAPI_TOKEN is set, otherwise falls back to HuggingFace.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    deapi_key = os.environ.get("DEAPI_TOKEN", "").strip()
    hf_key = os.environ.get("HF_TOKEN", "").strip()

    if deapi_key:
        model = os.environ.get("DEAPI_MODEL", "Flux_2_Klein_4B_BF16")
        try:
            img_bytes = _deapi_generate(
                prompt,
                api_key=deapi_key,
                width=width,
                height=height,
                model=model,
            )
            out_path.write_bytes(img_bytes)
            return "ok", "deapi"
        except Exception as e:
            return "fail", str(e)

    # Pollinations fallback — free, no key needed
    try:
        img_bytes = _pollinations_generate(prompt, width=width, height=height)
        out_path.write_bytes(img_bytes)
        return "ok", "pollinations"
    except Exception as e:
        return "fail", str(e)
