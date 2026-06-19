"""Edge TTS — free, no API key. Returns audio + sentence-level timestamps."""
from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from typing import TypedDict

VOICE = "en-US-ChristopherNeural"


class SentenceTiming(TypedDict):
    text: str
    offset_ms: int
    duration_ms: int


class WordTiming(TypedDict):
    text: str
    offset_ms: int
    duration_ms: int


def _ffprobe_duration(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    ).strip()
    return float(out)


async def _synthesize_with_timing(
    text: str, out_path: Path, voice: str, rate: str = "-15%",
) -> tuple[list[SentenceTiming], list[WordTiming]]:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice, rate=rate)
    sentences: list[SentenceTiming] = []
    words: list[WordTiming] = []

    with open(out_path, "wb") as audio_file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_file.write(chunk["data"])
            elif chunk["type"] == "SentenceBoundary":
                sentences.append(
                    SentenceTiming(
                        text=chunk["text"],
                        offset_ms=int(chunk["offset"]) // 10_000,
                        duration_ms=int(chunk["duration"]) // 10_000,
                    )
                )
            elif chunk["type"] == "WordBoundary":
                words.append(
                    WordTiming(
                        text=chunk["text"],
                        offset_ms=int(chunk["offset"]) // 10_000,
                        duration_ms=int(chunk["duration"]) // 10_000,
                    )
                )

    return sentences, words


def synthesize_full(
    text: str, out_path: Path, voice: str | None = None,
) -> tuple[float, list[SentenceTiming], list[WordTiming]]:
    """TTS the full narration. Returns (duration_seconds, sentence_timings, word_timings)."""
    voice = voice or os.environ.get("EDGE_TTS_VOICE", VOICE)
    rate = os.environ.get("EDGE_TTS_RATE", "-15%")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sentences, words = asyncio.run(_synthesize_with_timing(text, out_path, voice, rate=rate))
    dur = _ffprobe_duration(out_path)
    return dur, sentences, words
