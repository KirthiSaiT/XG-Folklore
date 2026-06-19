"""FFmpeg: images + audio + captions → vertical Short MP4 (9:16).

Effects:
  - Ken Burns zoom (alternating zoom-in / zoom-out per scene)
  - Fadeblack crossfade between scenes (horror vibe)
  - Creepster font captions at the bottom
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FONTS_DIR = REPO_ROOT / "assets" / "fonts"
DEFAULT_FONT_FILE = "CreepsterCaps.ttf"
DEFAULT_FONT_NAME = "Creepster"

FPS = 30
FADE_DUR = 0.5
ZOOM_AMOUNT = 0.15


def render_vertical_short(
    image_paths: list[Path],
    total_duration: float,
    audio_path: Path,
    srt_path: Path,
    out_video: Path,
    *,
    width: int = 1080,
    height: int = 1920,
    font_file: str = DEFAULT_FONT_FILE,
    font_name: str = DEFAULT_FONT_NAME,
) -> None:
    if not image_paths:
        raise ValueError("No images")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found; install it (brew install ffmpeg)")

    out_video = Path(out_video)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_video.parent / "_tmp_render"
    tmp.mkdir(parents=True, exist_ok=True)

    n = len(image_paths)
    clip_dur = (total_duration + (n - 1) * FADE_DUR) / n if n > 1 else total_duration
    frames_per_clip = max(int(clip_dur * FPS), 2)

    # ── 1. Pre-scale images ──────────────────────────────────────────
    for i, src in enumerate(image_paths):
        dst = tmp / f"img_{i + 1:02d}.png"
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
                "-i", str(src),
                "-vf", (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black"),
                "-update", "1", "-frames:v", "1",
                str(dst),
            ],
            check=True,
        )

    # ── 2. Generate zoompan clips ────────────────────────────────────
    for i in range(n):
        src_img = tmp / f"img_{i + 1:02d}.png"
        clip = tmp / f"clip_{i + 1:02d}.mp4"
        zoom_rate = ZOOM_AMOUNT / frames_per_clip

        if i % 2 == 0:
            zoom_expr = f"min(zoom+{zoom_rate:.8f},{1 + ZOOM_AMOUNT})"
        else:
            zoom_expr = f"if(eq(on,1),{1 + ZOOM_AMOUNT},max(zoom-{zoom_rate:.8f},1.0))"

        vf = (
            f"zoompan=z='{zoom_expr}':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={frames_per_clip}:s={width}x{height}:fps={FPS},"
            f"format=yuv420p"
        )

        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
                "-i", str(src_img),
                "-vf", vf,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "15",
                str(clip),
            ],
            check=True,
        )

    # ── 3. Prepare subtitles + font ──────────────────────────────────
    shutil.copyfile(srt_path, tmp / "captions.srt")

    font_path = FONTS_DIR / font_file
    rendered_font_name = "Arial"
    fontsdir_arg = ""
    if font_path.is_file():
        font_dir = tmp / "_fonts"
        font_dir.mkdir(exist_ok=True)
        shutil.copyfile(font_path, font_dir / font_path.name)
        rendered_font_name = font_name
        fontsdir_arg = ":fontsdir='_fonts'"
    else:
        print(f"   ⚠ Font {font_path} not found — using {rendered_font_name}")

    force_style = (
        f"FontName={rendered_font_name},"
        f"FontSize=22,"
        f"PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,"
        f"BackColour=&HAA000000,"
        f"BorderStyle=4,Outline=2,Bold=1,"
        f"Shadow=1,Alignment=2,"
        f"MarginV=40,MarginL=20,MarginR=20"
    )

    # ── 4. Generate background music ─────────────────────────────────
    music_path = tmp / "music_bg.mp3"
    music_duration = total_duration + 3
    music_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi",
        "-i", (
            f"aevalsrc="
            f"'0.12*sin(2*PI*55*t)+0.08*sin(2*PI*82.5*t)+0.06*sin(2*PI*110*t)"
            f"+0.04*sin(2*PI*55*t)*sin(2*PI*0.3*t)+0.03*sin(2*PI*165*t)"
            f"|"
            f"0.12*sin(2*PI*55*t+0.05)+0.08*sin(2*PI*82.5*t+0.05)+0.06*sin(2*PI*110*t+0.05)"
            f"+0.04*sin(2*PI*55*t+0.05)*sin(2*PI*0.3*t)+0.03*sin(2*PI*165*t+0.05)"
            f":c=stereo:s=44100'"
        ),
        "-af", f"volume=0.25,afade=t=in:ss=0:d=2,afade=t=out:st={music_duration-3:.1f}:d=3",
        "-t", str(music_duration),
        str(music_path),
    ]
    subprocess.run(music_cmd, check=True)

    # ── 5. Build xfade chain + subtitles ─────────────────────────────
    inputs: list[str] = []
    for i in range(n):
        inputs += ["-i", f"clip_{i + 1:02d}.mp4"]
    inputs += ["-i", str(audio_path.resolve())]
    inputs += ["-i", str(music_path.resolve())]
    audio_idx = n
    music_idx = n + 1

    filter_parts: list[str] = []

    cinematic_grade = (
        "eq=contrast=1.3:saturation=0.72:brightness=-0.03,"
        "colorchannelmixer=rr=0.94:rg=0:rb=0.04:gr=0:gg=0.96:gb=0.03:br=0.03:bg=0:bb=1.07,"
        "vignette=angle=PI/4,"
        "noise=alls=5:allf=t+u"
    )

    if n == 1:
        filter_parts.append(
            f"[0:v]subtitles=captions.srt{fontsdir_arg}:"
            f"force_style='{force_style}'[subbed]"
        )
    else:
        prev = "[0:v]"
        for i in range(n - 1):
            offset = (i + 1) * (clip_dur - FADE_DUR)
            next_v = f"[{i + 1}:v]"
            out = f"[x{i}]"
            filter_parts.append(
                f"{prev}{next_v}xfade=transition=fadeblack:"
                f"duration={FADE_DUR:.4f}:offset={offset:.4f}{out}"
            )
            prev = out
        filter_parts.append(
            f"{prev}subtitles=captions.srt{fontsdir_arg}:"
            f"force_style='{force_style}'[subbed]"
        )

    filter_parts.append(f"[subbed]{cinematic_grade}[final]")

    filter_parts.append(
        f"[{audio_idx}:a][{music_idx}:a]amix=inputs=2:duration=first:weights=1.8 0.55[audio_mix]"
    )

    fc = ";\n".join(filter_parts)

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        *inputs,
        "-filter_complex", fc,
        "-map", "[final]", "-map", "[audio_mix]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(out_video.resolve()),
    ]
    subprocess.run(cmd, check=True, cwd=str(tmp))

    # ── cleanup ──────────────────────────────────────────────────────
    shutil.rmtree(tmp, ignore_errors=True)
