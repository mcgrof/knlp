"""Render static evidence slides, stock narration, captions and transcript.

Requires ffmpeg, ffprobe, Pillow and edge-tts. Run with a disposable --work
directory. The narration is public text sent to the stock TTS service.
"""

import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import textwrap

import edge_tts
from PIL import Image, ImageDraw, ImageFont


def run(*args):
    subprocess.run(list(map(str, args)), check=True, stdout=subprocess.DEVNULL)


SPEED = 1.2


def seconds(stamp):
    h, m, s = stamp.replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def stamp(value):
    ms = round(value * 1000)
    return f"{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02}.{ms%1000:03}"


async def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    args = p.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    data = json.loads(args.data.read_text())
    font = lambda size: ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size
    )
    offset, captions, clips = 0.0, ["WEBVTT\n"], []
    transcript = [
        "# " + data["title"],
        "Published " + data["date"] + ". Stock synthetic narration (Aria).",
        "",
    ]
    for i, slide in enumerate(data["slides"]):
        base = args.work / str(i)
        audio, frame, clip = (
            base.with_suffix(".mp3"),
            base.with_suffix(".png"),
            base.with_suffix(".mp4"),
        )
        subtitles = edge_tts.SubMaker()
        with audio.open("wb") as stream:
            async for part in edge_tts.Communicate(
                slide["narration"], "en-US-AriaNeural", rate="+0%"
            ).stream():
                if part["type"] == "audio":
                    stream.write(part["data"])
                elif part["type"] in ("WordBoundary", "SentenceBoundary"):
                    subtitles.feed(part)
        raw = subtitles.get_srt()
        for block in raw.strip().split("\n\n"):
            lines = block.splitlines()
            if len(lines) < 3:
                continue
            start, end = lines[1].split(" --> ")
            captions.append(
                f"{stamp((offset + seconds(start)) / SPEED)} --> {stamp((offset + seconds(end)) / SPEED)}\n"
                + "\n".join(lines[2:])
                + "\n"
            )
        duration = float(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    str(audio),
                ],
                text=True,
            )
        )
        canvas = Image.new("RGB", (1280, 720), "#0b1714")
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, 1280, 10), fill="#61d6a4")
        draw.text((68, 48), "KNLP / OCTOBER 2026", font=font(24), fill="#61d6a4")
        y = 116
        for line in textwrap.wrap(slide["title"], width=39):
            draw.text((68, y), line, font=font(47), fill="#f0f7f3")
            y += 61
        y += 38
        for point in slide["points"]:
            draw.ellipse((68, y + 13, 78, y + 23), fill="#61d6a4")
            for line in textwrap.wrap(point, width=58):
                draw.text((99, y), line, font=font(29), fill="#dae8e1")
                y += 41
            y += 24
        draw.text(
            (68, 657),
            "knlp.io/decode/ · workload-specific serving evidence",
            font=font(21),
            fill="#98b7a7",
        )
        draw.text((1185, 657), f"{i+1}/7", font=font(21), fill="#98b7a7")
        canvas.save(frame)
        run(
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-loop",
            "1",
            "-i",
            frame,
            "-i",
            audio,
            "-t",
            duration,
            "-r",
            "15",
            "-c:v",
            "libx264",
            "-tune",
            "stillimage",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            clip,
        )
        clips.append(clip)
        transcript.extend(["## " + slide["title"], slide["narration"], ""])
        offset += duration
    concat = args.work / "concat.txt"
    concat.write_text("\n".join("file '" + str(c) + "'" for c in clips) + "\n")
    run(
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat,
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        args.work / "normal-speed.mp4",
    )
    run(
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        args.work / "normal-speed.mp4",
        "-vf",
        f"setpts=PTS/{SPEED}",
        "-af",
        f"atempo={SPEED}",
        "-r",
        "15",
        "-c:v",
        "libx264",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-movflags",
        "+faststart",
        here / "update.mp4",
    )
    (here / "captions.vtt").write_text("\n".join(captions))
    (here / "transcript.md").write_text("\n".join(transcript))
    Image.open(args.work / "0.png").save(here / "poster.png")
    print(
        json.dumps(
            {
                "duration_seconds": offset / SPEED,
                "playback_rate": SPEED,
                "slides": len(clips),
                "voice": "en-US-AriaNeural",
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
