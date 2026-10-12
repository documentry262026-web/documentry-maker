import os
import re
import json
import tempfile
import subprocess
from pathlib import Path
import streamlit as st
import requests
from groq import Groq
from faster_whisper import WhisperModel
import imageio_ffmpeg

st.set_page_config(page_title="AI Voiceover Video Maker", page_icon="🎬", layout="centered")
st.title("🎬 AI Voiceover to Video Generator")

def setting(k, d=None):
    try:
        return st.secrets.get(k, os.environ.get(k, d))
    except Exception:
        return os.environ.get(k, d)

def keylist(v):
    if isinstance(v, str): 
        v = [v]
    return [str(x).strip() for x in (v or []) if str(x).strip()]

GK = keylist(setting("GROQ_KEYS", []))
PK = str(setting("PIXABAY_API_KEY", "") or "").strip()
PIXABAY_VIDEO_API = "https://pixabay.com/api/videos/"
PIXABAY_IMAGE_API = "https://pixabay.com/api/"

def show_error(title, e):
    st.error(title)
    st.code(f"{type(e).__name__}: {str(e)[:1500]}")

def ffmpeg():
    return imageio_ffmpeg.get_ffmpeg_exe()

def run(cmd, timeout=300):
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return res

def transcribe_audio(path, size):
    model = WhisperModel(size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(path, task="transcribe", beam_size=5)
    full_text = " ".join([seg.text.strip() for seg in segments])
    
    # Calculate exact total duration from segments
    duration = 0.0
    for seg in segments:
        if seg.end > duration:
            duration = seg.end
    if duration <= 0:
        duration = 10.0 # Fallback
    return full_text, duration

def get_groq_keywords(transcript):
    if not GK:
        raise RuntimeError("GROQ_KEYS missing from Streamlit Secrets")
    client = Groq(api_key=GK[0])
    prompt = (
        "Extract 3 to 5 highly relevant visual search keywords in English for stock video search "
        "based on the following transcript. Keep it concise, separated by commas, with no extra text or quotes.\n\n"
        f"Transcript: {transcript}"
    )
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60,
        temperature=0.3
    )
    keywords = r.choices[0].message.content.strip().replace('"', '').replace("'", "")
    return keywords if keywords else "cinematic nature background"

def fetch_pixabay_media(query, orientation):
    if not PK:
        raise RuntimeError("PIXABAY_API_KEY missing from Secrets")
    
    # Try fetching Video first
    try:
        r = requests.get(PIXABAY_VIDEO_API, params={
            "key": PK, "q": query, "video_type": "film",
            "orientation": orientation, "safesearch": "true", "per_page": 3
        }, timeout=20)
        if r.status_code == 200:
            hits = r.json().get("hits", [])
            if hits:
                for quality in ["medium", "small", "large"]:
                    item = hits[0].get("videos", {}).get(quality, {})
                    if item.get("url"):
                        return {"type": "video", "url": item["url"]}
    except Exception:
        pass

    # Fallback to Image if video is not available
    try:
        r = requests.get(PIXABAY_IMAGE_API, params={
            "key": PK, "q": query, "image_type": "photo",
            "orientation": orientation, "safesearch": "true", "per_page": 3
        }, timeout=20)
        if r.status_code == 200:
            hits = r.json().get("hits", [])
            if hits:
                image_url = hits[0].get("largeImageURL") or hits[0].get("webformatURL")
                if image_url:
                    return {"type": "image", "url": image_url}
    except Exception:
        pass

    raise RuntimeError("No media found on Pixabay for the given query.")

# UI Components
st.header("1. Upload Voiceover")
audio_file = st.file_uploader(
    "Upload audio or video file (MP3, WAV, MP4, AAC, etc.)",
    type=["mp3", "wav", "m4a", "aac", "mp4", "mov", "avi", "webm", "flac"]
)
whisper_size = st.selectbox("Whisper Model Size", ["tiny", "base", "small"], index=1)

st.header("2. Video Settings")
ratio = st.radio("Output Aspect Ratio", ["9:16 (Shorts/Reels)", "16:9 (Landscape)"], horizontal=True)
width, height = (720, 1280) if "9:16" in ratio else (1280, 720)
custom_keyword = st.text_input("Custom Visual Keywords (Optional - leave blank to auto-detect via AI)", "")

st.header("3. Generate Final Video")
if st.button("Generate Video", disabled=audio_file is None):
    # Use a secure temporary directory and ensure automatic cleanup to prevent storage full errors
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            # Save uploaded file
            ext = Path(audio_file.name).suffix or ".mp3"
            voice_path = os.path.join(tmp_dir, f"voice{ext}")
            Path(voice_path).write_bytes(audio_file.getvalue())

            st.info("🎙️ Transcribing voiceover via Whisper...")
            transcript, duration = transcribe_audio(voice_path, whisper_size)
            st.write(f"**Detected Duration:** {duration:.2f} seconds")

            st.info("🤖 Extracting smart visual keywords via Groq...")
            search_query = custom_keyword.strip() if custom_keyword.strip() else get_groq_keywords(transcript)
            st.write(f"**Search Query:** `{search_query}`")

            st.info("🔍 Fetching media from Pixabay...")
            media_info = fetch_pixabay_media(search_query, "vertical" if width < height else "horizontal")
            
            media_file_path = os.path.join(tmp_dir, "media_file" + (".mp4" if media_info["type"] == "video" else ".jpg"))
            r_media = requests.get(media_info["url"], timeout=60)
            r_media.raise_for_status()
            Path(media_file_path).write_bytes(r_media.content)

            st.info("🎬 Rendering final video with motion graphics, fades, and audio sync...")
            final_path = os.path.join(tmp_dir, "final_output.mp4")
            exe = ffmpeg()

            # Build efficient FFmpeg filter graph for Zoom-in/Zoom-out & Fades
            # Safe zoompan/scale animation with audio/video crossfade
            fade_duration = min(1.0, duration / 4)
            out_fade_start = max(0, duration - fade_duration)

            if media_info["type"] == "video":
                vf = (
                    f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                    f"crop={width}:{height},setsar=1,"
                    f"fade=t=in:st=0:d={fade_duration},fade=t=out:st={out_fade_start}:d={fade_duration}"
                )
                cmd = [
                    exe, "-y", "-stream_loop", "-1", "-i", media_file_path,
                    "-i", voice_path,
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-t", str(duration),
                    "-vf", vf,
                    "-af", f"afade=in:st=0:d={fade_duration},afade=out:st={out_fade_start}:d={fade_duration}",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                    "-threads", "2", "-c:a", "aac", "-b:a", "128k",
                    "-shortest", "-movflags", "+faststart", final_path
                ]
            else:
                # For static image: apply smooth zoompan (ken burns motion effect) + fade
                zoom_filter = (
                    f"scale=8000:-1,"
                    f"zoompan=z='min(zoom+0.001,1.25)':d={int(duration*30)}:s={width}x{height}:fps=30,"
                    f"fade=t=in:st=0:d={fade_duration},fade=t=out:st={out_fade_start}:d={fade_duration}"
                )
                cmd = [
                    exe, "-y", "-loop", "1", "-i", media_file_path,
                    "-i", voice_path,
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-t", str(duration),
                    "-vf", zoom_filter,
                    "-af", f"afade=in:st=0:d={fade_duration},afade=out:st={out_fade_start}:d={fade_duration}",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                    "-threads", "2", "-c:a", "aac", "-b:a", "128k",
                    "-shortest", "-movflags", "+faststart", final_path
                ]

            result = run(cmd, timeout=400)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg render failed:\n{result.stderr[-2000:]}")

            if not os.path.exists(final_path) or os.path.getsize(final_path) < 1000:
                raise RuntimeError("Generated video file is empty or missing.")

            final_bytes = Path(final_path).read_bytes()
            st.success("✅ Video successfully generated!")
            st.video(final_bytes)
            st.download_button(
                "📥 Download Final Video",
                final_bytes,
                file_name="ai_voiceover_short.mp4",
                mime="video/mp4"
            )

        except Exception as e:
            show_error("Video Generation Failed", e)

st.caption("💡 Tip: Make sure your `GROQ_KEYS` and `PIXABAY_API_KEY` are added in your Streamlit Cloud Secrets.")
