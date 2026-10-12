import os
import re
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

def run(cmd, timeout=400):
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return res

# फाइल की असली लंबाई (Exact Duration) निकालने का फंक्शन
def get_exact_duration(path):
    exe = ffmpeg()
    result = subprocess.run([exe, "-i", path], capture_output=True, text=True)
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", result.stderr)
    if match:
        h, m, s = match.groups()
        return float(h) * 3600 + float(m) * 60 + float(s)
    return 10.0

def transcribe_audio(path, size):
    model = WhisperModel(size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(path, task="transcribe", beam_size=5)
    
    seg_list = list(segments)
    full_text = " ".join([seg.text.strip() for seg in seg_list])
    return full_text

# एकदम स्ट्रिक्ट कीवर्ड फंक्शन (कोई फालतू डिफ़ॉल्ट बैकग्राउंड नहीं)
def get_groq_keywords(transcript):
    if not GK:
        raise RuntimeError("GROQ_KEYS missing from Streamlit Secrets")
    
    if not transcript or not transcript.strip():
        raise RuntimeError("Whisper Error: Transcript is empty. No voiceover detected to generate keywords.")

    client = Groq(api_key=GK[0])
    prompt = (
        "Analyze the following transcript and extract 2 or 3 highly specific visual search keywords in English "
        "for fetching a stock video that exactly matches the context. "
        "Only provide the keywords separated by spaces, no commas, no extra text, no quotes.\n\n"
        f"Transcript: {transcript}"
    )
    
    r = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=30,
        temperature=0.2
    )
    
    keywords = r.choices[0].message.content.strip().replace('"', '').replace("'", "")
    
    # अगर ग्रॉक ने कीवर्ड नहीं दिया, तो डिफ़ॉल्ट वीडियो लगाने के बजाय सीधा एरर दो!
    if not keywords or keywords.lower() in ["none", "null", "empty"]:
        raise RuntimeError("Groq AI Error: Failed to extract relevant keywords for this voiceover. Cannot proceed without accurate context.")
        
    return keywords

def fetch_pixabay_media(query, orientation):
    if not PK:
        raise RuntimeError("PIXABAY_API_KEY missing from Secrets")
    
    # 1. पहले वीडियो ढूँढेंगे
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

    # 2. अगर वीडियो नहीं मिला, तो फोटो ढूँढेंगे
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

    # अगर उस कीवर्ड का कुछ नहीं मिला, तो सीधा एरर!
    raise RuntimeError(f"Pixabay Error: No video or image found on Pixabay for the strict keywords: '{query}'.")

# ================= UI Components ================= #
st.header("1. Upload Voiceover")
audio_file = st.file_uploader(
    "Upload audio or video file (MP3, WAV, MP4, AAC, etc.)",
    type=["mp3", "wav", "m4a", "aac", "mp4", "mov", "avi", "webm", "flac"]
)
whisper_size = st.selectbox("Whisper Model Size", ["tiny", "base", "small"], index=1)

st.header("2. Video Settings")
ratio = st.radio("Output Aspect Ratio", ["9:16 (Shorts/Reels)", "16:9 (Landscape)"], horizontal=True)
width, height = (720, 1280) if "9:16" in ratio else (1280, 720)
custom_keyword = st.text_input("Custom Visual Keywords (Optional - Override AI if you want)", "")

st.header("3. Generate Final Video")
if st.button("Generate Video", disabled=audio_file is None):
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            ext = Path(audio_file.name).suffix or ".mp3"
            voice_path = os.path.join(tmp_dir, f"voice{ext}")
            Path(voice_path).write_bytes(audio_file.getvalue())

            duration = get_exact_duration(voice_path)
            st.write(f"**Exact Audio Duration:** {duration:.2f} seconds")

            st.info("🎙️ Transcribing voiceover via Whisper...")
            transcript = transcribe_audio(voice_path, whisper_size)

            st.info("🤖 Extracting strict visual keywords via Groq (openai/gpt-oss-20b)...")
            search_query = custom_keyword.strip() if custom_keyword.strip() else get_groq_keywords(transcript)
            st.write(f"**Final Search Query:** `{search_query}`")

            st.info("🔍 Fetching accurate media from Pixabay...")
            media_info = fetch_pixabay_media(search_query, "vertical" if width < height else "horizontal")
            
            media_file_path = os.path.join(tmp_dir, "media_file" + (".mp4" if media_info["type"] == "video" else ".jpg"))
            r_media = requests.get(media_info["url"], timeout=60)
            r_media.raise_for_status()
            Path(media_file_path).write_bytes(r_media.content)

            st.info("🎬 Rendering final video with motion graphics, fades, and audio sync...")
            final_path = os.path.join(tmp_dir, "final_output.mp4")
            exe = ffmpeg()

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

            result = run(cmd, timeout=600)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg render failed:\n{result.stderr[-2000:]}")

            if not os.path.exists(final_path) or os.path.getsize(final_path) < 1000:
                raise RuntimeError("Generated video file is empty or missing.")

            final_bytes = Path(final_path).read_bytes()
            st.success("✅ Video successfully generated with accurate visuals!")
            st.video(final_bytes)
            st.download_button(
                "📥 Download Final Video",
                final_bytes,
                file_name="ai_voiceover_final.mp4",
                mime="video/mp4"
            )

        except Exception as e:
            show_error("Process Stopped / Error", e)
