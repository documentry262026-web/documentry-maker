import os
import re
import math
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

def get_exact_duration(path):
    exe = ffmpeg()
    result = subprocess.run([exe, "-i", path], capture_output=True, text=True)
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if match:
        h, m, s = match.groups()
        return float(h) * 3600 + float(m) * 60 + float(s)
    return 10.0

def transcribe_and_chunk(path, size, total_audio_duration, target_length=10.0):
    model = WhisperModel(size, device="cpu", compute_type="int8")
    
    # 🌟 URDU FIX RESTORED: यह लाइन उर्दू को रोकेगी और सिर्फ हिंदी लिखवाएगी
    segments, _ = model.transcribe(
        path, 
        language="hi", 
        task="transcribe", 
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
        initial_prompt="यह देवनागरी लिपि में लिखी गई शुद्ध हिंदी है।" # <- इसे मैंने गलती से हटा दिया था!
    )
    
    seg_list = list(segments)
    chunks = []
    num_chunks = math.ceil(total_audio_duration / target_length)
    
    last_valid_text = ""
    
    for i in range(num_chunks):
        start_time = i * target_length
        end_time = min((i + 1) * target_length, total_audio_duration)
        dur = end_time - start_time
        
        if dur <= 0:
            continue
            
        chunk_text = ""
        for seg in seg_list:
            if seg.start < end_time and seg.end > start_time:
                chunk_text += " " + seg.text.strip()
                
        chunk_text = chunk_text.strip()
        
        if not chunk_text:
            if last_valid_text:
                chunk_text = last_valid_text
            else:
                full_text = " ".join([s.text.strip() for s in seg_list]).strip()
                chunk_text = full_text
        else:
            last_valid_text = chunk_text
            
        chunks.append({
            "text": chunk_text,
            "start": start_time,
            "end": end_time,
            "duration": dur
        })
                
    return chunks

# 🚀 100% PURE, NO BACKUP KEYWORD GENERATOR
def get_groq_keywords(transcript):
    if not GK:
        raise RuntimeError("GROQ_KEYS missing from Streamlit Secrets")

    if not transcript:
        raise RuntimeError("Transcript is totally empty. No words detected by Whisper.")

    client = Groq(api_key=GK[0])
    model_name = "openai/gpt-oss-20b" 

    prompt = (
        "Give me exactly 3 English visual search keywords for this text. "
        "Only output the keywords, no sentences.\n\n"
        f"Text: {transcript}"
    )

    r = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You output only English keywords. Never return an empty string."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200, 
        temperature=0.2
    )
    
    raw_output = r.choices[0].message.content or ""
    raw_output = raw_output.strip()
    
    clean_text = re.sub(r'[^a-zA-Z\s]', ' ', raw_output)
    words = [w for w in clean_text.split() if len(w) > 1]
    final_keywords = " ".join(words[-4:]) 
    
    # ❌ NO BACKUP: सीधा एरर
    if not final_keywords:
        raise RuntimeError(f"Model '{model_name}' gave empty or invalid keywords. Raw output was: '{raw_output}'")
        
    return final_keywords

def fetch_pixabay_media(query, orientation):
    if not PK:
        raise RuntimeError("PIXABAY_API_KEY missing from Secrets")
    
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

    raise RuntimeError(f"Exact Error: No video/image found on Pixabay for the query: '{query}'")

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

st.header("3. Generate Final Video")
if st.button("Generate Cinematic Video", disabled=audio_file is None):
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            exe = ffmpeg()
            ext = Path(audio_file.name).suffix or ".mp3"
            voice_path = os.path.join(tmp_dir, f"voice{ext}")
            Path(voice_path).write_bytes(audio_file.getvalue())

            total_audio_duration = get_exact_duration(voice_path)
            st.write(f"**Total Audio Duration:** {total_audio_duration:.2f} seconds")

            st.info("🎙️ Transcribing and analyzing scenes...")
            chunks = transcribe_and_chunk(voice_path, whisper_size, total_audio_duration, target_length=10.0)
            
            st.success(f"✅ Voiceover split strictly into {len(chunks)} cinematic scenes!")

            scene_files = []
            
            for i, chunk in enumerate(chunks):
                st.write(f"---")
                st.write(f"**Scene {i+1} ({chunk['duration']:.1f}s):** {chunk['text']}")
                
                query = get_groq_keywords(chunk['text'])
                st.write(f"**AI Search Query:** `{query}`")
                
                media_info = fetch_pixabay_media(query, "vertical" if width < height else "horizontal")
                
                media_file = os.path.join(tmp_dir, f"media_{i}" + (".mp4" if media_info["type"] == "video" else ".jpg"))
                r_media = requests.get(media_info["url"], timeout=60)
                r_media.raise_for_status()
                Path(media_file).write_bytes(r_media.content)

                out_scene = os.path.join(tmp_dir, f"scene_{i}.mp4")
                dur = chunk['duration']
                
                fade_in = "fade=t=in:st=0:d=0.5"
                fade_out = f"fade=t=out:st={dur-0.5}:d=0.5"

                if media_info["type"] == "video":
                    vf = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1,{fade_in},{fade_out}"
                    cmd = [
                        exe, "-y", "-stream_loop", "-1", "-i", media_file,
                        "-t", str(dur), "-vf", vf,
                        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "27",
                        "-pix_fmt", "yuv420p", "-threads", "2", out_scene
                    ]
                else:
                    vf = f"scale=8000:-1,zoompan=z='min(zoom+0.001,1.25)':d={int(dur*30)}:s={width}x{height}:fps=30,{fade_in},{fade_out}"
                    cmd = [
                        exe, "-y", "-loop", "1", "-i", media_file,
                        "-t", str(dur), "-vf", vf,
                        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "27",
                        "-pix_fmt", "yuv420p", "-threads", "2", out_scene
                    ]

                result = run(cmd, timeout=300)
                if result.returncode != 0:
                    raise RuntimeError(f"Scene {i+1} render failed:\n{result.stderr[-1000:]}")
                
                scene_files.append(f"file 'scene_{i}.mp4'")

            st.info("🎬 Mixing all scenes with Voiceover...")
            
            list_path = os.path.join(tmp_dir, "list.txt")
            with open(list_path, "w") as f:
                f.write("\n".join(scene_files))

            final_path = os.path.join(tmp_dir, "final_output.mp4")
            
            concat_cmd = [
                exe, "-y", "-f", "concat", "-safe", "0", "-i", list_path,
                "-i", voice_path,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "128k",
                "-shortest", "-movflags", "+faststart", final_path
            ]
            
            concat_result = run(concat_cmd, timeout=300)
            if concat_result.returncode != 0:
                raise RuntimeError(f"Final mix failed:\n{concat_result.stderr[-1000:]}")

            final_bytes = Path(final_path).read_bytes()
            st.success("✅ Cinematic Multi-Clip Video Successfully Generated!")
            st.video(final_bytes)
            st.download_button(
                "📥 Download Final Video",
                final_bytes,
                file_name="ai_cinematic_short.mp4",
                mime="video/mp4"
            )

        except Exception as e:
            show_error("Process Stopped / Error", e)
