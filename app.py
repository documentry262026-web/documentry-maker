import os, re, json, tempfile, subprocess, traceback
from pathlib import Path
import streamlit as st
import requests

try:
    from groq import Groq
except Exception:
    Groq = None
try:
    from google import genai
except Exception:
    genai = None
try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None
try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None

st.set_page_config(page_title="AI Diagnostic Lab", page_icon="🧪")
st.title("🧪 AI Video Generator — Tests")

def setting(k, default=None):
    try:
        return st.secrets.get(k, os.environ.get(k, default))
    except Exception:
        return os.environ.get(k, default)

def keys(v):
    if isinstance(v, str):
        v = [v]
    return [str(x).strip() for x in (v or []) if str(x).strip() and "YOUR_" not in str(x) and "PASTE_NEW_" not in str(x)]

GROQ_KEYS = keys(setting("GROQ_KEYS", []))
GEMINI_KEYS = keys(setting("GEMINI_KEYS", []))
PIXABAY_API_KEY = str(setting("PIXABAY_API_KEY", "") or "").strip()
GROQ_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
GEMINI_MODELS = ["gemini-3.8-flash", "gemini-3.7-flash"]
PIXABAY_API = "https://pixabay.com/api/videos/"

def error(title, e):
    st.error(title)
    st.code(f"{type(e).__name__}: {str(e)[:1500]}")
    st.code(traceback.format_exc()[-3000:])

def ffmpeg():
    if not imageio_ffmpeg:
        raise RuntimeError("imageio-ffmpeg import failed")
    return imageio_ffmpeg.get_ffmpeg_exe()

def run(cmd, timeout=120):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

# 1. CONFIG / LIBRARIES
st.header("1. Configuration")
st.write("Groq keys:", len(GROQ_KEYS), "| Gemini keys:", len(GEMINI_KEYS),
         "| Pixabay key:", bool(PIXABAY_API_KEY))
st.write("Groq SDK:", bool(Groq), "| Gemini SDK:", bool(genai),
         "| Whisper:", bool(WhisperModel), "| FFmpeg package:", bool(imageio_ffmpeg))
with st.expander("Secrets format"):
    st.code('''GROQ_KEYS = ["NEW_GROQ_KEY_1", "NEW_GROQ_KEY_2"]
GEMINI_KEYS = ["NEW_GEMINI_KEY_1", "NEW_GEMINI_KEY_2"]
PIXABAY_API_KEY = "NEW_PIXABAY_KEY"''', language="toml")

# 2. GROQ
st.header("2. Groq API")
if st.button("Test Groq"):
    results = []
    if not Groq or not GROQ_KEYS:
        st.error("Groq SDK missing or no keys configured")
    else:
        for i, key in enumerate(GROQ_KEYS, 1):
            for model in GROQ_MODELS:
                try:
                    r = Groq(api_key=key).chat.completions.create(
                        model=model,
                        messages=[{"role":"user","content":"Reply OK"}],
                        max_tokens=10)
                    results.append(f"Key {i} / {model}: PASS - {r.choices[0].message.content}")
                except Exception as e:
                    results.append(f"Key {i} / {model}: FAIL - {type(e).__name__}: {str(e)[:250]}")
        st.code("\n".join(results))

# 3. GEMINI
st.header("3. Gemini API")
if st.button("Test Gemini"):
    results = []
    if not genai or not GEMINI_KEYS:
        st.error("Gemini SDK missing or no keys configured")
    else:
        for i, key in enumerate(GEMINI_KEYS, 1):
            for model in GEMINI_MODELS:
                try:
                    r = genai.Client(api_key=key).models.generate_content(
                        model=model, contents="Reply OK")
                    results.append(f"Key {i} / {model}: PASS - {(r.text or '')[:150]}")
                except Exception as e:
                    results.append(f"Key {i} / {model}: FAIL - {type(e).__name__}: {str(e)[:250]}")
        st.code("\n".join(results))

# 4. WHISPER
st.header("4. Whisper transcription")
audio = st.file_uploader("Upload audio/video", type=["mp3","wav","m4a","aac","mp4","mov","avi","webm","flac"])
size = st.selectbox("Whisper model", ["tiny","base","small"])
lang = st.selectbox("Language", ["Auto","Hindi","English"])
if st.button("Test Whisper", disabled=audio is None):
    try:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "input" + (Path(audio.name).suffix or ".mp3"))
            with open(p, "wb") as f:
                f.write(audio.getbuffer())
            model = WhisperModel(size, device="cpu", compute_type="int8")
            opts = {"word_timestamps": True}
            if lang != "Auto":
                opts["language"] = {"Hindi":"hi","English":"en"}[lang]
            segments, info = model.transcribe(p, **opts)
            data = [{"start":round(s.start,2),"end":round(s.end,2),"text":s.text.strip()} for s in segments]
        st.session_state["segments"] = data
        st.success(f"Whisper PASS | Language: {info.language} | Duration: {info.duration:.1f}s")
        st.json(data)
        st.download_button("Download transcript", "\n".join(x["text"] for x in data), "transcript.txt")
    except Exception as e:
        error("Whisper failed", e)

# 5. PIXABAY
st.header("5. Pixabay search/download")
query = st.text_input("Visual search phrase", "city traffic")
orientation = st.radio("Orientation", ["horizontal","vertical"], horizontal=True)

def pixabay(q):
    if not PIXABAY_API_KEY:
        raise RuntimeError("PIXABAY_API_KEY missing in Secrets")
    r = requests.get(PIXABAY_API, params={"key":PIXABAY_API_KEY,"q":q,
        "video_type":"film","orientation":orientation,"safesearch":"true","per_page":5}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Pixabay HTTP {r.status_code}: {r.text[:400]}")
    hits = r.json().get("hits", [])
    if not hits:
        raise RuntimeError("No videos found")
    return hits

if st.button("Search Pixabay"):
    try:
        hits = pixabay(query)
        st.session_state["hits"] = hits
        st.success(f"Pixabay PASS: {len(hits)} results")
        for h in hits:
            st.write(f"ID {h.get('id')} | {h.get('duration')}s | {h.get('tags')}")
    except Exception as e:
        error("Pixabay search failed", e)

if st.button("Download first Pixabay clip"):
    try:
        hits = st.session_state.get("hits") or pixabay(query)
        v = hits[0].get("videos", {})
        v = v.get("medium") or v.get("small") or v.get("large")
        if not v or not v.get("url"):
            raise RuntimeError("No video URL in search result")
        r = requests.get(v["url"], timeout=90)
        r.raise_for_status()
        if len(r.content) < 1000:
            raise RuntimeError("Video download too small")
        st.success(f"Download PASS: {len(r.content):,} bytes")
        st.download_button("Download sample video", r.content, "pixabay_test.mp4", "video/mp4")
    except Exception as e:
        error("Pixabay download failed", e)

# 6. FFMPEG + CAPTIONS
st.header("6. FFmpeg and captions")
if st.button("Test FFmpeg"):
    try:
        exe = ffmpeg()
        r = run([exe, "-version"], 20)
        if r.returncode:
            raise RuntimeError(r.stderr[-1500:])
        st.success("FFmpeg PASS")
        st.code(r.stdout[:800])
        f = run([exe, "-filters"], 30)
        st.write("Subtitle filter available:", bool(re.search(r"\b(subtitles|ass)\b", f.stdout)))
    except Exception as e:
        error("FFmpeg failed", e)

def make_srt(data):
    def stamp(t):
        ms = int(t*1000)
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        s, ms = divmod(ms, 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    out = []
    for i, x in enumerate(data, 1):
        if x["text"].strip():
            out += [str(i), f'{stamp(x["start"])} --> {stamp(x["end"])}', x["text"], ""]
    return "\n".join(out)

if st.button("Test subtitle creation", disabled="segments" not in st.session_state):
    try:
        srt = make_srt(st.session_state["segments"])
        if not srt.strip():
            raise RuntimeError("No captions generated")
        st.success("Caption file PASS")
        st.download_button("Download captions.srt", srt, "captions.srt", "application/x-subrip")
        st.text_area("Caption preview", srt, height=180)
    except Exception as e:
        error("Caption creation failed", e)

# 7. END-TO-END VIDEO
st.header("7. Complete video test")
ratio = st.radio("Output", ["16:9","9:16"], horizontal=True)

if st.button("Generate short test video", disabled=audio is None):
    try:
        exe = ffmpeg()
        with tempfile.TemporaryDirectory() as d:
            audio_path = os.path.join(d, "audio" + (Path(audio.name).suffix or ".mp3"))
            with open(audio_path, "wb") as f:
                f.write(audio.getbuffer())

            st.info("Step 1: Whisper transcription")
            model = WhisperModel(size, device="cpu", compute_type="int8")
            opts = {"word_timestamps": False}
            if lang != "Auto":
                opts["language"] = {"Hindi":"hi","English":"en"}[lang]
            segments, info = model.transcribe(audio_path, **opts)
            data = [{"start":float(s.start),"end":float(s.end),"text":s.text.strip()} for s in segments if s.text.strip()]
            if not data:
                raise RuntimeError("Whisper returned no speech")
            duration = max(x["end"] for x in data)

            st.info("Step 2: Pixabay visual")
            hits = pixabay(query)
            v = hits[0].get("videos", {})
            v = v.get("medium") or v.get("small") or v.get("large")
            if not v or not v.get("url"):
                raise RuntimeError("Pixabay has no usable video")
            video_path = os.path.join(d, "visual.mp4")
            r = requests.get(v["url"], timeout=90)
            r.raise_for_status()
            Path(video_path).write_bytes(r.content)

            st.info("Step 3: Captions")
            srt_path = os.path.join(d, "captions.srt")
            Path(srt_path).write_text(make_srt(data), encoding="utf-8")

            st.info("Step 4: FFmpeg rendering")
            w, h = (1280,720) if ratio == "16:9" else (720,1280)
            final_path = os.path.join(d, "final.mp4")
            sub = srt_path.replace("\\","/").replace(":","\\:").replace("'","\\'")
            vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1,subtitles='{sub}'"
            cmd = [exe,"-y","-stream_loop","-1","-i",video_path,"-i",audio_path,
                   "-t",str(duration),"-vf",vf,"-c:v","libx264","-preset","ultrafast",
                   "-crf","27","-threads","2","-c:a","aac","-shortest",
                   "-movflags","+faststart",final_path]
            result = run(cmd, 900)
            if result.returncode:
                raise RuntimeError("FFmpeg render failed:\n"+result.stderr[-4000:])
            if not os.path.exists(final_path) or os.path.getsize(final_path) < 1000:
                raise RuntimeError("Output video missing/empty")
            final_bytes = Path(final_path).read_bytes()
        st.success("End-to-end PASS")
        st.video(final_bytes)
        st.download_button("Download test video", final_bytes, "test_video.mp4", "video/mp4")
    except Exception as e:
        error("End-to-end video test failed", e)

st.divider()
st.caption("Testing only. Free APIs still have quotas and model access restrictions. Never share API keys in screenshots.")
