import os
import re
import json
import copy
import tempfile
import subprocess
from pathlib import Path
import streamlit as st
import requests
from groq import Groq
from faster_whisper import WhisperModel
import imageio_ffmpeg

st.set_page_config(page_title="AI Voiceover Video", page_icon="🎬")
st.title("🎬 Hindi Voiceover to Shorts")

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
PAPI = "https://pixabay.com/api/videos/"

def show_error(title, e):
    st.error(title)
    st.code(f"{type(e).__name__}: {str(e)[:1500]}")

def ffmpeg():
    return imageio_ffmpeg.get_ffmpeg_exe()

def run(cmd, timeout=120):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

def transcribe(path, size, language):
    model = WhisperModel(size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        path, language=language, task="transcribe",
        word_timestamps=True, beam_size=5,
        initial_prompt="यह शुद्ध हिंदी भाषा है। इसे देवनागरी लिपि में ही लिखें।"
    )
    data = []
    for s in segments:
        words = []
        for w in (s.words or []):
            words.append({
                "start": float(w.start), "end": float(w.end),
                "text": w.word.strip()
            })
        data.append({
            "start": float(s.start), "end": float(s.end),
            "text": s.text.strip(), "words": words
        })
    if not data or not any(s["words"] for s in data):
        raise RuntimeError("No speech or word timestamps detected")
    return data, info

def signature(data):
    words = [w for s in data for w in s.get("words", [])]
    return json.dumps(
        [[w["start"], w["end"], w["text"]] for w in words],
        ensure_ascii=False
    )

def correct_hindi(data):
    data = copy.deepcopy(data)
    sig = signature(data)
    cache = st.session_state.setdefault("hindi_cache", {})
    
    if sig in cache:
        converted = cache[sig]
    else:
        if not GK:
            raise RuntimeError("GROQ_KEYS missing from Streamlit Secrets")
        words = [w for s in data for w in s["words"]]
        client = Groq(api_key=GK[0])
        
        prompt = (
            "You are an expert Hindi proofreader and subtitle editor. "
            "Whisper speech-to-text often misinterprets Hindi audio and outputs words in Urdu script (Perso-Arabic) "
            "or broken Latin/Roman script. Your job is to convert every single word into natural, correctly spelled "
            "Hindi in Devanagari script. Use context to fix spelling mistakes. Do not translate or summarize. "
            "Preserve the exact count, order, and meaning of the words. "
            "Return ONLY a JSON array of strings with exactly one item per input word, with no markdown code block formatting if possible, or parseable as JSON. "
            "Words: " + json.dumps([w["text"] for w in words], ensure_ascii=False)
        )
        
        r = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=min(8192, max(500, len(words) * 12 + 200)),
            temperature=0.1
        )
        raw = r.choices[0].message.content or ""
        match = re.search(r"\[[\s\S]*\]", raw)
        if not match:
            raise RuntimeError("Groq did not return a valid JSON word list")
        
        try:
            converted = json.loads(match.group())
        except Exception as json_err:
            raise RuntimeError(f"Failed to parse Groq JSON output: {json_err} | Raw: {raw[:200]}")

        if (not isinstance(converted, list) or
            len(converted) != len(words) or
            not all(isinstance(x, str) for x in converted)):
            raise RuntimeError("Hindi correction word count or format mismatch")
            
        cache[sig] = converted

    i = 0
    for s in data:
        for w in s["words"]:
            w["text"] = converted[i] if i < len(converted) else w["text"]
            i += 1
        s["text"] = " ".join(w["text"] for w in s["words"])
    return data

def make_ass(data, normal, highlight, font_size, width, height):
    def color(c):
        c = c.lstrip("#")
        return "&H00" + c[4:6] + c[2:4] + c[0:2]
    def stamp(t):
        cs = max(0, round(t * 100))
        h, cs = divmod(cs, 360000)
        m, cs = divmod(cs, 6000)
        s, cs = divmod(cs, 100)
        return f"{h}:{m:02}:{s:02}.{cs:02}"

    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,Noto Sans Devanagari,{font_size},{color(highlight)},{color(normal)},&H80000000,&H80000000,-1,0,1,3,1,2,35,35,{max(45,int(height*.09))},1
[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    lines = [head]
    words = [w for s in data for w in s.get("words", []) if w["text"].strip()]
    
    for i in range(0, len(words), 3):
        group = words[i:i+3]
        start = group[0]["start"]
        end = group[-1]["end"]
        parts = []
        for j, w in enumerate(group):
            nxt = group[j+1]["start"] if j+1 < len(group) else w["end"]
            duration = max(1, round((nxt - w["start"]) * 100))
            word = w["text"].replace("{", "(").replace("}", ")")
            parts.append("{\\k" + str(duration) + "}" + word)
        lines.append(
            f"Dialogue: 0,{stamp(start)},{stamp(end)},Default,,0,0,0,,"
            + " ".join(parts)
        )
    return "\n".join(lines)

def pixabay(q, orientation):
    if not PK:
        raise RuntimeError("PIXABAY_API_KEY missing from Secrets")
    r = requests.get(PAPI, params={
        "key": PK, "q": q, "video_type": "film",
        "orientation": orientation, "safesearch": "true", "per_page": 5
    }, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Pixabay HTTP {r.status_code}: {r.text[:300]}")
    hits = r.json().get("hits", [])
    if not hits:
        raise RuntimeError("No Pixabay clips found")
    return hits

def download_clip(hits, path):
    for quality in ["medium", "small", "large"]:
        item = hits[0].get("videos", {}).get(quality, {})
        if item.get("url"):
            r = requests.get(item["url"], timeout=90)
            r.raise_for_status()
            if len(r.content) > 1000:
                Path(path).write_bytes(r.content)
                return
    raise RuntimeError("No downloadable Pixabay clip found")

st.caption("Groq model llama-3.3-70b-versatile is used to ensure pure Devanagari Hindi transcription formatting.")

# 1. Voiceover
st.header("1. Voiceover")
audio = st.file_uploader(
    "Upload voiceover audio or video",
    type=["mp3", "wav", "m4a", "aac", "mp4", "mov", "avi", "webm", "flac"]
)
size = st.selectbox("Whisper model", ["tiny", "base", "small"], index=1)
lang = st.selectbox("Speech language", ["Hindi", "English"])
lang_code = {"Hindi": "hi", "English": "en"}[lang]

if st.button("Transcribe audio", disabled=audio is None):
    try:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "input" + (Path(audio.name).suffix or ".mp3"))
            Path(p).write_bytes(audio.getvalue())
            data, info = transcribe(p, size, lang_code)
            if lang_code == "hi":
                data = correct_hindi(data)
        st.session_state["segments"] = data
        st.success(f"Transcription & Correction complete | Detected: {info.language}")
        st.json([{"start": s["start"], "end": s["end"], "text": s["text"]} for s in data])
    except Exception as e:
        show_error("Transcription failed", e)

# 2. Visuals
st.header("2. Visuals")
query = st.text_input("Visual keywords", "city traffic")
orientation = st.radio("Orientation", ["horizontal", "vertical"], horizontal=True)

if st.button("Search Pixabay"):
    try:
        hits = pixabay(query, orientation)
        st.session_state["hits"] = hits
        st.success(f"Found {len(hits)} clips")
        for h in hits:
            st.write(f'ID {h.get("id")} | {h.get("duration")} sec | {h.get("tags")}')
    except Exception as e:
        show_error("Pixabay search failed", e)

if st.button("Download sample clip"):
    try:
        hits = st.session_state.get("hits") or pixabay(query, orientation)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "sample.mp4")
            download_clip(hits, p)
            b = Path(p).read_bytes()
        st.download_button("Save sample video", b, "pixabay_test.mp4", "video/mp4")
    except Exception as e:
        show_error("Pixabay download failed", e)

# 3. Caption design
st.header("3. Caption design")
normal = st.color_picker("Normal caption colour", "#FFFFFF")
highlight = st.color_picker("Spoken word highlight colour", "#FFFF00")
font_size = st.slider("Caption font size", 30, 64, 48)
ratio = st.radio("Output format", ["9:16 Shorts", "16:9"], horizontal=True)
width, height = (720, 1280) if ratio == "9:16 Shorts" else (1280, 720)

if st.button("Create corrected captions", disabled="segments" not in st.session_state):
    try:
        data = correct_hindi(st.session_state["segments"])
        st.session_state["segments"] = data
        ass = make_ass(data, normal, highlight, font_size, width, height)
        st.success("Corrected Devanagari Hindi captions ready")
        st.download_button(
            "Download coloured captions (.ass)", ass,
            "captions.ass", "text/plain"
        )
        st.text_area("Hindi caption preview", "\n".join(s["text"] for s in data), height=150)
    except Exception as e:
        show_error("Caption correction failed", e)

# 4. FFmpeg check
st.header("4. FFmpeg")
if st.button("Check FFmpeg"):
    try:
        exe = ffmpeg()
        r = run([exe, "-version"], 20)
        if r.returncode:
            raise RuntimeError(r.stderr[-1000:])
        f = run([exe, "-filters"], 30)
        st.success("FFmpeg available")
        st.write("Subtitle filter available:", bool(re.search(r"\b(subtitles|ass)\b", f.stdout)))
    except Exception as e:
        show_error("FFmpeg check failed", e)

# 5. Final video
st.header("5. Generate final video")
if st.button("Generate video", disabled=audio is None):
    try:
        exe = ffmpeg()
        with tempfile.TemporaryDirectory() as d:
            audio_path = os.path.join(d, "voice" + (Path(audio.name).suffix or ".mp3"))
            Path(audio_path).write_bytes(audio.getvalue())

            st.info("Transcribing voiceover...")
            data, info = transcribe(audio_path, size, lang_code)
            st.info("Converting/Correcting Hindi Devanagari script via Groq...")
            data = correct_hindi(data)
            st.session_state["segments"] = data
            duration = max(w["end"] for s in data for w in s["words"])
            if duration <= 0:
                raise RuntimeError("Invalid audio duration")

            st.info("Downloading Pixabay visual...")
            hits = st.session_state.get("hits")
            if not hits:
                hits = pixabay(query, orientation)
            video_path = os.path.join(d, "visual.mp4")
            download_clip(hits, video_path)

            ass_path = os.path.join(d, "captions.ass")
            ass = make_ass(data, normal, highlight, font_size, width, height)
            Path(ass_path).write_text(ass, encoding="utf-8")

            final_path = os.path.join(d, "final.mp4")
            sub = ass_path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
            vf = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1,subtitles='{sub}'"

            cmd = [
                exe, "-y", "-stream_loop", "-1", "-i", video_path,
                "-i", audio_path,
                "-map", "0:v:0", "-map", "1:a:0",
                "-t", str(duration), "-vf", vf,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "27",
                "-threads", "2", "-c:a", "aac", "-b:a", "128k",
                "-shortest", "-movflags", "+faststart", final_path
            ]
            result = run(cmd, 900)
            if result.returncode:
                raise RuntimeError("FFmpeg render failed:\n" + result.stderr[-3000:])
            if not os.path.exists(final_path) or os.path.getsize(final_path) < 1000:
                raise RuntimeError("Final video is missing or empty")

            check = run([exe, "-i", final_path], 20)
            if not re.search(r"Audio:", check.stderr, re.I):
                raise RuntimeError("Final video has no audio stream")
            final_bytes = Path(final_path).read_bytes()

        st.success("Video ready: voiceover + visuals + captions")
        st.video(final_bytes)
        st.download_button(
            "Download final video", final_bytes,
            "final_video.mp4", "video/mp4"
        )
        st.download_button(
            "Download captions", ass, "captions.ass", "text/plain"
        )
    except Exception as e:
        show_error("Video generation failed", e)

st.caption("Ensure GROQ_KEYS and PIXABAY_API_KEY are configured in your Streamlit Cloud Secrets.")
