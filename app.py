import os, re, shutil, subprocess, tempfile
from pathlib import Path

import requests
import streamlit as st
import imageio_ffmpeg
from faster_whisper import WhisperModel
from groq import Groq
from google import genai

st.set_page_config(page_title="Voiceover to Video Studio", page_icon="🎬", layout="centered")
st.title("🎬 Voiceover-to-Video Studio")
st.caption("Searches separate Pixabay clips for each narration scene and burns timed captions into the MP4.")

PIXABAY_URL = "https://pixabay.com/api/videos/"
GROQ_DEFAULT_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
GEMINI_DEFAULT_MODELS = ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.5-flash"]

def secret(name, default=None):
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    return os.environ.get(name, value)

def secret_list(name, env_name=None):
    value = secret(name, None)
    if value is None and env_name:
        value = secret(env_name, None)
    if value is None:
        return []
    if isinstance(value, str):
        value = re.split(r"[,\n]+", value)
    elif not isinstance(value, (list, tuple)):
        value = [value]
    return [str(x).strip() for x in value if str(x).strip()]

def get_keys():
    groq_keys = secret_list("GROQ_KEYS", "GROQ_API_KEYS")
    gemini_keys = secret_list("GEMINI_KEYS", "GEMINI_API_KEYS")
    if not groq_keys and secret("GROQ_API_KEY"):
        groq_keys = [str(secret("GROQ_API_KEY")).strip()]
    if not gemini_keys and secret("GEMINI_API_KEY"):
        gemini_keys = [str(secret("GEMINI_API_KEY")).strip()]
    pixabay_key = str(secret("PIXABAY_API_KEY", "") or "").strip()
    return groq_keys, gemini_keys, pixabay_key

def run_cmd(args, timeout=600):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout or "Command failed")[-2500:])
    return result

def get_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    return imageio_ffmpeg.get_ffmpeg_exe()

def check_subtitles_filter(ffmpeg):
    result = subprocess.run([ffmpeg, "-hide_banner", "-filters"], capture_output=True, text=True)
    output = (result.stdout or "") + (result.stderr or "")
    if " subtitles " not in output and " ass " not in output:
        raise RuntimeError("FFmpeg lacks the subtitles/ASS filter. Add ffmpeg and fonts-dejavu-core to packages.txt, then redeploy.")

def make_keyword(text, groq_keys, gemini_keys):
    prompt = (
        "Create a concise English Pixabay stock-footage search phrase of 2 to 5 words for this narration. "
        "Name concrete visible objects, places, or actions. Return only the phrase, no explanation or quotes.\n"
        f"NARRATION: {text}"
    )
    groq_models = secret_list("GROQ_MODELS") or GROQ_DEFAULT_MODELS
    gemini_models = secret_list("GEMINI_MODELS") or GEMINI_DEFAULT_MODELS
    for key in groq_keys:
        for model in groq_models:
            try:
                client = Groq(api_key=key)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Return only a concise English stock-footage search phrase."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2, max_tokens=30,
                )
                phrase = response.choices[0].message.content.strip().strip("\"'`")
                phrase = re.sub(r"[^A-Za-z0-9 ,'-]", " ", phrase)
                phrase = " ".join(phrase.split()[:5]).strip(" ,")
                if phrase:
                    return phrase
            except Exception:
                pass
    for key in gemini_keys:
        for model in gemini_models:
            try:
                response = genai.Client(api_key=key).models.generate_content(
                    model=model, contents=prompt
                )
                phrase = (response.text or "").strip().strip("\"'`")
                phrase = re.sub(r"[^A-Za-z0-9 ,'-]", " ", phrase)
                phrase = " ".join(phrase.split()[:5]).strip(" ,")
                if phrase:
                    return phrase
            except Exception:
                pass
    raise RuntimeError("All Groq/Gemini attempts failed. Check rotated keys, model IDs, quotas, and network access.")

def find_pixabay_clip(keyword, key, orientation, used_ids):
    params = {
        "key": key, "q": keyword, "lang": "en",
        "video_type": "film", "orientation": orientation,
        "safesearch": "true", "order": "popular", "per_page": 15
    }
    response = requests.get(PIXABAY_URL, params=params, timeout=(10, 30))
    response.raise_for_status()
    hits = response.json().get("hits", [])

    if not hits:
        params.pop("orientation", None)
        response = requests.get(PIXABAY_URL, params=params, timeout=(10, 30))
        response.raise_for_status()
        hits = response.json().get("hits", [])

    query_words = {w.lower() for w in re.findall(r"[A-Za-z]{3,}", keyword)}
    hits.sort(
        key=lambda h: len(
            query_words & {
                w.lower() for w in re.findall(r"[A-Za-z]{3,}", h.get("tags", ""))
            }
        ),
        reverse=True
    )

    for allow_reuse in (False, True):
        for hit in hits:
            if not allow_reuse and str(hit.get("id")) in used_ids:
                continue
            videos = hit.get("videos") or {}
            video = videos.get("large") or videos.get("medium") or videos.get("small")
            if video and video.get("url"):
                return hit, video["url"]
    return None, None

def download_clip(url, path):
    with requests.get(url, stream=True, timeout=(10, 60)) as response:
        response.raise_for_status()
        with open(path, "wb") as f:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    if not os.path.exists(path) or os.path.getsize(path) < 10000:
        raise RuntimeError("Pixabay returned an empty or invalid video download.")

def transcribe(ffmpeg, input_path, workdir, model_size, language):
    wav = os.path.join(workdir, "audio_for_whisper.wav")
    run_cmd([
        ffmpeg, "-y", "-i", input_path, "-vn", "-ac", "1",
        "-ar", "16000", "-c:a", "pcm_s16le", wav
    ])
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    lang = {"Hindi": "hi", "English": "en"}.get(language)
    segments, info = model.transcribe(
        wav, language=lang, word_timestamps=True, vad_filter=True
    )
    words = []
    for segment in segments:
        for word in (segment.words or []):
            text = (word.word or "").strip()
            if text and word.end > word.start:
                words.append({
                    "start": float(word.start),
                    "end": float(word.end),
                    "text": text
                })
    if not words:
        raise RuntimeError("Whisper found no speech. Check the uploaded file and audio.")
    return words, getattr(info, "language", None)

def make_scenes(words, seconds):
    scenes, group = [], []
    group_start = words[0]["start"]
    for word in words:
        if group and word["end"] - group_start >= seconds:
            scenes.append({
                "start": group[0]["start"],
                "end": group[-1]["end"],
                "text": " ".join(w["text"] for w in group)
            })
            group = []
            group_start = word["start"]
        group.append(word)
        if word["end"] - group_start >= seconds and re.search(r"[.!?।]$", word["text"]):
            scenes.append({
                "start": group[0]["start"],
                "end": group[-1]["end"],
                "text": " ".join(w["text"] for w in group)
            })
            group = []
            group_start = word["end"]
    if group:
        scenes.append({
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "text": " ".join(w["text"] for w in group)
        })
    return [s for s in scenes if s["end"] - s["start"] >= 0.25]

def ass_time(value):
    value = max(0.0, float(value))
    return f"{int(value//3600)}:{int(value%3600//60):02d}:{int(value%60):02d}.{int(value%1*100):02d}"

def write_captions(words, width, height, path, highlight=True):
    font_size = 64 if width > height else 54
    margin_v = 100 if width > height else 250
    header = f"""[Script Info]
ScriptType: v4.00+
WrapStyle: 2
ScaledBorderAndShadow: yes
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,{font_size},&H00FFFFFF,&H0000FFFF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,4,2,2,60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    for i in range(0, len(words), 5):
        batch = words[i:i+5]
        for highlight_index, current in enumerate(batch):
            pieces = []
            for j, word in enumerate(batch):
                safe = word["text"].replace("{", r"\{").replace("}", r"\}")
                if highlight and j == highlight_index:
                    pieces.append(r"{\c&H0000FFFF&}" + safe + r"{\c&H00FFFFFF&}")
                else:
                    pieces.append(safe)
            lines.append(
                f"Dialogue: 0,{ass_time(current['start'])},{ass_time(current['end'])},Default,,0,0,0,,{' '.join(pieces)}\n"
            )
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines))

def make_scene(ffmpeg, source, output, duration, width, height, fade):
    vf = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1,fps=30"
    if fade:
        vf += ",fade=t=in:st=0:d=0.25"
    run_cmd([
        ffmpeg, "-y", "-stream_loop", "-1", "-i", source,
        "-t", f"{duration:.3f}", "-vf", vf, "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
        "-pix_fmt", "yuv420p", "-threads", "2", output
    ], timeout=600)
    if not os.path.exists(output) or os.path.getsize(output) < 10000:
        raise RuntimeError("A scene failed to render.")

def render_final(ffmpeg, scenes, audio, captions, output):
    concat = os.path.join(os.path.dirname(output), "concat_list.txt")
    with open(concat, "w", encoding="utf-8") as f:
        for scene in scenes:
            f.write("file '" + str(Path(scene).resolve()).replace("'", "'\\''") + "'\n")
    cap = str(Path(captions).resolve()).replace("\\", "/").replace(":", r"\\:")
    run_cmd([
        ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", concat, "-i", audio,
        "-vf", f"subtitles='{cap}'",
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart", output
    ], timeout=1800)
    if not os.path.exists(output) or os.path.getsize(output) < 50000:
        raise RuntimeError("Final MP4 was not created correctly.")

# ---------------- UI ----------------
ratio = st.radio(
    "Video format", ["YouTube 16:9", "Shorts/Reels 9:16"], horizontal=True
)
width, height = (1920, 1080) if ratio == "YouTube 16:9" else (1080, 1920)
orientation = "horizontal" if width > height else "vertical"
language = st.selectbox("Voiceover language", ["Auto-detect", "Hindi", "English"])
model_size = st.selectbox("Whisper model (CPU)", ["tiny", "base", "small"], index=1)
scene_seconds = st.slider("Approximate visual change (seconds)", 4, 10, 6)
highlight = st.checkbox("Highlight the currently spoken word", value=True)
uploaded = st.file_uploader(
    "Upload voiceover or video with audio",
    type=["mp3", "wav", "m4a", "aac", "flac", "ogg", "mp4", "mov", "mkv", "webm"]
)

with st.expander("Streamlit Secrets configuration"):
    st.code(
        'GROQ_KEYS = ["ROTATED_GROQ_KEY_1", "ROTATED_GROQ_KEY_2"]\n'
        'GEMINI_KEYS = ["ROTATED_GEMINI_KEY_1", "ROTATED_GEMINI_KEY_2"]\n'
        'PIXABAY_API_KEY = "YOUR_PIXABAY_KEY"\n'
        '# Optional: GROQ_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]\n'
        '# Optional: GEMINI_MODELS = ["gemini-3.6-flash", "gemini-3.7-flash"]',
        language="toml"
    )
st.info("Put rotated keys in Streamlit Cloud → Settings → Secrets. Never put real API keys in app.py or a public GitHub repository.")

if st.button("🚀 Generate video", type="primary", disabled=uploaded is None):
    groq_keys, gemini_keys, pixabay_key = get_keys()
    if not groq_keys and not gemini_keys:
        st.error("No Groq/Gemini keys found. Add GROQ_KEYS and/or GEMINI_KEYS to Streamlit Secrets.")
        st.stop()
    if not pixabay_key:
        st.error("PIXABAY_API_KEY is missing from Streamlit Secrets.")
        st.stop()

    ffmpeg = get_ffmpeg()
    try:
        check_subtitles_filter(ffmpeg)
    except Exception as e:
        st.error(str(e))
        st.stop()

    progress = st.progress(0)
    status = st.empty()
    report = []
    try:
        with tempfile.TemporaryDirectory(prefix="voiceover_video_") as workdir:
            ext = uploaded.name.rsplit(".", 1)[-1].lower()
            input_path = os.path.join(workdir, "input_voiceover." + ext)
            with open(input_path, "wb") as f:
                f.write(uploaded.getbuffer())

            status.info("1/4 — Transcribing audio…")
            words, detected_language = transcribe(
                ffmpeg, input_path, workdir, model_size, language
            )
            scenes = make_scenes(words, scene_seconds)
            if not scenes:
                raise RuntimeError("No scenes could be created from the transcription.")

            # Cover pauses between scenes so the visuals do not end too early.
            for scene_index in range(len(scenes) - 1):
                scenes[scene_index]["end"] = max(
                    scenes[scene_index]["end"], scenes[scene_index + 1]["start"]
                )

            report.append(
                f"Detected language: {detected_language or 'unknown'}; "
                f"{len(words)} words; {len(scenes)} scenes."
            )
            progress.progress(10)
            scene_paths, used_ids = [], set()

            for i, scene in enumerate(scenes):
                status.info(f"2/4 — Finding footage for scene {i+1}/{len(scenes)}…")
                keyword = make_keyword(scene["text"], groq_keys, gemini_keys)
                hit, video_url = find_pixabay_clip(
                    keyword, pixabay_key, orientation, used_ids
                )
                if not hit or not video_url:
                    raise RuntimeError(
                        f"No Pixabay clip found for scene {i+1}, search phrase {keyword!r}."
                    )
                used_ids.add(str(hit.get("id")))
                src = os.path.join(workdir, f"source_{i:04d}.mp4")
                out = os.path.join(workdir, f"scene_{i:04d}.mp4")
                download_clip(video_url, src)
                make_scene(
                    ffmpeg, src, out, max(0.35, scene["end"] - scene["start"]),
                    width, height, i > 0
                )
                scene_paths.append(out)
                report.append(
                    f"Scene {i+1}: {scene['start']:.1f}-{scene['end']:.1f}s | "
                    f"Query: {keyword} | Pixabay tags: {hit.get('tags', 'n/a')}"
                )
                progress.progress(10 + int((i+1) / len(scenes) * 65))

            status.info("3/4 — Building timed captions…")
            captions = os.path.join(workdir, "captions.ass")
            write_captions(words, width, height, captions, highlight)
            progress.progress(80)

            status.info("4/4 — Rendering MP4 with captions and original audio…")
            final_path = os.path.join(workdir, "FINAL_VIDEO.mp4")
            render_final(ffmpeg, scene_paths, input_path, captions, final_path)
            progress.progress(100)
            status.success("Video rendered successfully.")
            st.video(final_path)
            with open(final_path, "rb") as f:
                st.download_button(
                    "⬇️ Download final MP4", data=f.read(),
                    file_name="voiceover_stock_video.mp4", mime="video/mp4"
                )
            with st.expander("Scene report — verify visual relevance"):
                st.text("\n".join(report))
    except Exception as e:
        status.empty()
        st.error(f"Video generation failed: {e}")
        st.caption("The app stops on a failed scene instead of silently inserting unrelated generic footage.")
