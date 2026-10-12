import os,re,tempfile,subprocess,traceback
from pathlib import Path
import streamlit as st
import requests

try:
    from groq import Groq
except Exception:
    Groq=None
try:
    from google import genai
except Exception:
    genai=None
try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel=None
try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg=None

st.set_page_config(page_title="AI Video Diagnostic Lab",page_icon="🎬")
st.title("🎬 AI Voiceover to Video — Diagnostic Lab")

def setting(k,d=None):
    try:
        return st.secrets.get(k,os.environ.get(k,d))
    except Exception:
        return os.environ.get(k,d)

def keylist(v):
    if isinstance(v,str):
        v=[v]
    return [str(x).strip() for x in (v or []) if str(x).strip()]

GK=keylist(setting("GROQ_KEYS",[]))
MK=keylist(setting("GEMINI_KEYS",[]))
PK=str(setting("PIXABAY_API_KEY","") or "").strip()
GM=["openai/gpt-oss-20b","openai/gpt-oss-120b"]
MM=["gemini-3.8-flash","gemini-3.7-flash"]
PAPI="https://pixabay.com/api/videos/"

def show_error(title,e):
    st.error(title)
    st.code(f"{type(e).__name__}: {str(e)[:1200]}")

def ffmpeg():
    if not imageio_ffmpeg:
        raise RuntimeError("imageio-ffmpeg is not installed")
    return imageio_ffmpeg.get_ffmpeg_exe()

def run(cmd,timeout=120):
    return subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)

def transcribe(path,model_size,language):
    model=WhisperModel(model_size,device="cpu",compute_type="int8")
    # Force Hindi instead of automatic Urdu/Hindi detection
    segments,info=model.transcribe(
        path,language=language,task="transcribe",beam_size=5
    )
    data=[{"start":float(s.start),"end":float(s.end),
           "text":s.text.strip()} for s in segments if s.text.strip()]
    if not data:
        raise RuntimeError("Whisper returned no speech")
    return data,info

def make_srt(data):
    def stamp(t):
        ms=int(t*1000)
        h,ms=divmod(ms,3600000)
        m,ms=divmod(ms,60000)
        s,ms=divmod(ms,1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    out=[]
    for i,x in enumerate(data,1):
        out.extend([str(i),f'{stamp(x["start"])} --> {stamp(x["end"])}',x["text"],""])
    return "\n".join(out)

def pixabay(q,orientation):
    if not PK:
        raise RuntimeError("PIXABAY_API_KEY missing from Secrets")
    r=requests.get(PAPI,params={
        "key":PK,"q":q,"video_type":"film",
        "orientation":orientation,"safesearch":"true","per_page":5
    },timeout=30)
    if r.status_code!=200:
        raise RuntimeError(f"Pixabay HTTP {r.status_code}: {r.text[:300]}")
    hits=r.json().get("hits",[])
    if not hits:
        raise RuntimeError("No matching videos found")
    return hits

def download_clip(hits,path):
    for quality in ["medium","small","large"]:
        item=hits[0].get("videos",{}).get(quality,{})
        if item.get("url"):
            r=requests.get(item["url"],timeout=90)
            r.raise_for_status()
            if len(r.content)>1000:
                Path(path).write_bytes(r.content)
                return
    raise RuntimeError("No downloadable Pixabay clip found")

# 1. CONFIGURATION
st.header("1. Configuration")
st.write("Groq keys:",len(GK),"| Gemini keys:",len(MK),
         "| Pixabay key configured:",bool(PK))
st.write("Groq SDK:",bool(Groq),"| Gemini SDK:",bool(genai),
         "| Whisper:",bool(WhisperModel),"| FFmpeg package:",bool(imageio_ffmpeg))
with st.expander("Streamlit Secrets format"):
    st.code('''GROQ_KEYS = ["YOUR_NEW_GROQ_KEY_1", "YOUR_NEW_GROQ_KEY_2"]
GEMINI_KEYS = ["YOUR_NEW_GEMINI_KEY_1", "YOUR_NEW_GEMINI_KEY_2"]
PIXABAY_API_KEY = "YOUR_NEW_PIXABAY_KEY"''')

# 2. GROQ API
st.header("2. Groq API test")
if st.button("Test Groq"):
    if not Groq or not GK:
        st.error("Groq SDK missing or no keys configured")
    else:
        results=[]
        for i,k in enumerate(GK,1):
            for model_name in GM:
                try:
                    r=Groq(api_key=k).chat.completions.create(
                        model=model_name,
                        messages=[{"role":"user","content":"Reply OK"}],
                        max_tokens=10)
                    results.append(f"Key {i} | {model_name}: PASS")
                except Exception as e:
                    results.append(f"Key {i} | {model_name}: FAIL - {type(e).__name__}: {str(e)[:180]}")
        st.code("\n".join(results))

# 3. GEMINI API
st.header("3. Gemini API test")
if st.button("Test Gemini"):
    if not genai or not MK:
        st.error("Gemini SDK missing or no keys configured")
    else:
        results=[]
        for i,k in enumerate(MK,1):
            for model_name in MM:
                try:
                    genai.Client(api_key=k).models.generate_content(
                        model=model_name,contents="Reply OK")
                    results.append(f"Key {i} | {model_name}: PASS")
                except Exception as e:
                    results.append(f"Key {i} | {model_name}: FAIL - {type(e).__name__}: {str(e)[:180]}")
        st.code("\n".join(results))

# 4. UPLOAD + WHISPER
st.header("4. Whisper transcription")
audio=st.file_uploader(
    "Upload voiceover audio or video",
    type=["mp3","wav","m4a","aac","mp4","mov","avi","webm","flac"]
)
size=st.selectbox("Whisper model",["tiny","base","small"])
lang=st.selectbox("Transcription language",["Hindi","English"])
lang_code={"Hindi":"hi","English":"en"}[lang]

if st.button("Test Whisper",disabled=audio is None):
    try:
        with tempfile.TemporaryDirectory() as d:
            p=os.path.join(d,"input"+(Path(audio.name).suffix or ".mp3"))
            Path(p).write_bytes(audio.getvalue())
            data,info=transcribe(p,size,lang_code)
        st.session_state["segments"]=data
        st.success(f"Whisper PASS | Requested: {lang} | Detected: {info.language}")
        st.json(data)
        st.download_button("Download transcript",
            "\n".join(x["text"] for x in data),"transcript.txt")
    except Exception as e:
        show_error("Whisper failed",e)

# 5. PIXABAY
st.header("5. Pixabay search and download")
query=st.text_input("Visual search keywords","city traffic")
orientation=st.radio("Video orientation",["horizontal","vertical"],horizontal=True)

if st.button("Search Pixabay"):
    try:
        hits=pixabay(query,orientation)
        st.session_state["hits"]=hits
        st.success(f"Search PASS: {len(hits)} clips")
        for h in hits:
            st.write(f'ID {h.get("id")} | {h.get("duration")} sec | {h.get("tags")}')
    except Exception as e:
        show_error("Pixabay search failed",e)

if st.button("Download first clip"):
    try:
        hits=st.session_state.get("hits") or pixabay(query,orientation)
        with tempfile.TemporaryDirectory() as d:
            p=os.path.join(d,"sample.mp4")
            download_clip(hits,p)
            b=Path(p).read_bytes()
        st.success(f"Download PASS: {len(b):,} bytes")
        st.download_button("Download sample clip",b,"pixabay_test.mp4","video/mp4")
    except Exception as e:
        show_error("Pixabay download failed",e)

# 6. FFMPEG + CAPTIONS
st.header("6. FFmpeg and captions")
if st.button("Test FFmpeg"):
    try:
        exe=ffmpeg()
        r=run([exe,"-version"],20)
        if r.returncode:
            raise RuntimeError(r.stderr[-1000:])
        st.success("FFmpeg PASS")
        st.code(r.stdout[:500])
        f=run([exe,"-filters"],30)
        st.write("Subtitle filter available:",
                 bool(re.search(r"\b(subtitles|ass)\b",f.stdout)))
    except Exception as e:
        show_error("FFmpeg failed",e)

if st.button("Create subtitle file",
             disabled="segments" not in st.session_state):
    try:
        srt=make_srt(st.session_state["segments"])
        st.text_area("Caption preview",srt,height=180)
        st.download_button("Download captions.srt",srt,
                           "captions.srt","application/x-subrip")
        st.success("Caption creation PASS")
    except Exception as e:
        show_error("Caption creation failed",e)

# 7. COMPLETE VIDEO
st.header("7. Complete video test")
ratio=st.radio("Output format",["16:9","9:16"],horizontal=True)

if st.button("Generate complete test video",disabled=audio is None):
    try:
        exe=ffmpeg()
        with tempfile.TemporaryDirectory() as d:
            # Save uploaded file without changing its original format
            audio_path=os.path.join(d,"voice"+(Path(audio.name).suffix or ".mp3"))
            Path(audio_path).write_bytes(audio.getvalue())

            st.info("Step 1/4: Hindi transcription")
            data,info=transcribe(audio_path,size,lang_code)
            duration=max(x["end"] for x in data)
            if duration<=0:
                raise RuntimeError("Invalid audio duration")

            st.info(f"Transcription language: {info.language}")
            st.session_state["segments"]=data

            st.info("Step 2/4: Downloading Pixabay visual")
            hits=pixabay(query,orientation)
            video_path=os.path.join(d,"visual.mp4")
            download_clip(hits,video_path)

            st.info("Step 3/4: Creating captions")
            srt_path=os.path.join(d,"captions.srt")
            Path(srt_path).write_text(make_srt(data),encoding="utf-8")

            st.info("Step 4/4: Rendering video with voiceover")
            w,h=(1280,720) if ratio=="16:9" else (720,1280)
            final_path=os.path.join(d,"final.mp4")
            sub=srt_path.replace("\\","/").replace(":","\\:").replace("'","\\'")
            vf=f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1,subtitles='{sub}'"

            cmd=[
                exe,"-y",
                "-stream_loop","-1","-i",video_path,
                "-i",audio_path,
                "-map","0:v:0",
                "-map","1:a:0",
                "-t",str(duration),
                "-vf",vf,
                "-c:v","libx264","-preset","ultrafast","-crf","27",
                "-threads","2",
                "-c:a","aac","-b:a","128k",
                "-shortest","-movflags","+faststart",
                final_path
            ]
            result=run(cmd,900)
            if result.returncode:
                raise RuntimeError("FFmpeg failed:\n"+result.stderr[-3000:])
            if not os.path.exists(final_path) or os.path.getsize(final_path)<1000:
                raise RuntimeError("Output video missing or empty")

            # Verify final file actually contains an audio stream
            check=run([exe,"-i",final_path],20)
            info_text=check.stderr
            if not re.search(r"Audio:",info_text,re.I):
                raise RuntimeError("Final video has no audio stream")
            final_bytes=Path(final_path).read_bytes()

        st.success("COMPLETE VIDEO PASS — audio stream detected")
        st.video(final_bytes)
        st.download_button("Download final video",final_bytes,
                           "final_video.mp4","video/mp4")
        st.download_button("Download captions",make_srt(data),
                           "captions.srt","application/x-subrip")
    except Exception as e:
        show_error("Complete video test failed",e)

st.divider()
st.caption("API quotas apply. Never expose API keys in source code or screenshots.")
