import os
import glob
import time
import subprocess
import streamlit as st
import numpy as np
import random
import requests
from google import genai
from groq import Groq
from faster_whisper import WhisperModel
import imageio_ffmpeg 

# --- MAGIC FIX FOR PILLOW ANTIALIAS ERROR ---
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

# --- FFMPEG BYPASS ---
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

# --- STREAMLIT MEMORY ---
if 'video_ready' not in st.session_state:
    st.session_state.video_ready = False
    st.session_state.video_path = None

# --- GROQ API KEYS ---
GROQ_KEYS = [
    "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"
    "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",
    "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"
]
GROQ_MODELS = ['openai/gpt-oss-20b', 'openai/gpt-oss-120b']

# --- GEMINI BACKUP KEYS ---
GEMINI_KEYS = [
    "AQ.Ab8RN6IglBEm55inC7zLAOlN1HQ8Qr4JOCOQrbGwL-1vYpXSXQ",
    "AQ.Ab8RN6JPeFn--zNyZFTnQJzj9M8lMwl4PfX4R1Qt3SFKoQ4D_w",
    "AQ.Ab8RN6IVQYNwTLU-r1GGLqI8Jcf-eNva9G0WfwRy6mUcegUTMQ",
    "AQ.Ab8RN6KCsY2r4jW9nu8C8RbAY8k9vn7SMbNwzAayQBDQj1ahUA",
    "AQ.Ab8RN6IF_rUC9PBAFwfqXOPcjp8pAvQyhdiQ903Nx1Qjs9XlzQ",
    "AQ.Ab8RN6Ka2qaZ8MmcqlRf5_7uWblcVZrdYIXCrQaAu3kekUYoAA",
    "AQ.Ab8RN6IpSUMTKhGEuJK4GWtaeYYE2tezI1wunXOBIomhQAkQaw",
    "AQ.Ab8RN6LS63wOgSxVNGl8y1w7hW0hb-cRwy3QgpmwTLbpgPF2Yg",
    "AQ.Ab8RN6JQgJgnDoXG-lTryMKol-lwYeZhgq3NXMVNouYtu_bssw",
    "AQ.Ab8RN6IYjqQiuJ_4mI91fUoebTEzbHfD_YTCoEAqDl_WwUZEgQ"
]
GEMINI_MODELS = ['gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-flash']

PIXABAY_API_KEY = "57871281-bac58345c5fba7b07f0655556"

def cleanup_temp_files():
    print("🧹 Cleaning up Temp files...")
    files_to_delete = glob.glob("temp_*") + glob.glob("processed_scene_*.mp4") + ["captions.ass", "concat_list.txt"]
    for file_path in files_to_delete:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass

def load_audio_direct(file_path):
    cmd = [FFMPEG_EXE, "-i", file_path, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def analyze_audio(audio_path, progress_bar, status_text):
    status_text.text("🎙️ AI ऑडियो सुन रहा है और सही सीन्स बना रहा है...")
    progress_bar.progress(10)
    
    audio_array = load_audio_direct(audio_path)
    whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, info = whisper_model.transcribe(audio_array, word_timestamps=True, language="hi")
    
    words_data, scenes, current_scene_words = [], [], []
    scene_start = 0.0
    
    for segment in segments:
        for word in segment.words:
            words_data.append(word)
            current_scene_words.append(word.word)
            if word.end - scene_start >= 6.0: # Har 6 second ka ek scene
                scenes.append({"start": scene_start, "end": word.end, "text": " ".join(current_scene_words)})
                scene_start = word.end
                current_scene_words = []
    if current_scene_words:
        scenes.append({"start": scene_start, "end": words_data[-1].end, "text": " ".join(current_scene_words)})
    return words_data, scenes

# --- STRICT AI ENGINE (NO FAKE FALLBACKS) ---
def get_ai_keyword(scene_text, status_text):
    prompt = f"Based on this exact Hindi text snippet: '{scene_text}', give me strictly ONE or TWO English words representing a visual B-roll search query for Pixabay. Return ONLY the words, nothing else."
    
    # Try Groq
    for api_key in GROQ_KEYS:
        try:
            client = Groq(api_key=api_key)
            for model_name in GROQ_MODELS:
                try:
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model_name,
                        temperature=0.3,
                        max_tokens=8
                    )
                    kw = chat_completion.choices[0].message.content.strip().replace('"', '')
                    if kw: return kw
                except Exception:
                    continue
        except Exception:
            continue
            
    # Try Gemini Backup
    for api_key in GEMINI_KEYS:
        try:
            client = genai.Client(api_key=api_key)
            for model_name in GEMINI_MODELS:
                try:
                    response = client.models.generate_content(model=model_name, contents=prompt)
                    if response and response.text:
                        kw = response.text.strip().replace('"', '')
                        if kw: return kw
                except Exception:
                    continue
        except Exception:
            continue
            
    raise Exception("AI Keyword generation failed. Please check your API keys.")

def get_video_for_scene(scene_text, orientation, status_text):
    keyword = get_ai_keyword(scene_text, status_text)
    status_text.text(f"🔍 Searching Pixabay for: '{keyword}'...")
    
    url = f"https://pixabay.com/api/videos/?key={PIXABAY_API_KEY}&q={keyword}&video_type=film&orientation={orientation}&safesearch=true&order=popular"
    
    res = requests.get(url, timeout=10).json() 
    if not res.get('hits'):
        # Fallback query without orientation filter if strict orientation fails
        fallback_url = f"https://pixabay.com/api/videos/?key={PIXABAY_API_KEY}&q={keyword}&video_type=film&safesearch=true&order=popular"
        res = requests.get(fallback_url, timeout=10).json()

    if res.get('hits'):
        video_url = res['hits'][0]['videos']['medium']['url']
        vid_res = requests.get(video_url, timeout=20) 
        filename = f"temp_{abs(hash(keyword))}.mp4"
        with open(filename, 'wb') as f:
            f.write(vid_res.content)
        return filename
    else:
        raise Exception(f"Pixabay par '{keyword}' ke liye koi video nahi mili!")

def generate_ass_subtitles(words_data, target_w, target_h, output_file="captions.ass"):
    font_size = 65 if target_w == 1920 else 55
    margin_v = 120 if target_h == 1080 else 250 
    
    ass_content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {target_w}
PlayResY: {target_h}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,3,2,10,10,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"""
    def format_time(seconds):
        h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
        return f"{h}:{m:02d}:{s:02d}.{int((seconds % 1) * 100):02d}"

    for i in range(0, len(words_data), 5):
        batch = words_data[i:i+5]
        for highlight_idx in range(len(batch)):
            text_line = " ".join([f"{{\\c&H00FFFF&}}{w.word.strip()}{{\\c&HFFFFFF&}}" if idx == highlight_idx else w.word.strip() for idx, w in enumerate(batch)])
            ass_content += f"Dialogue: 0,{format_time(batch[highlight_idx].start)},{format_time(batch[highlight_idx].end)},Default,,0,0,0,,{text_line.strip()}\n"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(ass_content)

# --- WEB GUI ---
st.set_page_config(page_title="AI Documentary Maker", page_icon="🎬", layout="centered")
st.title("🎬 Pure Docu-Studio AI (Strict & Clean)")
st.write("100% Matching Visuals & Burned Subtitles - No Fake Fallbacks.")

ratio_choice = st.radio("🎥 वीडियो का साइज़ चुनें:", ["16:9 (YouTube Standard)", "9:16 (Shorts / Reels)"], index=0)
target_w, target_h = (1920, 1080) if "16:9" in ratio_choice else (1080, 1920)
orientation = "horizontal" if "16:9" in ratio_choice else "vertical"

uploaded_file = st.file_uploader("यहाँ अपनी ऑडियो फाइल अपलोड करें (Max: 500MB)", type=["mp3", "wav", "mp4", "m4a", "aac", "mov", "avi"])

if uploaded_file and st.button("🔥 Generate Strict Documentary"):
    st.session_state.video_ready = False 
    
    file_ext = uploaded_file.name.split('.')[-1]
    temp_input_file = f"temp_input.{file_ext}"
    
    with open(temp_input_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    progress_bar = st.progress(0)
    status_text = st.empty()
    processed_files_list = []
    
    try:
        words_data, scenes = analyze_audio(temp_input_file, progress_bar, status_text)
        
        for i, scene in enumerate(scenes):
            status_text.text(f"🎬 Scene {i+1}/{len(scenes)}: Script text -> '{scene['text'][:30]}...'")
            
            duration = scene['end'] - scene['start']
            vid_path = get_video_for_scene(scene['text'], orientation, status_text)
            out_path = f"processed_scene_{i}.mp4"
            fade_filter = ",fade=t=in:st=0:d=0.3" if i > 0 else ""
            
            if vid_path and os.path.exists(vid_path):
                cmd = f'"{FFMPEG_EXE}" -y -stream_loop -1 -i "{vid_path}" -t {duration} -vf "scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h},setsar=1{fade_filter}" -r 24 -c:v libx264 -preset ultrafast -threads 2 -an "{out_path}"'
                subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
            processed_files_list.append(out_path)
            progress_bar.progress(10 + int((i+1)/len(scenes) * 40))
            
        status_text.text("🎞️ सीन्स को जोड़ा जा रहा है...")
        with open("concat_list.txt", "w") as f:
            for pf in processed_files_list:
                f.write(f"file '{pf}'\n")
        
        status_text.text("📝 कैप्शंस (Yellow Highlight) वीडियो पर जलाए जा रहे हैं...")
        generate_ass_subtitles(words_data, target_w, target_h)
        
        final_output = "FINAL_DOCUMENTARY.mp4"
        
        # FFmpeg Subtitle Burn Command
        cmd = f'"{FFMPEG_EXE}" -y -f concat -safe 0 -i concat_list.txt -i "{temp_input_file}" -vf "ass=captions.ass" -c:v libx264 -preset ultrafast -crf 28 -threads 2 -c:a aac -shortest "{final_output}"'
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if os.path.exists(final_output):
            progress_bar.progress(100)
            status_text.success("✅ डॉक्यूमेंट्री बिल्कुल सही तरीके से तैयार है!")
            st.session_state.video_path = final_output
            st.session_state.video_ready = True
        else:
            raise Exception("FFmpeg final rendering failed.")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    finally:
        cleanup_temp_files()

if st.session_state.video_ready and st.session_state.video_path and os.path.exists(st.session_state.video_path):
    st.markdown("### 🎬 तुम्हारी फाइनल डॉक्यूमेंट्री:")
    st.video(st.session_state.video_path)
    
    with open(st.session_state.video_path, "rb") as file:
        st.download_button(
            label="⬇️ Download Final Video", 
            data=file, 
            file_name="My_Documentary.mp4", 
            mime="video/mp4"
        )
