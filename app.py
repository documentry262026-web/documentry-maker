import os
import glob
import time
import random
import requests
import subprocess
import streamlit as st
import numpy as np
from google import genai
from groq import Groq
from faster_whisper import WhisperModel
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, ColorClip
from moviepy.video.fx.all import crop

# --- MAGIC FIX FOR PILLOW ANTIALIAS ERROR ---
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

# --- API KEYS ---
GROQ_KEYS = [
    "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",
    "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"
]
GROQ_MODELS = ['openai/gpt-oss-120b', 'openai/gpt-oss-20b']

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

# --- SYSTEM FUNCTIONS ---
def cleanup_temp_files():
    print("🧹 Cleaning up RAM & Storage...")
    files_to_delete = glob.glob("temp_*") + glob.glob("processed_scene_*.mp4") + ["captions.ass", "concat_list.txt"]
    for file_path in files_to_delete:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass

def load_audio_direct(file_path):
    cmd = ["ffmpeg", "-i", file_path, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def analyze_audio(audio_path, progress_bar, status_text):
    status_text.text("🎙️ AI फाइल सुन रहा है और शुद्ध हिंदी समझ रहा है...")
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
            if word.end - scene_start >= 7.0:
                scenes.append({"start": scene_start, "end": word.end, "text": " ".join(current_scene_words)})
                scene_start = word.end
                current_scene_words = []
    if current_scene_words:
        scenes.append({"start": scene_start, "end": words_data[-1].end, "text": " ".join(current_scene_words)})
    return words_data, scenes

# --- HYBRID AI ENGINE (GROQ FIRST, GEMINI SECOND) ---
def get_ai_keyword(scene_text, status_text):
    prompt = f"Based on this Hindi text: '{scene_text}', give me exactly ONE short english search phrase (max 2 words) to find documentary B-roll footage. Only return the words, no symbols."
    
    # PHASE 1: GROQ API (Superfast)
    for attempt in range(2): 
        for idx, api_key in enumerate(GROQ_KEYS):
            try:
                client = Groq(api_key=api_key)
                for model_name in GROQ_MODELS:
                    try:
                        chat_completion = client.chat.completions.create(
                            messages=[{"role": "user", "content": prompt}],
                            model=model_name,
                            temperature=0.5,
                            max_tokens=10
                        )
                        keyword = chat_completion.choices[0].message.content.strip().replace('"', '')
                        return keyword
                    except Exception as e:
                        error_msg = str(e).lower()
                        if "429" in error_msg or "rate" in error_msg:
                            status_text.text(f"⏳ Groq Limit on Key {idx+1}. Cooldown (5s)...")
                            time.sleep(5)
                            continue
                        elif "401" in error_msg or "auth" in error_msg:
                            break # Move to next key
                        else:
                            continue
            except Exception:
                continue
    
    # PHASE 2: GEMINI API (Fallback if Groq completely fails)
    status_text.text("🔄 Groq busy/failed. Switching to Gemini Backup Engine...")
    for attempt in range(2): 
        for idx, api_key in enumerate(GEMINI_KEYS):
            try:
                client = genai.Client(api_key=api_key)
                for model_name in GEMINI_MODELS:
                    try:
                        response = client.models.generate_content(model=model_name, contents=prompt)
                        if response and response.text:
                            return response.text.strip().replace('"', '')
                    except Exception as e:
                        error_msg = str(e).lower()
                        if "429" in error_msg or "rate" in error_msg or "quota" in error_msg:
                            time.sleep(5)
                            continue 
                        elif "401" in error_msg or "unauthenticated" in error_msg:
                            break 
                        else:
                            continue
            except Exception:
                continue

    raise Exception("FINAL ERROR: Both Groq and Gemini APIs failed or are blocked. Please check your keys.")

def get_video_for_scene(scene_text, orientation, status_text):
    try:
        keyword = get_ai_keyword(scene_text, status_text)
        url = f"https://pixabay.com/api/videos/?key={PIXABAY_API_KEY}&q={keyword}&video_type=film&orientation={orientation}&safesearch=true&order=popular"
        
        res = requests.get(url, timeout=10).json() 
        
        if not res.get('hits'):
            fallback_url = f"https://pixabay.com/api/videos/?key={PIXABAY_API_KEY}&q={keyword}&video_type=film&safesearch=true&order=popular"
            res = requests.get(fallback_url, timeout=10).json()

        if res.get('hits'):
            video_url = res['hits'][0]['videos']['medium']['url']
            vid_res = requests.get(video_url, timeout=20) 
            filename = f"temp_{keyword.replace(' ', '_')}.mp4"
            with open(filename, 'wb') as f:
                f.write(vid_res.content)
            return filename
    except Exception as e:
        if "FINAL ERROR" in str(e):
            raise e
        print(f"Pixabay Video Fetch Failed: {e}")
        return None
    return None

def apply_cinematic_effects(clip, duration, target_w, target_h):
    effect = random.choice(['zoom_in', 'zoom_out'])
    if effect == 'zoom_in':
        zoomed = clip.resize(lambda t: 1 + 0.03 * t)
    else:
        zoomed = clip.resize(lambda t: 1.15 - 0.03 * t)
    
    zoomed = zoomed.set_position(('center', 'center'))
    comp = CompositeVideoClip([zoomed], size=(target_w, target_h)).subclip(0, duration)
    return comp

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

st.set_page_config(page_title="AI Documentary Maker", page_icon="🎬", layout="centered")
st.title("🎬 Pure Docu-Studio AI (Groq + Gemini)")
st.write("Upload Audio/Video. Superfast AI with Aspect Ratio settings!")

ratio_choice = st.radio("🎥 वीडियो का साइज़ चुनें (Aspect Ratio):", ["16:9 (YouTube Standard)", "9:16 (Shorts / Reels)"], index=0)
target_w, target_h = (1920, 1080) if "16:9" in ratio_choice else (1080, 1920)
orientation = "horizontal" if "16:9" in ratio_choice else "vertical"

uploaded_file = st.file_uploader("यहाँ अपनी फाइल अपलोड करें (Max: 500MB)", type=["mp3", "wav", "mp4", "m4a", "aac", "mov", "avi"])

if uploaded_file and st.button("🔥 Generate Documentary Video"):
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
            status_text.text(f"🎬 Scene {i+1}/{len(scenes)} प्रोसेस हो रहा है... (Powered by Groq)")
            
            duration = scene['end'] - scene['start']
            
            vid_path = get_video_for_scene(scene['text'], orientation, status_text)
            
            if vid_path:
                clip = VideoFileClip(vid_path)
                clip_ratio = clip.w / clip.h
                target_ratio = target_w / target_h
                
                if clip_ratio > target_ratio:
                    clip = clip.resize(height=target_h)
                    clip = crop(clip, x_center=clip.w/2, width=target_w, height=target_h)
                else:
                    clip = clip.resize(width=target_w)
                    clip = crop(clip, y_center=clip.h/2, width=target_w, height=target_h)
                    
                if clip.duration < duration: 
                    clip = clip.loop(duration=duration)
                
                clip = apply_cinematic_effects(clip, duration, target_w, target_h)
                if i > 0: clip = clip.fadein(0.5)
            else:
                clip = ColorClip(size=(target_w, target_h), color=(0,0,0), duration=duration)
                if i > 0: clip = clip.fadein(0.5)
                
            out_path = f"processed_scene_{i}.mp4"
            clip.write_videofile(out_path, fps=24, codec="libx264", audio=False, logger=None)
            processed_files_list.append(out_path)
            
            clip.close()
            del clip
                
            progress_bar.progress(10 + int((i+1)/len(scenes) * 40))
            
        status_text.text("🎞️ सारी क्लिप्स को बिना RAM यूज़ किए जोड़ा जा रहा है...")
        with open("concat_list.txt", "w") as f:
            for pf in processed_files_list:
                f.write(f"file '{pf}'\n")
        
        status_text.text("📝 कैप्शंस (Yellow Highlight) वीडियो पर जलाए जा रहे हैं...")
        generate_ass_subtitles(words_data, target_w, target_h)
        
        final_output = "FINAL_DOCUMENTARY.mp4"
        cmd = f'ffmpeg -y -f concat -safe 0 -i concat_list.txt -i {temp_input_file} -vf "ass=captions.ass" -c:v libx264 -preset fast -c:a aac -shortest {final_output}'
        subprocess.run(cmd, shell=True)
        
        progress_bar.progress(100)
        status_text.success("✅ बधाई हो! तुम्हारी 100% Genuine लंबी डॉक्यूमेंट्री तैयार है!")
        
        with open(final_output, "rb") as file:
            st.download_button(label="⬇️ Download Final Video", data=file, file_name="My_Documentary.mp4", mime="video/mp4")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
    finally:
        cleanup_temp_files()
