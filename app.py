import os
import random
import requests
import subprocess
import streamlit as st
import numpy as np
from google import genai
from faster_whisper import WhisperModel
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips

# --- MAGIC FIX FOR PILLOW ANTIALIAS ERROR ---
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

PIXABAY_API_KEY = "57871281-bac58345c5fba7b07f0655556"
GEMINI_API_KEY = "AQ.Ab8RN6IVQYNwTLU-r1GGLqI8Jcf-eNva9G0WfwRy6mUcegUTMQ"
client = genai.Client(api_key=GEMINI_API_KEY)

# --- NEW: BYPASS 'av' ERROR USING DIRECT FFMPEG DECODING ---
def load_audio_direct(file_path):
    cmd = [
        "ffmpeg", "-i", file_path,
        "-f", "s16le", "-ac", "1", "-ar", "16000", "-"
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def analyze_audio(audio_path, progress_bar, status_text):
    status_text.text("🎙️ AI फाइल सुन रहा है और शुद्ध हिंदी लिख रहा है...")
    progress_bar.progress(10)
    
    # Bypass Error: AI को डायरेक्ट numpy array भेजा जा रहा है
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

def get_video_for_scene(scene_text, duration):
    prompt = f"Based on this Hindi text: '{scene_text}', give me exactly ONE short english search phrase (max 2 words) to find documentary B-roll footage. Only return the words."
    try:
        response = client.models.generate_content(model='gemini-3.6-flash', contents=prompt)
    except Exception:
        response = client.models.generate_content(model='gemini-flash', contents=prompt)
        
    keyword = response.text.strip().replace('"', '')
    url = f"https://pixabay.com/api/videos/?key={PIXABAY_API_KEY}&q={keyword}&video_type=film&orientation=horizontal&safesearch=true&order=popular"
    res = requests.get(url).json()
    
    if res.get('hits'):
        video_url = res['hits'][0]['videos']['medium']['url']
        vid_res = requests.get(video_url)
        filename = f"temp_{keyword.replace(' ', '_')}.mp4"
        with open(filename, 'wb') as f:
            f.write(vid_res.content)
        return filename
    return None

def apply_cinematic_effects(clip, duration):
    effect = random.choice(['zoom_in', 'zoom_out'])
    if effect == 'zoom_in':
        clip = clip.resize(lambda t: 1 + 0.03 * t)
    else:
        clip = clip.resize(lambda t: 1.15 - 0.03 * t)
    return clip.set_position(('center', 'center')).subclip(0, duration)

def generate_ass_subtitles(words_data, output_file="captions.ass"):
    ass_content = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,65,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,3,2,10,10,120,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
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
st.title("🎬 Pure Docu-Studio AI (Pro)")
st.write("Upload your Audio or Video file, and let AI build a cinematic documentary!")

uploaded_file = st.file_uploader("यहाँ अपनी फाइल अपलोड करें (Max: 500MB)", type=["mp3", "wav", "mp4", "m4a", "aac", "mov", "avi"])

if uploaded_file and st.button("🔥 Generate Documentary Video"):
    file_ext = uploaded_file.name.split('.')[-1]
    temp_input_file = f"temp_input.{file_ext}"
    
    with open(temp_input_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        words_data, scenes = analyze_audio(temp_input_file, progress_bar, status_text)
        video_clips = []
        
        for i, scene in enumerate(scenes):
            status_text.text(f"🎬 Scene {i+1}/{len(scenes)} के लिए क्लिप्स डाउनलोड हो रही हैं...")
            duration = scene['end'] - scene['start']
            vid_path = get_video_for_scene(scene['text'], duration)
            
            if vid_path:
                clip = VideoFileClip(vid_path).resize((1920, 1080))
                if clip.duration < duration: clip = clip.loop(duration=duration)
                clip = apply_cinematic_effects(clip, duration)
                if i > 0: clip = clip.fadein(0.5)
                video_clips.append(clip)
            progress_bar.progress(10 + int((i+1)/len(scenes) * 40))
            
        status_text.text("🎞️ विजुअल्स, फेट्स, और ऑडियो को जोड़ा जा रहा है...")
        final_video = concatenate_videoclips(video_clips, method="compose")
        final_video = final_video.set_audio(AudioFileClip(temp_input_file))
        
        temp_out = "temp_render.mp4"
        final_video.write_videofile(temp_out, fps=24, codec="libx264", audio_codec="aac", logger=None)
        
        status_text.text("📝 कैप्शंस (Yellow Highlight) वीडियो पर जलाए जा रहे हैं...")
        generate_ass_subtitles(words_data)
        
        final_output = "FINAL_DOCUMENTARY.mp4"
        subprocess.run(f'ffmpeg -y -i {temp_out} -vf "ass=captions.ass" -c:a copy {final_output}', shell=True)
        
        progress_bar.progress(100)
        status_text.success("✅ बधाई हो! तुम्हारी डॉक्यूमेंट्री तैयार है!")
        
        with open(final_output, "rb") as file:
            st.download_button(label="⬇️ Download Final Video", data=file, file_name="My_Documentary.mp4", mime="video/mp4")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
