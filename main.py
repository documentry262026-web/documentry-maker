import os
import math
import requests
import google.generativeai as genai
from faster_whisper import WhisperModel
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
import subprocess

# --- 1. CONFIGURATION & API KEYS ---
# यहाँ अपनी पिक्साबे और जेमिनी की एपीआई की (API Key) डालो
PIXABAY_API_KEY = "57871281-bac58345c5fba7b07f0655556"
GEMINI_API_KEY = "YAHAN_APNI_GEMINI_KEY_DALO"

genai.configure(api_key=GEMINI_API_KEY)

# तुम्हारी स्ट्रिक्ट डिमांड के हिसाब से मॉडल्स सेट किए गए हैं
def get_gemini_model():
    try:
        return genai.GenerativeModel('gemini-3.6-flash')
    except Exception:
        print("Fallback: gemini-3.6-flash respond नहीं कर रहा, gemini-flash यूज़ कर रहे हैं।")
        return genai.GenerativeModel('gemini-flash')

model = get_gemini_model()

# --- 2. AUDIO TO TEXT & TIMESTAMPS (WHISPER) ---
def analyze_audio(audio_path):
    print("🎙️ ऑडियो सुना जा रहा है और टाइमस्टैम्प्स निकाले जा रहे हैं (CPU Mode)...")
    # CPU पर चलने वाला फास्ट मॉडल
    whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, info = whisper_model.transcribe(audio_path, word_timestamps=True)
    
    words_data = []
    scenes = []
    current_scene_words = []
    scene_start = 0.0
    
    # हर 8-10 सेकंड का एक 'सीन' (Scene) बनाएंगे ताकि विजुअल्स बदलते रहें
    for segment in segments:
        for word in segment.words:
            words_data.append(word)
            current_scene_words.append(word.word)
            
            if word.end - scene_start >= 8.0:
                scenes.append({
                    "start": scene_start,
                    "end": word.end,
                    "text": " ".join(current_scene_words)
                })
                scene_start = word.end
                current_scene_words = []
                
    if current_scene_words:
        scenes.append({"start": scene_start, "end": words_data[-1].end, "text": " ".join(current_scene_words)})
        
    return words_data, scenes

# --- 3. CONTEXTUAL VISUALS (GEMINI + PIXABAY) ---
def get_video_for_scene(scene_text, duration):
    # जेमिनी से कॉन्टेक्स्ट समझना (जैसे एलियन है तो एलियन का कीवर्ड)
    prompt = f"Based on this Hindi text: '{scene_text}', give me exactly ONE short english search phrase (max 2 words) to find documentary B-roll footage. Only return the words, nothing else."
    response = model.generate_content(prompt)
    keyword = response.text.strip().replace('"', '')
    
    print(f"🎬 सीन के लिए वीडियो खोज रहे हैं: {keyword} ({round(duration, 1)} sec)")
    
    url = f"https://pixabay.com/api/videos/?key={PIXABAY_API_KEY}&q={keyword}&video_type=film&orientation=horizontal&safesearch=true"
    res = requests.get(url).json()
    
    if res['hits']:
        video_url = res['hits'][0]['videos']['medium']['url']
        vid_res = requests.get(video_url)
        filename = f"temp_{keyword.replace(' ', '_')}.mp4"
        with open(filename, 'wb') as f:
            f.write(vid_res.content)
        return filename
    return None

# --- 4. CINEMATIC EFFECTS (ZOOM) ---
def add_ken_burns_effect(clip, duration):
    # डॉक्यूमेंट्री स्टाइल हल्का ज़ूम-इन इफ़ेक्ट
    def zoom(t):
        return 1 + 0.03 * t  # धीरे-धीरे ज़ूम होगा
    
    # वीडियो को रिसाइज़ करना और कट करना
    clip = clip.resize(zoom).set_position(('center', 'center'))
    clip = clip.subclip(0, duration)
    return clip

# --- 5. NON-INTRUSIVE CAPTIONS (ASS FORMAT) ---
def generate_ass_subtitles(words_data, output_file="captions.ass"):
    print("📝 डॉक्यूमेंट्री स्टाइल वर्ड-बाय-वर्ड कैप्शंस तैयार हो रहे हैं...")
    
    # ASS हेडर: MarginV=80 का मतलब कैप्शंस स्क्रीन के एकदम नीचे रहेंगे, बीच में नहीं।
    ass_content = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,55,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,2,2,10,10,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    def format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int((seconds % 1) * 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    # हर 4-5 शब्दों का एक बैच बनाएंगे ताकि स्क्रीन कंजस्टेड ना हो
    batch_size = 5
    for i in range(0, len(words_data), batch_size):
        batch = words_data[i:i+batch_size]
        start_time = format_time(batch[0].start)
        end_time = format_time(batch[-1].end)
        
        # वर्ड-बाय-वर्ड हाईलाइट लॉजिक (पीला रंग: {\c&H00FFFF&})
        for highlight_idx in range(len(batch)):
            text_line = ""
            for idx, w in enumerate(batch):
                clean_word = w.word.strip()
                if idx == highlight_idx:
                    text_line += f"{{\\c&H00FFFF&}}{clean_word}{{\\c&HFFFFFF&}} "
                else:
                    text_line += f"{clean_word} "
            
            w_start = format_time(batch[highlight_idx].start)
            w_end = format_time(batch[highlight_idx].end)
            ass_content += f"Dialogue: 0,{w_start},{w_end},Default,,0,0,0,,{text_line.strip()}\n"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(ass_content)

# --- 6. MAIN ENGINE ---
def create_documentary(audio_file):
    print("🚀 डॉक्यूमेंट्री इंजन चालू हो गया है...")
    words_data, scenes = analyze_audio(audio_file)
    
    video_clips = []
    for scene in scenes:
        duration = scene['end'] - scene['start']
        vid_path = get_video_for_scene(scene['text'], duration)
        
        if vid_path:
            clip = VideoFileClip(vid_path).resize((1920, 1080))
            if clip.duration < duration:
                clip = clip.loop(duration=duration)
            
            clip = add_ken_burns_effect(clip, duration)
            video_clips.append(clip)
    
    print("🎞️ विजुअल्स को जोड़ा जा रहा है...")
    final_video = concatenate_videoclips(video_clips, method="compose")
    
    audio = AudioFileClip(audio_file)
    final_video = final_video.set_audio(audio)
    
    temp_output = "temp_render.mp4"
    final_video.write_videofile(temp_output, fps=24, codec="libx264", audio_codec="aac")
    
    generate_ass_subtitles(words_data)
    
    print("🔥 फाइनल रेंडरिंग: वीडियो, ऑडियो और वर्ड-बाय-वर्ड कैप्शंस को मिलाया जा रहा है (No '0000' Error)...")
    # FFmpeg से कैप्शंस बर्न करना (CPU पर सबसे फास्ट और रिलायबल तरीका)
    final_output = "FINAL_DOCUMENTARY.mp4"
    cmd = f'ffmpeg -y -i {temp_output} -vf "ass=captions.ass" -c:a copy {final_output}'
    subprocess.run(cmd, shell=True)
    
    print(f"✅ काम खत्म! तुम्हारी वीडियो तैयार है: {final_output}")

# --- START SCRIPT ---
if __name__ == "__main__":
    # बस अपनी वॉइसओवर फाइल यहाँ अपलोड करो और नाम बदलो
    VOICEOVER_FILE = "my_voice.mp3" 
    
    if os.path.exists(VOICEOVER_FILE):
        create_documentary(VOICEOVER_FILE)
    else:
        print("❌ एरर: अपनी वॉइसओवर ऑडियो फाइल को फोल्डर में डालो और उसका नाम 'my_voice.mp3' रखो!")
