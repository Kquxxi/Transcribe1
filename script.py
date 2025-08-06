import whisperx
from dotenv import load_dotenv
load_dotenv()
import os
from moviepy.editor import VideoFileClip
from whisperx.diarize import DiarizationPipeline
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.VideoClip import VideoClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import textwrap


# Kolory dla mówców
SPEAKER_COLORS = [
    "yellow", "cyan", "magenta", "lime", "orange", "deepskyblue", "violet", "salmon"
]

def get_speaker_color(speaker):
    idx = int(speaker.replace("SPEAKER_", ""))
    return SPEAKER_COLORS[idx % len(SPEAKER_COLORS)]

# GŁÓWNA FUNKCJA KARAOKE CLIP: jeden klip na subsegment, highlight płynnie za słowem
def make_karaoke_clip(display_text, words, seg_start, seg_end, highlight_color, font_path=None, fontsize=65, height=1080):
    try:
        font = ImageFont.truetype(font_path or "arialbd.ttf", fontsize)
    except IOError:
        font = ImageFont.load_default()

    words_in_display = display_text.split()
    if len(words_in_display) != len(words):
        print(f"UWAGA: Niezgodność liczby słów: {len(words_in_display)} (display) != {len(words)} (words). Dopasowanie highlightu może być niedokładne.")

    wrapped_lines = textwrap.wrap(display_text, width=25)
    text_width = max((font.getbbox(line)[2] - font.getbbox(line)[0]) for line in wrapped_lines) if wrapped_lines else 0
    line_spacing = fontsize + 10
    text_height = len(wrapped_lines) * line_spacing + 30

    word_positions = []
    y_offset = 15
    for line in wrapped_lines:
        words_in_line = line.split()
        line_width = font.getbbox(line)[2] - font.getbbox(line)[0]
        current_x = (int(text_width) - line_width) / 2 + 20
        for w in words_in_line:
            word_width = font.getbbox(w)[2] - font.getbbox(w)[0]
            word_positions.append((current_x, y_offset, word_width))
            current_x += font.getbbox(w + " ")[2] - font.getbbox(" ")[0]
        y_offset += line_spacing

    word_times = [(w['start'], w['end']) for w in words]

    def make_frame(t):
        idx = None
        abs_time = t + seg_start
        for i, (start, end) in enumerate(word_times):
            if start <= abs_time < end:
                idx = i
                break
            if abs_time >= end:
                idx = i
        if idx is None:
            idx = 0

        img = Image.new("RGBA", (int(text_width) + 40, int(text_height)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        if highlight_color and idx < len(word_positions):
            x, y, w_width = word_positions[idx]
            draw.rectangle((x - 5, y - 5, x + w_width + 5, y + fontsize + 5), fill=highlight_color)
        # Tekst
        y_offset_draw = 15
        for line in wrapped_lines:
            line_width = font.getbbox(line)[2] - font.getbbox(line)[0]
            current_x = (int(text_width) - line_width) / 2 + 20
            draw.text((current_x, y_offset_draw), line, font=font, fill="white", stroke_width=3, stroke_fill="black")
            y_offset_draw += line_spacing

        rgba = np.array(img)
        rgb = rgba[...,:3]
        alpha = rgba[...,3] / 255.0  # 0-1 float
        return rgb, alpha

    duration = seg_end - seg_start
    def make_color_frame(t):
        rgb, alpha = make_frame(t)
        return rgb

    def make_mask_frame(t):
        rgb, alpha = make_frame(t)
        return alpha

    color_clip = VideoClip(make_color_frame, ismask=False, duration=duration)
    mask_clip  = VideoClip(make_mask_frame, ismask=True,  duration=duration)
    karaoke_clip = color_clip.set_mask(mask_clip)
    return karaoke_clip.set_start(seg_start).set_position(("center", height * 0.5))
    
def add_captions(video_path, output_path, subtitle_path, hf_token):
    video = VideoFileClip(video_path)
    audio_tmp = "temp_audio.wav"
    video.audio.write_audiofile(audio_tmp)

    device = "cpu"
    model = whisperx.load_model("medium", device=device, compute_type="float32")
    result = model.transcribe(audio_tmp, language="pl")
    align_model, metadata = whisperx.load_align_model(language_code="pl", device=device)
    aligned_result = whisperx.align(result["segments"], align_model, metadata, audio_tmp, device)

    diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarization = diarize_model(audio_tmp)
    final = whisperx.assign_word_speakers(diarization, aligned_result)

    width, height = video.size
    new_width = height * 9 // 16
    if new_width < width:
        video = video.crop(width=new_width, height=height, x_center=width // 2, y_center=height // 2)

    subtitle_clips = []
    srt_lines = []
    idx = 1

    for seg in final["segments"]:
        words = seg.get("words", [])
        if not words:
            continue
        # Podział na subfragmenty (podzielone po interpunkcji)
        sub_segs = []
        cur = []
        for w in words:
            cur.append(w)
            if w['word'].strip().endswith((',', '.')):
                sub_segs.append(cur); cur = []
        if cur: sub_segs.append(cur)

        for sub in sub_segs:
            display_text = " ".join([w['word'].strip('., ') for w in sub])
            seg_start = sub[0]['start']
            seg_end = sub[-1]['end']
            # Najczęstszy speaker w subsegmencie
            speakers = [w.get("speaker", "SPEAKER_00") for w in sub]
            main_speaker = max(set(speakers), key=speakers.count)
            color = get_speaker_color(main_speaker)
            karaoke_clip = make_karaoke_clip(display_text, sub, seg_start, seg_end, color, height=height)
            subtitle_clips.append(karaoke_clip)

        # SRT (bez zmian)
        seg_text = seg["text"].strip()
        s = seg['start']; e = seg['end']
        start_srt = f"{int(s//3600):02}:{int((s%3600)//60):02}:{int(s%60):02},{int((s%1)*1000):03}"
        end_srt   = f"{int(e//3600):02}:{int((e%3600)//60):02}:{int(e%60):02},{int((e%1)*1000):03}"
        srt_lines.append(f"{idx}\n{start_srt} --> {end_srt}\n{seg_text}\n\n")
        idx += 1

    with open(subtitle_path, "w", encoding="utf-8") as f:
        f.writelines(srt_lines)

    final_video = CompositeVideoClip([video, *subtitle_clips])
    final_video.write_videofile(output_path, fps=video.fps)
    os.remove(audio_tmp)

def main():
    input_folder = "clips"
    output_folder = "output"
    subtitle_folder = "subtitles"
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Brak tokena HF_TOKEN w pliku .env!")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(subtitle_folder, exist_ok=True)

    for fn in os.listdir(input_folder):
        if fn.lower().endswith(".mp4"):
            in_p = os.path.join(input_folder, fn)
            out_p = os.path.join(output_folder, f"captioned_{fn}")
            sub_p = os.path.join(subtitle_folder, f"{os.path.splitext(fn)[0]}.srt")
            print("Processing:", fn)
            add_captions(in_p, out_p, sub_p, hf_token)
    print("Wszystkie klipy zostały przetworzone!")

if __name__ == '__main__':
    main()
