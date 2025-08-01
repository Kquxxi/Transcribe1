import os
import whisper
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.VideoClip import ImageClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import textwrap

# Załaduj model Whisper (medium balansuje dokładność i prędkość)
model = whisper.load_model("medium")

# Foldery wejścia/wyjścia
input_folder = "clips"
output_folder = "output"
subtitle_folder = "subtitles"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(subtitle_folder, exist_ok=True)

# Funkcja tworząca obraz z tekstem (napisy), z dynamicznym tłem dla aktualnego słowa
def make_text_image(text, highlight_word_index, fontsize=65, font_path=None, color='white', highlight_color='yellow', stroke_color='black', stroke_width=3, max_width=800):
    try:
        font = ImageFont.truetype(font_path or "arialbd.ttf", fontsize)
    except IOError:
        font = ImageFont.load_default()

    wrapped_lines = textwrap.wrap(text, width=25)
    
    text_width = 0
    if wrapped_lines:
        text_width = max((font.getbbox(line)[2] - font.getbbox(line)[0]) for line in wrapped_lines)
    
    line_spacing = fontsize + 10
    text_height = len(wrapped_lines) * line_spacing + 30
    
    img = Image.new("RGBA", (int(text_width) + 40, int(text_height)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    y_offset = 15
    current_word_index = 0

    for line in wrapped_lines:
        words_in_line = line.split()
        
        line_bbox = font.getbbox(line)
        line_width = line_bbox[2] - line_bbox[0]
        current_x = (int(text_width) - line_width) / 2 + 20

        for word in words_in_line:
            word_bbox = font.getbbox(word)
            word_width = word_bbox[2] - word_bbox[0]

            if current_word_index == highlight_word_index:
                highlight_box = (current_x - 5, y_offset - 5, current_x + word_width + 5, y_offset + fontsize + 5)
                draw.rectangle(highlight_box, fill=highlight_color)

            draw.text((current_x, y_offset), word, font=font, fill=color, stroke_width=stroke_width, stroke_fill=stroke_color)
            
            advance = font.getbbox(word + " ")[2] - font.getbbox(" ")[0]
            current_x += advance
            
            current_word_index += 1
            
        y_offset += line_spacing

    return img

# Funkcja generująca napisy ze śledzeniem aktualnie wypowiadanego słowa
def add_captions(video_path, output_path, subtitle_path):
    video = VideoFileClip(video_path)
    audio_tmp = "temp_audio.wav"
    video.audio.write_audiofile(audio_tmp)

    transcription = model.transcribe(audio_tmp, language="pl", word_timestamps=True)

    width, height = video.size
    new_width = height * 9 // 16
    if new_width < width:
        video = video.crop(width=new_width, height=height, x_center=width//2, y_center=height//2)

    subtitle_clips = []
    srt_lines = []

    for idx, segment in enumerate(transcription["segments"], start=1):
        words_in_segment = segment.get("words", [])
        if not words_in_segment:
            continue

        # --- NOWA LOGIKA: DZIELENIE NA SUB-FRAGMENTY ---
        sub_segments = []
        current_sub_segment = []
        for word_info in words_in_segment:
            current_sub_segment.append(word_info)
            # Jeśli słowo kończy się na przecinek lub kropkę, kończymy sub-fragment.
            if word_info['word'].strip().endswith((',', '.')):
                sub_segments.append(current_sub_segment)
                current_sub_segment = []
        
        # Dodajemy ostatni, niedokończony sub-fragment.
        if current_sub_segment:
            sub_segments.append(current_sub_segment)

        # Przetwarzamy każdy sub-fragment osobno.
        for sub_segment_words in sub_segments:
            # Tworzymy czysty tekst dla tego fragmentu, usuwając interpunkcję.
            display_text = " ".join([w['word'].strip('., ') for w in sub_segment_words])

            # Tworzymy klipy dla każdego słowa w tym fragmencie.
            for local_index, word_info in enumerate(sub_segment_words):
                word_start = word_info["start"]
                word_end = word_info["end"]

                # Przekazujemy do funkcji rysującej tylko tekst bieżącego fragmentu.
                img = make_text_image(display_text, highlight_word_index=local_index, fontsize=65)
                array_img = np.array(img)

                subtitle = ImageClip(array_img).with_duration(word_end - word_start)
                subtitle = subtitle.with_start(word_start).with_position(("center", height * 0.5))
                subtitle_clips.append(subtitle)
        
        # Tworzenie napisów SRT (bez zmian)
        segment_text = segment["text"].strip()
        segment_start = segment["start"]
        segment_end = segment["end"]
        start_srt = '{:02}:{:02}:{:02},{:03}'.format(int(segment_start // 3600), int((segment_start % 3600) // 60), int(segment_start % 60), int((segment_start % 1)*1000))
        end_srt = '{:02}:{:02}:{:02},{:03}'.format(int(segment_end // 3600), int((segment_end % 3600) // 60), int(segment_end % 60), int((segment_end % 1)*1000))
        srt_lines.append(f"{idx}\n{start_srt} --> {end_srt}\n{segment_text}\n\n")

    with open(subtitle_path, "w", encoding="utf-8") as f:
        f.writelines(srt_lines)

    final = CompositeVideoClip([video, *subtitle_clips])
    final.write_videofile(output_path, fps=video.fps)

    os.remove(audio_tmp)

# Główna pętla
def main():
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".mp4"):
            in_path = os.path.join(input_folder, filename)
            out_path = os.path.join(output_folder, f"captioned_{filename}")
            subtitle_path = os.path.join(subtitle_folder, f"{os.path.splitext(filename)[0]}.srt")
            print(f"Przetwarzam: {filename}")
            add_captions(in_path, out_path, subtitle_path)
    print("Wszystkie klipy zostały przetworzone!")

if __name__ == '__main__':
    main()