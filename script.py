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
def make_text_image(text, highlight_word, fontsize=65, font_path=None, color='white', highlight_color='yellow', stroke_color='black', stroke_width=3, max_width=800):
    try:
        font = ImageFont.truetype(font_path or "arialbd.ttf", fontsize)
    except IOError:
        font = ImageFont.load_default()

    wrapped_lines = textwrap.wrap(text, width=25)

    # ---- POCZĄTEK POPRAWIONEGO FRAGMENTU ----

    # Poprzednia metoda (sumowanie wymiarów z `getbbox`) nie uwzględniała stałych odstępów między liniami,
    # co powodowało ucinanie tekstu przy dłuższych sentencjach.

    # Poprawne obliczenie wymiarów obrazka
    text_width = 0
    if wrapped_lines:
        # Obliczanie maksymalnej szerokości linii tekstu
        text_width = max(font.getbbox(line)[2] - font.getbbox(line)[0] for line in wrapped_lines)

    # Obliczanie wysokości na podstawie liczby linii i odstępu używanego w pętli (`fontsize + 10`).
    # Dodatkowy padding (+30) powiększa box, dając więcej przestrzeni wokół tekstu.
    text_height = len(wrapped_lines) * (fontsize + 10) + 30

    # Stworzenie obrazka z odpowiednio dużym buforem pionowym i poziomym
    img = Image.new("RGBA", (text_width + 40, text_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # ---- KONIEC POPRAWIONEGO FRAGMENTU ----

    y_offset = 15 # Ustawienie marginesu górnego dla lepszego wyśrodkowania w pionie
    for line in wrapped_lines:
        line_width = font.getbbox(line)[2] - font.getbbox(line)[0]
        # Wyśrodkowanie każdej linii tekstu w poziomie
        x_offset = (text_width - line_width) / 2 + 20
        words_in_line = line.split()
        current_x = x_offset
        for word in words_in_line:
            word_width = font.getbbox(word + " ")[2] - font.getbbox(word + " ")[0]
            if word.strip("[]") == highlight_word:
                # Rysowanie tła dla podświetlonego słowa
                highlight_box = (current_x - 5, y_offset - 5, current_x + word_width + 5, y_offset + fontsize + 5)
                draw.rectangle(highlight_box, fill=highlight_color)
            # Rysowanie tekstu z obrysem
            draw.text((current_x, y_offset), word + " ", font=font, fill=color, stroke_width=stroke_width, stroke_fill=stroke_color)
            current_x += word_width
        # Przesunięcie do następnej linii ze stałym odstępem
        y_offset += fontsize + 10

    return img

# Funkcja generująca napisy ze śledzeniem aktualnie wypowiadanego słowa
def add_captions(video_path, output_path, subtitle_path):
    video = VideoFileClip(video_path)
    audio_tmp = "temp_audio.wav"
    video.audio.write_audiofile(audio_tmp)

    transcription = model.transcribe(audio_tmp, language="pl", word_timestamps=True)

    # Kadrowanie do 9:16
    width, height = video.size
    new_width = height * 9 // 16
    if new_width < width:
        video = video.crop(width=new_width, height=height, x_center=width//2, y_center=height//2)

    subtitle_clips = []
    srt_lines = []

    for idx, segment in enumerate(transcription["segments"], start=1):
        segment_text = segment["text"].strip()
        words = segment.get("words", [])
        if not words:
            continue

        segment_start = segment["start"]
        segment_end = segment["end"]

        for word_info in words:
            word_text = word_info["word"].strip()
            word_start = word_info["start"]
            word_end = word_info["end"]

            img = make_text_image(segment_text, word_text, fontsize=65)
            array_img = np.array(img)

            subtitle = ImageClip(array_img).with_duration(word_end - word_start)
            subtitle = subtitle.with_start(word_start).with_position(("center", height * 0.5))
            subtitle_clips.append(subtitle)

        # Tworzenie napisów SRT
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