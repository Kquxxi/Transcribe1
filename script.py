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

    # Obliczanie wymiarów obrazka
    text_width = 0
    if wrapped_lines:
        text_width = max(font.getbbox(line)[2] - font.getbbox(line)[0] for line in wrapped_lines)

    line_spacing = fontsize + 10
    text_height = len(wrapped_lines) * line_spacing + 30

    img = Image.new("RGBA", (text_width + 40, text_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    y_offset = 15
    
    # Inicjalizujemy licznik słów
    current_word_index = 0

    for line in wrapped_lines:
        line_width = font.getbbox(line)[2] - font.getbbox(line)[0]
        x_offset = (text_width - line_width) / 2 + 20

        words_in_line = line.split()
        current_x = x_offset

        for word in words_in_line:
            word_width = font.getbbox(word + " ")[2] - font.getbbox(word + " ")[0]

            # WARUNEK PODŚWIETLENIA: Sprawdzamy indeks bieżącego słowa
            if current_word_index == highlight_word_index:
                highlight_box = (current_x - 5, y_offset - 5, current_x + word_width + 5, y_offset + fontsize + 5)
                draw.rectangle(highlight_box, fill=highlight_color)

            draw.text((current_x, y_offset), word + " ", font=font, fill=color, stroke_width=stroke_width, stroke_fill=stroke_color)
            current_x += word_width

            # Zwiększamy licznik dla następnego słowa
            current_word_index += 1
            
        y_offset += line_spacing

    return img

