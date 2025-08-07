# script.py
import os
import re
import json
import logging
import unicodedata
import textwrap
import numpy as np

from dotenv import load_dotenv
import whisperx
from whisperx.diarize import DiarizationPipeline

from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.VideoClip import VideoClip
from PIL import Image, ImageDraw, ImageFont

# =========================
# KONFIG - ustaw raz i zapomnij
# =========================
CLIPS_FOLDER   = "clips"
OUTPUT_FOLDER  = "output"
SRT_FOLDER     = "subtitles"
BADWORDS_PATH  = "badwords.json"

# Wideo / eksport
TARGET_HEIGHT  = 1080        # docelowa wysokość (lanczos)
CRF            = 18          # niżej = lepsza jakość, większy plik
PRESET         = "slow"      # ultrafast..veryslow

# Napisy
FONT_PATH      = None        # np. "C:/Windows/Fonts/arialbd.ttf"
FONTSIZE       = 65
WRAP_WIDTH     = 25

# Cenzura audio
PAD_MS         = 80          # padding ciszy wokół przekleństwa
FADE_MS        = 20          # krótkie fade-in/out
PROB_THRESHOLD = None        # np. 0.4, jeśli słowa mają 'prob'

# =========================
# Logowanie
# =========================
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("captioner")

# =========================
# Ładowanie przekleństw
# =========================
def _normalize(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()

def load_badwords(path=BADWORDS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    bad = {_normalize(w.strip()) for w in raw if str(w).strip()}
    log.info(f"Załadowano {len(bad)} przekleństw z {path}")
    return bad

WORD_RE = re.compile(r"\b[\w\-']+\b", re.UNICODE)

def is_bad_token(token: str, badwords_set) -> bool:
    return _normalize(token) in badwords_set

def censor_token(token: str) -> str:
    n = len(token)
    if n <= 2: return "*" * n
    if n <= 4: return token[0] + "*" * (n - 1)
    return token[0] + "*" * (n - 2) + token[-1]

def censor_text_line(line: str, badwords_set) -> str:
    out, i = [], 0
    for m in WORD_RE.finditer(line):
        out.append(line[i:m.start()])
        tok = m.group(0)
        out.append(censor_token(tok) if is_bad_token(tok, badwords_set) else tok)
        i = m.end()
    out.append(line[i:])
    return "".join(out)

# =========================
# Kolory mówców
# =========================
SPEAKER_COLORS = ["yellow", "cyan", "magenta", "lime", "orange", "deepskyblue", "violet", "salmon"]

def get_speaker_color(speaker: str) -> str:
    try:
        idx = int(str(speaker).replace("SPEAKER_", ""))
    except Exception:
        idx = 0
    return SPEAKER_COLORS[idx % len(SPEAKER_COLORS)]

# =========================
# Karaoke clip (napisy z maską alfa)
# =========================
def make_karaoke_clip(words, seg_start, seg_end, highlight_color,
                      badwords_set, font_path=None, fontsize=65, height=1080, wrap_width=25):
    # cenzura per-token (zachowuje indeksy względem słów)
    censored_tokens = [
        censor_token(w['word'].strip('.,?!:;')) if is_bad_token(w['word'], badwords_set)
        else w['word'].strip('.,?!:;')
        for w in words
    ]
    display_text_censored = " ".join(censored_tokens)

    try:
        font = ImageFont.truetype(font_path or "arialbd.ttf", fontsize)
    except IOError:
        font = ImageFont.load_default()

    wrapped_lines = textwrap.wrap(display_text_censored, width=wrap_width)
    text_width = max((font.getbbox(line)[2] - font.getbbox(line)[0]) for line in wrapped_lines) if wrapped_lines else 0
    line_spacing = fontsize + 10
    text_height = len(wrapped_lines) * line_spacing + 30

    # pozycje highlightu dla każdego tokenu
    word_positions = []
    y_offset = 15
    for line in wrapped_lines:
        words_in_line = line.split()
        line_width = font.getbbox(line)[2] - font.getbbox(line)[0]
        current_x = (int(text_width) - line_width) / 2 + 20
        for wtok in words_in_line:
            w_width = font.getbbox(wtok)[2] - font.getbbox(wtok)[0]
            word_positions.append((current_x, y_offset, w_width))
            current_x += font.getbbox(wtok + " ")[2] - font.getbbox(" ")[0]
        y_offset += line_spacing

    # czasy słów (oryginalne, niecenzurowane tokeny 1:1)
    word_times = [(w['start'], w['end']) for w in words]

    def make_frame_rgba(t):
        # wskaźnik aktywnego słowa
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

        # render RGBA
        img = Image.new("RGBA", (int(text_width) + 40, int(text_height)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # highlight
        if highlight_color and 0 <= idx < len(word_positions):
            x, y, w_width = word_positions[idx]
            draw.rectangle((x - 5, y - 5, x + w_width + 5, y + fontsize + 5), fill=highlight_color)

        # tekst (biały + obrys)
        y_draw = 15
        for line in wrapped_lines:
            line_width = font.getbbox(line)[2] - font.getbbox(line)[0]
            current_x = (int(text_width) - line_width) / 2 + 20
            draw.text((current_x, y_draw), line, font=font, fill="white", stroke_width=3, stroke_fill="black")
            y_draw += line_spacing

        rgba = np.array(img)
        rgb = rgba[..., :3]
        alpha = rgba[..., 3] / 255.0
        return rgb, alpha

    duration = max(0.0, float(seg_end - seg_start))
    def make_color_frame(t): return make_frame_rgba(t)[0]
    def make_mask_frame(t):  return make_frame_rgba(t)[1]

    color_clip = VideoClip(make_color_frame, ismask=False, duration=duration)
    mask_clip  = VideoClip(make_mask_frame,  ismask=True,  duration=duration)
    return color_clip.set_mask(mask_clip).set_start(seg_start).set_position(("center", height * 0.5))

# =========================
# Cenzura audio (wyciszanie)
# =========================
def censor_audio(audio_path, word_segments, output_audio_path, badwords_set,
                 pad_ms=80, fade_ms=20, prob_threshold=None):
    import soundfile as sf
    import librosa

    y, sr = librosa.load(audio_path, sr=None)
    N = len(y)
    mask = np.ones(N, dtype=np.float32)

    def to_samples(t): return int(max(0, min(N, t * sr)))

    for w in word_segments:
        if prob_threshold is not None and float(w.get("prob", 1.0)) < prob_threshold:
            continue
        if not is_bad_token(w["word"], badwords_set):
            continue

        s = to_samples(max(0.0, w['start'] - pad_ms/1000.0))
        e = to_samples(min(N/sr, w['end'] + pad_ms/1000.0))
        if e <= s:
            continue

        fade = int(fade_ms * sr / 1000)
        if fade > 0 and e - s > 2 * fade:
            mask[s:s+fade] *= np.linspace(1.0, 0.0, fade)
            mask[s+fade:e-fade] = 0.0
            mask[e-fade:e] *= np.linspace(0.0, 1.0, fade)
        else:
            mask[s:e] = 0.0

    y_censored = y * mask
    sf.write(output_audio_path, y_censored, sr)
    log.info(f"Zapisano ocenzurowane audio: {output_audio_path}")

# =========================
# Główna logika
# =========================
def add_captions(video_path, output_path, subtitle_path, hf_token, badwords_set):
    temp_audio = "temp_audio.wav"
    temp_censored = "temp_audio_censored.wav"

    video = VideoFileClip(video_path)
    try:
        # wyciągnij audio
        video.audio.write_audiofile(temp_audio)

        # ASR + align + diarization
        device = "cpu"
        model = whisperx.load_model("medium", device=device, compute_type="float32")
        result = model.transcribe(temp_audio, language="pl")

        align_model, metadata = whisperx.load_align_model(language_code="pl", device=device)
        aligned = whisperx.align(result["segments"], align_model, metadata, temp_audio, device)

        diarize_model = DiarizationPipeline(device=device) if not hf_token else DiarizationPipeline(use_auth_token=hf_token, device=device)
        diar = diarize_model(temp_audio)
        final = whisperx.assign_word_speakers(diar, aligned)

        # cenzura audio po wszystkich słowach
        all_words = []
        for seg in final["segments"]:
            all_words.extend(seg.get("words", []) or [])
        censor_audio(temp_audio, all_words, temp_censored, badwords_set, PAD_MS, FADE_MS, PROB_THRESHOLD)

        # crop do 9:16 (środek)
        width, height = video.size
        new_width = height * 9 // 16
        if new_width < width:
            video = video.crop(width=new_width, height=height, x_center=width // 2, y_center=height // 2)

        subtitle_clips = []
        srt_lines = []
        idx = 1

        # podział na subsegmenty po interpunkcji
        def split_on_punct(words_list):
            subs, cur = [], []
            for w in words_list:
                cur.append(w)
                if w['word'].strip().endswith(('.', ',', '?', '!')):
                    subs.append(cur); cur = []
            if cur: subs.append(cur)
            return subs

        for seg in final["segments"]:
            words = seg.get("words", []) or []
            if not words:
                continue

            # karaoke subsegmentami
            for sub in split_on_punct(words):
                seg_start = sub[0]['start']
                seg_end   = sub[-1]['end']
                speakers = [w.get("speaker", "SPEAKER_00") for w in sub]
                main_speaker = max(set(speakers), key=speakers.count) if speakers else "SPEAKER_00"
                color = get_speaker_color(main_speaker)

                clip = make_karaoke_clip(sub, seg_start, seg_end, color, badwords_set, FONT_PATH, FONTSIZE, height, WRAP_WIDTH)
                subtitle_clips.append(clip)

            # SRT z cenzurą
            seg_text = censor_text_line(seg["text"].strip(), badwords_set)
            s = seg['start']; e = seg['end']
            srt_lines.append(
                f"{idx}\n"
                f"{int(s//3600):02}:{int((s%3600)//60):02}:{int(s%60):02},{int((s%1)*1000):03} --> "
                f"{int(e//3600):02}:{int((e%3600)//60):02}:{int(e%60):02},{int((e%1)*1000):03}\n"
                f"{seg_text}\n\n"
            )
            idx += 1

        with open(subtitle_path, "w", encoding="utf-8") as f:
            f.writelines(srt_lines)

        # podmień audio na ocenzurowane i sklej
        base = video.set_audio(AudioFileClip(temp_censored))
        final_video = CompositeVideoClip([base, *subtitle_clips])

        final_video.write_videofile(
            output_path,
            fps=video.fps,
            codec="libx264",
            audio_codec="aac",
            audio_bitrate="192k",
            ffmpeg_params=[
                "-preset", PRESET,
                "-crf", str(CRF),
                "-vf", f"scale=-2:{TARGET_HEIGHT}:flags=lanczos"
            ]
        )
        log.info(f"Zapisano wideo: {output_path}")

    finally:
        for p in (temp_audio, temp_censored):
            if os.path.exists(p):
                os.remove(p)

# =========================
# Start
# =========================
def main():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")  # jeśli masz gated modele pyannote/HF
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(SRT_FOLDER, exist_ok=True)
    badwords_set = load_badwords(BADWORDS_PATH)

    for fn in os.listdir(CLIPS_FOLDER):
        if fn.lower().endswith(".mp4"):
            in_p = os.path.join(CLIPS_FOLDER, fn)
            out_p = os.path.join(OUTPUT_FOLDER, f"captioned_{fn}")
            sub_p = os.path.join(SRT_FOLDER, f"{os.path.splitext(fn)[0]}.srt")
            log.info(f"Processing: {fn}")
            add_captions(in_p, out_p, sub_p, hf_token, badwords_set)
    log.info("Wszystkie klipy zostały przetworzone!")

if __name__ == "__main__":
    main()
