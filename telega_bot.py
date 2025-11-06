# telega_asr_tts.py
import os
import json
import wave
import shutil
import subprocess
import tempfile
from pathlib import Path
import traceback

import telebot
import numpy as np
import soundfile as sf
from vosk import Model as VoskModel, KaldiRecognizer
import whisper as openai_whisper

# Optional libs (import failures are allowed — we'll try available TTS backends)
try:
    import torch
except Exception:
    torch = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    from huggingface_hub import login as hf_login
    from transformers import pipeline as hf_pipeline
except Exception:
    hf_pipeline = None
    hf_login = None

# ---------------- CONFIG ----------------
BOT_TOKEN = "8594216688:AAGQ-ZCOgJI-0Dk4cKKwjFbkJPsYyHE4C_8"
VOSK_MODEL_PATH = r"C:\Users\User\Downloads\vosk-model-kz-0.42\vosk-model-kz-0.42"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

TMP_DIR = Path(tempfile.gettempdir()) / "tg_asr_tts_bot"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- INIT ----------------
bot = telebot.TeleBot(BOT_TOKEN)

if not os.path.exists(VOSK_MODEL_PATH):
    raise FileNotFoundError(f"Vosk модельі табылмады: {VOSK_MODEL_PATH}")

print("📦 Vosk моделін жүктеу...")
vosk_model = VoskModel(VOSK_MODEL_PATH)
print("✅ Vosk дайын")

print("📦 Whisper моделін жүктеу (GPU үшін device='cuda'):")
device = "cuda"
try:
    whisper_model = openai_whisper.load_model("large-v3", device=device)
    print("✅ Whisper дайын (GPU)")
except Exception as e:
    print("⚠️ Whisper GPU іске қосылмады, CPU режиміне ауысамыз:", e)
    whisper_model = openai_whisper.load_model("large-v3", device="cpu")
    print("✅ Whisper дайын (CPU)")

# ---------------- TTS INIT ----------------
silero_tts = None
if torch is not None:
    try:
        silero_tts = torch.hub.load('snakers4/silero-models', 'silero_tts', language='multi', speaker='bayan')
        print("✅ Silero TTS дайын (multi/speaker=bayan)")
    except Exception as e:
        print("ℹ Silero TTS жүктелмеді:", e)
        silero_tts = None

hf_tts = None
if HF_TOKEN and hf_pipeline is not None:
    try:
        if hf_login:
            try:
                hf_login(HF_TOKEN)
            except Exception:
                pass
        HF_MODEL_ID = "facebook/mms-tts-kaz"
        hf_tts = hf_pipeline("text-to-speech", model=HF_MODEL_ID, token=HF_TOKEN)
        print("✅ Hugging Face TTS pipeline дайын:", HF_MODEL_ID)
    except Exception as e:
        print("ℹ HF TTS pipeline қолданылмады:", e)
        hf_tts = None

# ---------------- HELPERS ----------------
def run_ffmpeg_convert(in_path: str, out_path: str, sr: int = 16000):
    subprocess.run(
        ["ffmpeg", "-y", "-i", in_path, "-ar", str(sr), "-ac", "1", out_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

def transcribe_vosk(wav_path: str) -> str:
    with wave.open(wav_path, "rb") as wf:
        rec = KaldiRecognizer(vosk_model, wf.getframerate())
        rec.SetWords(False)
        text = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                text += (res.get("text", "") + " ")
        final = json.loads(rec.FinalResult())
        text += final.get("text", "")
    return text.strip()

def transcribe_whisper(wav_path: str) -> str:
    result = whisper_model.transcribe(wav_path, language="kk")
    if isinstance(result, dict):
        return result.get("text", "").strip()
    return str(result).strip()

def save_float_audio_as_pcm16(wav_path: str, float_array: np.ndarray, sr: int):
    if float_array.dtype not in (np.float32, np.float64):
        float_array = float_array.astype(np.float32)
    maxv = np.max(np.abs(float_array)) if float_array.size > 0 else 1.0
    if maxv == 0:
        maxv = 1.0
    int16 = np.int16(float_array / maxv * 32767)
    sf.write(wav_path, int16, sr, subtype="PCM_16")

def tts_silero(text: str, out_wav: str) -> bool:
    if silero_tts is None:
        return False
    try:
        if hasattr(silero_tts, "save_wav"):
            silero_tts.save_wav(text=text, speaker='bayan', sample_rate=48000, audio_path=out_wav)
            return True
        else:
            audio = silero_tts.apply_tts(text=text, speaker='bayan')
            if hasattr(audio, "cpu"):
                audio = audio.cpu().numpy()
            save_float_audio_as_pcm16(out_wav, np.asarray(audio), 48000)
            return True
    except Exception as e:
        print("Silero TTS error:", e)
        return False

def tts_hf_pipeline(text: str, out_wav: str) -> bool:
    if hf_tts is None:
        return False
    try:
        res = hf_tts(text)
        if isinstance(res, dict) and "audio" in res:
            save_float_audio_as_pcm16(out_wav, np.asarray(res["audio"]), res.get("sampling_rate", 48000))
            return True
        return False
    except Exception as e:
        print("HF TTS error:", e)
        return False

def tts_gtts(text: str, out_wav: str) -> bool:
    if gTTS is None:
        return False
    try:
        tmp_mp3 = out_wav + ".mp3"
        try:
            gTTS(text=text, lang="kk").save(tmp_mp3)
        except Exception:
            gTTS(text=text, lang="ru").save(tmp_mp3)
        subprocess.run([
            "ffmpeg", "-y", "-i", tmp_mp3,
            "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "1", out_wav
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(tmp_mp3)
        return os.path.exists(out_wav)
    except Exception as e:
        print("gTTS error:", e)
        return False

def make_tts_any(text: str, out_wav: str) -> bool:
    if tts_silero(text, out_wav):
        return True
    if tts_hf_pipeline(text, out_wav):
        return True
    if tts_gtts(text, out_wav):
        return True
    return False

# ---------------- BOT HANDLERS ----------------
@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    bot.reply_to(message, "Сәлем! Аудио жібер — мен оны мәтінге айналдырып, қайта қазақша сөйлеймін (бірнеше TTS бар).")

@bot.message_handler(content_types=['voice', 'audio'])
def handle_voice(message):
    username = message.from_user.username or message.from_user.first_name or "Бейтаныс"
    user_id = message.from_user.id
    print(f"\n🔵 {username} (id={user_id}) жаңа аудио жіберді.")

    tmp_base = TMP_DIR / f"{message.message_id}"
    tmp_base.mkdir(parents=True, exist_ok=True)
    ogg_path = str(tmp_base / "voice.ogg")
    wav_path = str(tmp_base / "voice_16k.wav")
    reply_wav = str(tmp_base / "reply_48k.wav")

    try:
        file_id = message.voice.file_id if hasattr(message, "voice") else message.audio.file_id
        file_info = bot.get_file(file_id)
        data = bot.download_file(file_info.file_path)
        with open(ogg_path, "wb") as f:
            f.write(data)

        run_ffmpeg_convert(ogg_path, wav_path, sr=16000)

        vosk_text = transcribe_vosk(wav_path)
        try:
            whisper_text = transcribe_whisper(wav_path)
        except Exception:
            whisper_text = ""

        chosen = whisper_text.strip() if (whisper_text and len(whisper_text) > len(vosk_text)) else vosk_text.strip()
        if not chosen:
            chosen = "Сөйлеу табылмады."

        print(f"🗣 Танылған мәтін (Vosk): {vosk_text}")
        print(f"🤖 Танылған мәтін (Whisper): {whisper_text}")
        print(f"✅ Таңдалған нәтиже: {chosen}")

        reply_text = (
            f"🎙 Vosk нәтижесі:\n{vosk_text}\n\n"
            f"🧠 Whisper нәтижесі:\n{whisper_text}\n\n"
            f"✅ Соңғы нәтиже:\n{chosen}"
        )
        bot.reply_to(message, reply_text)

        ok = make_tts_any(chosen, reply_wav)
        if ok:
            with open(reply_wav, "rb") as af:
                bot.send_voice(message.chat.id, af, reply_to_message_id=message.message_id)
                print(f"📤 {username} пайдаланушысына жауап жіберілді.\n")
        else:
            bot.reply_to(message, "⚠️ TTS жасау мүмкін болмады.")

    except Exception as e:
        tb = traceback.format_exc()
        print("Handle voice exception:", e, tb)
        bot.reply_to(message, f"⚠️ Қате: {e}")
    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)

# ---------------- RUN ----------------
if __name__ == "__main__":
    print("🤖 Бот іске қосылды.")
    bot.infinity_polling()
